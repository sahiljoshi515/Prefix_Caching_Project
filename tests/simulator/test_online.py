"""Online-mode tests: a real ``vllm serve`` process driven through the
OpenAI API, with the simulator standing in for the model.

The server is launched twice: once with the default multiprocessing
front-end (``MQLLMEngine``) and once in-process (``AsyncLLMEngine``), since
each picks the executor through its own ``_get_executor_cls``.
"""
import asyncio

import psutil
import pytest
import pytest_asyncio
import requests
from prometheus_client.parser import text_string_to_metric_families

from ..utils import RemoteOpenAIServer
from .conftest import write_trace

MODEL_NAME = "gpt2-sim"
# gpt2 has no chat template, so the server gets a trivial literal one.
CHAT_TEMPLATE = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n"
                 "{% endfor %}assistant:")
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 6
TRACE = {
    **{
        f"cmpl-{i}": f"completion number {i} from the trace."
        for i in range(4)
    },
    **{f"chat-{i}": f"chat reply {i} from the trace."
       for i in range(4)},
    **{f"conc-{i}": f"concurrent answer number {i}"
       for i in range(16)},
    "batch-1": "batched answer.",
}


@pytest.fixture(scope="module")
def trace_file(tmp_path_factory):
    return write_trace(
        tmp_path_factory.mktemp("trace") / "online.jsonl", TRACE)


@pytest.fixture(scope="module",
                params=[
                    pytest.param([], id="mp-frontend"),
                    pytest.param(["--disable-frontend-multiprocessing"],
                                 id="inproc-frontend"),
                ])
def server(request, gpt2_dir, trace_file):
    args = [
        "--served-model-name", MODEL_NAME, "--host", "127.0.0.1",
        "--max-model-len", "512", "--max-num-seqs", "32",
        "--enable-prefix-caching", "--chat-template", CHAT_TEMPLATE,
        "--disable-log-requests", *request.param
    ]
    env = {
        "VLLM_USE_SIMULATOR": "1",
        "VLLM_SIMULATOR_TRACE": trace_file,
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_CPU_KVCACHE_SPACE": "1",
    }
    with RemoteOpenAIServer(gpt2_dir, args, env_dict=env,
                            max_wait_seconds=240) as remote:
        children = psutil.Process(remote.proc.pid).children(recursive=True)
        yield remote
    for child in children:  # Only matters if the server had to be killed.
        if child.is_running():
            child.terminate()


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def completion_kwargs(**overrides):
    kwargs = dict(model=MODEL_NAME,
                  prompt="Hello",
                  max_tokens=32,
                  temperature=0)
    kwargs.update(overrides)
    return kwargs


@pytest.mark.asyncio
async def test_completion_with_x_request_id_returns_trace(client):
    completion = await client.completions.create(**completion_kwargs(
        extra_headers={"X-Request-Id": "cmpl-1"}))
    choice = completion.choices[0]
    assert choice.text == TRACE["cmpl-1"]
    assert choice.finish_reason == "stop"
    assert completion.id == "cmpl-cmpl-1"


@pytest.mark.asyncio
async def test_completion_without_id_stops_at_max_tokens(client):
    completion = await client.completions.create(**completion_kwargs(
        max_tokens=9))
    assert completion.choices[0].finish_reason == "length"
    assert completion.usage.completion_tokens == 9


@pytest.mark.asyncio
async def test_chat_completion_with_request_id_field(client):
    chat = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "Hi"
        }],
        max_tokens=32,
        extra_body={"request_id": "chat-1"})
    assert chat.choices[0].message.content == TRACE["chat-1"]
    assert chat.choices[0].finish_reason == "stop"
    assert chat.id == "chatcmpl-chat-1"


@pytest.mark.asyncio
async def test_chat_header_overrides_body_request_id(client):
    chat = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "Hi"
        }],
        max_tokens=32,
        extra_headers={"X-Request-Id": "chat-2"},
        extra_body={"request_id": "chat-3"})
    assert chat.choices[0].message.content == TRACE["chat-2"]


@pytest.mark.asyncio
async def test_completion_streaming_matches_trace(client):
    stream = await client.completions.create(**completion_kwargs(
        stream=True, extra_headers={"X-Request-Id": "cmpl-2"}))
    text, finish_reasons = "", []
    async for chunk in stream:
        text += chunk.choices[0].text
        if chunk.choices[0].finish_reason is not None:
            finish_reasons.append(chunk.choices[0].finish_reason)
    assert text == TRACE["cmpl-2"]
    assert finish_reasons == ["stop"]


@pytest.mark.asyncio
async def test_chat_streaming_matches_trace(client):
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "Hi"
        }],
        max_tokens=32,
        stream=True,
        extra_body={"request_id": "chat-3"})
    text, finish_reasons = "", []
    async for chunk in stream:
        text += chunk.choices[0].delta.content or ""
        if chunk.choices[0].finish_reason is not None:
            finish_reasons.append(chunk.choices[0].finish_reason)
    assert text == TRACE["chat-3"]
    assert finish_reasons == ["stop"]


@pytest.mark.asyncio
async def test_batch_prompts_share_the_header_id(client):
    completion = await client.completions.create(**completion_kwargs(
        prompt=["a", "b"], extra_headers={"X-Request-Id": "batch-1"}))
    texts = [c.text for c in sorted(completion.choices, key=lambda c: c.index)]
    assert texts == [TRACE["batch-1"]] * 2


@pytest.mark.asyncio
async def test_concurrent_requests_get_their_own_trace(client):
    results = await asyncio.gather(*[
        client.completions.create(**completion_kwargs(
            prompt=f"prompt {i}", extra_headers={"X-Request-Id": f"conc-{i}"}))
        for i in range(16)
    ])
    for i, completion in enumerate(results):
        assert completion.choices[0].text == TRACE[f"conc-{i}"]
        assert completion.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_metrics_report_prefix_cache_hits(client, server):
    for _ in range(3):  # Same long prompt three times -> cache hits.
        await client.completions.create(**completion_kwargs(
            prompt=LONG_PROMPT, extra_headers={"X-Request-Id": "cmpl-3"}))
    hit_rate, families = 0.0, {}
    for _ in range(20):
        families = {
            family.name: family
            for family in text_string_to_metric_families(
                requests.get(server.url_for("metrics")).text)
        }
        hit_rate = families["vllm:gpu_prefix_cache_hit_rate"].samples[0].value
        if hit_rate > 0:
            break
        await asyncio.sleep(0.5)
    assert hit_rate > 0
    assert "vllm:num_preemptions" in families
    config_labels = families["vllm:cache_config_info"].samples[0].labels
    assert config_labels["enable_prefix_caching"] == "True"


@pytest.mark.asyncio
async def test_health_models_and_request_id_echo(client, server):
    assert requests.get(server.url_for("health")).status_code == 200
    models = await client.models.list()
    assert [m.id for m in models.data] == [MODEL_NAME]
    raw = await client.completions.with_raw_response.create(
        **completion_kwargs(extra_headers={"X-Request-Id": "cmpl-9"}))
    assert raw.headers["x-request-id"] == "cmpl-9"
