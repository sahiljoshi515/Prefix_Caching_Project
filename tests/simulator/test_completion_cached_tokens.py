"""``/v1/completions`` reports cached prompt tokens (spec 006 FR-003).

Token-id prompts go through the completions endpoint, which at the base
commit never reported ``prompt_tokens_details``. With
``--enable-prompt-tokens-details`` it now does, as chat does; without the
flag the response is unchanged. A real ``vllm serve`` on the simulator.
"""
import pytest
import requests

from ..utils import RemoteOpenAIServer
from .conftest import write_trace

pytestmark = pytest.mark.m3

MODEL_NAME = "gpt2-sim"
# 64 token ids: four full 16-token blocks.
PROMPT = list(range(1000, 1064))


def completion(server, request_id):
    response = requests.post(server.url_for("v1/completions"),
                             headers={"X-Request-Id": request_id},
                             json={
                                 "model": MODEL_NAME,
                                 "prompt": PROMPT,
                                 "max_tokens": 3,
                                 "temperature": 0,
                             })
    response.raise_for_status()
    return response.json()["usage"]


@pytest.fixture(scope="module",
                params=[
                    pytest.param(["--enable-prompt-tokens-details"],
                                 id="details-on"),
                    pytest.param([], id="details-off"),
                ])
def server(request, gpt2_dir, tmp_path_factory):
    trace = write_trace(
        tmp_path_factory.mktemp("trace") / "trace.jsonl", {
            "tok-1": [11, 12],
            "tok-2": [11, 12],
        })
    args = [
        "--served-model-name", MODEL_NAME, "--host", "127.0.0.1",
        "--max-model-len", "512", "--block-size", "16",
        "--enable-prefix-caching", "--disable-log-requests", *request.param
    ]
    env = {
        "VLLM_USE_SIMULATOR": "1",
        "VLLM_SIMULATOR_TRACE": trace,
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_CPU_KVCACHE_SPACE": "1",
    }
    with RemoteOpenAIServer(gpt2_dir, args, env_dict=env,
                            max_wait_seconds=240) as remote:
        remote.details = bool(request.param)
        yield remote


def test_repeated_token_prompt_reports_its_cached_tokens(server):
    first = completion(server, "tok-1")
    second = completion(server, "tok-2")
    # The simulator replays two tokens and then EOS.
    assert first["completion_tokens"] == 3
    assert first.get("prompt_tokens_details") is None  # nothing cached yet
    if server.details:
        # All four full blocks are cached (vLLM v0 counts every one).
        assert second["prompt_tokens_details"]["cached_tokens"] == 64
    else:
        assert second.get("prompt_tokens_details") is None
