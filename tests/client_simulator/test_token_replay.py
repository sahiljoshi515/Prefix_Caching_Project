"""Token transport, recorded schedules and mixed global concurrency."""
import asyncio
import io
import json

import httpx
import pytest

from client_simulator import client_sim as client
from client_simulator.process_raw_traces import write_token_traces

pytestmark = pytest.mark.m3


def request(rid="bailian:x:1:0", timestamp=100, prompt=35):
    return {
        "request_id": rid,
        "conversation_id": "bailian:x:root",
        "turn_index": 0,
        "timestamp_s": timestamp,
        "prompt_token_ids": [5] * prompt,
        "response_token_ids": [7, 8],
        "eos_token_id": 0,
        "block_size": 16,
        "request_type": "text"
    }


def transport(seen, wrong=False):

    def respond(req):
        path = req.url.path
        if path == "/health":
            return httpx.Response(200, json={})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test"}]})
        if path == "/tokenize":
            return httpx.Response(200,
                                  json={
                                      "count": 2,
                                      "tokens": [7, 8],
                                      "max_model_len": 128
                                  })
        data = json.loads(req.content)
        seen.append((req.headers["X-Request-Id"], data))
        choice: dict = ({
            "logprobs": {
                "tokens": ["token_id:99"]
                if wrong else ["token_id:7", "token_id:8", "token_id:0"]
            }
        } if path == "/v1/completions" else {
            "message": {
                "content": "answer"
            }
        })
        choice["finish_reason"] = "stop"
        return httpx.Response(200,
                              json={
                                  "choices": [choice],
                                  "usage": {
                                      "prompt_tokens":
                                      len(data.get("prompt", [0] * 32)),
                                      "completion_tokens":
                                      3
                                  }
                              })

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
async def test_recorded_times_exact_tokens_and_unknown_api_cache(monkeypatch):
    seen: list = []
    output = io.StringIO()
    now = [1000.0]
    original_sleep = asyncio.sleep

    async def sleep(delay):
        now[0] += delay
        await original_sleep(0)

    monkeypatch.setattr(client.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(client.asyncio, "sleep", sleep)
    async with httpx.AsyncClient(transport=transport(seen)) as http:
        replay = client.ReplayClient(http, "http://test/v1", "test", output)
        summary = await replay.run_tokens(
            [request(), request("bailian:x:2:0", 100.2)])
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert summary["completed"] == 2
    assert [r["scheduled_s"] for r in rows] == pytest.approx([0, .2])
    assert all(r["answer_match"] and r["cached_tokens"] is None for r in rows)
    assert seen[0][1]["prompt"] == [5] * 35
    assert seen[0][1]["max_tokens"] == 3
    assert not seen[0][1]["add_special_tokens"]


@pytest.mark.asyncio
async def test_context_skip_does_not_suppress_later_token_request():
    output = io.StringIO()
    seen: list = []
    async with httpx.AsyncClient(transport=transport(seen)) as http:
        replay = client.ReplayClient(http, "http://test/v1", "test", output)
        summary = await replay.run_tokens(
            [request(prompt=200),
             request("bailian:x:2:1")])
    assert summary["skipped_context"] == summary["completed"] == 1
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_wrong_tokens_are_failed():
    async with httpx.AsyncClient(transport=transport([], wrong=True)) as http:
        replay = client.ReplayClient(http, "http://test/v1", "test",
                                     io.StringIO())
        assert (await replay.run_tokens([request()]))["failed"] == 1


@pytest.mark.asyncio
async def test_mixed_chat_and_tokens_share_global_limit(tmp_path):
    tokens, answers = tmp_path / "tokens.jsonl", tmp_path / "answers.jsonl"
    write_token_traces([request(), request("bailian:x:2:0")], tokens, answers)
    chat = tmp_path / "chat.jsonl"
    chat.write_text(
        json.dumps({
            "conversation_id":
            "sharegpt:a",
            "messages": [{
                "role": "user",
                "content": "hello"
            }, {
                "role": "assistant",
                "content": "answer"
            }]
        }) + "\n")
    seen: list = []
    active, maximum = 0, 0
    sync = transport(seen)

    async def respond(req):
        nonlocal active, maximum
        if req.url.path.endswith("completions"):
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            active -= 1
        return sync.handler(req)

    works = [{
        "name": "bailian",
        "format": "tokens",
        "trace": str(tokens),
        "responses": str(answers)
    }, {
        "name": "sharegpt",
        "format": "chat",
        "trace": str(chat),
        "conversation_rate": 1e9
    }]
    output = io.StringIO()
    async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)) as http:
        replay = client.ReplayClient(http,
                                     "http://test/v1",
                                     "test",
                                     output,
                                     max_concurrency=1)
        summary = await client.run_workloads(replay, works)
    assert summary["completed"] == 3
    assert maximum == 1
    assert set(summary["workloads"]) == {"bailian", "sharegpt"}
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert {r["workload"] for r in rows} == {"bailian", "sharegpt"}
