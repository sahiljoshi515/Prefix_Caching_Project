"""Token-level plan replay (specs/006-m3-workload-eviction FR-003).

The server is an httpx mock that answers like the simulator: the recorded
tokens, then EOS.
"""
import asyncio
import io
import json

import httpx
import pytest

from analysis.events import client_request_id, parse_request_id, workload_of
from client_simulator.plan_replay import PlanReplay, group_sessions
from client_simulator.schema import TokenRequest

pytestmark = pytest.mark.m3

MODEL = "m"


def req(rid, turn, prompt, output, arrival=None, after=None):
    return TokenRequest(request_id=rid,
                        session_id=rid.rsplit(":", 1)[0],
                        turn_index=turn,
                        prompt_token_ids=prompt,
                        output_token_ids=output,
                        arrival_s=arrival,
                        after_s=after)


class FakeServer:
    """Records each completion request and its arrival time."""

    def __init__(self, max_model_len=100, cached=3, wrong=()):
        self.max_model_len, self.cached, self.wrong = (max_model_len, cached,
                                                       set(wrong))
        self.calls = []
        self.clock = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200)
        if path == "/v1/models":
            return httpx.Response(200,
                                  json={
                                      "data": [{
                                          "id":
                                          MODEL,
                                          "max_model_len":
                                          self.max_model_len
                                      }]
                                  })
        body = json.loads(request.content)
        rid = request.headers["X-Request-Id"]
        self.calls.append((rid, body, self.clock()))
        generated = body["max_tokens"] - (1 if rid in self.wrong else 0)
        return httpx.Response(200,
                              json={
                                  "choices": [{
                                      "text": "x",
                                      "finish_reason": "stop"
                                  }],
                                  "usage": {
                                      "prompt_tokens": len(body["prompt"]),
                                      "completion_tokens": generated,
                                      "prompt_tokens_details": {
                                          "cached_tokens": self.cached
                                      }
                                  }
                              })


def replay(sessions, server, stop_at=None):
    out = io.StringIO()

    async def go():
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(server.handler)) as http:
            client = PlanReplay(http,
                                "http://test/v1",
                                MODEL,
                                out,
                                stop_at=stop_at)
            server.clock = client.elapsed
            return await client.run(sessions)

    summary = asyncio.run(go())
    rows = [json.loads(line) for line in out.getvalue().splitlines()]
    return summary, {r["request_id"]: r for r in rows}


def test_requests_carry_token_prompts_budget_and_id():
    sessions = group_sessions(
        [req("bailian-chat:1:0", 0, [5, 6, 7], [8, 9], arrival=0.0)])
    server = FakeServer()
    summary, rows = replay(sessions, server)
    [(rid, body, _)] = server.calls
    assert rid == "bailian-chat:1:0"
    assert body["prompt"] == [5, 6, 7]
    assert body["max_tokens"] == 3  # two recorded tokens and EOS
    assert body["model"] == MODEL and body["temperature"] == 0
    row = rows["bailian-chat:1:0"]
    assert row["status"] == "completed"
    assert row["workload"] == "bailian-chat"
    assert row["conversation_id"] == "bailian-chat:1"
    assert (row["prompt_tokens"], row["cached_tokens"]) == (3, 3)
    assert row["cached_fraction"] == 1.0
    assert summary["completed"] == 1


def test_closed_loop_waits_think_time_after_completion():
    sessions = group_sessions([
        req("sharegpt:a:0", 0, [1], [2], arrival=0.0),
        req("sharegpt:a:1", 1, [1, 2], [3], after=0.15),
    ])
    server = FakeServer()
    _, rows = replay(sessions, server)
    first, second = server.calls
    assert second[2] - first[2] >= 0.15
    assert rows["sharegpt:a:1"]["after_s"] == 0.15


def test_open_loop_turn_goes_at_its_time_but_after_its_parent():
    sessions = group_sessions([
        req("bailian-chat:1:0", 0, [1], [2], arrival=0.0),
        req("bailian-chat:1:1", 1, [1, 2], [3], arrival=0.2),
        req("bailian-api:9:0", 0, [4], [5], arrival=0.05),
    ])
    server = FakeServer()
    _, rows = replay(sessions, server)
    times = {rid: t for rid, _, t in server.calls}
    assert times["bailian-api:9:0"] >= 0.05
    assert times["bailian-chat:1:1"] >= 0.2
    assert [rid for rid, _, _ in server.calls
            ] == ["bailian-chat:1:0", "bailian-api:9:0", "bailian-chat:1:1"]
    assert rows["bailian-chat:1:1"]["planned_s"] == 0.2


def test_closed_loop_rows_carry_the_plans_send_time():
    first = req("sharegpt:a:0", 0, [1], [2], arrival=0.0)
    second = req("sharegpt:a:1", 1, [1, 2], [3], after=0.05)
    second.meta["planned_s"] = 0.05
    _, rows = replay(group_sessions([first, second]), FakeServer())
    assert rows["sharegpt:a:0"]["planned_s"] == 0.0
    assert rows["sharegpt:a:1"]["planned_s"] == 0.05


def test_requests_not_ready_by_the_stop_are_not_sent():
    # The guard stops at 0.1 s: the second session's first request (0.2 s)
    # and both of its later turns are recorded as stopped, never sent; the
    # first session is untouched.
    sessions = group_sessions([
        req("sharegpt:a:0", 0, [1], [2], arrival=0.0),
        req("sharegpt:a:1", 1, [1, 2], [3], after=0.01),
        req("sharegpt:b:0", 0, [4], [5], arrival=0.2),
        req("sharegpt:b:1", 1, [4, 5], [6], after=0.01),
        req("bailian-chat:9:0", 0, [7], [8], arrival=0.3),
    ])
    server = FakeServer()
    summary, rows = replay(sessions, server, stop_at=0.1)
    assert [rid
            for rid, _, _ in server.calls] == ["sharegpt:a:0", "sharegpt:a:1"]
    assert summary["completed"] == 2 and summary["stopped"] == 3
    for rid in ("sharegpt:b:0", "sharegpt:b:1", "bailian-chat:9:0"):
        assert rows[rid]["status"] == "stopped"
        assert rows[rid]["sent_s"] is None


def test_wrong_output_length_fails_and_ends_a_closed_loop_session():
    sessions = group_sessions([
        req("sharegpt:a:0", 0, [1], [2, 3], arrival=0.0),
        req("sharegpt:a:1", 1, [1, 2, 3], [4], after=0.0),
    ])
    summary, rows = replay(sessions, FakeServer(wrong={"sharegpt:a:0"}))
    assert rows["sharegpt:a:0"]["status"] == "failed"
    assert "expected 2 and EOS" in rows["sharegpt:a:0"]["error"]
    assert "sharegpt:a:1" not in rows
    assert summary["abandoned_turns"] == 1


def test_context_overflow_is_skipped_and_open_loop_goes_on():
    sessions = group_sessions([
        req("bailian-coder:1:0", 0, [1] * 8, [2, 2], arrival=0.0),
        req("bailian-coder:1:1", 1, [1] * 3, [2], arrival=0.0),
    ])
    server = FakeServer(max_model_len=10)
    summary, rows = replay(sessions, server)
    # 8 prompt + 2 output + EOS = 11 > 10.
    assert rows["bailian-coder:1:0"]["status"] == "skipped_context"
    assert rows["bailian-coder:1:1"]["status"] == "completed"
    assert [rid for rid, _, _ in server.calls] == ["bailian-coder:1:1"]


def test_group_sessions_checks_the_arrival_fields():
    with pytest.raises(ValueError, match="first request needs arrival_s"):
        group_sessions([req("s:a:0", 0, [1], [1], after=1.0)])
    with pytest.raises(ValueError, match="exactly one"):
        group_sessions([
            req("s:a:0", 0, [1], [1], arrival=0.0),
            req("s:a:1", 1, [1], [1], arrival=1.0, after=1.0)
        ])


@pytest.mark.parametrize("engine_id,client_id", [
    ("chatcmpl-sharegpt:a:3", "sharegpt:a:3"),
    ("cmpl-bailian-chat:12:0-0", "bailian-chat:12:0"),
    ("cmpl-agentbank:alfworld-alfworld_0:2-0",
     "agentbank:alfworld-alfworld_0:2"),
    ("cmpl-x", "cmpl-x"),
    ("other", "other"),
])
def test_engine_ids_of_both_endpoints_map_back_to_the_client_id(
        engine_id, client_id):
    assert client_request_id(engine_id) == client_id


def test_completion_ids_parse_into_session_turn_and_workload():
    assert parse_request_id("cmpl-bailian-chat:12:3-0") == ("bailian-chat:12",
                                                            3)
    assert workload_of("cmpl-agentbank:t-1:0-0") == "agentbank"
    assert workload_of("chatcmpl-sharegpt:a:0") == "sharegpt"
