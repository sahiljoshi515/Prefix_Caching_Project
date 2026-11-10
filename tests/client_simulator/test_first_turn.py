"""Conversations whose first turns are history only (re-run branches)."""
import io
import json

import httpx
import pytest

from client_simulator import client_sim as client

pytestmark = pytest.mark.m3

MESSAGES = [{
    "role": role,
    "content": role[0] + str(turn)
} for turn in range(3) for role in ("user", "assistant")]


def write_row(tmp_path, **fields):
    path = tmp_path / "conversations.jsonl"
    row = {"conversation_id": "c", "messages": MESSAGES, **fields}
    path.write_text(json.dumps(row) + "\n")
    return path


def test_first_turn_is_read_from_the_trace(tmp_path):
    path = write_row(tmp_path, first_turn=2)
    conversation = next(client.load_conversations(path))
    assert conversation.first_turn == 2


@pytest.mark.parametrize("first_turn", [-1, 3, "1", 1.0, True])
def test_an_invalid_first_turn_is_rejected(tmp_path, first_turn):
    with pytest.raises(ValueError, match="invalid first_turn"):
        path = write_row(tmp_path, first_turn=first_turn)
        list(client.load_conversations(path))


def server(fail):
    sent = []

    def respond(req):
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 3, "max_model_len": 64})
        request_id = req.headers["X-Request-Id"]
        sent.append(request_id)
        if request_id in fail:
            return httpx.Response(500, text="boom")
        choice = {
            "message": {
                "content": "a" + request_id.rsplit(":", 1)[1]
            },
            "finish_reason": "stop"
        }
        body = {"choices": [choice], "usage": {"prompt_tokens": 3}}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(respond), sent


async def replay(tmp_path, mode, fail=()):
    path = write_row(tmp_path, first_turn=1)
    conversation = next(client.load_conversations(path))
    mock, sent = server(fail)
    results = io.StringIO()
    async with httpx.AsyncClient(transport=mock) as http:
        replayer = client.ReplayClient(http,
                                       "http://test/v1",
                                       "test",
                                       results,
                                       mode=mode)
        await replayer.replay_conversation(conversation, 0.0)
    rows = [json.loads(line) for line in results.getvalue().splitlines()]
    return replayer.counts, sent, rows


@pytest.mark.asyncio
async def test_multi_mode_sends_only_the_new_turns(tmp_path):
    counts, sent, rows = await replay(tmp_path, "multi")
    assert sent == ["c:1", "c:2"]
    assert [r["status"] for r in rows] == ["completed", "completed"]
    assert counts["abandoned_turns"] == 0


@pytest.mark.asyncio
async def test_a_failed_first_new_turn_abandons_only_the_rest(tmp_path):
    counts, sent, rows = await replay(tmp_path, "multi", fail={"c:1"})
    assert sent == ["c:1"]
    assert [r["status"] for r in rows] == ["failed"]
    assert counts["abandoned_turns"] == 1


@pytest.mark.asyncio
async def test_single_mode_sends_the_first_new_turn(tmp_path):
    counts, sent, rows = await replay(tmp_path, "single")
    assert sent == ["c:1"]
    assert counts["completed"] == 1
