"""Replaying a trace's recorded pauses (CC-Bench's turn_gaps_s)."""
import json

import pytest

from client_simulator import client_sim as client
from client_simulator.schema import Conversation, Message

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


def test_pauses_are_read_from_the_trace(tmp_path):
    path = write_row(tmp_path, turn_gaps_s=[None, 0.5, None])
    conversation = next(client.load_conversations(path))
    assert conversation.turn_gaps_s == [None, 0.5, None]


@pytest.mark.parametrize(
    "gaps",
    [[None, 1.0], [None, -1.0, 0], [None, "1", 0], [None, True, 0], {}])
def test_invalid_pauses_are_rejected(tmp_path, gaps):
    with pytest.raises(ValueError, match="invalid turn_gaps_s"):
        list(client.load_conversations(write_row(tmp_path, turn_gaps_s=gaps)))


def replayer(**options):
    return client.ReplayClient(None,
                               "http://test/v1",
                               "test",
                               None,
                               think_time=5,
                               **options)


CONVERSATION = Conversation("c", [Message(**m) for m in MESSAGES],
                            turn_gaps_s=[None, 0.25, None])


def test_think_time_is_used_unless_recorded_pauses_are_asked_for():
    assert replayer().pause_before(CONVERSATION, 1) == 5


def test_a_recorded_pause_replaces_the_think_time():
    replay = replayer(recorded_pauses=True)
    assert replay.pause_before(CONVERSATION, 1) == 0.25
    # A turn without a timestamp falls back to the think time.
    assert replay.pause_before(CONVERSATION, 2) == 5


def test_long_pauses_are_capped_and_counted():
    replay = replayer(recorded_pauses=True, max_pause=1)
    assert replay.pause_before(CONVERSATION, 1) == 0.25
    assert replay.pause_before(CONVERSATION, 2) == 1
    assert replay.counts["pauses_capped"] == 1
