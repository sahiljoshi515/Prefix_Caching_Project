"""CC-Bench adapter on a handcrafted Claude Code log (spec 006, SC-002).

Row 2 of the fixture has a session with a tool call, meta and slash-command
messages, an API error and a final unanswered prompt; a /compact boundary
that starts a new context; a sub-agent sidechain; and an orphan answer.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from client_simulator.adapters import ccbench as adapter
from client_simulator.process_raw_traces import write_traces

pytestmark = pytest.mark.m3
FIXTURE = Path(__file__).parent / "fixtures/ccbench_small.jsonl"


def rows():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


def by_id(seed=0):
    return {c.conversation_id: c for c in adapter.conversations(rows(), seed)}


def roles_and_text(conversation):
    return [(m.role, m.content) for m in conversation.messages]


def test_each_root_starts_its_own_conversation():
    assert sorted(by_id()) == [
        "ccbench:2:0", "ccbench:2:1", "ccbench:2:2", "ccbench:3:0"
    ]


def test_session_tool_calls_and_dropped_events():
    assert roles_and_text(by_id()["ccbench:2:0"]) == [
        ("user", "Build a gomoku game."),
        # Thinking is dropped; the response's blocks are one message.
        ("assistant", "I'll plan first.\n\n"
         '<tool_use name="Write">{"file_path": "index.html"}</tool_use>'),
        # The meta caveat, the /model command and the API error are dropped.
        ("user", "<tool_result>File created</tool_result>"),
        ("assistant", "Done."),
        # "Add a win message." is never answered in this thread.
    ]


def test_compact_boundary_starts_a_new_context():
    assert roles_and_text(by_id()["ccbench:2:1"]) == [
        ("user", "This session is being continued from a previous "
         "conversation. Summary: gomoku."),
        ("assistant", "Adding the message.\n\n"
         '<tool_use name="Edit">{"old": "a", "new": "b"}</tool_use>'),
        ("user", "<tool_result>ok\n\n[image]</tool_result>"),
        ("assistant", "Win message added."),
    ]


def test_sidechain_is_its_own_conversation():
    assert roles_and_text(by_id()["ccbench:2:2"]) == [
        ("user", "Search the code for the board size."),
        ("assistant", "It is 15x15."),
    ]


def test_counts_of_dropped_parts(capsys):
    list(adapter.conversations(rows()))
    printed = capsys.readouterr().out
    counts = Counter(json.loads(printed.split("CC-Bench: ", 1)[1]))
    assert counts["dropped_system_events"] == 1
    assert counts["dropped_meta_or_api_error"] == 2
    assert counts["dropped_local_commands"] == 1
    assert counts["unanswered_final_users_removed"] == 1
    assert counts["leading_assistant_messages_removed"] == 1
    assert counts["skipped_threads_without_exchange"] == 1
    assert counts["replay_requests"] == 6


def test_the_order_is_seeded_and_keeps_a_rows_threads_together():
    orders = {
        tuple(c.conversation_id for c in adapter.conversations(rows(), s))
        for s in range(10)
    }
    assert orders == {
        ("ccbench:2:0", "ccbench:2:1", "ccbench:2:2", "ccbench:3:0"),
        ("ccbench:3:0", "ccbench:2:0", "ccbench:2:1", "ccbench:2:2"),
    }


def test_reads_the_parquet_file_and_writes_replay_traces(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    raw = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows()), raw)
    adapter.check_download(raw)  # present: no download

    client, server = tmp_path / "c.jsonl", tmp_path / "s.jsonl"
    write_traces(adapter.load_conversations(raw), client, server)
    answers = [json.loads(line) for line in server.read_text().splitlines()]
    assert len(answers) == 6
    assert {
        "request_id": "ccbench:2:2:0",
        "response": "It is 15x15."
    } in answers
