"""CC-Bench adapter on a handcrafted Claude Code log (spec 006, SC-002).

Row 2 of the fixture has a session with a tool call, meta and slash-command
messages, an API error and a final unanswered prompt; a /compact boundary
that starts a new context; a sub-agent sidechain; and an orphan answer.
Row 4 has a branch, a resumed session's copied events (some with the same
uuids, some with new ones), a /exit stored as a list of blocks and a
synthetic API error.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from client_simulator import client_sim as client
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
        "ccbench:2:0", "ccbench:2:1", "ccbench:2:2", "ccbench:3:0",
        "ccbench:4:0", "ccbench:4:0b1"
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


def test_a_resumed_session_continues_its_conversation():
    main = by_id()["ccbench:4:0"]
    # Copies with the same uuids are skipped; copies with new uuids repeat
    # the conversation word for word, so their new turns extend it instead
    # of replaying it.
    assert [m.content for m in main.messages] == [
        "Fix the bug.", "Reading the code.", "Run the tests.", "Tests fail.",
        "Try again.", "Tests pass.", "Commit it.", "Committed.", "Deploy it.",
        "Deployed."
    ]
    assert main.first_turn == 0
    assert [r.request_id for r in client.iter_requests(main)
            ] == [f"ccbench:4:0:{turn}" for turn in range(5)]


def test_a_branch_replays_only_its_new_turns():
    branch = by_id()["ccbench:4:0b1"]
    # The branch's history is its own path: no event of the other branch,
    # and the first turn, already replayed, is history only.
    assert branch.first_turn == 1
    assert [m.content for m in branch.messages] == [
        "Fix the bug.", "Reading the code.", "Run the tests.",
        "Running them now.", "Good.", "All green."
    ]
    assert [r.request_id for r in client.iter_requests(branch)
            ] == ["ccbench:4:0b1:1", "ccbench:4:0b1:2"]
    assert [m.content for m in next(client.iter_requests(branch)).messages
            ] == ["Fix the bug.", "Reading the code.", "Run the tests."]


def test_counts_of_dropped_parts(capsys):
    list(adapter.conversations(rows()))
    printed = capsys.readouterr().out
    counts = Counter(json.loads(printed.split("CC-Bench: ", 1)[1]))
    assert counts["dropped_system_events"] == 1
    # Row 2's meta caveat and API error, and row 4's synthetic error.
    assert counts["dropped_meta_or_api_error"] == 3
    # Row 2's /model, and row 4's list-form /exit and its output.
    assert counts["dropped_local_commands"] == 3
    assert counts["duplicate_events_skipped"] == 2
    assert counts["continued_paths"] == 1
    assert counts["forked_branches"] == 1
    assert counts["unanswered_final_users_removed"] == 1
    assert counts["leading_assistant_messages_removed"] == 1
    assert counts["paths_without_new_turns"] == 1
    assert counts["replay_requests"] == 13


def test_the_order_is_seeded_and_keeps_a_rows_threads_together():
    orders = set()
    for seed in range(10):
        ids = [c.conversation_id for c in adapter.conversations(rows(), seed)]
        assert ids == [
            c.conversation_id for c in adapter.conversations(rows(), seed)
        ]
        row_order = [i.split(":")[1] for i in ids]
        assert row_order == sorted(row_order, key=row_order.index)
        assert sorted(ids, key=ids.index) == ids
        orders.add(tuple(dict.fromkeys(row_order)))
    assert len(orders) > 1


def test_reads_the_parquet_file_and_writes_replay_traces(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    raw = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows()), raw)
    adapter.check_download(raw)  # present: no download

    client_path, server = tmp_path / "c.jsonl", tmp_path / "s.jsonl"
    write_traces(adapter.load_conversations(raw), client_path, server)
    answers = [json.loads(line) for line in server.read_text().splitlines()]
    assert len(answers) == 13
    assert {
        "request_id": "ccbench:2:2:0",
        "response": "It is 15x15."
    } in answers
    ids = {a["request_id"] for a in answers}
    assert "ccbench:4:0b1:1" in ids and "ccbench:4:0b1:0" not in ids
    # The client reads first_turn back and replays the same requests.
    replayed = [
        r.request_id for c in client.load_conversations(client_path)
        for r in client.iter_requests(c)
    ]
    assert sorted(replayed) == sorted(ids)
