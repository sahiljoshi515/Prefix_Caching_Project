"""AgentBank adapter on a handcrafted fixture (spec 006, SC-002)."""
import json
from pathlib import Path

import pytest

from client_simulator.adapters import agentbank as adapter
from client_simulator.process_raw_traces import write_traces

pytestmark = pytest.mark.m3
FIXTURE = Path(__file__).parent / "fixtures/agentbank_small.jsonl"


def records():
    for line in FIXTURE.read_text().splitlines():
        row = json.loads(line)
        yield row.pop("subset"), row


def by_id(seed=0):
    return {
        c.conversation_id: c
        for c in adapter.conversations(records(), seed)
    }


def roles_and_text(conversation):
    return [(m.role, m.content) for m in conversation.messages]


def test_roles_ids_merging_and_unanswered_turns():
    conversations = by_id()
    assert sorted(conversations) == [
        "agentbank:alfworld:alfworld_0", "agentbank:mbpp:mbpp_7",
        "agentbank:mbpp_before:mbpp_7"
    ]
    assert roles_and_text(conversations["agentbank:alfworld:alfworld_0"]) == [
        ("user", "You are in a room. Your task is to: put a cup in the sink."),
        ("assistant", "Thought: find a cup.\nAction: go to table 1"),
        ("user", "Observation: On the table 1, you see a cup 1."),
        ("assistant", "Action: take cup 1 from table 1"),
    ]
    # The final observation has no answer, so it is not replayed.
    assert roles_and_text(conversations["agentbank:mbpp:mbpp_7"]) == [
        ("user", "Write a function that adds two numbers."),
        ("assistant", "def add(a, b):\n    return a + b"),
    ]
    # Consecutive same-role messages are merged with a blank line.
    assert roles_and_text(conversations["agentbank:mbpp_before:mbpp_7"]) == [
        ("user", "Write a function that adds two numbers.\n\nUse Python."),
        ("assistant", "def add(a, b): return a + b"),
    ]


def test_the_order_is_seeded():
    first = [c.conversation_id for c in adapter.conversations(records(), 0)]
    assert first == [
        c.conversation_id for c in adapter.conversations(records(), 0)
    ]
    orders = {
        tuple(c.conversation_id for c in adapter.conversations(records(), s))
        for s in range(10)
    }
    assert len(orders) > 1


def test_an_unknown_role_is_an_error():
    bad = [("webshop", {
        "id": "w1",
        "conversations": [{
            "from": "system",
            "value": "x"
        }]
    })]
    with pytest.raises(ValueError, match="agentbank:webshop:w1"):
        list(adapter.conversations(bad))


def test_reads_the_parquet_subsets_and_writes_replay_traces(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = {}
    for subset, record in records():
        rows.setdefault(subset, []).append(record)
    message = pa.struct([("from", pa.string()), ("value", pa.string())])
    schema = pa.schema([("id", pa.string()),
                        ("conversations", pa.list_(message))])
    for subset in adapter.SUBSETS:
        table = pa.Table.from_pylist(rows.get(subset, []), schema=schema)
        pq.write_table(table, tmp_path / f"{subset}.parquet")
    adapter.check_download(tmp_path)  # every subset present: no download

    client, server = tmp_path / "c.jsonl", tmp_path / "s.jsonl"
    write_traces(adapter.load_conversations(tmp_path), client, server)
    answers = [json.loads(line) for line in server.read_text().splitlines()]
    assert sorted(a["request_id"] for a in answers) == [
        "agentbank:alfworld:alfworld_0:0", "agentbank:alfworld:alfworld_0:1",
        "agentbank:mbpp:mbpp_7:0", "agentbank:mbpp_before:mbpp_7:0"
    ]
