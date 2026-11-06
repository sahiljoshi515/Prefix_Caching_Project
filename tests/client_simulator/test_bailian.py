"""Small, hand-checkable Bailian synthesis and paired-file checks."""
import json
from pathlib import Path

import pytest

from client_simulator.adapters import qwenbailian as adapter
from client_simulator.client_sim import load_tokens
from client_simulator.process_raw_traces import write_token_traces

pytestmark = pytest.mark.m3
FIXTURE = Path(__file__).parent / "fixtures/bailian_small.jsonl"


class Tokenizer:
    all_special_ids = [0]
    eos_token_id = 0

    def get_vocab(self):
        return {str(i): i for i in range(128)}

    def decode(self, ids):
        return "ordinary" if ids != [1] else " "


def requests(seed=0):
    return list(adapter.load_token_requests(FIXTURE, Tokenizer(), seed))


def test_shared_full_blocks_lengths_roots_and_partial_tails():
    rows = requests()
    assert [len(r["prompt_token_ids"]) for r in rows] == [35, 49, 33, 16]
    assert [len(r["response_token_ids"]) for r in rows] == [3, 4, 2, 1]
    assert rows[0]["prompt_token_ids"][:32] == rows[1]["prompt_token_ids"][:32]
    assert rows[0]["prompt_token_ids"][:32] == rows[2]["prompt_token_ids"][:32]
    assert rows[0]["conversation_id"] == rows[1]["conversation_id"]
    assert rows[0]["conversation_id"] != rows[2]["conversation_id"]
    assert [r["timestamp_s"] for r in rows] == [100, 100.1, 100.1, 100.3]
    assert [r["turn_index"] for r in rows] == [0, 1, 0, 0]
    assert all(0 not in r["prompt_token_ids"] + r["response_token_ids"]
               for r in rows)
    assert all(1 not in r["prompt_token_ids"] + r["response_token_ids"]
               for r in rows)
    assert rows[0]["prompt_token_ids"][32] != rows[2]["prompt_token_ids"][32]


def test_seed_is_reproducible_and_changes_tokens():
    assert requests() == requests()
    assert requests(
        1)[0]["prompt_token_ids"] != requests()[0]["prompt_token_ids"]


def test_paired_export_load_and_mismatched_ids(tmp_path):
    client, server = tmp_path / "requests.jsonl", tmp_path / "responses.jsonl"
    assert write_token_traces(requests(), client, server) == 4
    assert list(load_tokens(client, server)) == requests()
    lines = server.read_text().splitlines()
    wrong = json.loads(lines[0])
    wrong["request_id"] = "wrong"
    lines[0] = json.dumps(wrong)
    server.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="Invalid paired"):
        list(load_tokens(client, server))


@pytest.mark.parametrize("change", [{
    "hash_ids": [1]
}, {
    "parent_chat_id": 999
}, {
    "timestamp": -1
}, {
    "input_length": 0
}])
def test_invalid_raw_record_is_rejected(tmp_path, change):
    row = json.loads(FIXTURE.read_text().splitlines()[0])
    row.update(change)
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        list(adapter.load_token_requests(path, Tokenizer()))


def test_block_size_mismatch():
    with pytest.raises(ValueError, match="block size 16"):
        list(adapter.load_token_requests(FIXTURE, Tokenizer(), block_size=32))
