"""Shared runner sampling and completion-aware usable reuse."""
import json
from pathlib import Path

import pytest

from analysis import events, metrics
from analysis import run_experiment as run
from client_simulator.process_raw_traces import write_token_traces

pytestmark = pytest.mark.m3


@pytest.fixture
def config(tmp_path):
    inputs, answers = tmp_path / "tokens.jsonl", tmp_path / "answers.jsonl"
    write_token_traces([{
        "request_id": "bailian:x:1:0",
        "conversation_id": "bailian:root",
        "turn_index": 0,
        "timestamp_s": 10,
        "prompt_token_ids": [1] * 35,
        "response_token_ids": [2],
        "eos_token_id": 0,
        "block_size": 16,
        "request_type": "text"
    }], inputs, answers)
    chat = tmp_path / "chat.jsonl"
    chat_answers = tmp_path / "chat_answers.jsonl"
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
    chat_answers.write_text(
        json.dumps({
            "request_id": "sharegpt:a:0",
            "response": "answer"
        }) + "\n")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({
            "workloads": [{
                "name": "bailian",
                "format": "tokens",
                "trace": inputs.name,
                "responses": answers.name
            }, {
                "name": "sharegpt",
                "format": "chat",
                "trace": chat.name,
                "responses": chat_answers.name
            }]
        }))
    return run.read_config(path)


def test_prepare_freezes_and_merges_mixed_inputs(config, tmp_path):
    out = tmp_path / "experiment"
    sample = run.prepare(config, out)
    assert sample["request_ids"] == ["bailian:x:1:0", "sharegpt:a:0"]
    assert len(events.load_results(sample["responses"])) == 2
    assert run.prepare(config, out) == sample
    config["capacity_tokens"] = 32768
    with pytest.raises(ValueError, match="different settings"):
        run.prepare(config, out)


def test_changed_frozen_file_is_rejected(config, tmp_path):
    out = tmp_path / "experiment"
    sample = run.prepare(config, out)
    with open(sample["responses"], "a") as target:
        target.write("changed")
    with pytest.raises(ValueError, match="Frozen sample changed"):
        run.prepare(config, out)


def test_overlapping_ids_in_mixed_workloads_are_rejected(config, tmp_path):
    config["workloads"][1] = {**config["workloads"][0], "name": "other"}
    with pytest.raises(ValueError, match="repeated IDs"):
        run.prepare(config, tmp_path / "experiment")


def test_completion_id_and_parent_root_mapping():
    owner = "cmpl-bailian:x:10:0-0"
    hitter = "cmpl-bailian:x:11:1-0"
    assert events.client_request_id(owner) == "bailian:x:10:0"
    log = [
        events.Event(0, "insert", 1, 0, owner, "allocate"),
        events.Event(1, "hit", 1, 0, hitter, "allocate")
    ]
    roots = {"bailian:x:10:0": "root", "bailian:x:11:1": "root"}
    assert metrics.hit_sources(log, roots) == {("intra", "allocate"): 1}


def test_missing_api_counts_stay_unknown_and_hits_after_miss_are_not_usable(
        tmp_path):
    raw = {
        "request_id": "bailian:x:1:0",
        "conversation_id": "bailian:root",
        "turn_index": 0,
        "workload": "bailian",
        "prompt_tokens": 40,
        "cached_tokens": None,
        "status": "completed"
    }
    (tmp_path / "client.jsonl").write_text(json.dumps(raw) + "\n")
    log = [{"ev": "header", "block_size": 16}]
    for kind, h in [("miss", 1), ("hit", 2)]:
        log.append({
            "t": 1,
            "ev": kind,
            "hash": h,
            "block": h,
            "req": "cmpl-bailian:x:1:0-0",
            "op": "allocate"
        })
    log.append({"ev": "end"})
    (tmp_path / "events.jsonl").write_text("".join(
        json.dumps(r) + "\n" for r in log))
    (tmp_path / "client.jsonl.metrics.after.txt"
     ).write_text("vllm:gpu_prefix_cache_hit_rate 0.5\n")
    summary = run.analyze(tmp_path)
    assert summary["valid_measurement"]
    assert summary["token_reused_fraction"] is None
    assert summary["usable_reused_fraction"] == 0
    assert summary["requests_with_hits_after_miss"] == 1
    assert summary["sc004_gap_signs"] == {"unknown": 1}
    assert events.load_results(
        tmp_path / "client.jsonl")[0]["cached_tokens"] is None


def test_token_copy_handles_uncapped_and_limited_inputs(config, tmp_path):
    work = config["workloads"][0]
    client, server = Path(work["trace"]), Path(work["responses"])
    row, answer = json.loads(client.read_text()), json.loads(
        server.read_text())
    row["request_id"] = answer["request_id"] = "bailian:x:2:0"
    row["timestamp_s"] += 1
    with client.open("a") as target:
        target.write(json.dumps(row) + "\n")
    with server.open("a") as target:
        target.write(json.dumps(answer) + "\n")
    full = run.prepare(config, tmp_path / "full")
    assert full["workloads"][0]["limit"] == 2
    assert len(events.load_results(full["responses"])) == 3
    work["limit"] = 1
    limited = run.prepare(config, tmp_path / "limited")
    assert limited["workloads"][0]["limit"] == 1
    assert len(events.load_results(limited["responses"])) == 2
