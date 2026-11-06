"""Check plot populations without depending on local experiment outputs."""
import json

import pytest

from analysis.plot_experiment import load_data
from analysis.run_m2_task2 import write_csv

pytestmark = pytest.mark.m3


@pytest.fixture
def comparison(tmp_path):
    experiment = {
        "config": {
            "capacity_tokens": 65536,
            "block_size": 16
        },
        "checksums": {}
    }
    (tmp_path / "experiment.json").write_text(json.dumps(experiment))
    policies = []
    for policy in ("lru", "fifo", "lfu"):
        name = f"{policy}-bs16-cap65536"
        folder = tmp_path / name
        folder.mkdir()
        policies.append({"name": name, "policy": policy, "status": "finished"})
        summary = {
            "valid_measurement": True,
            "requests_completed": 2,
            "requests_excluded": {
                "skipped_context": 1
            },
            "blocks": 3,
            "event_counts": {
                "hit": 2
            },
            "usable_reused_fraction": .25,
            "block_hit_rate_log": .5
        }
        (folder / "summary.json").write_text(json.dumps(summary))
        (folder / "run.json").write_text(json.dumps({"sample": experiment}))
        raw = [{
            "request_id": "a",
            "status": "completed"
        }, {
            "request_id": "b",
            "status": "completed"
        }, {
            "request_id": "c",
            "status": "skipped_context"
        }]
        (folder / "client.jsonl").write_text("".join(
            json.dumps(r) + "\n" for r in raw))
        write_csv(folder / "requests.csv", [{
            "request_id": "a",
            "prompt_tokens": 32,
            "usable_cached_tokens": 0,
            "usable_cached_fraction": 0
        }, {
            "request_id": "b",
            "prompt_tokens": 32,
            "usable_cached_tokens": 16,
            "usable_cached_fraction": .5
        }])
        write_csv(folder / "blocks.csv", [{
            "hash": 1,
            "hits": 0
        }, {
            "hash": 2,
            "hits": 2
        }, {
            "hash": 3,
            "hits": 0
        }])
        write_csv(folder / "reuse_intervals.csv", [{
            "interval_s": 0
        }, {
            "interval_s": 1.5
        }])
        write_csv(folder / "hit_rate_windows.csv", [{
            "start_s": 0,
            "lookups": 4,
            "hit_rate": .5
        }])
    write_csv(tmp_path / "summary.csv", policies)
    return tmp_path


def test_zero_values_and_exclusion_populations(comparison):
    _, data, _ = load_data(comparison)
    for values in data.values():
        assert values["cached"] == [0, .5]
        assert values["hits"] == [0, 2, 0]
        assert values["intervals"] == [0, 1.5]
        assert values["summary"]["requests_excluded"] == {"skipped_context": 1}


def test_changed_usable_reuse_is_rejected(comparison):
    path = comparison / "lfu-bs16-cap65536/summary.json"
    summary = json.loads(path.read_text())
    summary["usable_reused_fraction"] = .5
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="overall usable reuse"):
        load_data(comparison)


def test_changed_time_windows_are_rejected(comparison):
    path = comparison / "fifo-bs16-cap65536/hit_rate_windows.csv"
    write_csv(path, [{"start_s": 0, "lookups": 4, "hit_rate": .1}])
    with pytest.raises(ValueError, match="Time windows disagree"):
        load_data(comparison)
