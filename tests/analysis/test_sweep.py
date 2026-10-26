"""The Task 3 sweep without a server (specs/005-m2-cache-tuning, US2).

Grid expansion, token capacity to block count, feasibility, resuming, the
summary rows and the eviction metrics (FR-007), worked out by hand. Replays
are faked by writing Task 2's handcrafted run folder.
"""
import csv
import json

import pytest

from analysis import metrics as m
from analysis import run_m2_task3 as task3
from analysis import sweep
from analysis.sweep import Config

from .test_metrics import LOG
from .test_run_m2_task2 import conversation, make_run, write_jsonl

pytestmark = pytest.mark.m2

# --- Eviction metrics (FR-007) ----------------------------------------------


def test_eviction_count_and_rate():
    # LOG: 4 misses, 4 hits and 1 eviction (hash 4 at t=7).
    assert m.eviction_count(LOG) == 1
    assert m.eviction_rate(LOG) == 1000 * 1 / 8


def test_no_lookups_and_no_evictions():
    assert m.eviction_count([]) == 0
    assert m.eviction_rate([]) is None
    no_evictions = [e for e in LOG if e.ev != "evict"]
    assert m.eviction_count(no_evictions) == 0
    assert m.eviction_rate(no_evictions) == 0.0


# --- Configurations and the grid --------------------------------------------


def test_capacity_is_in_tokens():
    config = Config("lfu", 32, 65536)
    assert config.num_blocks == 2048
    assert config.name == "lfu-bs32-cap65536"
    assert config.server_args() == [
        "--block-size", "32", "--num-gpu-blocks-override", "2048"
    ]
    assert config.env() == {"VLLM_EVICTION_POLICY": "lfu"}
    # Same tokens at every block size (US2, scenario 2).
    assert {Config("lru", b, 65536).num_blocks * b
            for b in (8, 16, 128)} == {65536}


def test_grid_order_is_capacity_then_block_size_then_policy():
    grid = sweep.expand_grid(["lru", "lfu"], [32, 16], [256, 128])
    assert [c.name for c in grid] == [
        "lru-bs16-cap128", "lfu-bs16-cap128", "lru-bs32-cap128",
        "lfu-bs32-cap128", "lru-bs16-cap256", "lfu-bs16-cap256",
        "lru-bs32-cap256", "lfu-bs32-cap256"
    ]


@pytest.mark.parametrize("policies,block_sizes,capacities,message", [
    (["mru"], [16], [1024], "unknown policy 'mru'"),
    (["lru"], [24], [1024], "block size 24"),
    (["lru"], [16, 128], [1024, 1000], "1000 tokens.*block size 16"),
    (["lru"], [128], [64], "64 tokens.*block size 128"),
    (["lru", "lru"], [16], [1024], "repeated policy"),
    ([], [16], [1024], "at least one policy"),
])
def test_bad_grids_are_refused(policies, block_sizes, capacities, message):
    with pytest.raises(ValueError, match=message):
        sweep.expand_grid(policies, block_sizes, capacities)


def test_cache_must_hold_one_max_model_len_request():
    assert sweep.feasible(Config("lru", 16, 8192), 8192)
    assert not sweep.feasible(Config("lru", 16, 4096), 8192)


# --- Running and resuming ---------------------------------------------------


class FakeReplay:
    """Writes Task 2's handcrafted run folder, like a finished replay."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, config, run_dir):
        self.calls.append(config.name)
        if config.name == self.fail_on:
            run_dir.mkdir(parents=True)  # interrupted: no manifest
            raise KeyboardInterrupt
        make_run(run_dir)
        metrics = run_dir / "client.jsonl.metrics.after.txt"
        metrics.write_text(metrics.read_text() +
                           'vllm:num_preemptions_total{model_name="m"} 3.0\n')
        (run_dir / "run.json").write_text(
            json.dumps({
                "commit": "abc",
                "kv_cache": {
                    "num_gpu_blocks": config.num_blocks,
                    "block_size": config.block_size
                },
                "client_summary": {
                    "completed": 3,
                    "skipped_context": 1
                },
                "finished_at_utc": "2026-10-23T00:00:00+00:00",
            }))


def analyze(run_dir):
    from analysis import run_m2_task2
    return run_m2_task2.analyze_run(run_dir, window_s=5)


def run(configs, out, replay, **kwargs):
    return sweep.run_sweep(configs,
                           out,
                           replay,
                           analyze,
                           max_model_len=1024,
                           log=lambda line: None,
                           **kwargs)


GRID = sweep.expand_grid(["lru", "fifo"], [16], [512, 2048])


def test_rows_for_finished_and_infeasible_configurations(tmp_path):
    replay = FakeReplay()
    rows = run(GRID, tmp_path, replay)
    # 512 tokens cannot hold a 1,024-token request: never replayed.
    assert replay.calls == ["lru-bs16-cap2048", "fifo-bs16-cap2048"]
    assert [(r["name"], r["status"]) for r in rows] == [
        ("lru-bs16-cap512", "infeasible"),
        ("fifo-bs16-cap512", "infeasible"),
        ("lru-bs16-cap2048", "finished"),
        ("fifo-bs16-cap2048", "finished"),
    ]
    assert rows[0]["block_hit_rate"] is None
    row = rows[2]
    assert (row["policy"], row["block_size"], row["capacity_tokens"],
            row["num_blocks"]) == ("lru", 16, 2048, 128)
    assert row["kv_blocks_reported"] == 128
    assert row["block_hit_rate"] == 0.5
    assert row["evictions"] == 1
    assert row["eviction_rate_per_1k_lookups"] == 125.0
    assert row["preemptions"] == 3.0
    assert row["requests_completed"] == 3
    assert row["requests_skipped_context"] == 1
    assert row["sc003_pass"] is True
    assert row["commit"] == "abc"


def test_finished_runs_are_not_replayed_again(tmp_path):
    run(GRID, tmp_path, FakeReplay())
    again = FakeReplay()
    rows = run(GRID, tmp_path, again)
    assert again.calls == []
    assert [r["status"] for r in rows][2:] == ["finished", "finished"]


def test_interrupted_run_is_removed_and_replayed(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        run(GRID, tmp_path, FakeReplay(fail_on="fifo-bs16-cap2048"))
    partial = tmp_path / "fifo-bs16-cap2048"
    (partial / "stale.txt").write_text("from the interrupted run")
    resumed = FakeReplay()
    rows = run(GRID, tmp_path, resumed)
    assert resumed.calls == ["fifo-bs16-cap2048"]
    assert not (partial / "stale.txt").exists()
    assert rows[3]["status"] == "finished"


def test_summary_is_cached_unless_reanalyzed(tmp_path):
    run(GRID, tmp_path, FakeReplay())
    summary = tmp_path / "lru-bs16-cap2048" / "summary.json"
    summary.write_text(json.dumps({"evictions": 99}))
    assert run(GRID, tmp_path, FakeReplay())[2]["evictions"] == 99
    rows = run(GRID, tmp_path, FakeReplay(), reanalyze=True)
    assert rows[2]["evictions"] == 1


def test_missing_runs_are_reported_without_replay(tmp_path):
    replay = FakeReplay()
    rows = run(GRID, tmp_path, replay, replay_missing=False)
    assert replay.calls == []
    assert [r["status"] for r in rows][2:] == ["missing", "missing"]


# --- The Task 3 command -----------------------------------------------------


def test_full_grid_is_the_spec_minimum():
    args = task3.parse_args([])
    # SC-003: at least 5 capacities x 3 block sizes x 3 policies.
    assert len(args.configs) == 45
    assert (args.limit, args.seed, args.conversation_rate,
            args.think_time) == (1000, 0, 1.0, 5.0)
    assert max(c.capacity_tokens for c in args.configs) > 114_964 * 16
    assert all(sweep.feasible(c, args.max_model_len) for c in args.configs)


def test_smoke_grid_and_overrides():
    args = task3.parse_args(["--grid", "smoke"])
    assert len(args.configs) == 6
    assert (args.limit, args.conversation_rate,
            args.think_time) == (20, float("inf"), 0.0)
    args = task3.parse_args([
        "--grid", "smoke", "--policies", "lfu", "--capacities", "16384",
        "--limit", "5"
    ])
    assert [c.name for c in args.configs] == ["lfu-bs16-cap16384"]
    assert args.limit == 5


def test_bad_capacity_is_a_usage_error():
    with pytest.raises(SystemExit):
        task3.parse_args(["--capacities", "1000"])


@pytest.fixture
def data(tmp_path):
    conversations = write_jsonl(
        tmp_path / "c.jsonl", [conversation("s:A", 2),
                               conversation("s:B", 1)])
    responses = write_jsonl(tmp_path / "r.jsonl", [{
        "request_id": f"{cid}:{t}",
        "response": "x"
    } for cid, n in (("s:A", 2), ("s:B", 1)) for t in range(n)])
    return [
        "--conversations",
        str(conversations), "--responses",
        str(responses)
    ]


def test_main_runs_the_grid_and_writes_summary_csv(tmp_path, data,
                                                   monkeypatch):
    calls = []

    def fake_run_setting(setting, args, sample, run_dir, extra_server_args,
                         env, manifest_updates):
        calls.append((setting, extra_server_args, env, manifest_updates))
        FakeReplay()(Config(env["VLLM_EVICTION_POLICY"],
                            int(extra_server_args[1]),
                            manifest_updates["capacity_tokens"]), run_dir)

    monkeypatch.setattr(task3.task2, "run_setting", fake_run_setting)
    out = tmp_path / "sweep"
    argv = data + [
        "--grid", "smoke", "--limit", "2", "--capacities", "8192",
        "--policies", "lru", "lfu", "--out",
        str(out)
    ]
    task3.main(argv)
    assert [(c[0], c[2]) for c in calls] == [
        ("multi", {
            "VLLM_EVICTION_POLICY": "lru"
        }),
        ("multi", {
            "VLLM_EVICTION_POLICY": "lfu"
        }),
    ]
    assert calls[1][1] == [
        "--block-size", "16", "--num-gpu-blocks-override", "512"
    ]
    assert calls[1][3]["eviction_policy"] == "lfu"
    with (out / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["name"]
            for r in rows] == ["lru-bs16-cap8192", "lfu-bs16-cap8192"]
    sweep_file = json.loads((out / "sweep.json").read_text())
    assert sweep_file["sample"]["turns"] == 3
    # Same command again: nothing is replayed.
    calls.clear()
    task3.main(argv)
    assert calls == []


def test_resuming_with_other_settings_is_refused(tmp_path, data, monkeypatch):
    monkeypatch.setattr(
        task3.task2, "run_setting", lambda *a, **k: FakeReplay()
        (Config("lru", 16, 8192), a[3]))
    out = tmp_path / "sweep"
    base = data + [
        "--grid", "smoke", "--policies", "lru", "--capacities", "8192",
        "--out",
        str(out)
    ]
    task3.main(base + ["--limit", "2"])
    with pytest.raises(SystemExit, match="other settings"):
        task3.main(base + ["--limit", "1"])


def test_other_folders_are_not_used(tmp_path, data):
    (tmp_path / "unrelated.txt").write_text("x")
    with pytest.raises(SystemExit, match="not a sweep folder"):
        task3.main(data + ["--grid", "smoke", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="nothing to analyze"):
        task3.main(data + ["--analyze-only", "--out", str(tmp_path / "new")])
