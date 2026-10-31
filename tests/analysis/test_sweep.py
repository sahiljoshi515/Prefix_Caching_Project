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
    cached = dict.fromkeys(sweep.CURRENT_SUMMARY_FIELDS, 0.5)
    summary.write_text(json.dumps({**cached, "evictions": 99}))
    assert run(GRID, tmp_path, FakeReplay())[2]["evictions"] == 99
    rows = run(GRID, tmp_path, FakeReplay(), reanalyze=True)
    assert rows[2]["evictions"] == 1


def test_a_summary_from_an_older_analysis_is_redone(tmp_path):
    """One written before the usable reused fraction existed."""
    run(GRID, tmp_path, FakeReplay())
    summary = tmp_path / "lru-bs16-cap2048" / "summary.json"
    summary.write_text(json.dumps({"evictions": 99}))
    row = run(GRID, tmp_path, FakeReplay())[2]
    assert row["evictions"] == 1
    assert row["usable_reused_fraction"] is not None


def test_missing_runs_are_reported_without_replay(tmp_path):
    replay = FakeReplay()
    rows = run(GRID, tmp_path, replay, replay_if=lambda config: False)
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
    # No figures: these tests drive the sweep, and must not need
    # matplotlib; the figure tests use importorskip.
    return [
        "--conversations",
        str(conversations), "--responses",
        str(responses), "--no-figures"
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


# --- Results: figure 4's capacity, the best-configuration table, figures ----


def result(policy,
           block_size,
           capacity,
           reused,
           status="finished",
           vllm_reused=None):
    return {
        "name": Config(policy, block_size, capacity).name,
        "policy": policy,
        "block_size": block_size,
        "capacity_tokens": capacity,
        "status": status,
        "usable_reused_fraction": reused,
        "token_reused_fraction":
        reused if vllm_reused is None else vllm_reused,
        "block_hit_rate": reused,
        "evictions": 0,
    }


RESULTS = [
    # 1,024 tokens: spread 0.3; 2,048: spread 0.1; 4,096: LFU missing.
    result("lru", 16, 1024, 0.5),
    result("fifo", 16, 1024, 0.2),
    result("lfu", 16, 1024, 0.4),
    result("lru", 16, 2048, 0.8),
    result("fifo", 16, 2048, 0.7),
    result("lfu", 16, 2048, 0.75),
    result("lru", 16, 4096, 0.9),
    result("fifo", 16, 4096, 0.1),
    result("lfu", 16, 4096, None, status="missing"),
    result("lru", 32, 1024, 0.6),
    result("lru", 16, 512, None, status="infeasible"),
]


def test_pressure_capacity_is_where_policies_differ_most():
    assert task3.pressure_capacity(RESULTS, 16) == 1024
    assert task3.pressure_capacity(RESULTS, 64) is None


def test_pressure_capacity_ties_go_to_the_smaller_cache():
    tied = [
        result(p, 16, c, v) for c in (2048, 1024)
        for p, v in (("lru", 0.5), ("fifo", 0.4), ("lfu", 0.45))
    ]
    assert task3.pressure_capacity(tied, 16) == 1024


def test_best_configuration_per_capacity():
    best = task3.best_by_capacity(RESULTS)
    assert [(b["capacity_tokens"], b["best"], b["configurations"])
            for b in best] == [(1024, "lru-bs32-cap1024", 4),
                               (2048, "lru-bs16-cap2048", 3),
                               (4096, "lru-bs16-cap4096", 2)]
    assert best[0]["next_best"] == "lru-bs16-cap1024"
    assert best[0]["margin"] == pytest.approx(0.1)
    table = task3.markdown_table(best).splitlines()
    assert table[0].startswith("| capacity_tokens | best |")
    assert "| 1024 | lru-bs32-cap1024 | lru | 32 | 0.6000 |" in table[2]


def test_figures_are_drawn_from_the_runs(tmp_path):
    pytest.importorskip("matplotlib")
    grid = sweep.expand_grid(list(sweep.POLICIES), [16, 32], [2048, 4096])
    rows = run(grid, tmp_path, FakeReplay())
    written = task3.make_figures(tmp_path, tmp_path / "figures", rows, 16,
                                 2048)
    names = sorted(p.name for p in written)
    assert len(names) == 10  # five figures, PDF and PNG
    assert "fig4a_hits_per_residency_bs16_cap2048.png" in names
    assert "fig3_evictions_vs_capacity.pdf" in names
    assert all(p.stat().st_size > 0 for p in written)


def test_runs_are_ranked_by_usable_reuse_not_vllm_cached_tokens():
    """FIFO's vLLM-reported reuse can be higher while less of it is a
    contiguous prefix that vLLM can skip."""
    rows = [
        result("lru", 16, 1024, 0.76),
        result("fifo", 16, 1024, 0.38, vllm_reused=0.77),
    ]
    assert task3.best_by_capacity(rows)[0]["best"] == "lru-bs16-cap1024"


def test_unmeasured_runs_are_not_ranked():
    """A finished run with no completed request has no reused fraction."""
    rows = RESULTS + [
        result("lfu", 16, 4096, None),
        result("lfu", 32, 8192, None)
    ]
    assert task3.pressure_capacity(rows, 16) == 1024
    best = task3.best_by_capacity(rows)
    assert [b["capacity_tokens"] for b in best] == [1024, 2048, 4096]


def test_legend_names_a_policy_missing_from_the_first_panel(tmp_path):
    pytest.importorskip("matplotlib")
    from analysis import plots
    rows = [result("lru", 16, c, 0.5) for c in (1024, 2048)]
    rows.append(result("lfu", 16, 2048, 0.4))
    names = []
    original = plots.plt.Figure.legend

    def spy(fig, handles, labels, **kwargs):
        names.extend(labels)
        return original(fig, handles, labels, **kwargs)

    plots.plt.Figure.legend = spy
    try:
        plots.panel_lines_plot(rows, "block_size", ["block_hit_rate"],
                               "capacity_tokens", tmp_path / "f")
    finally:
        plots.plt.Figure.legend = original
    assert names == ["LRU", "LFU"]


def fake_run_setting(setting, args, sample, run_dir, extra_server_args, env,
                     manifest_updates):
    FakeReplay()(Config(env["VLLM_EVICTION_POLICY"], int(extra_server_args[1]),
                        manifest_updates["capacity_tokens"]), run_dir)


def test_analyze_only_reuses_the_recorded_grid(tmp_path, data, monkeypatch):
    monkeypatch.setattr(task3.task2, "run_setting", fake_run_setting)
    out = tmp_path / "sweep"
    task3.main(data + [
        "--grid", "smoke", "--limit", "2", "--policies", "lfu", "--capacities",
        "8192", "16384", "--out",
        str(out)
    ])
    # Only --out: the folder's grid and sample settings, not the full grid.
    task3.main(data + ["--analyze-only", "--out", str(out)])
    with (out / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["name"], r["status"]) for r in rows] == [
        ("lfu-bs16-cap8192", "finished"),
        ("lfu-bs16-cap16384", "finished"),
    ]


def test_a_narrower_resume_keeps_every_recorded_run(tmp_path, data,
                                                    monkeypatch):
    calls = []

    def counting(*args, **kwargs):
        calls.append(kwargs["env"]["VLLM_EVICTION_POLICY"])
        fake_run_setting(*args, **kwargs)

    monkeypatch.setattr(task3.task2, "run_setting", counting)
    out = tmp_path / "sweep"
    base = data + [
        "--grid", "smoke", "--limit", "2", "--capacities", "8192", "--out",
        str(out)
    ]
    task3.main(base + ["--policies", "lru", "lfu"])
    task3.main(base + ["--policies", "fifo"])
    assert calls == ["lru", "lfu", "fifo"]
    with (out / "summary.csv").open() as f:
        names = [r["name"] for r in csv.DictReader(f)]
    assert names == [
        "lru-bs16-cap8192", "fifo-bs16-cap8192", "lfu-bs16-cap8192"
    ]


def test_an_interrupted_sample_is_written_again(tmp_path, data, monkeypatch):
    monkeypatch.setattr(task3.task2, "run_setting", fake_run_setting)
    out = tmp_path / "sweep"
    out.mkdir()
    (out / "sample.conversations.jsonl").write_text("partial")
    task3.main(data + [
        "--grid", "smoke", "--limit", "2", "--policies", "lru", "--capacities",
        "8192", "--out",
        str(out)
    ])
    assert json.loads((out / "sweep.json").read_text())["sample"]["turns"] == 3
    assert (out / "sample.conversations.jsonl").read_text() != "partial"


def test_axes_not_given_come_from_the_folders_grid(tmp_path, data,
                                                   monkeypatch):
    """--capacities on a smoke folder adds that capacity at the folder's
    block size, not at the full preset's 8, 16 and 32."""
    monkeypatch.setattr(task3.task2, "run_setting", fake_run_setting)
    out = tmp_path / "sweep"
    task3.main(data + [
        "--grid", "smoke", "--limit", "2", "--policies", "lru", "fifo",
        "--capacities", "8192", "--out",
        str(out)
    ])
    args = task3.parse_args(
        data +
        ["--capacities", "16384", "--policies", "lru", "--out",
         str(out)])
    assert [c.name for c in args.configs] == ["lru-bs16-cap16384"]
    args = task3.parse_args(data +
                            ["--capacities", "16384", "--out",
                             str(out)])
    assert [c.name for c in args.configs
            ] == ["lru-bs16-cap16384", "fifo-bs16-cap16384"]


def test_a_folder_without_a_listed_grid_resumes_its_presets_grid(
        tmp_path, data, monkeypatch):
    """An interrupted full sweep from before sweep.json listed its grid
    resumes all 45 configurations, not only the run folders present."""
    monkeypatch.setattr(task3.task2, "run_setting", fake_run_setting)
    out = tmp_path / "sweep"
    task3.main(data + [
        "--grid", "full", "--limit", "2", "--policies", "lru", "--block-sizes",
        "16", "--capacities", "16384", "--out",
        str(out)
    ])
    recorded = json.loads((out / "sweep.json").read_text())
    del recorded["configs"]
    (out / "sweep.json").write_text(json.dumps(recorded))
    args = task3.parse_args(data + ["--out", str(out)])
    assert len(args.configs) == 45
    # A sample no preset uses needs --grid.
    recorded["settings"]["think_time"] = 1.5
    (out / "sweep.json").write_text(json.dumps(recorded))
    with pytest.raises(SystemExit):
        task3.parse_args(data + ["--out", str(out)])


def test_sweep_file_is_replaced_whole_and_only_when_it_changes(tmp_path):
    path = tmp_path / "sweep.json"
    task3.write_atomically(path, "a")
    before = path.stat().st_mtime_ns
    task3.write_atomically(path, "a")
    assert path.stat().st_mtime_ns == before
    task3.write_atomically(path, "b")
    assert path.read_text() == "b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sweep.json"]


def test_no_best_table_is_left_from_an_earlier_command(tmp_path, data,
                                                       monkeypatch):
    monkeypatch.setattr(task3.task2, "run_setting", fake_run_setting)
    out = tmp_path / "sweep"
    task3.main(data + [
        "--grid", "smoke", "--limit", "2", "--policies", "lru", "--capacities",
        "8192", "--out",
        str(out)
    ])
    assert (out / "best_by_capacity.md").exists()
    # Nothing to rank this time.
    monkeypatch.setattr(task3, "best_by_capacity", lambda rows: [])
    task3.main(data + ["--analyze-only", "--out", str(out)])
    assert not (out / "best_by_capacity.md").exists()
    assert not (out / "best_by_capacity.csv").exists()


def test_folders_without_a_recorded_grid_use_their_run_folders(tmp_path):
    (tmp_path / "sweep.json").write_text(
        json.dumps({
            "settings": {},
            "sample": {}
        }))
    for name in ("lfu-bs16-cap16384", "lru-bs16-cap16384", "lru-bs16-cap8192",
                 "figures"):
        (tmp_path / name).mkdir()
    recorded = task3.read_sweep_file(tmp_path)
    assert [c.name for c in task3.recorded_grid(recorded, tmp_path)] == [
        "lru-bs16-cap8192", "lru-bs16-cap16384", "lfu-bs16-cap16384"
    ]
