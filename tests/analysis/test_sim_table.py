"""The cache-replay result table of spec 007 (``analysis/sim_table.py``)."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from analysis.m3t2_study import load_config
from analysis.sim_table import (collect, engine_rows, gate_failures, load_rows,
                                render, select, verdict)

pytestmark = pytest.mark.m3


def row(policy,
        seed,
        usable,
        workloads=None,
        capacity=131072,
        window=(600.0, 1800.0),
        timing="model"):
    measured = {
        "usable_reused_fraction": usable,
        "workloads": {
            w: {
                "usable_reused_fraction": v
            }
            for w, v in (workloads or {}).items()
        },
    }
    return {
        "plan": f"data/processed/m3/balanced-human-s{seed}",
        "policy": policy,
        "capacity": capacity,
        "block_size": 16,
        "timing": timing,
        "window": list(window),
        "summary": {
            "usable_reused_fraction": 0.0,
            "window": measured
        },
    }


def test_rows_with_an_error_are_skipped(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(row("lru", 2, 0.5)) + "\n" + json.dumps({
            "plan": "p-s3",
            "policy": "lru",
            "error": "boom"
        }) + "\n")
    assert len(load_rows([path])) == 1


def test_the_measured_window_is_used_per_seed():
    whole, workloads = collect([
        row("lru", 2, 0.50, {"a": 0.4}),
        row("lru", 3, 0.60, {"a": 0.6}),
    ])
    assert whole[("lru", "131072")] == {"2": 50.0, "3": 60.0}
    assert workloads[("lru", "131072")]["a"] == [40.0, 60.0]


@pytest.mark.parametrize("diffs, expected", [
    ([1.0, 0.8, 2.0], "higher"),
    ([-1.0, -0.6, -0.9], "lower"),
    ([1.0, -0.1, 2.0], "tie"),
    ([0.2, 0.3, 0.4], "tie"),
])
def test_verdict_needs_every_seed_and_half_a_point(diffs, expected):
    assert verdict(diffs) == expected


def test_render_compares_with_a_baseline():
    text = render([
        row("lru", 2, 0.50),
        row("lru", 3, 0.52),
        row("race_rp", 2, 0.54),
        row("race_rp", 3, 0.55),
    ],
                  against="lru")
    assert "race_rp | 54.50" in text
    assert "race_rp | +4.00 +3.00 | mean +3.50 | higher" in text


def test_the_tuning_seed_is_left_out_unless_asked_for():
    rows = [row("lru", 1, 0.9), row("lru", 2, 0.5), row("lru", 3, 0.6)]
    assert [r["plan"][-1] for r in select(rows)] == ["2", "3"]
    assert [r["plan"][-1] for r in select(rows, ["1"])] == ["1"]
    assert "lru | 55.00" in render(rows)
    assert "lru | 90.00" in render(rows, seeds=["1"])


@pytest.mark.parametrize("other", [
    row("race_rp", 3, 0.5, window=(0.0, 1800.0)),
    row("race_rp", 3, 0.5, timing="engine:results/x"),
    {
        **row("race_rp", 3, 0.5),
        "plan": "data/processed/m3/chat-heavy-human-s3",
    },
])
def test_rows_of_different_setups_are_refused(other):
    with pytest.raises(ValueError, match="different setups"):
        render([row("lru", 2, 0.5), other])


def test_every_workload_below_the_gate_is_listed():
    base = {"a": {"2": 50.0, "3": 50.0}, "b": {"2": 20.0, "3": 20.0}}
    policy = {
        "a": {
            "2": 47.0,
            "3": 48.5
        },  # -2.25
        "b": {
            "2": 19.0,
            "3": 21.0
        },  # 0
    }
    assert gate_failures(policy, base) == [("a", pytest.approx(-2.25))]
    text = render([
        row("lru", 2, 0.5, {
            "a": 0.5,
            "b": 0.2
        }),
        row("lru", 3, 0.5, {
            "a": 0.5,
            "b": 0.2
        }),
        row("race_rp", 2, 0.6, {
            "a": 0.47,
            "b": 0.19
        }),
        row("race_rp", 3, 0.6, {
            "a": 0.485,
            "b": 0.21
        }),
    ],
                  against="lru")
    assert "-- workloads more than 2 points below lru (mean)" in text
    assert "race_rp | a -2.25" in text


# --- Engine runs -------------------------------------------------------------

WORKLOADS = load_config()["workloads"]


def engine_run(root, policy, seed, tenths, ceiling=False, finished=True):
    """A finished grid run folder with the request table the study reads:
    one request per workload in the window, 160 prompt tokens (10 blocks of
    16), the first `tenths` blocks usable."""
    run_dir = root / f"balanced-human-s{seed}-{policy}{'-ceiling' * ceiling}"
    (run_dir / "tables").mkdir(parents=True)
    n = len(WORKLOADS)
    pq.write_table(
        pa.table({
            "req": [f"{w}:x:0" for w in WORKLOADS],
            "workload": WORKLOADS,
            "session": [f"{w}:x" for w in WORKLOADS],
            "turn": [0] * n,
            "planned_s": [700.0] * n,
            "status": ["completed"] * n,
            "prompt_tokens": [160] * n,
            "prompt_blocks": [10] * n,
            "hits": [tenths] * n,
            "usable_hits": [tenths] * n,
            "hits_after_miss": [0] * n,
            "cached_tokens": [16 * tenths] * n,
            "preempted": [False] * n,
        }), run_dir / "tables" / "requests.parquet")
    run = {
        "plan_dir": f"data/processed/m3/balanced-human-s{seed}",
        "eviction_policy": policy,
        "capacity_tokens": 131072,
        "block_size": 16,
        "mix": "balanced",
        "pacing": "human",
        "seed": seed,
        "replicate": 0,
        "ceiling": ceiling,
    }
    if finished:
        run["finished_at_utc"] = "2026-11-10T00:00:00+00:00"
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir


def test_engine_runs_give_rows_like_the_replays(tmp_path):
    runs = [
        engine_run(tmp_path, "lru", 2, 3),
        engine_run(tmp_path, "race_rp", 2, 5),
        engine_run(tmp_path, "lru", 3, 4),
        engine_run(tmp_path, "race_rp", 3, 6),
    ]
    rows = engine_rows(runs)
    assert [r["policy"] for r in rows] == ["lru", "race_rp"] * 2
    assert {(r["timing"], tuple(r["window"]), r["capacity"])
            for r in rows} == {("engine", (600.0, 1800.0), 131072)}
    window = rows[0]["summary"]["window"]
    assert window["usable_reused_fraction"] == pytest.approx(0.3)
    assert window["workloads"]["sharegpt"]["usable_reused_fraction"] == (
        pytest.approx(0.3))
    text = render(rows, against="lru")
    assert "race_rp | +20.00 +20.00 | mean +20.00 | higher" in text
    assert "race_rp | none" in text  # no workload below the gate


def test_a_ceiling_run_is_its_own_capacity(tmp_path):
    row, = engine_rows([engine_run(tmp_path, "lru", 2, 9, ceiling=True)])
    assert (row["policy"], row["capacity"]) == ("lru", "ceiling")


def test_engine_and_replay_rows_are_not_mixed(tmp_path):
    rows = engine_rows([engine_run(tmp_path, "lru", 2, 3)])
    with pytest.raises(ValueError, match="different setups"):
        render(rows + [row("race_rp", 2, 0.5)])


def test_an_engine_run_needs_its_tables_and_to_have_finished(tmp_path):
    run_dir = engine_run(tmp_path, "lru", 2, 3)
    (run_dir / "tables" / "requests.parquet").unlink()
    with pytest.raises(ValueError, match="analysis.tables"):
        engine_rows([run_dir])
    with pytest.raises(ValueError, match="has not finished"):
        engine_rows([engine_run(tmp_path, "race_rp", 2, 3, finished=False)])
