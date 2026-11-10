"""The cache-replay result table of spec 007 (``analysis/sim_table.py``)."""
import json

import pytest

from analysis.sim_table import (collect, gate_failures, load_rows, render,
                                select, verdict)

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
        row("wa", 2, 0.54),
        row("wa", 3, 0.55),
    ],
                  against="lru")
    assert "wa | 54.50" in text
    assert "wa | +4.00 +3.00 | mean +3.50 | higher" in text


def test_the_tuning_seed_is_left_out_unless_asked_for():
    rows = [row("lru", 1, 0.9), row("lru", 2, 0.5), row("lru", 3, 0.6)]
    assert [r["plan"][-1] for r in select(rows)] == ["2", "3"]
    assert [r["plan"][-1] for r in select(rows, ["1"])] == ["1"]
    assert "lru | 55.00" in render(rows)
    assert "lru | 90.00" in render(rows, seeds=["1"])


@pytest.mark.parametrize("other", [
    row("wa", 3, 0.5, window=(0.0, 1800.0)),
    row("wa", 3, 0.5, timing="engine:results/x"),
    {
        **row("wa", 3, 0.5), "plan": "data/processed/m3/chat-heavy-human-s3"
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
        row("wa", 2, 0.6, {
            "a": 0.47,
            "b": 0.19
        }),
        row("wa", 3, 0.6, {
            "a": 0.485,
            "b": 0.21
        }),
    ],
                  against="lru")
    assert "-- workloads more than 2 points below lru (mean)" in text
    assert "wa | a -2.25" in text
