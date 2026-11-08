"""The cache-replay result table of spec 007 (``analysis/sim_table.py``)."""
import json

import pytest

from analysis.sim_table import collect, load_rows, render, verdict

pytestmark = pytest.mark.m3


def row(policy, seed, usable, workloads=None, capacity=131072):
    window = {
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
        "summary": {
            "usable_reused_fraction": 0.0,
            "window": window
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
