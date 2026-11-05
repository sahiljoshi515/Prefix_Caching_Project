"""The report's static figures (``analysis/m3t2_figures.py``), drawn from
the page's chart data: points 1, 2 and 4 with the new policies when they
ran, and point 7's two figures."""
import json
import math
import runpy
import sys

import matplotlib
import pytest

from analysis import m3t2_figures as mf
from analysis import m3t2_report as rp
from tests.analysis.test_m3t2_report_policies import (ALL_NEW, CAPS, NEW,
                                                      curve, summary)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.m3

BASE_FILES = ("point1_reuse_interval", "point2_reuse_portion",
              "point3_identification", "point4_capacity",
              "point5_alone_portion", "point5_alone_interval",
              "point6_alone_vs_mixed")
POINT7_FILES = ("point7_capacity_by_workload", "point7_evictions")
MIXED = ["lru", "lfu", "fifo", "ttdf", "race_rp"]
MIXED3 = ["lru", "lfu", "fifo", *ALL_NEW]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def baseline():
    return summary(engine=(), replay=(), tune=False, sketch=False)


def write(tmp_path, s):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(s))
    return path


def files(names):
    return sorted(f"{n}.{ext}" for n in names for ext in ("pdf", "png"))


def figures():
    return [plt.figure(n) for n in plt.get_fignums()]


def legend_texts(legend):
    return [t.get_text() for t in legend.get_texts()]


def policy_keys(ax):
    """The legend's policy entries, in order (the ceiling key aside)."""
    return [t for t in legend_texts(ax.get_legend()) if t != "No eviction"]


# --- main ---------------------------------------------------------------


def test_main_draws_every_point_with_point_7(tmp_path, capsys):
    out = tmp_path / "figures" / "task2"  # created, parents too
    mf.main(["--summary", str(write(tmp_path, summary())), "--out", str(out)])
    written = sorted(p.name for p in out.iterdir())
    assert written == files(BASE_FILES + POINT7_FILES)
    assert all((out / n).stat().st_size > 0 for n in written)
    printed = capsys.readouterr().out.split()
    assert sorted(printed) == sorted(str(out / n) for n in written)
    assert plt.get_fignums() == []  # each point's figures are closed


def test_main_without_new_policies_draws_six_points(tmp_path):
    out = tmp_path / "out"
    mf.main(["--summary", str(write(tmp_path, baseline())), "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == files(BASE_FILES)


def test_main_needs_the_isolated_runs(tmp_path):
    s = summary()
    del s["isolation"]
    with pytest.raises(SystemExit, match="no isolated runs"):
        mf.main([
            "--summary",
            str(write(tmp_path, s)), "--out",
            str(tmp_path / "out")
        ])
    assert not (tmp_path / "out").exists()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_runs_as_a_module(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "m3t2_figures", "--summary",
        str(write(tmp_path, baseline())), "--out",
        str(out)
    ])
    runpy.run_module("analysis.m3t2_figures", run_name="__main__")
    assert sorted(p.name for p in out.iterdir()) == files(BASE_FILES)


# --- Point 7 ----------------------------------------------------------------


def test_point7_draws_a_panel_per_workload(tmp_path):
    data = rp.build_data(summary())
    q = data["q7"]
    paths = mf.point7(data, plt, tmp_path)
    assert sorted(p.name for p in paths) == files(POINT7_FILES)
    assert all(p.exists() for p in paths)
    panels, bars = figures()
    axes = panels.axes
    assert len(axes) == 4
    for ax, g in zip(axes, rp.GROUPS):
        assert ax.get_title(loc="left") == f"{rp.LABELS[g]} in the mix"
        lines = ax.get_lines()
        assert [ln.get_label() for ln in lines
                ] == ["LRU", "LFU", "FIFO", "TTDF", "RACE-RP", "No eviction"]
        ttdf = lines[3]
        assert list(ttdf.get_xdata()) == CAPS
        assert list(ttdf.get_ydata()) == pytest.approx(
            q["panels"][g]["curve"]["ttdf"])
        assert ttdf.get_color() == mf.COLOR["ttdf"]
        assert list(lines[-1].get_ydata()) == [q["panels"][g]["ceiling"]] * 2
        assert ax.get_ylim() == pytest.approx(q["panels"][g]["yDomain"])
        assert ax.get_xscale() == "log"
        assert [t.get_text() for t in ax.get_xticklabels()] == ["32K", "128K"]
    assert legend_texts(panels.legends[0]) == [
        "LRU", "LFU", "FIFO", "TTDF", "RACE-RP", "No eviction"
    ]
    # Evictions: a bar per workload (and the whole mix) and engine policy.
    ax = bars.axes[0]
    assert len(ax.patches) == 5 * len(MIXED)
    top = max(q["evictions"][p][r]["max"] for p in MIXED
              for r in ("all", *rp.GROUPS))
    assert ax.get_xlim() == pytest.approx((0, 1.15 * top))
    assert [t.get_text() for t in ax.get_yticklabels()] == \
        ["Whole mix", "ShareGPT", "AgentBank", "CC-Bench", "Bailian"]
    labels = {t.get_text() for t in ax.texts}
    assert "465/s" in labels and "360/s" in labels  # TTDF: all, Bailian
    assert legend_texts(ax.get_legend()) == \
        ["LRU", "LFU", "FIFO", "TTDF", "RACE-RP"]


def test_point7_without_evictions_draws_only_the_panels(tmp_path):
    data = rp.build_data(summary())
    data["q7"]["evictions"] = {}
    paths = mf.point7(data, plt, tmp_path)
    assert sorted(p.name for p in paths) == \
        files(("point7_capacity_by_workload", ))
    assert len(figures()) == 1


def test_point7_evictions_only_once_a_new_policy_ran(tmp_path):
    # Replay curves for the new policies, engine runs for none: the
    # evictions are the baselines' only, so (as on the page) no chart.
    data = rp.build_data(summary(engine=()))
    assert data["q7"]["policies"] == MIXED
    assert set(data["q7"]["evictions"]) == set(rp.POLICIES)
    paths = mf.point7(data, plt, tmp_path)
    assert sorted(p.name for p in paths) == \
        files(("point7_capacity_by_workload", ))
    (panels, ) = figures()
    assert len(panels.axes[0].get_lines()) == len(MIXED) + 1
    plt.close("all")
    # One new policy with engine runs: the chart, with the baselines.
    data = rp.build_data(summary(engine=("race_rp", )))
    paths = mf.point7(data, plt, tmp_path)
    assert sorted(p.name for p in paths) == files(POINT7_FILES)
    _, bars = figures()
    assert len(bars.axes[0].patches) == 5 * 4
    assert policy_keys(bars.axes[0]) == ["LRU", "LFU", "FIFO", "RACE-RP"]


def test_main_without_new_engine_runs_has_no_evictions_figure(tmp_path):
    out = tmp_path / "out"
    mf.main([
        "--summary",
        str(write(tmp_path, summary(engine=()))), "--out",
        str(out)
    ])
    assert sorted(p.name for p in out.iterdir()) == \
        files(BASE_FILES + ("point7_capacity_by_workload", ))


# --- Points 1, 2 and 4 with the new policies --------------------------------


def test_point1_draws_the_mixed_policies(tmp_path):
    data = rp.build_data(summary())
    mf.point1(data, plt, tmp_path)
    (fig, ) = figures()
    ax = fig.axes[0]
    assert legend_texts(ax.get_legend()) == \
        ["LRU", "LFU", "FIFO", "TTDF", "RACE-RP"]
    assert [ln.get_marker() for ln in ax.get_lines()] == \
        ["o", "D", "s", "P", "*"]
    plt.close("all")
    mf.point1(rp.build_data(baseline()), plt, tmp_path)
    assert legend_texts(figures()[0].axes[0].get_legend()) == \
        ["LRU", "LFU", "FIFO"]


@pytest.mark.parametrize("s, height, pols",
                         [(summary(), 5.6, MIXED),
                          (baseline(), 4.6, ["lru", "lfu", "fifo"])],
                         ids=["new", "baseline"])
def test_point2_grows_with_the_policies(tmp_path, s, height, pols):
    mf.point2(rp.build_data(s), plt, tmp_path)
    (fig, ) = figures()
    assert fig.get_figheight() == pytest.approx(height)
    ax = fig.axes[0]
    assert len(ax.patches) == 5 * len(pols)
    assert policy_keys(ax) == [rp.POLICY_LABELS[p] for p in pols]
    assert legend_texts(ax.get_legend()).count("No eviction") == 1


def test_point4_draws_the_q4_policies_and_their_engine_runs(tmp_path):
    data = rp.build_data(summary())
    mf.point4(data, plt, tmp_path)
    (fig, ) = figures()
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert labels[:6] == [
        "LRU", "LFU", "FIFO", "TTDF", "RACE-RP", "Belady (offline)"
    ]
    assert ax.get_lines()[5].get_linestyle() == "--"
    # Two markers per engine mark (engine run, replay), plus two legend keys.
    assert len(ax.collections) == 2 * len(data["q4"]["engine"]) + 2
    assert {m["policy"] for m in data["q4"]["engine"]} >= set(NEW)


# --- Helpers ----------------------------------------------------------------


def bars_values(pols, rows, flat=()):
    """Seed spreads per policy and row; policies in `flat` have none."""
    return {
        p: {
            r: {
                "mean": 0.1 * (k + 1),
                "min": 0.1 * (k + 1) - (0 if p in flat else 0.02),
                "max": 0.1 * (k + 1) + (0 if p in flat else 0.02),
            }
            for r in rows
        }
        for k, p in enumerate(pols)
    }


def bar_centres(ax):
    return sorted(b.get_y() + b.get_height() / 2 for b in ax.patches)


def test_grouped_bars_any_number_of_policies_without_a_ceiling():
    rows = ["all", "sharegpt"]
    fig, ax = plt.subplots()
    ax.set_xlim(0, 500)
    mf._grouped_bars(ax,
                     rows,
                     bars_values(MIXED, rows, flat=("lfu", )),
                     None,
                     MIXED,
                     fmt=lambda v: f"{v:.2f}/s")
    h = 0.78 / 5
    assert len(ax.patches) == 10
    assert ax.patches[0].get_height() == pytest.approx(0.9 * h)
    expected = sorted(r + k * h for r in range(2) for k in (-2, -1, 0, 1, 2))
    assert bar_centres(ax) == pytest.approx(expected)
    # Seed ranges for every bar but LFU's; no dashed ceiling tick.
    assert len(ax.get_lines()) == 2 * 4
    assert ax.get_xlim() == (0, 500)
    assert ax.get_xlabel() == ""
    assert {t.get_text() for t in ax.texts} >= {"0.10/s", "0.50/s"}
    assert legend_texts(ax.get_legend()) == \
        ["LRU", "LFU", "FIFO", "TTDF", "RACE-RP"]


def test_grouped_bars_three_policies_with_a_ceiling_keep_the_old_layout():
    rows = ["all", "sharegpt", "bailian"]
    fig, ax = plt.subplots()
    ceiling = {r: {"mean": 0.9} for r in rows}
    mf._grouped_bars(ax, rows, bars_values(rp.POLICIES, rows), ceiling)
    assert bar_centres(ax) == pytest.approx(
        sorted(r + d for r in range(3) for d in (-0.26, 0.0, 0.26)))
    dashed = [ln for ln in ax.get_lines() if list(ln.get_xdata()) == [0.9] * 2]
    assert len(dashed) == 3
    assert list(dashed[0].get_ydata()) == pytest.approx([-0.39, 0.39])
    assert ax.get_xlim() == (0, 1.08)
    assert ax.get_xlabel().startswith("Usable prefix reuse")
    assert "10.0%" in {t.get_text() for t in ax.texts}
    assert policy_keys(ax) == ["LRU", "LFU", "FIFO"]
    assert legend_texts(ax.get_legend()).count("No eviction") == 1


def test_interval_axes_draw_the_bins_with_enough_reuses():
    bins = {p: curve([1.0, 0.9, 0.8, 0.5, 0.2, 0.1]) for p in MIXED}
    bins["race_rp"][1]["offered"] = 10  # too few: not drawn for anyone
    fig, ax = plt.subplots()
    mf._interval_axes(ax, bins, "A title", pols=MIXED)
    lines = ax.get_lines()
    assert len(lines) == 5
    # Bins 0-0.1, 1-10, 10-30, 30-300 (the open last bin is not drawn).
    assert list(lines[0].get_xdata()) == pytest.approx(
        [0.05, math.sqrt(10),
         math.sqrt(300), math.sqrt(9000)])
    assert list(lines[0].get_ydata()) == [1.0, 0.8, 0.5, 0.2]
    assert ax.get_title(loc="left") == "A title"
    # Without a policy list: the baselines.
    fig, ax = plt.subplots()
    mf._interval_axes(ax, bins)
    assert len(ax.get_lines()) == 3 and ax.get_title(loc="left") == ""


def test_tick_seconds():
    assert [mf._tick_secs(t) for t in mf.INTERVAL_TICKS] == \
        ["0.1 s", "1 s", "10 s", "1 min", "5 min", "20 min"]


def test_line_styles():
    assert mf.LINESTYLE == {"belady": "--", "sampled_lrfu": ":"}


def test_every_drawn_policy_has_a_colour_and_a_marker():
    for p in (*MIXED3, "belady"):
        assert p in mf.COLOR and p in mf.MARKER
    assert (mf.COLOR["sampled_lrfu"], mf.MARKER["sampled_lrfu"]) == \
        ("#008300", "X")
    assert len(set(mf.COLOR.values())) == len(mf.COLOR)
    assert len(set(mf.MARKER.values())) == len(mf.MARKER)


def test_figures_with_three_new_policies(tmp_path):
    data = rp.build_data(summary(engine=ALL_NEW, replay=ALL_NEW))
    labels = [rp.POLICY_LABELS[p] for p in MIXED3]
    mf.point1(data, plt, tmp_path)
    ax = figures()[0].axes[0]
    assert legend_texts(ax.get_legend()) == labels
    assert ax.get_lines()[-1].get_marker() == "X"
    assert ax.get_lines()[-1].get_color() == "#008300"
    # Sampled LRFU dotted (its green is near FIFO's), the others solid.
    assert [ln.get_linestyle() for ln in ax.get_lines()] == \
        ["-"] * 5 + [":"]
    plt.close("all")
    mf.point2(data, plt, tmp_path)
    (fig, ) = figures()
    assert fig.get_figheight() == pytest.approx(4.6 + 0.5 * 3)
    assert len(fig.axes[0].patches) == 5 * 6
    assert fig.axes[0].patches[0].get_height() == pytest.approx(0.9 * 0.13)
    plt.close("all")
    mf.point4(data, plt, tmp_path)
    lines = figures()[0].axes[0].get_lines()[:7]
    assert [ln.get_label() for ln in lines] == [*labels, "Belady (offline)"]
    assert [ln.get_linestyle() for ln in lines] == \
        ["-"] * 5 + [":", "--"]
    plt.close("all")
    mf.point7(data, plt, tmp_path)
    panels, bars = figures()
    assert [ln.get_label() for ln in panels.axes[0].get_lines()] == \
        [*labels, "No eviction"]
    assert panels.axes[0].get_lines()[5].get_linestyle() == ":"
    assert legend_texts(panels.legends[0]) == [*labels, "No eviction"]
    assert len(bars.axes[0].patches) == 5 * 6
    assert policy_keys(bars.axes[0]) == labels
