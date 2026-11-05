"""Milestone 3 Task 2: report figures from summary.json.

    python -m analysis.m3t2_figures --summary results/m3-study/summary.json \\
        --out report/milestone3/figures/task2

Draws the static (PDF and PNG) versions of the results page's charts for
the technical report, one figure per point (two for points 5 and 7),
from the
same chart data as the page (``analysis/m3t2_report.build_data``), so the
page and the report cannot disagree (constitution V: every figure from a
committed script). Policy colours follow the page's validated palette.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from analysis.m3t2_report import (GROUPS, LABELS, MIN_OFFERED, NEW_POLICIES,
                                  POLICIES, POLICY_LABELS, build_data, tokens)

COLOR = {
    "lru": "#2a78d6",
    "lfu": "#eb6834",
    "fifo": "#1baf7a",
    "ttdf": "#6250d6",
    "race_rp": "#e87ba4",
    "sampled_lrfu": "#008300",
    "belady": "#eda100",
}
MARKER = {
    "lru": "o",
    "lfu": "D",
    "fifo": "s",
    "ttdf": "P",
    "race_rp": "*",
    "sampled_lrfu": "X",
    "belady": "^"
}
# Sampled LRFU's green sits near FIFO's teal: a dotted line tells them apart.
LINESTYLE = {"belady": "--", "sampled_lrfu": ":"}
INK = "#111a1d"
INK_2 = "#46555a"
INTERVAL_TICKS = [0.1, 1, 10, 60, 300, 1200]
Row = Dict[str, Any]


def _save(fig, out: Path, name: str) -> List[Path]:
    paths = []
    for ext in ("pdf", "png"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    return paths


def _pct_axis(axis) -> None:
    axis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")


def _tick_secs(v: float) -> str:
    return f"{v:g} s" if v < 60 else f"{v / 60:g} min"


def _interval_axes(ax,
                   curves: Row,
                   title: str = "",
                   pols: Sequence[str] = POLICIES) -> None:
    """Share of reuses served by how long the block waited, per policy."""
    bins = curves[pols[0]]
    keep = [
        i for i in range(len(bins)) if bins[i]["hi"] is not None and all(
            curves[p][i]["offered"] >= MIN_OFFERED for p in pols)
    ]
    x = [
        (bins[i]["hi"] / 2 if bins[i]["lo"] <= 0 else math.sqrt(bins[i]["lo"] *
                                                                bins[i]["hi"]))
        for i in keep
    ]
    for p in pols:
        ax.plot(x, [curves[p][i]["share"] for i in keep],
                ls=LINESTYLE.get(p, "-"),
                color=COLOR[p],
                marker=MARKER[p],
                ms=4,
                lw=2,
                label=POLICY_LABELS[p])
    ax.set_xscale("log")
    ax.set_xticks(INTERVAL_TICKS, [_tick_secs(t) for t in INTERVAL_TICKS])
    ax.minorticks_off()
    ax.set_xlim(0.03, 1800)
    ax.set_ylim(0, 1.03)
    _pct_axis(ax.yaxis)
    ax.grid(axis="y", color="#dce2e1", lw=0.8)
    if title:
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")


def point1(data: Row, plt, out: Path) -> List[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _interval_axes(ax, data["q1"]["curves"], pols=data["mixedPolicies"])
    ax.set_xlabel("Time the block sat unused before it was needed again")
    ax.set_ylabel("Share of those reuses served")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Point 1 · Mixed workload: reuses served by wait",
                 loc="left",
                 fontweight="bold")
    return _save(fig, out, "point1_reuse_interval")


def _grouped_bars(ax,
                  rows: List[str],
                  values: Row,
                  ceiling: Optional[Row],
                  pols: Sequence[str] = POLICIES,
                  fmt=lambda v: f"{100 * v:.1f}%") -> None:
    """Horizontal bars per row and policy, seed ranges, ceiling ticks
    (none when `ceiling` is None)."""
    height = 0.78 / len(pols)
    for r, row in enumerate(rows):
        for k, p in enumerate(pols):
            v = values[p][row]
            y = r + (k - (len(pols) - 1) / 2) * height
            ax.barh(y,
                    v["mean"],
                    height * 0.9,
                    color=COLOR[p],
                    label=POLICY_LABELS[p] if r == 0 else None)
            if v["max"] > v["min"]:
                ax.plot([v["min"], v["max"]], [y, y],
                        color=INK,
                        lw=1,
                        alpha=0.6)
            ax.text(max(v["mean"], v["max"]) + 0.01 * ax.get_xlim()[1],
                    y,
                    fmt(v["mean"]),
                    va="center",
                    fontsize=7)
        if ceiling is None:
            continue
        c = ceiling[row]["mean"]
        half = len(pols) / 2 * height
        ax.plot([c, c], [r - half, r + half],
                color=INK,
                lw=1.5,
                ls=(0, (3, 2)),
                label="No eviction" if r == 0 else None)
    ax.set_yticks(range(len(rows)), [LABELS[r] for r in rows])
    ax.invert_yaxis()
    if ceiling is not None:
        ax.set_xlim(0, 1.08)
        _pct_axis(ax.xaxis)
        ax.set_xlabel("Usable prefix reuse (share of prompt tokens served)")
    ax.legend(frameon=False,
              ncol=6,
              loc="lower center",
              bbox_to_anchor=(0.5, 1.0),
              fontsize=8)


def point2(data: Row, plt, out: Path) -> List[Path]:
    q = data["q2"]
    pols = data["mixedPolicies"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6 + 0.5 * (len(pols) - 3)))
    _grouped_bars(ax, q["rows"], q["values"], q["ceiling"], pols)
    fig.suptitle("Point 2 · Mixed workload: reuse portion",
                 x=0.02,
                 ha="left",
                 fontweight="bold",
                 fontsize=10,
                 y=1.03)
    return _save(fig, out, "point2_reuse_portion")


def point3(data: Row, plt, out: Path) -> List[Path]:
    q = data["q3"]
    fig, (a, b) = plt.subplots(1,
                               2,
                               figsize=(9.6, 3.8),
                               gridspec_kw={"width_ratios": [1.3, 1]})
    models = q["models"]
    y = range(len(models))
    a.barh([i - 0.18 for i in y], [m["accuracy"] for m in models],
           0.34,
           color="#1c5cab",
           label="Requests labelled right")
    a.barh([i + 0.18 for i in y], [m["balanced"] for m in models],
           0.34,
           color="#86b6ef",
           label="Average of the four workloads")
    for i, m in enumerate(models):
        a.text(m["accuracy"] + 0.01,
               i - 0.18,
               f"{100 * m['accuracy']:.0f}%",
               va="center",
               fontsize=7)
        a.text(m["balanced"] + 0.01,
               i + 0.18,
               f"{100 * m['balanced']:.0f}%",
               va="center",
               fontsize=7)
    maj = q["majority"]
    a.axvline(maj["accuracy"],
              color=INK,
              lw=1.2,
              ls=(0, (3, 2)),
              label=f"Always guessing {LABELS[maj['label']]}")
    a.set_yticks(list(y), [m["label"] for m in models])
    a.invert_yaxis()
    a.set_xlim(0, 1.1)
    _pct_axis(a.xaxis)
    a.set_xlabel("Share right on unseen runs (seeds 2–4)")
    a.legend(frameon=False,
             fontsize=7,
             loc="lower center",
             bbox_to_anchor=(0.5, 1.0),
             ncol=2)
    c = q["confusion"]
    names = [LABELS[g] for g in c["classes"]]
    b.imshow(c["share"], cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            v = c["share"][i][j]
            b.text(j,
                   i,
                   f"{100 * v:.0f}%",
                   ha="center",
                   va="center",
                   fontsize=8,
                   color="white" if v > 0.5 else INK)
    b.set_xticks(range(len(names)), names, rotation=30, ha="right")
    b.set_yticks(range(len(names)), names)
    b.set_xlabel("Labelled as")
    b.set_ylabel("Really from")
    for spine in b.spines.values():
        spine.set_visible(False)
    fig.suptitle("Point 3 · Which workload sent a request",
                 x=0.02,
                 ha="left",
                 fontweight="bold",
                 fontsize=10,
                 y=1.05)
    fig.tight_layout()
    return _save(fig, out, "point3_identification")


def point4(data: Row, plt, out: Path) -> List[Path]:
    q = data["q4"]
    caps = q["capacities"]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for p in q["policies"]:
        ax.plot(caps,
                q["curve"][p],
                color=COLOR[p],
                marker=MARKER[p],
                ms=4,
                lw=2,
                ls=LINESTYLE.get(p, "-"),
                label=POLICY_LABELS[p] +
                (" (offline)" if p == "belady" else ""))
    for e in q["engine"]:
        ax.scatter(e["capacity"],
                   e["engine"],
                   s=70,
                   marker=MARKER[e["policy"]],
                   facecolors="none",
                   edgecolors=COLOR[e["policy"]],
                   lw=1.6,
                   zorder=3)
        ax.scatter(e["capacity"] * 1.06,
                   e["replay"],
                   s=22,
                   marker="x",
                   color=COLOR[e["policy"]],
                   lw=1.2,
                   zorder=3)
    ax.axhline(q["ceiling"],
               color=INK,
               lw=1.2,
               ls=(0, (3, 2)),
               label="No eviction")
    ax.axvline(q["fixed"], color="#7b878b", lw=0.8)
    ax.scatter([], [],
               s=60,
               marker="o",
               facecolors="none",
               edgecolors=INK_2,
               label="Engine run")
    ax.scatter([], [],
               s=22,
               marker="x",
               color=INK_2,
               label="Cache replay, same seeds")
    ax.set_xscale("log")
    ax.set_xticks(caps, [tokens(c) for c in caps])
    ax.minorticks_off()
    ax.set_ylim(*q["yDomain"])
    _pct_axis(ax.yaxis)
    ax.grid(axis="y", color="#dce2e1", lw=0.8)
    ax.set_xlabel("KV-cache capacity (tokens, log scale)")
    ax.set_ylabel("Usable prefix reuse")
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
    ax.set_title(
        "Point 4 · Mixed workload: cache capacity (lines: cache "
        "replay)",
        loc="left",
        fontweight="bold")
    return _save(fig, out, "point4_capacity")


def point5(data: Row, plt, out: Path) -> List[Path]:
    q = data["q5"]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    _grouped_bars(ax, list(GROUPS), q["values"], q["ceiling"])
    fig.suptitle("Point 5 · Each workload alone: reuse portion",
                 x=0.02,
                 ha="left",
                 fontweight="bold",
                 fontsize=10,
                 y=1.03)
    paths = _save(fig, out, "point5_alone_portion")
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), sharey=True)
    for ax, g in zip(axes.flat, GROUPS):
        _interval_axes(ax, q["curves"][g], f"{LABELS[g]} alone")
    axes[0][0].legend(frameon=False, loc="lower left", fontsize=7)
    fig.supxlabel("Time the block sat unused before it was needed again",
                  fontsize=9)
    fig.supylabel("Share of those reuses served", fontsize=9)
    fig.suptitle("Point 5 · Each workload alone: reuses served by wait",
                 x=0.02,
                 ha="left",
                 fontweight="bold",
                 fontsize=10)
    fig.tight_layout()
    return paths + _save(fig, out, "point5_alone_interval")


def point6(data: Row, plt, out: Path) -> List[Path]:
    q = data["q6"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = []
    for i, r in enumerate(q["rows"]):
        y = i + list(GROUPS).index(r["group"]) * 0.8
        c = COLOR[r["policy"]]
        ax.plot([r["alone"], r["mixed"]], [y, y], color=c, lw=2.2, alpha=0.7)
        ax.scatter(r["alone"],
                   y,
                   s=40,
                   marker=MARKER[r["policy"]],
                   facecolors="white",
                   edgecolors=c,
                   lw=1.6,
                   zorder=3)
        ax.scatter(r["mixed"],
                   y,
                   s=36,
                   marker=MARKER[r["policy"]],
                   color=c,
                   zorder=3)
        d = 100 * (r["mixed"] - r["alone"])
        ax.text(1.02, y, f"{d:+.1f} pts", va="center", fontsize=7)
        labels.append((y, f"{LABELS[r['group']]} · "
                       f"{POLICY_LABELS[r['policy']]}"))
    for k, g in enumerate(GROUPS):
        c = q["ceiling"][g]
        y0 = 3 * k + k * 0.8
        ax.plot([c, c], [y0 - 0.5, y0 + 2.5],
                color=INK,
                lw=1.3,
                ls=(0, (3, 2)))
    ax.set_yticks([y for y, _ in labels], [t for _, t in labels], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    _pct_axis(ax.xaxis)
    ax.set_xlabel("Usable prefix reuse: alone (hollow) → sharing the cache "
                  "(filled); dashed: no eviction")
    ax.set_title("Point 6 · Each workload alone against sharing the cache",
                 loc="left",
                 fontweight="bold")
    return _save(fig, out, "point6_alone_vs_mixed")


def point7(data: Row, plt, out: Path) -> List[Path]:
    """The new policies: each workload's usable reuse against cache size
    (cache replay), and evicted blocks per second by workload (engine)."""
    q = data["q7"]
    caps = q["capacities"]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2))
    for ax, g in zip(axes.flat, GROUPS):
        panel = q["panels"][g]
        for p in q["policies"]:
            ax.plot(caps,
                    panel["curve"][p],
                    ls=LINESTYLE.get(p, "-"),
                    color=COLOR[p],
                    marker=MARKER[p],
                    ms=4,
                    lw=1.8,
                    label=POLICY_LABELS[p])
        ax.axhline(panel["ceiling"],
                   color=INK,
                   lw=1.1,
                   ls=(0, (3, 2)),
                   label="No eviction")
        ax.set_xscale("log")
        ticks = [c for i, c in enumerate(caps) if i % 2 == 0]
        ax.set_xticks(ticks, [tokens(c) for c in ticks])
        ax.minorticks_off()
        ax.set_ylim(*panel["yDomain"])
        _pct_axis(ax.yaxis)
        ax.grid(axis="y", color="#dce2e1", lw=0.8)
        ax.set_title(f"{LABELS[g]} in the mix",
                     loc="left",
                     fontsize=9,
                     fontweight="bold")
    axes[1][0].set_xlabel("Cache (tokens, log scale)")
    axes[1][1].set_xlabel("Cache (tokens, log scale)")
    axes[0][0].set_ylabel("Usable prefix reuse")
    axes[1][0].set_ylabel("Usable prefix reuse")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles,
               labels,
               frameon=False,
               ncol=6,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.98),
               fontsize=8)
    fig.suptitle(
        "Point 7 · Each workload in the mix against cache size "
        "(cache replay)",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=10,
        y=1.05)
    fig.tight_layout()
    paths = _save(fig, out, "point7_capacity_by_workload")
    ev = q["evictions"]
    pols = [p for p in data["mixedPolicies"] if p in ev]
    # As on the page: only once a new policy has engine runs.
    if any(p in NEW_POLICIES for p in pols):
        rows = ["all", *GROUPS]
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        top = max(ev[p][r]["max"] for p in pols for r in rows)
        ax.set_xlim(0, 1.15 * top)
        _grouped_bars(ax, rows, ev, None, pols, fmt=lambda v: f"{v:.0f}/s")
        ax.set_xlabel("Blocks evicted per second, by the workload that last "
                      "used them")
        fig.suptitle("Point 7 · Evictions by workload (engine runs)",
                     x=0.02,
                     ha="left",
                     fontweight="bold",
                     fontsize=10,
                     y=1.03)
        paths += _save(fig, out, "point7_evictions")
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",
                        type=Path,
                        default=Path("results/m3-study/summary.json"))
    parser.add_argument("--out",
                        type=Path,
                        default=Path("report/milestone3/figures/task2"))
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False
    })
    data = build_data(json.loads(args.summary.read_text()))
    if not data["q5"]:
        raise SystemExit("summary.json has no isolated runs yet")
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    draws = [point1, point2, point3, point4, point5, point6]
    if data.get("q7"):
        draws.append(point7)
    for draw in draws:
        written += draw(data, plt, args.out)
        plt.close("all")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
