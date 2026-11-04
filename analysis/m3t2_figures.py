"""Milestone 3 Task 2, Phase 1: report figures from summary.json.

    python -m analysis.m3t2_figures --summary results/m3-study/summary.json \\
        --out report/milestone3/figures/task2

Draws the static (PDF and PNG) versions of the results page's main charts
for the technical report (constitution V: every figure from a committed
script). Workload colors follow the page's validated palette.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

COLORS = {
    "sharegpt": "#2a78d6",
    "bailian-chat": "#eb6834",
    "agentbank": "#1baf7a",
    "ccbench": "#eda100",
    "bailian-thinking": "#e87ba4",
    "bailian-coder": "#008300",
    "bailian-api": "#6250d6",
}
POLICY = {
    "ceiling": "No eviction",
    "lru": "LRU",
    "lfu": "LFU",
    "fifo": "FIFO",
    "belady": "Belady (offline)"
}
POLICY_STYLE = {
    "ceiling": ("#111a1d", "o", "none"),
    "lru": ("#263b44", "o", None),
    "lfu": ("#6f838b", "D", None),
    "fifo": ("#a2988a", "s", None),
    "belady": ("#0c5c70", "^", None)
}
Row = Dict[str, Any]


def _mids(edges: List[float]) -> List[float]:
    return [
        hi / 2 if lo <= 0 else math.sqrt(lo * hi)
        for lo, hi in zip(edges[:-1], edges[1:])
    ]


def _save(fig, out: Path, name: str) -> List[Path]:
    paths = []
    for ext in ("pdf", "png"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    return paths


def come_back(s: Row, plt, out: Path) -> List[Path]:
    cfg, cb = s["config"], s["q1"]["come_back"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for w in cfg["workloads"]:
        ax.plot(cb["grid"],
                cb[w]["came_back"],
                color=COLORS[w],
                lw=2,
                label=cfg["workload_labels"][w])
    for i, p in enumerate(cfg["policies"]):
        q = s["policy_views"][p]["eviction_ages"]["all"]["quantiles"]
        ax.axvline(q["p50"], color=POLICY_STYLE[p][0], lw=1, ls=":")
        # Staggered: LRU's and FIFO's medians are close.
        ax.text(q["p50"],
                1.02 + 0.07 * i,
                f"{POLICY[p]} evicts at {q['p50']:.0f} s",
                ha="center",
                fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("Time since the block was freed (s)")
    ax.set_ylabel("Share looked up again")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
    return _save(fig, out, "fig1_come_back")


def hit_by_interval(s: Row, plt, out: Path) -> List[Path]:
    cfg = s["config"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, p in zip(axes, cfg["policies"]):
        h = s["q1"]["hit_by_interval"][p]
        x = _mids(h["edges"])
        for w in cfg["workloads"]:
            ax.plot(x,
                    h[w]["usable_share"],
                    color=COLORS[w],
                    lw=1.8,
                    marker="o",
                    ms=3,
                    label=cfg["workload_labels"][w])
        ax.set_xscale("log")
        ax.set_title(POLICY[p], fontsize=10)
        ax.set_xlabel("Reuse interval (s)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Share of reuses served")
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(fontsize=7, frameon=False, loc="upper right")
    return _save(fig, out, "fig2_hit_by_interval")


def usable(s: Row, plt, out: Path) -> List[Path]:
    cfg = s["config"]
    rows = ["all"] + cfg["workloads"]
    labels = ["Whole mix"
              ] + [cfg["workload_labels"][w] for w in cfg["workloads"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, w in enumerate(rows):
        lru = s["headline"]["lru"][w]["usable"]["mean"]
        ceil = s["headline"]["ceiling"][w]["usable"]["mean"]
        ax.plot([lru, ceil], [i, i],
                color="#dce2e1",
                lw=6,
                zorder=1,
                solid_capstyle="round")
        for p in ("ceiling", "lru", "lfu", "fifo"):
            u = s["headline"][p][w]["usable"]
            color, marker, face = POLICY_STYLE[p]
            ax.plot([u["min"], u["max"]], [i, i], color=color, lw=1, zorder=2)
            ax.scatter([u["mean"]], [i],
                       color=face or color,
                       edgecolors=color,
                       marker=marker,
                       s=36,
                       zorder=3,
                       label=POLICY[p] if i == 0 else None)
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Usable reuse (share of prompt tokens served)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8,
              frameon=False,
              ncol=4,
              loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    return _save(fig, out, "fig3_usable_by_policy")


def lost(s: Row, plt, out: Path) -> List[Path]:
    cfg = s["config"]
    seeds = [str(x) for x in cfg["reported_seeds"]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    parts = [("kept", "Reused", "#1c5cab"), ("evicted", "Evicted", "#86b6ef"),
             ("cut_off", "Cached but cut off", "#d03b3b")]
    for ax, p in zip(axes, cfg["policies"]):
        for i, w in enumerate(cfg["workloads"]):
            tot = {
                k: sum(s["lost"][p][x][w][k] for x in seeds)
                for k, _, _ in parts
            }
            n = sum(tot.values()) or 1
            left = 0.0
            for k, label, color in parts:
                ax.barh(i,
                        tot[k] / n,
                        left=left,
                        color=color,
                        height=0.7,
                        label=label if i == 0 else None)
                left += tot[k] / n
        ax.set_title(POLICY[p], fontsize=10)
        ax.set_xlim(0, 1)
    axes[0].set_yticks(range(len(cfg["workloads"])),
                       [cfg["workload_labels"][w] for w in cfg["workloads"]])
    axes[0].invert_yaxis()
    axes[1].set_xlabel("Share of the blocks the no-eviction run reused")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    return _save(fig, out, "fig4_lost_reuse")


def capacity(s: Row, plt, out: Path) -> List[Path]:
    sim = (s.get("sim") or {}).get("balanced-human")
    if not sim:
        return []
    caps = sorted(int(k) for k in sim["lru"] if k != "ceiling")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for p in ("belady", "lru", "lfu", "fifo"):
        color, marker, _ = POLICY_STYLE[p]
        ax.plot(caps, [sim[p][str(c)]["all"]["mean"] for c in caps],
                color=color,
                marker=marker,
                ms=4,
                lw=1.8,
                label=POLICY[p])
    ceil = sim["ceiling"]["ceiling"]["all"]["mean"]
    ax.axhline(ceil, color="#7b878b", lw=1, ls="--", label="No eviction")
    ax.axvline(s["config"]["capacity_tokens"], color="#7b878b", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cache size (tokens)")
    ax.set_ylabel("Usable reuse (simulator)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, out, "fig5_capacity_sim")


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
    s = json.loads(args.summary.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for draw in (come_back, hit_by_interval, usable, lost, capacity):
        written += draw(s, plt, args.out)
        plt.close("all")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
