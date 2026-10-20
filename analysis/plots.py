"""Figures comparing workload settings (spec 004, User Story 3).

Each function takes ``{setting: data}`` and writes one PDF and one PNG.
"""
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.metrics import cdf  # noqa: E402

COLORS = {"single": "#1b6ca8", "multi": "#d1495b"}
LABELS = {"single": "single-turn", "multi": "multi-turn"}

plt.rcParams.update({
    "figure.figsize": (4.2, 2.8),
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})


def _style(setting: str) -> Dict[str, str]:
    return {
        "color": COLORS.get(setting, "#555555"),
        "label": LABELS.get(setting, setting)
    }


def _save(fig, path: Path) -> List[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=200)
        written.append(out)
    plt.close(fig)
    return written


def cdf_plot(values: Mapping[str, Sequence[float]],
             xlabel: str,
             path: Path,
             log_x: bool = False,
             symlog_x: bool = False) -> List[Path]:
    fig, ax = plt.subplots()
    for setting, data in values.items():
        if not data:
            continue
        xs, ys = cdf(data)
        ax.step(xs, ys, where="post", **_style(setting))
    if log_x:
        ax.set_xscale("log")
    elif symlog_x:
        ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    return _save(fig, path)


def by_turn_plot(means: Mapping[int, float],
                 counts: Mapping[int, int],
                 path: Path,
                 setting: str = "multi") -> List[Path]:
    """Mean cached fraction by turn index, with the request count."""
    turns = sorted(means)
    fig, ax = plt.subplots()
    ax.plot(turns, [means[t] for t in turns],
            marker="o",
            markersize=3,
            **_style(setting))
    ax.set_xlabel("turn index")
    ax.set_ylabel("mean cached fraction")
    ax.set_ylim(0, 1)
    twin = ax.twinx()
    twin.bar(turns, [counts[t] for t in turns],
             color="#999999",
             alpha=0.25,
             width=0.8)
    twin.set_ylabel("requests")
    twin.grid(False)
    ax.set_zorder(twin.get_zorder() + 1)
    ax.patch.set_visible(False)
    return _save(fig, path)


def over_time_plot(windows: Mapping[str, Sequence[Mapping[str, float]]],
                   path: Path) -> List[Path]:
    """Windowed block hit rate over time."""
    fig, ax = plt.subplots()
    for setting, rows in windows.items():
        ax.plot([r["start_s"] for r in rows], [r["hit_rate"] for r in rows],
                **_style(setting))
    ax.set_xlabel("time since first request (s)")
    ax.set_ylabel("block hit rate")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    return _save(fig, path)


def sources_plot(shares: Mapping[str, Mapping[str, float]],
                 path: Path) -> List[Path]:
    """Stacked share of hits by source for each setting."""
    settings = list(shares)
    categories = sorted({c for s in shares.values() for c in s})
    fig, ax = plt.subplots()
    bottom = [0.0] * len(settings)
    for category in categories:
        heights = [shares[s].get(category, 0.0) for s in settings]
        ax.bar([LABELS.get(s, s) for s in settings],
               heights,
               bottom=bottom,
               label=category)
        bottom = [b + h for b, h in zip(bottom, heights)]
    ax.set_ylabel("share of hits")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    return _save(fig, path)
