"""Figures comparing workload settings (spec 004, User Story 3) and cache
configurations (spec 005, User Story 3).

Each function takes ``{setting: data}`` (a setting may also be an eviction
policy) and writes one PDF and one PNG.
"""
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

from analysis.metrics import cdf  # noqa: E402

COLORS = {
    "single": "#1b6ca8",
    "multi": "#d1495b",
    "lru": "#1b6ca8",
    "fifo": "#e09f3e",
    "lfu": "#2a9d8f",
}
LABELS = {
    "single": "single-turn",
    "multi": "multi-turn",
    "lru": "LRU",
    "fifo": "FIFO",
    "lfu": "LFU",
}
MARKERS = {"lru": "o", "fifo": "s", "lfu": "^"}

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


def _compact(value: float, _position: Any = None) -> str:
    """1,500 -> "1.5k", 2,000,000 -> "2M"."""
    for divisor, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= divisor:
            return f"{value / divisor:g}{suffix}"
    return f"{value:g}"


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


def panel_lines_plot(
    rows: Sequence[Mapping[str, Any]],
    x: str,
    ys: Sequence[str],
    panel_by: str,
    path: Path,
    line_by: str = "policy",
    labels: Optional[Mapping[str, str]] = None,
    title: str = "{name} {value}",
    log2_x: bool = True,
    symlog_y: Sequence[str] = ()) -> List[Path]:
    """Small multiples: one row per metric in `ys`, one column per value of
    `panel_by`, one line per value of `line_by`, `x` on the x-axis. `title`
    formats each column's title from `name` (the label of `panel_by`) and
    `value`.

    Rows whose metric is None (infeasible or missing runs) are left out.
    """
    labels = labels or {}
    panels = sorted({r[panel_by] for r in rows})
    lines = list(dict.fromkeys(r[line_by] for r in rows))
    width, height = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(len(ys),
                             len(panels),
                             figsize=(0.65 * width * len(panels),
                                      0.75 * height * len(ys)),
                             sharex=True,
                             sharey="row",
                             squeeze=False)
    for col, panel in enumerate(panels):
        for row_index, y in enumerate(ys):
            ax = axes[row_index][col]
            for line in lines:
                points = sorted((r[x], r[y]) for r in rows
                                if r[panel_by] == panel and r[line_by] == line
                                and r[y] is not None)
                if points:
                    ax.plot([p[0] for p in points], [p[1] for p in points],
                            marker=MARKERS.get(line, "o"),
                            markersize=3,
                            **_style(line))
            if log2_x:
                ax.set_xscale("log", base=2)
            if y in symlog_y:
                ax.set_yscale("symlog", linthresh=1)
                values = [r[y] for r in rows if r[y]]
                # A range within one decade gets ticks at 1, 2 and 5 x 10^n,
                # so it is labelled at all; wider ones only at powers of 10.
                narrow = bool(values) and max(values) < 10 * min(values)
                ax.yaxis.set_major_locator(
                    ticker.SymmetricalLogLocator(
                        base=10,
                        linthresh=1,
                        subs=[1, 2, 5] if narrow else [1]))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(_compact))
            if row_index == 0:
                ax.set_title(title.format(name=labels.get(panel_by, panel_by),
                                          value=panel),
                             fontsize=8)
            if row_index == len(ys) - 1:
                ax.set_xlabel(labels.get(x, x))
            if col == 0:
                ax.set_ylabel(labels.get(y, y))
    # Every panel's lines, so a line missing from the first still shows.
    legend: Dict[str, Any] = {}
    for ax in fig.axes:
        for handle, name in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(name, handle)
    handles, names = list(legend.values()), list(legend)
    if handles:
        fig.legend(handles,
                   names,
                   loc="lower center",
                   ncol=len(names),
                   bbox_to_anchor=(0.5, 1.0),
                   frameon=False)
    return _save(fig, path)
