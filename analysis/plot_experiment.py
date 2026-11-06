"""Plot a saved policy comparison; never starts a server.

python -m analysis.plot_experiment --out results/bailian-500-cap65536 \
    --figures report/milestone3/figures/task1/bailian
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from analysis import metrics, plots
from analysis.run_experiment import checksum
from analysis.run_m2_task2 import git_state

STYLES = {"lru": "-", "lfu": "--", "fifo": ":"}
BARS = [("block_hit_rate_log", "Block hit rate", "%", 100),
        ("usable_reused_fraction", "Usable token reuse", "%", 100),
        ("eviction_rate_per_1k_lookups", "Eviction rate", "per 1,000 lookups",
         1)]
CDFS = [("cached", "usable fraction of prompt"), ("hits", "hits per block"),
        ("intervals", "reuse interval (s)")]


def read_csv(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def load_data(out):
    """Keep zero observations and reject inconsistent plotting populations."""
    experiment = json.loads((out / "experiment.json").read_text())
    for path, digest in experiment["checksums"].items():
        if checksum(path) != digest:
            raise ValueError(f"Frozen sample changed: {path}")
    rows = read_csv(out / "summary.csv")
    if not rows or any(r["status"] != "finished" for r in rows):
        raise ValueError("All policies must have finished before plotting")
    order = {"lru": 0, "lfu": 1, "fifo": 2}
    rows.sort(key=lambda r: order.get(r["policy"], 3))
    data, populations, inputs = {}, [], {
        str(out / "experiment.json"): checksum(out / "experiment.json")
    }
    for row in rows:
        folder = out / row["name"]
        summary = json.loads((folder / "summary.json").read_text())
        run = json.loads((folder / "run.json").read_text())
        if not summary["valid_measurement"] or run["sample"] != experiment:
            raise ValueError("Invalid measurement or inconsistent sample")
        raw = [
            json.loads(line)
            for line in (folder / "client.jsonl").read_text().splitlines()
        ]
        population = {r["request_id"]: r["status"] for r in raw}
        if len(population) != len(raw):
            raise ValueError("Duplicate request IDs")
        populations.append(population)
        requests = read_csv(folder / "requests.csv")
        hits = [int(r["hits"]) for r in read_csv(folder / "blocks.csv")]
        intervals = [
            float(r["interval_s"])
            for r in read_csv(folder / "reuse_intervals.csv")
        ]
        windows = [{k: float(v)
                    for k, v in r.items()}
                   for r in read_csv(folder / "hit_rate_windows.csv")]
        excluded = Counter(r["status"] for r in raw
                           if r["status"] != "completed")
        if (dict(excluded) != summary["requests_excluded"]
                or len(requests) != summary["requests_completed"]
                or {r["request_id"]
                    for r in requests} !=
            {r["request_id"]
             for r in raw if r["status"] == "completed"}
                or len(hits) != summary["blocks"]
                or sum(hits) != summary["event_counts"].get("hit", 0)):
            raise ValueError("CSV populations disagree with saved results")
        measured = [{
            "prompt_tokens": int(r["prompt_tokens"]),
            "usable_cached_tokens": int(r["usable_cached_tokens"])
        } for r in requests]
        reuse = metrics.token_reused_fraction(measured, "usable_cached_tokens")
        if not math.isclose(
                reuse, summary["usable_reused_fraction"], abs_tol=1e-12):
            raise ValueError("Request CSV disagrees with overall usable reuse")
        lookups = sum(r["lookups"] for r in windows)
        rate = sum(r["lookups"] * r["hit_rate"] for r in windows) / lookups
        if not math.isclose(rate, summary["block_hit_rate_log"],
                            abs_tol=1e-12):
            raise ValueError("Time windows disagree with overall hit rate")
        data[row["policy"]] = {
            "summary": summary,
            "cached": [float(r["usable_cached_fraction"]) for r in requests],
            "hits": hits,
            "intervals": intervals,
            "windows": windows
        }
        for name in ("requests.csv", "blocks.csv", "reuse_intervals.csv",
                     "hit_rate_windows.csv", "summary.json", "client.jsonl",
                     "run.json"):
            inputs[str(folder / name)] = checksum(folder / name)
    if any(p != populations[0] for p in populations[1:]):
        raise ValueError("Policies have different completed/excluded requests")
    return experiment, data, inputs


def bar_figure(data, specs, path, note):
    fig, axes = plots.plt.subplots(1,
                                   len(specs),
                                   squeeze=False,
                                   figsize=(2.5 * len(specs), 2.8))
    for ax, (key, title, unit, scale) in zip(axes[0], specs):
        values = [d["summary"][key] * scale for d in data.values()]
        bars = ax.bar([plots.LABELS.get(p, p) for p in data],
                      values,
                      color=[plots.COLORS.get(p, "#555555") for p in data])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.2 or 1)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.text(.02, -.04, note, fontsize=7)
    return plots._save(fig, path)


def cdf_figure(data, specs, path, note):
    fig, axes = plots.plt.subplots(1,
                                   len(specs),
                                   squeeze=False,
                                   figsize=(2.5 * len(specs), 2.8))
    for ax, (key, label) in zip(axes[0], specs):
        inset = ax.inset_axes([.32, .16, .64, .52]) if key == "hits" else None
        for policy, values in data.items():
            xs, ys = metrics.cdf(values[key])
            style = {
                **plots._style(policy), "linestyle": STYLES.get(policy, "-")
            }
            ax.step(xs, ys, where="post", **style)
            if inset is not None:
                inset.step(xs, ys, where="post", **style)
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        if key != "cached":
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(left=0)
        if inset is not None:
            zero_share = min(d["hits"].count(0) / len(d["hits"])
                             for d in data.values())
            tail = .005 if zero_share >= .995 else .02
            inset.set_xscale("symlog", linthresh=1)
            inset.set_xlim(left=0)
            inset.set_ylim(1 - tail, 1.0001)
            inset.set_title(f"Upper {100 * tail:g}%: same data", fontsize=6)
            inset.tick_params(labelsize=6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles,
               labels,
               loc="upper center",
               ncol=len(data),
               bbox_to_anchor=(.5, 1.06),
               fontsize=8)
    fig.tight_layout()
    fig.text(.02, -.04, note, fontsize=7)
    return plots._save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    experiment, data, inputs = load_data(args.out)
    figures = args.figures or args.out / "figures"
    settings = experiment["config"]
    counts = next(iter(data.values()))["summary"]
    excluded = sum(counts["requests_excluded"].values())
    note = (f"{counts['requests_completed']} completed/policy; "
            f"{excluded} excluded.\n"
            f"Cache: {settings['capacity_tokens']:,} tokens; "
            f"block size: {settings['block_size']}.")
    files = bar_figure(data, BARS, figures / "fig1_policy_comparison", note)
    files += cdf_figure(data, CDFS, figures / "fig2_reuse_profiles", note)
    for bar in BARS:
        files += bar_figure(data, [bar], figures / bar[0], note)
    for cdf in CDFS:
        files += cdf_figure(data, [cdf], figures / (cdf[0] + "_cdf"), note)
    files += plots.over_time_plot({p: d["windows"]
                                   for p, d in data.items()},
                                  figures / "fig3_hit_rate_over_time")
    record = {
        "settings": settings,
        "inputs_sha256": inputs,
        "plot_source_sha256": checksum(Path(__file__)),
        "plot_base_commit": git_state()["commit"],
        "completed_per_policy": counts["requests_completed"],
        "excluded_per_policy": counts["requests_excluded"],
        "zero_hit_blocks": {p: d["hits"].count(0)
                            for p, d in data.items()},
        "figures": [str(p) for p in files]
    }
    (figures / "plot_manifest.json").write_text(json.dumps(record, indent=2))
    print(f"Saved {len(files)} PNG/PDF files to {figures}")


if __name__ == "__main__":
    main()
