"""Tables of cache-replay results by policy and capacity (spec 007).

Reads the JSONL rows of ``python -m analysis.run_m3_task2 sim`` and prints,
for each capacity, every policy's usable reused fraction in the measured
window (mean over the plans' seeds, with its minimum and maximum) and each
workload's mean. With ``--against POLICY`` it adds, for every other policy,
the per-seed difference from that policy and the study's verdict: higher or
lower only when every seed agrees on the sign and the mean differs by at
least 0.5 points, otherwise a tie::

    python -m analysis.sim_table results/m3-explore/wa-heldout.jsonl \\
        results/m3-explore/heldout-baselines.jsonl --against lru
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TIE_POINTS = 0.5

Key = Tuple[str, str]  # (policy, capacity)


def load_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """The rows without an error, in file order."""
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row and "summary" in row:
                    rows.append(row)
    return rows


def measured(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The measured-window summary when there is one, else the run's."""
    window = summary.get("window")
    return window if isinstance(window, dict) else summary


def seed_of(row: Dict[str, Any]) -> str:
    return Path(row["plan"]).name.rsplit("-s", 1)[-1]


def collect(
    rows: Iterable[Dict[str, Any]]
) -> Tuple[Dict[Key, Dict[str, float]], Dict[Key, Dict[str, List[float]]]]:
    """Usable reuse in percent by (policy, capacity) and seed, and the
    workloads' values by (policy, capacity). A later row for the same
    policy, capacity and seed replaces an earlier one."""
    whole: Dict[Key, Dict[str, float]] = defaultdict(dict)
    per_seed_workloads: Dict[Key, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict))
    for row in rows:
        key = (row["policy"], str(row["capacity"]))
        window = measured(row["summary"])
        seed = seed_of(row)
        whole[key][seed] = 100 * window["usable_reused_fraction"]
        for workload, values in window.get("workloads", {}).items():
            fraction = values.get("usable_reused_fraction")
            if fraction is not None:
                per_seed_workloads[key][workload][seed] = 100 * fraction
    workloads = {
        key: {w: list(seeds.values())
              for w, seeds in by_workload.items()}
        for key, by_workload in per_seed_workloads.items()
    }
    return whole, workloads


def verdict(differences: List[float]) -> str:
    mean = statistics.fmean(differences)
    same_sign = all(d > 0 for d in differences) or all(d < 0
                                                       for d in differences)
    if same_sign and abs(mean) >= TIE_POINTS:
        return "higher" if mean > 0 else "lower"
    return "tie"


def render(rows: Iterable[Dict[str, Any]],
           against: Optional[str] = None) -> str:
    whole, workloads = collect(rows)
    capacities = sorted({c
                         for _, c in whole},
                        key=lambda c: (not c.isdigit(), int(c)
                                       if c.isdigit() else 0, c))
    lines = []
    for capacity in capacities:
        keys = sorted((k for k in whole if k[1] == capacity),
                      key=lambda k: -statistics.fmean(whole[k].values()))
        names = sorted({w for k in keys for w in workloads.get(k, {})})
        lines.append(f"\n== capacity {capacity}")
        lines.append("policy | mean | min | max | " + " | ".join(names) +
                     " | seeds")
        for key in keys:
            values = whole[key]
            cells = [
                f"{statistics.fmean(workloads[key][w]):.2f}" if workloads.get(
                    key, {}).get(w) else "-" for w in names
            ]
            lines.append(f"{key[0]} | {statistics.fmean(values.values()):.2f}"
                         f" | {min(values.values()):.2f} | "
                         f"{max(values.values()):.2f} | " + " | ".join(cells) +
                         " | " + ",".join(sorted(values)))
        base = whole.get((against, capacity)) if against else None
        if base:
            lines.append(f"-- against {against}, per seed")
            for key in keys:
                if key[0] == against:
                    continue
                seeds = sorted(set(whole[key]) & set(base))
                if not seeds:
                    continue
                diffs = [whole[key][s] - base[s] for s in seeds]
                lines.append(f"{key[0]} | " + " ".join(f"{d:+.2f}"
                                                       for d in diffs) +
                             f" | mean {statistics.fmean(diffs):+.2f} | "
                             f"{verdict(diffs)}")
    return "\n".join(lines).lstrip("\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=Path, nargs="+")
    parser.add_argument("--against",
                        help="Compare every policy with this one, per seed")
    args = parser.parse_args(argv)
    print(render(load_rows(args.rows), args.against))


if __name__ == "__main__":
    main()
