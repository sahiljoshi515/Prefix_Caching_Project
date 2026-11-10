"""Tables of cache-replay and engine results by policy and capacity
(spec 007).

Reads the JSONL rows of ``python -m analysis.run_m3_task2 sim``, or, with
``--engine``, finished engine runs (a grid's run folders, after
``python -m analysis.tables`` has built their tables), and prints,
for each capacity, every policy's usable reused fraction in the measured
window (mean over the plans' seeds, with its minimum and maximum) and each
workload's mean. With ``--against POLICY`` it adds, for every other policy,
the per-seed difference from that policy and the study's verdict: higher or
lower only when every seed agrees on the sign and the mean differs by at
least 0.5 points, otherwise a tie. It then lists, for every policy, the
workloads whose mean is more than 2 points below that policy's
(``plan.md``'s decision gate)::

    python -m analysis.sim_table results/m3-explore/wa-heldout.jsonl \\
        results/m3-explore/heldout-baselines.jsonl --against lru
    python -m analysis.sim_table --against lru \\
        --engine results/m3-task2/balanced-human-s{2,3,4}-*-cap131072

An engine run's usable reuse is the study's (``analysis.m3t2_study``,
window and workloads from ``analysis/m3t2_config.json``).

Seed 1 tunes the settings, so it is left out unless ``--seeds`` names it.
Rows of different mixes, timings, windows or block sizes are refused: their
numbers are not comparable.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TIE_POINTS = 0.5
GATE_POINTS = 2.0
TUNING_SEED = "1"

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


def engine_rows(run_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    """Rows like the replay's for finished engine runs: usable reuse in
    the study's window, overall and per workload, from each run's
    ``tables/requests.parquet``. A ceiling run's capacity is
    ``"ceiling"``, as in the replay."""
    from analysis import m3t2_study as study
    from analysis.run_m3_task2 import run_config
    cfg = study.load_config()
    rows = []
    for run_dir in map(Path, run_dirs):
        run = json.loads((run_dir / "run.json").read_text())
        if "finished_at_utc" not in run:
            raise ValueError(f"{run_dir} has not finished")
        if not (run_dir / "tables" / "requests.parquet").exists():
            raise ValueError(f"{run_dir} has no tables; run python -m "
                             "analysis.tables on it first")
        config = run_config(run_dir)
        usable = study.usable_by_workload(study.load_requests(run_dir),
                                          cfg["workloads"], cfg["w0_s"],
                                          cfg["stop_s"], config.block_size)
        rows.append({
            "plan": run["plan_dir"],
            "policy": config.policy,
            "capacity":
            "ceiling" if config.ceiling else config.capacity_tokens,
            "block_size": config.block_size,
            "timing": "engine",
            "window": [cfg["w0_s"], cfg["stop_s"]],
            "summary": {
                "window": {
                    "usable_reused_fraction": usable["all"]["fraction"],
                    "workloads": {
                        w: {
                            "usable_reused_fraction": usable[w]["fraction"]
                        }
                        for w in cfg["workloads"]
                    },
                }
            },
        })
    return rows


def measured(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The measured-window summary when there is one, else the run's."""
    window = summary.get("window")
    return window if isinstance(window, dict) else summary


def seed_of(row: Dict[str, Any]) -> str:
    return Path(row["plan"]).name.rsplit("-s", 1)[-1]


def setup_of(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """What must match for rows to share a table: the mix (the plan's name
    without its seed), the timing, the window and the block size."""
    window = row.get("window")
    return (Path(row["plan"]).name.rsplit("-s", 1)[0], row.get("timing"),
            None if window is None else tuple(window), row.get("block_size"))


def select(rows: Iterable[Dict[str, Any]],
           seeds: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """The rows of `seeds` (default: all but the tuning seed), after
    checking that they share one setup."""
    wanted = None if seeds is None else {str(s) for s in seeds}
    rows = [
        row for row in rows
        if (seed_of(row) != TUNING_SEED if wanted is None else seed_of(row) in
            wanted)
    ]
    setups = sorted({setup_of(row) for row in rows}, key=repr)
    if len(setups) > 1:
        raise ValueError("rows from different setups (mix, timing, window, "
                         "block size): " + "; ".join(map(repr, setups)))
    return rows


def collect(
    rows: Iterable[Dict[str, Any]]
) -> Tuple[Dict[Key, Dict[str, float]], Dict[Key, Dict[str, List[float]]]]:
    """Usable reuse in percent by (policy, capacity) and seed, and the
    workloads' values by (policy, capacity). A later row for the same
    policy, capacity and seed replaces an earlier one."""
    whole, per_seed = collect_by_seed(rows)
    workloads = {
        key: {w: list(seeds.values())
              for w, seeds in by_workload.items()}
        for key, by_workload in per_seed.items()
    }
    return whole, workloads


def collect_by_seed(
    rows: Iterable[Dict[str, Any]]
) -> Tuple[Dict[Key, Dict[str, float]], Dict[Key, Dict[str, Dict[str,
                                                                 float]]]]:
    """As `collect`, with each workload's values by seed."""
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
    return whole, per_seed_workloads


def verdict(differences: List[float]) -> str:
    mean = statistics.fmean(differences)
    same_sign = all(d > 0 for d in differences) or all(d < 0
                                                       for d in differences)
    if same_sign and abs(mean) >= TIE_POINTS:
        return "higher" if mean > 0 else "lower"
    return "tie"


def gate_failures(
        policy: Dict[str, Dict[str, float]],
        base: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
    """The workloads whose mean, over the seeds both have, is more than
    GATE_POINTS below the baseline's, with the difference."""
    failures = []
    for workload in sorted(policy):
        seeds = sorted(set(policy[workload]) & set(base.get(workload, {})))
        if not seeds:
            continue
        diff = statistics.fmean(policy[workload][s] - base[workload][s]
                                for s in seeds)
        if diff < -GATE_POINTS:
            failures.append((workload, diff))
    return failures


def render(rows: Iterable[Dict[str, Any]],
           against: Optional[str] = None,
           seeds: Optional[Iterable[str]] = None) -> str:
    whole, by_seed = collect_by_seed(select(rows, seeds))
    capacities = sorted({c
                         for _, c in whole},
                        key=lambda c: (not c.isdigit(), int(c)
                                       if c.isdigit() else 0, c))
    lines = []
    for capacity in capacities:
        keys = sorted((k for k in whole if k[1] == capacity),
                      key=lambda k: -statistics.fmean(whole[k].values()))
        names = sorted({w for k in keys for w in by_seed.get(k, {})})
        lines.append(f"\n== capacity {capacity}")
        lines.append("policy | mean | min | max | " + " | ".join(names) +
                     " | seeds")
        for key in keys:
            values = whole[key]
            cells = [
                f"{statistics.fmean(by_seed[key][w].values()):.2f}"
                if by_seed.get(key, {}).get(w) else "-" for w in names
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
                shared = sorted(set(whole[key]) & set(base))
                if not shared:
                    continue
                diffs = [whole[key][s] - base[s] for s in shared]
                lines.append(f"{key[0]} | " + " ".join(f"{d:+.2f}"
                                                       for d in diffs) +
                             f" | mean {statistics.fmean(diffs):+.2f} | "
                             f"{verdict(diffs)}")
            lines.append(f"-- workloads more than {GATE_POINTS:g} points "
                         f"below {against} (mean)")
            for key in keys:
                if key[0] == against:
                    continue
                failures = gate_failures(by_seed.get(key, {}),
                                         by_seed.get((against, capacity), {}))
                lines.append(f"{key[0]} | " +
                             (", ".join(f"{w} {d:+.2f}"
                                        for w, d in failures) or "none"))
    return "\n".join(lines).lstrip("\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows",
                        type=Path,
                        nargs="*",
                        help="JSONL rows of run_m3_task2 sim")
    parser.add_argument("--engine",
                        type=Path,
                        nargs="+",
                        default=[],
                        metavar="RUN_DIR",
                        help="Finished engine runs, with their tables")
    parser.add_argument("--against",
                        help="Compare every policy with this one, per seed")
    parser.add_argument("--seeds",
                        nargs="+",
                        help="Only these seeds (default: all but seed "
                        f"{TUNING_SEED}, the tuning seed)")
    args = parser.parse_args(argv)
    if not args.rows and not args.engine:
        parser.error("give JSONL rows, --engine run folders, or both")
    rows = load_rows(args.rows) + engine_rows(args.engine)
    print(render(rows, args.against, args.seeds))


if __name__ == "__main__":
    main()
