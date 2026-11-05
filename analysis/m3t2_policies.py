"""TTDF and RACE-RP on the mixed workload: tuning, evictions, overhead, CSV.

Milestone 3 Task 2's two policies (``vllm/core/reuse_evictors.py``) are
compared with LRU, LFU and FIFO on the shared cache by
``analysis/m3t2_study.py``; this module holds what is particular to them:

- **Tuning (seed 1 only).** ``tuning`` reads the seed-1 cache-replay sweep
  (``results/m3-sim/tune-seed1.jsonl``: TTDF's time constants and weights,
  RACE-RP's tables L and bits P) and picks, per policy, the setting with the
  highest usable reuse averaged over the five tuning capacities; a tie keeps
  the specified setting. Held-out seeds never enter the choice.
- **Policy names.** ``policy_specs`` maps the page's names (``ttdf``,
  ``race_rp`` as specified; ``ttdf_tuned``, ``race_rp_tuned`` when tuning
  picked something else) to the replay's policy strings.
- **Evictions per workload** (``eviction_view``) and **overhead**
  (``overhead_view``: per-eviction time, candidates per scan, metadata) from
  the engine runs and the replay.
- **CSV tables** of everything the page plots (``write_csvs``)::

    python -m analysis.m3t2_policies select    # held-out sweep policies
    python -m analysis.m3t2_policies csv --out report/milestone3/data/task2
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

Row = Dict[str, Any]

NEW_POLICIES = ("ttdf", "race_rp", "sampled_lrfu")
SKETCH = "results/m3-race/sketch-L16-P6.json"
# The specified settings, as replay policy strings.
SPEC = {
    "ttdf": "ttdf",
    "race_rp": f"race_rp:sketch={SKETCH}",
    "sampled_lrfu": "sampled_lrfu"
}
SPEC_SETTINGS = {
    "ttdf": {
        "tau_short": 5.0,
        "tau_long": 120.0,
        "w_short": 0.7,
        "w_long": 0.3
    },
    "race_rp": {
        "L": 16,
        "P": 6
    },
    "sampled_lrfu": {
        "half_life": 512.0,
        "sample_size": 4,
        "seed": 0
    },
}
TUNING_PLAN = "balanced-human-s1"
TUNING_CAPACITIES = (32768, 65536, 131072, 262144, 393216)
TUNING_SWEEP = Path("results/m3-sim/tune-seed1.jsonl")
OVERHEAD_SWEEP = Path("results/m3-sim/overhead.jsonl")
STATS_FILE = "evictor-stats.json"
SKETCH_NAME = re.compile(r"sketch-L(\d+)-P(\d+)\.json$")


def family(policy: str) -> str:
    """The policy a replay policy string runs: ``ttdf:tau_short=2`` ->
    ``ttdf``."""
    return policy.split(":", 1)[0]


def settings(policy: str) -> Row:
    """A policy string's settings, defaults filled in: TTDF's four numbers,
    RACE-RP's L and P read from its sketch file's name, or Sampled LRFU's
    half-life, sample size and seed."""
    name, _, text = policy.partition(":")
    given = dict(item.split("=", 1) for item in text.split(",") if "=" in item)
    if name == "ttdf":
        return {
            k: float(given.get(k, v))
            for k, v in SPEC_SETTINGS["ttdf"].items()
        }
    if name == "race_rp":
        m = SKETCH_NAME.search(given.get("sketch", ""))
        if m is None:
            raise ValueError(f"{policy!r} names no sketch-L<L>-P<P>.json")
        return {"L": int(m[1]), "P": int(m[2])}
    if name == "sampled_lrfu":
        spec = SPEC_SETTINGS["sampled_lrfu"]
        return {
            "half_life": float(given.get("half_life", spec["half_life"])),
            "sample_size": int(given.get("sample_size", spec["sample_size"])),
            "seed": int(given.get("seed", spec["seed"])),
        }
    return {}


def label(policy: str) -> str:
    """A setting in words: "tau 5 s / 120 s, w 0.7 / 0.3", "L 16, P 6" or
    "half-life 512, 4 drawn"."""
    s = settings(policy)
    if family(policy) == "ttdf":
        return (f"tau {s['tau_short']:g} s / {s['tau_long']:g} s, "
                f"w {s['w_short']:g} / {s['w_long']:g}")
    if family(policy) == "race_rp":
        return f"L {s['L']}, P {s['P']}"
    if family(policy) == "sampled_lrfu":
        return (f"half-life {s['half_life']:,.0f}, {s['sample_size']} drawn" +
                (f", seed {s['seed']}" if s["seed"] else ""))
    return policy


def read_rows(path: Path) -> List[Row]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "summary" in row and "error" not in row:
                rows.append(row)
    return rows


def tuning(rows: Iterable[Row],
           plan: str = TUNING_PLAN,
           capacities: Sequence[int] = TUNING_CAPACITIES) -> Optional[Row]:
    """Seed-1 usable reuse (measurement window) of every setting tried, by
    capacity, and the setting picked per policy (module docstring). A
    policy whose specified setting lacks complete rows is left out; None
    when every policy is."""
    usable: Dict[str, Dict[int, float]] = {}
    for r in rows:
        if Path(r["plan"]).name != plan or not r.get("window"):
            continue
        cap = r.get("capacity_tokens")
        if cap in capacities:
            usable.setdefault(
                r["policy"],
                {})[cap] = (r["summary"]["window"]["usable_reused_fraction"])
    complete = {
        p: [v[c] for c in capacities]
        for p, v in usable.items() if all(c in v for c in capacities)
    }
    out: Row = {
        "plan": plan,
        "capacities": list(capacities),
        "lru": complete.get("lru"),
        "candidates": {},
        "selected": {},
        "spec": dict(SPEC),
    }
    for name in NEW_POLICIES:
        rows_ = [{
            "policy": p,
            "label": label(p),
            "settings": settings(p),
            "usable": v,
            "mean": float(np.mean(v)),
        } for p, v in complete.items() if family(p) == name]
        spec = [c for c in rows_ if c["settings"] == SPEC_SETTINGS[name]]
        if not spec:
            continue
        # Best mean first; equal means by policy string, so the order of
        # the sweep's rows does not matter.
        rows_.sort(key=lambda c: (-c["mean"], c["policy"]))
        # A tie keeps the specified setting.
        pick = spec[0] if spec[0]["mean"] >= rows_[0]["mean"] else rows_[0]
        out["candidates"][name] = rows_
        out["selected"][name] = {
            **pick, "is_spec": pick is spec[0],
            "spec_mean": spec[0]["mean"]
        }
    return out if out["selected"] else None


def policy_specs(tune: Optional[Row]) -> Dict[str, str]:
    """Page name -> replay policy string: the specified settings, plus a
    ``_tuned`` entry for each policy whose seed-1 pick differs."""
    specs = dict(SPEC)
    for name, pick in ((tune or {}).get("selected") or {}).items():
        if not pick["is_spec"]:
            specs[f"{name}_tuned"] = pick["policy"]
    return specs


def eviction_view(per_seed: Dict[str, List[Row]],
                  groups: Dict[str, Sequence[str]], seconds: float) -> Row:
    """Evicted blocks per second of the window by the group whose request
    last used them, per policy (seed mean and per-seed values), with the
    share of those blocks looked up again later (regretted). `per_seed`
    holds each seed's ``m3t2_study.evictions_matrix``."""
    out: Row = {}
    for policy, mats in per_seed.items():
        view: Row = {}
        for key in list(groups) + ["all"]:
            rates, regrets = [], []
            for m in mats:
                names = m["rows_evictor"]
                members = (names if key == "all" else
                           [w for w in names if w in groups[key]])
                cols = [names.index(w) for w in members]
                counts = [sum(row[c] for row in m["counts"]) for c in cols]
                total = sum(counts)
                rates.append(total / seconds)
                regret = sum(n * (m["regretted_share"].get(w) or 0.0)
                             for n, w in zip(counts, members))
                regrets.append(regret / total if total else None)
            view[key] = {
                "per_s":
                float(np.mean(rates)),
                "seeds":
                rates,
                "regretted":
                (None if None in regrets else float(np.mean(regrets))),
            }
        out[policy] = view
    return out


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def overhead_view(runs: Path, run_names: Dict[str, List[str]],
                  rows: Iterable[Row], specs: Dict[str, str]) -> Row:
    """Per-eviction time and metadata: from the engine runs' evictor stats
    (`run_names`: policy -> run folders), and from replays timed the same
    way for every policy (`rows`, e.g. ``results/m3-sim/overhead.jsonl``),
    averaged over seeds per policy and capacity."""
    engine: Row = {}
    for policy, names in run_names.items():
        stats = [
            json.loads((runs / n / STATS_FILE).read_text()) for n in names
            if (runs / n / STATS_FILE).exists()
        ]
        if not stats:
            continue
        engine[policy] = {
            "runs": len(stats),
            "evict_us": _mean([s["evict_us_mean"] for s in stats]),
            "candidates_per_scan":
            _mean([s["candidates_per_scan"] for s in stats]),
            "max_candidates": max(s["max_candidates"] for s in stats),
            "metadata_kib": _mean([s["metadata_bytes"] / 1024 for s in stats]),
            # The same, by part (counters, history, sketch, bookkeeping).
            "metadata_kib_parts": {
                part: _mean([
                    8 * s.get("metadata_words", {}).get(part, 0) / 1024
                    for s in stats
                ])
                for part in sorted(
                    {k
                     for s in stats for k in s.get("metadata_words", {})})
            },
            "evictions": _mean([s["evictions"] for s in stats]),
        }
    names = {v: k for k, v in specs.items()}
    replay: Row = {}
    for r in rows:
        name = names.get(r["policy"], r["policy"])
        s = r["summary"]
        if not s.get("evictions") or "evict_seconds" not in s:
            continue
        cell = replay.setdefault(name,
                                 {}).setdefault(str(r["capacity_tokens"]), {
                                     "evict_us": [],
                                     "metadata_kib": [],
                                     "candidates_per_scan": []
                                 })
        cell["evict_us"].append(1e6 * s["evict_seconds"] / s["evictions"])
        ev = s.get("evictor") or {}
        cell["metadata_kib"].append(ev["metadata_bytes"] /
                                    1024 if "metadata_bytes" in ev else None)
        cell["candidates_per_scan"].append(ev.get("candidates_per_scan"))
    for cells in replay.values():
        for cap, cell in cells.items():
            cells[cap] = {k: _mean(v) for k, v in cell.items()}
    return {"engine": engine, "replay": replay}


# --- CSV ----------------------------------------------------------------


def _write(path: Path, header: Sequence[str], rows: Iterable[Sequence]):
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(header)
        for row in rows:
            out.writerow(["" if v is None else v for v in row])


def write_csvs(summary: Row, out: Path) -> List[Path]:
    """The page's numbers as CSV files in `out`."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    cw = summary.get("capacity_window") or {}
    if cw:
        path = out / "capacity_by_workload.csv"
        rows = []
        for policy, groups in cw.get("group_seeds", {}).items():
            for group, per_cap in groups.items():
                for cap, vals in zip(cw["capacities"], per_cap):
                    for seed, v in zip(cw["seeds"], vals or []):
                        rows.append([policy, group, cap, seed, v])
        _write(path, [
            "policy", "workload", "capacity_tokens", "seed",
            "usable_prefix_reuse"
        ], rows)
        written.append(path)
        path = out / "evictions_by_capacity.csv"
        rows = []
        for policy, groups in cw.get("evictions", {}).items():
            for group, per_cap in groups.items():
                for cap, v in zip(cw["capacities"], per_cap):
                    rows.append([policy, group, cap, v])
        _write(path,
               ["policy", "workload", "capacity_tokens", "evictions_per_s"],
               rows)
        written.append(path)
    mg = summary.get("mixed_groups") or {}
    if mg:
        path = out / "engine_128k_usable.csv"
        seeds = summary["config"]["reported_seeds"]
        rows = []
        for group, cells in mg.items():
            for policy, cell in cells.items():
                if not isinstance(cell, dict) or not cell.get("usable"):
                    continue
                for seed, v in zip(seeds, cell["usable"]["seeds"]):
                    rows.append([policy, group, seed, v])
        _write(path, ["policy", "workload", "seed", "usable_prefix_reuse"],
               rows)
        written.append(path)
    mi = summary.get("mixed_intervals") or {}
    if mi:
        path = out / "engine_128k_served_by_wait.csv"
        rows = [[
            policy, c["lo"], c["hi"], c["offered"], c["served"], c["share"]
        ] for policy, view in mi.items() for c in view["curve"]]
        _write(path, [
            "policy", "wait_from_s", "wait_to_s", "reuses_offered",
            "reuses_served", "share_served"
        ], rows)
        written.append(path)
    npol = summary.get("new_policies") or {}
    if npol.get("evictions"):
        path = out / "engine_128k_evictions.csv"
        rows = [[policy, group, cell["per_s"], cell["regretted"]]
                for policy, view in npol["evictions"].items()
                for group, cell in view.items()]
        _write(
            path,
            ["policy", "workload", "evictions_per_s", "share_looked_up_again"],
            rows)
        written.append(path)
    if npol.get("overhead"):
        path = out / "overhead.csv"
        oh = npol["overhead"]
        rows = [[
            "engine", policy, 131072, c["evict_us"], c["metadata_kib"],
            c["candidates_per_scan"]
        ] for policy, c in oh["engine"].items()]
        rows += [[
            "replay", policy,
            int(cap), c["evict_us"], c["metadata_kib"],
            c["candidates_per_scan"]
        ] for policy, cells in oh["replay"].items()
                 for cap, c in cells.items()]
        _write(path, [
            "source", "policy", "capacity_tokens", "evict_us_mean",
            "metadata_kib", "candidates_per_scan"
        ], rows)
        written.append(path)
    if npol.get("tuning"):
        path = out / "tuning_seed1.csv"
        tune = npol["tuning"]
        rows = [[name, c["label"], c["policy"], *c["usable"], c["mean"]]
                for name, cands in tune["candidates"].items() for c in cands]
        _write(path, ["policy", "setting", "replay_policy"] +
               [f"usable_{c}" for c in tune["capacities"]] + ["mean"], rows)
        written.append(path)
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select",
                            help="Print the held-out sweep's "
                            "policy strings (specified and tuned)")
    select.add_argument("--tuning", type=Path, default=TUNING_SWEEP)
    export = sub.add_parser("csv", help="Write the page's numbers as CSV")
    export.add_argument("--summary",
                        type=Path,
                        default=Path("results/m3-study/summary.json"))
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "select":
        tune = tuning(read_rows(args.tuning))
        if tune is None:
            raise SystemExit(f"{args.tuning} lacks complete seed-1 rows for "
                             "the specified settings")
        for name, policy in policy_specs(tune).items():
            print(f"{name}\t{policy}")
    else:
        summary = json.loads(args.summary.read_text())
        for path in write_csvs(summary, args.out):
            print(path)


if __name__ == "__main__":
    main()
