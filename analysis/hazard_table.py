"""Survival table for the ``wa_probe`` evictor (exploratory, spec 007).

From one run's cache event log, usually the training seed's no-eviction
engine run, this measures for every block the time between consecutive
accesses (a lookup hit, an insertion or a deduplication), labelled with the
workload of the request that made the earlier access. Blocks still waiting
at the end of the log are right-censored. It writes Kaplan-Meier survival
curves per workload and pooled, on a log-spaced age grid::

    python -m analysis.hazard_table \\
        --run results/m3-task2/balanced-human-s1-ceiling \\
        --out results/m3-explore/hazard-s1.json

``vllm/core/wa_probe.py`` reads the file (``table=<file>``).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from analysis.events import Event, load_events, log_path
from analysis.survival import kaplan_meier, survival_at
from vllm.core.block.cache_events import workload_label

ACCESS_EVENTS = ("hit", "insert", "dedupe")
# Ages from 10 ms to 1 h, plus 0.
AGES = np.concatenate(([0.0], np.logspace(-2, np.log10(3600.0), 240)))


def access_gaps(
    events: Iterable[Event]
) -> Tuple[Dict[Optional[str], List[float]], Dict[Optional[str], List[bool]]]:
    """Durations and observed flags per label of the earlier access."""
    last: Dict[int, Tuple[float, Optional[str]]] = {}
    durations: Dict[Optional[str], List[float]] = defaultdict(list)
    observed: Dict[Optional[str], List[bool]] = defaultdict(list)
    end = 0.0
    for event in events:
        end = max(end, event.t)
        if event.ev not in ACCESS_EVENTS:
            continue
        label = workload_label(event.req)
        previous = last.get(event.hash)
        if previous is not None:
            durations[previous[1]].append(event.t - previous[0])
            observed[previous[1]].append(True)
        last[event.hash] = (event.t, label)
    for t, label in last.values():
        durations[label].append(end - t)
        observed[label].append(False)
    return durations, observed


def curve(durations: List[float], flags: List[bool]) -> List[float]:
    times, survival = kaplan_meier(durations, flags)
    return [float(s) for s in survival_at(times, survival, AGES)]


def build(run_dir: Path) -> dict:
    log = load_events(log_path(run_dir))
    durations, observed = access_gaps(log.events)
    labels = sorted(k for k in durations if k is not None)
    pooled_d = [d for k in durations for d in durations[k]]
    pooled_o = [o for k in observed for o in observed[k]]
    return {
        "source": str(run_dir),
        "ages": [float(a) for a in AGES],
        "survival": {k: curve(durations[k], observed[k])
                     for k in labels},
        "pooled": curve(pooled_d, pooled_o),
        "counts": {
            str(k): {
                "gaps": len(durations[k]),
                "observed": int(sum(observed[k]))
            }
            for k in durations
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    table = build(args.run)
    if not table["counts"]:
        raise SystemExit(f"No block accesses in {args.run}'s event log")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table))
    for label, counts in table["counts"].items():
        print(label, counts)


if __name__ == "__main__":
    main()
