"""Check the fast cache simulator against engine runs (spec 007, plan).

For each engine run folder (``analysis/run_m3_task2.py replay``), the plan
is simulated at the run's policy and capacity twice: with the run's own
timing (``engine:<run>``) and with the modelled timing used for sweeps.
Reported per run:

- whether both saw the same prompt blocks (the multiset of lookup hashes);
- block hit rate, usable reused fraction and vLLM's cached share, overall
  and per workload, with the differences;
- evictions, and how many evicted blocks match position by position and
  as multisets (engine timing only).

::

    python -m analysis.validate_cachesim results/m3-validate/* \\
        --out results/m3-validate/validation.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from analysis import cachesim
from analysis import events as ev
from analysis import metrics as m


def engine_summary(run_dir: Path) -> Dict[str, Any]:
    log = ev.load_events(ev.log_path(run_dir))
    rows, _ = m.cached_fractions(ev.load_results(run_dir / "client.jsonl"))
    m.add_usable_fractions(rows, m.prefix_hits(log.events),
                           log.header["block_size"])
    per: Dict[str, Counter] = {}
    for e in log.events:
        if e.ev in ("hit", "miss") and e.req is not None:
            c = per.setdefault(ev.workload_of(e.req), Counter())
            c["lookups"] += 1
            c["hits"] += e.ev == "hit"
    for r in rows:
        c = per.setdefault(ev.workload_of(r["request_id"]), Counter())
        c["prompt_tokens"] += r["prompt_tokens"]
        c["usable_tokens"] += r["usable_cached_tokens"]
        c["vllm_cached_tokens"] += min(r["cached_tokens"], r["prompt_tokens"])
    total: Counter = Counter()
    for c in per.values():
        total.update(c)
    return {
        "lookup_hashes":
        Counter(e.hash for e in log.events if e.ev in ("hit", "miss")),
        "evicted": [e.hash for e in log.events if e.ev == "evict"],
        "summary":
        _fractions(total),
        "workloads": {w: _fractions(c)
                      for w, c in sorted(per.items())},
        "completed":
        len(rows),
    }


def _fractions(c: Counter) -> Dict[str, Any]:
    prompt = c["prompt_tokens"]
    return {
        "block_hit_rate": c["hits"] / c["lookups"] if c["lookups"] else None,
        "usable_reused_fraction":
        c["usable_tokens"] / prompt if prompt else None,
        "vllm_cached_share":
        c["vllm_cached_tokens"] / prompt if prompt else None,
        "lookups": c["lookups"],
    }


def _diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("block_hit_rate", "usable_reused_fraction", "vllm_cached_share")
    return {
        k: (b[k] -
            a[k]) if a.get(k) is not None and b.get(k) is not None else None
        for k in keys
    }


def validate(run_dir: Path, scratch: Path) -> Dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text())
    plan_dir = Path(run["plan_dir"])
    policy, capacity = run["eviction_policy"], run["capacity_tokens"]
    block_size = run["block_size"]
    env = {"VLLM_WA_LABELS": run["wa_labels"]} if run.get("wa_labels") else {}
    engine = engine_summary(run_dir)
    out: Dict[str, Any] = {
        "run": str(run_dir),
        "policy": policy,
        "capacity_tokens": capacity,
        "engine": {
            "summary": engine["summary"],
            "workloads": engine["workloads"],
            "evictions": len(engine["evicted"]),
            "completed": engine["completed"],
        },
    }
    for timing in ("engine", "model"):
        sim_dir = scratch / f"{run_dir.name}-{timing}"
        result = cachesim.simulate(
            plan_dir,
            policy,
            capacity,
            block_size,
            f"engine:{run_dir}" if timing == "engine" else "model",
            env,
            out=sim_dir)
        sim = engine_summary(sim_dir)
        entry: Dict[str, Any] = {
            "summary":
            sim["summary"],
            "difference":
            _diff(engine["summary"], sim["summary"]),
            "workload_differences": {
                w: _diff(engine["workloads"][w], s)
                for w, s in sim["workloads"].items()
                if w in engine["workloads"]
            },
            "evictions":
            len(sim["evicted"]),
            "same_lookup_blocks":
            sim["lookup_hashes"] == engine["lookup_hashes"],
            "lookup_multiset_overlap":
            (sum((sim["lookup_hashes"] & engine["lookup_hashes"]).values()) /
             max(1, sum(engine["lookup_hashes"].values()))),
            "counts": {
                k: v
                for k, v in result.summary.items()
                if k in ("deferred", "decode_out_of_blocks", "not_sent",
                         "decode_times_interpolated", "never_allocated")
            },
        }
        if timing == "engine":
            a, b = engine["evicted"], sim["evicted"]
            same = sum(x == y for x, y in zip(a, b))
            entry["evicted_same_position"] = same / max(1, len(a))
            entry["evicted_multiset_overlap"] = (sum(
                (Counter(a) & Counter(b)).values()) / max(1, len(a)))
            first: List[int] = [
                i for i, (x, y) in enumerate(zip(a, b)) if x != y
            ][:1]
            entry["first_eviction_mismatch"] = first[0] if first else None
        out[f"sim_{timing}_timing"] = entry
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--scratch",
                        type=Path,
                        default=Path("results/m3-validate-sim"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    results = [validate(r, args.scratch) for r in args.runs]
    text = json.dumps(results, indent=2)
    if args.out:
        args.out.write_text(text)
    for r in results:
        e, s = r["engine"]["summary"], r["sim_engine_timing"]
        print(
            f"{Path(r['run']).name}: engine usable "
            f"{e['usable_reused_fraction']:.4f}, sim (engine timing) "
            f"{s['summary']['usable_reused_fraction']:.4f}, sim (model) "
            f"{r['sim_model_timing']['summary']['usable_reused_fraction']:.4f}"
            f"; evictions {r['engine']['evictions']} vs {s['evictions']}, "
            f"same position {s.get('evicted_same_position', 0):.4f}")


if __name__ == "__main__":
    main()
