"""Milestone 2 Task 3: sweep KV-cache capacity, block size and policy.

Replays the same multi-turn ShareGPT sample once per configuration, each on
a fresh simulator server with ``--block-size``, ``--num-gpu-blocks-override``
and ``VLLM_EVICTION_POLICY`` set and the cache event log on, then analyzes
each run with Task 2's metrics and writes one row per configuration to
``<out>/summary.csv``::

    python -m analysis.run_m2_task3 --grid full --out results/m2-task3
    # Interrupted? Run the same command again: finished runs are kept.
    python -m analysis.run_m2_task3 --grid smoke --out results/smoke-task3
    # Recompute every run's metrics and summary.csv without replaying:
    python -m analysis.run_m2_task3 --out results/m2-task3 --analyze-only

Capacities are in tokens, so the block count is capacity / block size and
block sizes compare at equal cache size. See ``specs/005-m2-cache-tuning``.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from analysis import run_m2_task2 as task2
from analysis import sweep

# Spec 005, Assumptions: capacities span the multi-turn working set of
# Task 2's sample, 114,964 distinct blocks of 16 tokens (1.84M tokens), from
# far below it to above it. Powers of two, so every block size divides them;
# the smallest that vLLM accepts is max_model_len (8,192 tokens).
GRIDS: Dict[str, Dict[str, Any]] = {
    "full": {
        "policies": list(sweep.POLICIES),
        "block_sizes": [8, 16, 32],
        "capacities": [2**14, 2**16, 2**18, 2**20, 2**21],
    },
    # Spec 005 US2's independent test: 2 x 1 x 3 on 20 conversations,
    # small enough to evict, no pacing.
    "smoke": {
        "policies": list(sweep.POLICIES),
        "block_sizes": [16],
        "capacities": [2**13, 2**14],
        "limit": 20,
        "conversation_rate": float("inf"),
        "think_time": 0.0,
    },
}
# Fields of the sweep that every run of one output folder must share.
SWEEP_FIELDS = ("model", "limit", "seed", "conversation_rate", "think_time",
                "max_concurrency", "max_model_len")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="full")
    parser.add_argument("--policies", nargs="+", choices=sweep.POLICIES)
    parser.add_argument("--block-sizes",
                        nargs="+",
                        type=int,
                        choices=sweep.BLOCK_SIZES)
    parser.add_argument("--capacities",
                        nargs="+",
                        type=int,
                        help="KV-cache capacities in tokens")
    parser.add_argument(
        "--conversations",
        type=Path,
        default=Path("data/processed/sharegpt.conversations.jsonl"))
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/processed/sharegpt.responses.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    # Task 2's sample and pacing (spec 004, Assumptions) unless the grid
    # preset or these options say otherwise.
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conversation-rate", type=float)
    parser.add_argument("--think-time", type=float)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--window",
                        type=float,
                        default=10.0,
                        help="Hit-rate window in seconds")
    parser.add_argument("--out", type=Path, default=Path("results/m2-task3"))
    parser.add_argument("--analyze-only",
                        action="store_true",
                        help="Recompute metrics and summary.csv, no replay")
    args = parser.parse_args(argv)
    preset = GRIDS[args.grid]
    for name, default in (("policies", None), ("block_sizes", None),
                          ("capacities", None), ("limit", 1000),
                          ("conversation_rate", 1.0), ("think_time", 5.0)):
        if getattr(args, name) is None:
            setattr(args, name, preset.get(name, default))
    if args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        args.configs = sweep.expand_grid(args.policies, args.block_sizes,
                                         args.capacities)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def sweep_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {field: getattr(args, field) for field in SWEEP_FIELDS}


def prepare_out(args: argparse.Namespace) -> Dict[str, Any]:
    """Create or reopen the output folder; return the sample description.

    A new folder gets the sample and `sweep.json`. An existing one is only
    resumed with the same sample and pacing, so all its runs compare.
    """
    sweep_file = args.out / "sweep.json"
    settings = json.loads(json.dumps(sweep_settings(args)))  # inf -> Infinity
    if sweep_file.exists():
        recorded = json.loads(sweep_file.read_text())
        if recorded["settings"] != settings:
            raise SystemExit(f"{args.out} holds a sweep with other settings "
                             f"({recorded['settings']}); use a new --out")
        return recorded["sample"]
    if args.analyze_only:
        raise SystemExit(f"{sweep_file} not found; nothing to analyze")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not a sweep folder; use a new --out")
    args.out.mkdir(parents=True, exist_ok=True)
    sample = task2.write_sample(args.conversations, args.responses, args.limit,
                                args.out)
    sweep_file.write_text(
        json.dumps({
            "settings": settings,
            "sample": sample
        }, indent=2))
    return sample


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sample = prepare_out(args)

    def replay(config: sweep.Config, run_dir: Path) -> None:
        # Pass the policy explicitly so a shell's VLLM_EVICTION_POLICY
        # never leaks into a run.
        task2.run_setting("multi",
                          args,
                          sample,
                          run_dir,
                          extra_server_args=config.server_args(),
                          env=config.env(),
                          manifest_updates={
                              "experiment": "m2-task3",
                              "eviction_policy": config.policy,
                              "block_size": config.block_size,
                              "capacity_tokens": config.capacity_tokens,
                              "num_gpu_blocks": config.num_blocks,
                          })

    rows = sweep.run_sweep(
        args.configs,
        args.out,
        replay,
        lambda run_dir: task2.analyze_run(run_dir, args.window),
        args.max_model_len,
        reanalyze=args.analyze_only,
        replay_missing=not args.analyze_only)
    task2.write_csv(args.out / "summary.csv", rows)
    print(f"Wrote {len(rows)} rows to {args.out / 'summary.csv'}")


if __name__ == "__main__":
    os.environ.pop("VLLM_EVICTION_POLICY", None)
    main()
