"""Milestone 2 Task 3: sweep KV-cache capacity, block size and policy.

Replays the same multi-turn ShareGPT sample once per configuration, each on
a fresh simulator server with ``--block-size``, ``--num-gpu-blocks-override``
and ``VLLM_EVICTION_POLICY`` set and the cache event log on, then analyzes
each run with Task 2's metrics, writes one row per configuration to
``<out>/summary.csv``, the best configuration per capacity to
``<out>/best_by_capacity.csv`` and ``.md``, and spec 005's figures to
``<out>/figures/``::

    python -m analysis.run_m2_task3 --grid full --out results/m2-task3
    # Interrupted? Run the same command again: finished runs are kept.
    python -m analysis.run_m2_task3 --grid smoke --out results/smoke-task3
    # Recompute every run's metrics and summary.csv without replaying:
    python -m analysis.run_m2_task3 --out results/m2-task3 --analyze-only

``sweep.json`` records the sample, the pacing and every configuration the
folder has been asked to run. Without ``--grid`` or axis options, an
existing folder is resumed or analyzed with its recorded grid; with them,
the new configurations are added, and an axis not given takes the
folder's values (``--capacities 131072`` adds that capacity for every
recorded policy and block size). ``summary.csv`` always covers every
recorded configuration.

Capacities are in tokens, so the block count is capacity / block size and
block sizes compare at equal cache size. See ``specs/005-m2-cache-tuning``.
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# Figure labels.
LABELS = {
    "capacity_tokens": "KV-cache capacity (tokens)",
    "block_size": "block size (tokens)",
    "block_hit_rate": "block hit rate",
    "usable_reused_fraction": "usable reused token fraction",
    "evictions": "evictions",
}
# A run folder's name, `sweep.Config.name`.
RUN_NAME = re.compile(r"([a-z]+)-bs(\d+)-cap(\d+)")
# What `task2.write_sample` leaves in a folder before `sweep.json` exists,
# and what an interrupted first write of `sweep.json` leaves.
SAMPLE_FILES = {
    "sample.conversations.jsonl", "sample.responses.jsonl", "sweep.json.tmp"
}
# A grid preset's sample and pacing, with the defaults of presets that do
# not set them.
PACING_DEFAULTS = (("limit", 1000), ("conversation_rate", 1.0), ("think_time",
                                                                 5.0))
# Fields of the sweep that every run of one output folder must share.
SWEEP_FIELDS = ("model", "limit", "seed", "conversation_rate", "think_time",
                "max_concurrency", "max_model_len")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid",
                        choices=sorted(GRIDS),
                        help="Grid preset (default: the output folder's "
                        "recorded grid, or full for a new folder)")
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
    parser.add_argument("--figures",
                        type=Path,
                        help="Figure folder (default: <out>/figures)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--figure-block-size",
        type=int,
        default=16,
        help="Block size of figure 4 (the policies' CDFs); vLLM's default")
    parser.add_argument(
        "--pressure-capacity",
        type=int,
        help="Capacity of figure 4 (default: where the policies' usable "
        "reused fractions differ most at --figure-block-size)")
    args = parser.parse_args(argv)
    recorded = read_sweep_file(args.out)
    axes_given = (args.grid is not None or args.policies is not None
                  or args.block_sizes is not None
                  or args.capacities is not None)
    # Folders from before sweep.json listed its configurations.
    listed = recorded is not None and "configs" in recorded
    preset_name = args.grid or "full"
    if recorded is not None and args.grid is None:
        # A folder's recorded sample and pacing come before the presets'.
        for name, _ in PACING_DEFAULTS:
            if getattr(args, name) is None:
                setattr(args, name, recorded["settings"][name])
        if not listed:
            matched = preset_for(recorded["settings"])
            if matched is None:
                parser.error(f"{args.out} records no grid and its sample "
                             "matches no preset; pass --grid")
            preset_name = matched
    preset = GRIDS[preset_name]
    for name, default in PACING_DEFAULTS:
        if getattr(args, name) is None:
            setattr(args, name, preset.get(name, default))
    if args.limit <= 0:
        parser.error("--limit must be positive")
    recorded_configs = recorded_grid(recorded, args.out)
    if listed and not axes_given:
        args.configs = recorded_configs
    else:
        # Axes not given come from the folder's grid, or from the preset
        # for a new folder, with --grid, or for a folder that lists none.
        base = (grid_axes(recorded_configs)
                if listed and args.grid is None else preset)
        for name in ("policies", "block_sizes", "capacities"):
            if getattr(args, name) is None:
                setattr(args, name, base[name])
        try:
            args.configs = sweep.expand_grid(args.policies, args.block_sizes,
                                             args.capacities)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def preset_for(settings: Dict[str, Any]) -> Optional[str]:
    """The grid preset whose pacing a folder was run with (presets differ
    in pacing; the sample size is often overridden)."""
    for name, grid in GRIDS.items():
        if all(
                settings.get(field) == grid.get(field, default)
                for field, default in PACING_DEFAULTS if field != "limit"):
            return name
    return None


def grid_axes(configs: Sequence[sweep.Config]) -> Dict[str, List[Any]]:
    """The policies, block sizes and capacities a grid uses."""
    return {
        "policies":
        [p for p in sweep.POLICIES if any(c.policy == p for c in configs)],
        "block_sizes":
        sorted({c.block_size
                for c in configs}),
        "capacities":
        sorted({c.capacity_tokens
                for c in configs}),
    }


def read_sweep_file(out: Path) -> Optional[Dict[str, Any]]:
    path = out / "sweep.json"
    return json.loads(path.read_text()) if path.exists() else None


def recorded_grid(recorded: Optional[Dict[str, Any]],
                  out: Path) -> List[sweep.Config]:
    """The configurations a folder has been asked to run so far.

    Folders from before `sweep.json` listed them: the run folders present,
    which `parse_args` completes with the matching preset's grid.
    """
    if recorded is None:
        return []
    if "configs" in recorded:
        return [sweep.Config(*c) for c in recorded["configs"]]
    return sweep.grid_order(
        sweep.Config(m.group(1), int(m.group(2)), int(m.group(3)))
        for m in (RUN_NAME.fullmatch(p.name) for p in out.iterdir())
        if m and m.group(1) in sweep.POLICIES)


def sweep_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {field: getattr(args, field) for field in SWEEP_FIELDS}


def prepare_out(
        args: argparse.Namespace) -> Tuple[Dict[str, Any], List[sweep.Config]]:
    """Create or reopen the output folder; return the sample description
    and every configuration recorded for it, this invocation's included.

    A new folder gets the sample and `sweep.json`. An existing one is only
    resumed with the same sample and pacing, so all its runs compare.
    """
    sweep_file = args.out / "sweep.json"
    settings = json.loads(json.dumps(sweep_settings(args)))  # inf -> Infinity
    recorded = read_sweep_file(args.out)
    if recorded is not None:
        if recorded["settings"] != settings:
            raise SystemExit(f"{args.out} holds a sweep with other settings "
                             f"({recorded['settings']}); use a new --out")
        sample = recorded["sample"]
    elif args.analyze_only:
        raise SystemExit(f"{sweep_file} not found; nothing to analyze")
    else:
        if args.out.exists():
            contents = {p.name for p in args.out.iterdir()}
            if not contents <= SAMPLE_FILES:
                raise SystemExit(
                    f"{args.out} is not a sweep folder; use a new --out")
            # Only an interrupted sample: write it again.
            for name in contents:
                (args.out / name).unlink()
        args.out.mkdir(parents=True, exist_ok=True)
        sample = task2.write_sample(args.conversations, args.responses,
                                    args.limit, args.out)
    configs = sweep.grid_order(
        recorded_grid(recorded, args.out) + list(args.configs))
    if not args.analyze_only:
        write_atomically(
            sweep_file,
            json.dumps(
                {
                    "settings":
                    settings,
                    "sample":
                    sample,
                    "configs": [[c.policy, c.block_size, c.capacity_tokens]
                                for c in configs],
                },
                indent=2))
    return sample, configs


def write_atomically(path: Path, text: str) -> None:
    """Write `path` only if it changes, through a rename, so an
    interruption never leaves it half written."""
    if path.exists() and path.read_text() == text:
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def finished(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r["status"] == "finished"]


# What runs are ranked by: the share of prompt tokens served from a
# contiguous cached prefix. Not vLLM's cached_tokens, which overcounts when
# a policy leaves a miss before cached blocks (FIFO; see spec 004).
RANK_BY = "usable_reused_fraction"


def measured(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Finished runs with a usable reused fraction (None when no request
    completed), the ones that can be ranked."""
    return [r for r in finished(rows) if r[RANK_BY] is not None]


def pressure_capacity(rows: Sequence[Dict[str, Any]],
                      block_size: int) -> Optional[int]:
    """The capacity at which the policies' usable reused fractions spread
    most, at one block size: where eviction policy matters. Only
    capacities that every policy finished count."""
    by_capacity: Dict[int, Dict[str, float]] = {}
    for r in measured(rows):
        if r["block_size"] == block_size:
            by_capacity.setdefault(r["capacity_tokens"],
                                   {})[r["policy"]] = r[RANK_BY]
    policies = {r["policy"] for r in rows}
    spreads = {
        capacity: max(values.values()) - min(values.values())
        for capacity, values in by_capacity.items() if set(values) == policies
    }
    if not spreads:
        return None
    # Ties go to the smaller cache.
    return max(sorted(spreads), key=lambda c: spreads[c])


def best_by_capacity(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per capacity, the configuration with the highest usable reused
    fraction, and its margin over the next best."""
    table = []
    for capacity in sorted({r["capacity_tokens"] for r in measured(rows)}):
        ranked = sorted(
            (r for r in measured(rows) if r["capacity_tokens"] == capacity),
            key=lambda r: -r[RANK_BY])
        best = ranked[0]
        table.append({
            "capacity_tokens":
            capacity,
            "best":
            best["name"],
            "policy":
            best["policy"],
            "block_size":
            best["block_size"],
            RANK_BY:
            best[RANK_BY],
            "block_hit_rate":
            best["block_hit_rate"],
            "evictions":
            best["evictions"],
            "next_best":
            ranked[1]["name"] if len(ranked) > 1 else None,
            "margin":
            best[RANK_BY] - ranked[1][RANK_BY] if len(ranked) > 1 else None,
            "configurations":
            len(ranked),
        })
    return table


def markdown_table(rows: Sequence[Dict[str, Any]]) -> str:

    def cell(value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [
        "| " + " | ".join(cell(r[k]) for k in header) + " |" for r in rows
    ]
    return "\n".join(lines) + "\n"


def make_figures(out: Path, figures: Path, rows: Sequence[Dict[str, Any]],
                 block_size: int, capacity: Optional[int]) -> List[Path]:
    """Spec 005, User Story 3, figures 1 to 4 (5 is the table)."""
    from analysis import events as ev
    from analysis import metrics as m
    from analysis import plots  # matplotlib only when drawing

    done = finished(rows)
    written: List[Path] = []
    if not done:
        return written
    written += plots.panel_lines_plot(done,
                                      "capacity_tokens",
                                      ["block_hit_rate", RANK_BY],
                                      "block_size",
                                      figures / "fig1_vs_capacity",
                                      labels=LABELS,
                                      title="block size {value}")
    written += plots.panel_lines_plot(done,
                                      "block_size",
                                      ["block_hit_rate", RANK_BY],
                                      "capacity_tokens",
                                      figures / "fig2_vs_block_size",
                                      labels=LABELS,
                                      title="{value:,} tokens")
    written += plots.panel_lines_plot(done,
                                      "capacity_tokens", ["evictions"],
                                      "block_size",
                                      figures / "fig3_evictions_vs_capacity",
                                      labels=LABELS,
                                      title="block size {value}",
                                      symlog_y=["evictions"])
    if capacity is None:
        return written
    residencies, intervals = {}, {}
    for r in done:
        if r["block_size"] == block_size and r["capacity_tokens"] == capacity:
            events = ev.load_events(out / r["name"] / "events.jsonl").events
            residencies[r["policy"]] = m.hits_per_residency(events)
            intervals[r["policy"]] = [
                i["interval_s"] for i in m.reuse_intervals(events)
                if i["interval_s"] > 0
            ]
    suffix = f"bs{block_size}_cap{capacity}"
    written += plots.cdf_plot(residencies,
                              "hits per residency",
                              figures / f"fig4a_hits_per_residency_{suffix}",
                              symlog_x=True)
    written += plots.cdf_plot(intervals,
                              "reuse interval (s)",
                              figures / f"fig4b_reuse_interval_{suffix}",
                              log_x=True)
    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sample, configs = prepare_out(args)
    # Only this invocation's configurations are replayed; the others the
    # folder recorded are summarized (or listed as missing).
    requested = set(args.configs)

    def replay(config: sweep.Config, run_dir: Path) -> None:
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
        configs,
        args.out,
        replay,
        lambda run_dir: task2.analyze_run(run_dir, args.window),
        args.max_model_len,
        reanalyze=args.analyze_only,
        replay_if=lambda config: not args.analyze_only and config in requested)
    task2.write_csv(args.out / "summary.csv", rows)
    print(f"Wrote {len(rows)} rows to {args.out / 'summary.csv'}")
    best = best_by_capacity(rows)
    if best:
        task2.write_csv(args.out / "best_by_capacity.csv", best)
        (args.out / "best_by_capacity.md").write_text(markdown_table(best))
    else:  # Nothing to rank: do not leave an earlier table behind.
        for name in ("best_by_capacity.csv", "best_by_capacity.md"):
            (args.out / name).unlink(missing_ok=True)
    if args.no_figures:
        return
    capacity = args.pressure_capacity or pressure_capacity(
        rows, args.figure_block_size)
    written = make_figures(args.out, args.figures or args.out / "figures",
                           rows, args.figure_block_size, capacity)
    print(f"Wrote {len(written)} figure files; figure 4 at block size "
          f"{args.figure_block_size}, capacity {capacity}")


if __name__ == "__main__":
    main()
