"""Milestone 2 Task 2: replay ShareGPT single-turn and multi-turn, analyze.

For each setting, starts a fresh simulator server (empty cache) with the
cache event log on, replays the same conversation sample with the client
simulator, stops the server, then writes metrics CSVs and a manifest to
``<out>/<setting>/`` and the comparison figures to ``<out>/figures/``::

    python -m analysis.run_m2_task2 --limit 1000 --out results/m2-task2
    # Recompute metrics and figures from existing run folders:
    python -m analysis.run_m2_task2 --out results/m2-task2 --analyze-only

See ``specs/004-m2-prefix-sharing-analysis``.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from analysis import events as ev
from analysis import metrics as m
from analysis.server import SimulatorServer

SETTINGS = ("single", "multi")
HIT_RATE_METRIC = "vllm:gpu_prefix_cache_hit_rate"
# SC-003: the log and vLLM's metric agree to 4 decimal places.
SC003_TOLERANCE = 5e-5
# Turns with fewer requests are left out of the by-turn figure.
MIN_REQUESTS_PER_TURN = 10


def git_state() -> Dict[str, Any]:

    def git(*args: str) -> str:
        return subprocess.run(["git", *args],
                              capture_output=True,
                              text=True,
                              check=False).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))
    }


def write_sample(conversations: Path, responses: Path, limit: int,
                 out: Path) -> Dict[str, Any]:
    """Copy the first `limit` conversations and only their responses.

    The server tokenizes every trace response at startup, so it gets the
    sample's responses rather than the whole 330k-turn file.
    """
    ids = set()
    turns = 0
    sample_conversations = out / "sample.conversations.jsonl"
    with conversations.open() as src, sample_conversations.open("w") as dst:
        for line in islice(src, limit):
            row = json.loads(line)
            ids.add(row["conversation_id"])
            turns += len(row["messages"]) // 2
            dst.write(line)
    sample_responses = out / "sample.responses.jsonl"
    kept = 0
    with responses.open() as src, sample_responses.open("w") as dst:
        for line in src:
            request_id = json.loads(line)["request_id"]
            if request_id.rpartition(":")[0] in ids:
                dst.write(line)
                kept += 1
    if kept != turns:
        raise ValueError(f"{responses} has {kept} responses for {turns} "
                         "sampled turns")
    return {
        "source_conversations": str(conversations),
        "source_responses": str(responses),
        "limit": limit,
        "conversations": len(ids),
        "turns": turns,
        "conversations_file": str(sample_conversations),
        "responses_file": str(sample_responses),
    }


def run_setting(
        setting: str,
        args: argparse.Namespace,
        sample: Dict[str, Any],
        run_dir: Path,
        extra_server_args: Sequence[str] = (),
        env: Optional[Dict[str, str]] = None,
        manifest_updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Replay one setting on a fresh server into `run_dir`.

    `extra_server_args`, `env` and `manifest_updates` let other experiments
    (Task 3's sweep) change the cache configuration and record it.
    """
    run_dir.mkdir(parents=True)
    server_args = [
        "--enable-prefix-caching", "--enable-prompt-tokens-details",
        "--max-model-len",
        str(args.max_model_len), *extra_server_args
    ]
    client_output = run_dir / "client.jsonl"
    manifest: Dict[str, Any] = {
        "experiment": "m2-task2",
        "setting": setting,
        **git_state(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample": sample,
        "model": args.model,
        "server_args": server_args,
        "kv_cache_space_gib": os.environ.get("VLLM_CPU_KVCACHE_SPACE", "4"),
        "eviction_policy": "lru",
        "client": {
            "mode": setting,
            "conversation_rate": args.conversation_rate,
            "think_time_s": args.think_time,
            "seed": args.seed,
            "max_concurrency": args.max_concurrency,
        },
        **(manifest_updates or {}),
    }
    with SimulatorServer(args.model,
                         Path(sample["responses_file"]),
                         log_path=run_dir / "server.log",
                         event_log=run_dir / "events.jsonl",
                         args=server_args,
                         env=env,
                         port=args.port) as server:
        manifest["kv_cache"] = server.kv_cache_blocks()
        client = subprocess.run([
            sys.executable, "-m", "client_simulator.client_sim", "--trace",
            sample["conversations_file"], "--model", args.model, "--base-url",
            server.base_url, "--mode", setting, "--limit",
            str(sample["conversations"]), "--conversation-rate",
            str(args.conversation_rate), "--think-time",
            str(args.think_time), "--seed",
            str(args.seed), "--max-concurrency",
            str(args.max_concurrency), "--output",
            str(client_output), "--capture-metrics"
        ],
                                capture_output=True,
                                text=True,
                                check=False)
    (run_dir / "client.out").write_text(client.stdout + client.stderr)
    client_run = json.loads(Path(str(client_output) + ".run.json").read_text())
    # Exit code 1 only means some requests were skipped or failed; those
    # are counted in the summary. Anything else is a broken run.
    if client.returncode not in (0, 1) or client_run["status"] != "finished":
        raise RuntimeError(f"client failed ({client.returncode}); see "
                           f"{run_dir / 'client.out'}")
    manifest["client_summary"] = client_run.get("summary")
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "mean": statistics.fmean(ordered),
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": ordered[-1]
    }


def analyze_run(run_dir: Path, window_s: float) -> Dict[str, Any]:
    """Write the metrics CSVs and summary.json of one run folder."""
    log = ev.load_events(run_dir / "events.jsonl")
    events = log.events
    block_size = log.header["block_size"]
    rows, excluded = m.cached_fractions(
        ev.load_results(run_dir / "client.jsonl"))
    gaps = m.cached_token_gaps(rows, events, block_size)
    for row, gap in zip(rows, gaps):
        row["logged_hits"] = gap["logged_hits"]
        row["gap_tokens"] = gap["gap_tokens"]
    m.add_usable_fractions(rows, m.prefix_hits(events), block_size)
    hits = m.hits_per_block(events)
    intervals = m.reuse_intervals(events)
    windows = m.windowed_hit_rate(events, window_s)
    sources = m.hit_sources(events)
    write_csv(run_dir / "requests.csv", rows)
    write_csv(run_dir / "blocks.csv", ({
        "hash": h,
        "hits": n
    } for h, n in hits.items()))
    write_csv(run_dir / "reuse_intervals.csv", intervals)
    write_csv(run_dir / "hit_rate_windows.csv", windows)

    logged_rate = m.block_hit_rate(events)
    metric_rate = ev.read_metric(run_dir / "client.jsonl.metrics.after.txt",
                                 HIT_RATE_METRIC)
    gap_signs = Counter(
        "zero" if g["gap_tokens"] ==
        0 else "negative" if g["gap_tokens"] < 0 else "positive" for g in gaps)
    total_hits = sum(sources.values())
    summary = {
        "requests_completed":
        len(rows),
        "requests_excluded":
        dict(excluded),
        "log_complete":
        log.complete,
        "event_counts":
        dict(Counter(e.ev for e in events)),
        "mean_cached_fraction":
        statistics.fmean(r["cached_fraction"] for r in rows) if rows else None,
        "token_reused_fraction":
        m.token_reused_fraction(rows),
        "mean_cached_fraction_by_turn":
        m.mean_by_turn(rows),
        # vLLM's cached_tokens overcounts when a hit follows a miss (FIFO);
        # these count only the contiguous cached prefix (spec 004).
        "mean_usable_cached_fraction":
        statistics.fmean(r["usable_cached_fraction"]
                         for r in rows) if rows else None,
        "usable_reused_fraction":
        m.token_reused_fraction(rows, "usable_cached_tokens"),
        "mean_usable_cached_fraction_by_turn":
        m.mean_by_turn(rows, "usable_cached_fraction"),
        "requests_with_hits_after_miss":
        sum(r["hits_after_miss"] > 0 for r in rows),
        "blocks":
        len(hits),
        "blocks_with_hits_share":
        sum(n > 0 for n in hits.values()) / len(hits) if hits else None,
        "hits_per_block":
        quantiles(list(hits.values())),
        "hits_per_residency":
        quantiles(m.hits_per_residency(events)),
        "reuse_interval_s":
        quantiles([r["interval_s"] for r in intervals]),
        "reuse_requests_between":
        quantiles([r["requests_between"] for r in intervals]),
        "block_hit_rate_log":
        logged_rate,
        "block_hit_rate_metric":
        metric_rate,
        "evictions":
        m.eviction_count(events),
        "eviction_rate_per_1k_lookups":
        m.eviction_rate(events),
        "sc003_pass":
        logged_rate is not None and metric_rate is not None
        and abs(logged_rate - metric_rate) < SC003_TOLERANCE,
        "sc004_gap_signs":
        dict(gap_signs),
        "hit_sources": {
            f"{who}/{op}": n / total_hits
            for (who, op), n in sorted(sources.items(), key=str)
        } if total_hits else {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not log.complete:
        print(f"warning: {run_dir}/events.jsonl has no end line; the "
              "engine was killed and the tail of the log may be lost")
    return summary


def make_figures(out: Path, figures: Path, settings: Sequence[str],
                 window_s: float) -> List[Path]:
    from analysis import plots  # matplotlib only when drawing

    cached, hits, seconds, distance, windows, shares = {}, {}, {}, {}, {}, {}
    by_turn = None
    for setting in settings:
        run_dir = out / setting
        events = ev.load_events(run_dir / "events.jsonl").events
        rows, _ = m.cached_fractions(ev.load_results(run_dir / "client.jsonl"))
        intervals = m.reuse_intervals(events)
        cached[setting] = [r["cached_fraction"] for r in rows]
        hits[setting] = list(m.hits_per_block(events).values())
        seconds[setting] = [
            r["interval_s"] for r in intervals if r["interval_s"] > 0
        ]
        distance[setting] = [r["requests_between"] for r in intervals]
        windows[setting] = m.windowed_hit_rate(events, window_s)
        shares[setting] = json.loads(
            (run_dir / "summary.json").read_text())["hit_sources"]
        if setting == "multi":
            counts = Counter(r["turn_index"] for r in rows)
            means = m.mean_by_turn(rows)
            by_turn = ({
                t: v
                for t, v in means.items() if counts[t] >= MIN_REQUESTS_PER_TURN
            }, counts)
    written = []
    written += plots.cdf_plot(cached, "cached fraction of prompt",
                              figures / "fig1_cached_fraction_cdf")
    written += plots.cdf_plot(hits,
                              "hits per block",
                              figures / "fig2_hits_per_block_cdf",
                              symlog_x=True)
    written += plots.cdf_plot(seconds,
                              "reuse interval (s)",
                              figures / "fig3_reuse_interval_cdf",
                              log_x=True)
    written += plots.cdf_plot(distance,
                              "requests started between reuses",
                              figures / "fig3b_reuse_distance_cdf",
                              symlog_x=True)
    if by_turn is not None:
        written += plots.by_turn_plot(by_turn[0], by_turn[1],
                                      figures / "fig4_cached_by_turn")
    written += plots.over_time_plot(windows,
                                    figures / "fig5_hit_rate_over_time")
    written += plots.sources_plot(shares, figures / "fig6_hit_sources")
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--conversations",
        type=Path,
        default=Path("data/processed/sharegpt.conversations.jsonl"))
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/processed/sharegpt.responses.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conversation-rate", type=float, default=1.0)
    parser.add_argument("--think-time", type=float, default=5.0)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--window",
                        type=float,
                        default=10.0,
                        help="Hit-rate window in seconds")
    parser.add_argument("--settings",
                        nargs="+",
                        choices=SETTINGS,
                        default=list(SETTINGS))
    parser.add_argument("--out", type=Path, default=Path("results/m2-task2"))
    parser.add_argument("--figures",
                        type=Path,
                        help="Figure folder (default: <out>/figures)")
    parser.add_argument("--analyze-only",
                        action="store_true",
                        help="Recompute metrics and figures, no replay")
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.analyze_only:
        if args.out.exists():
            raise SystemExit(f"{args.out} exists; choose a new --out or "
                             "pass --analyze-only")
        args.out.mkdir(parents=True)
        sample = write_sample(args.conversations, args.responses, args.limit,
                              args.out)
        for setting in args.settings:
            print(f"Replaying {setting} ({sample['conversations']} "
                  "conversations)...")
            run_setting(setting, args, sample, args.out / setting)
    for setting in args.settings:
        summary = analyze_run(args.out / setting, args.window)
        print(f"{setting}: {json.dumps(summary)}")
    figures = make_figures(args.out, args.figures or args.out / "figures",
                           args.settings, args.window)
    print(f"Wrote {len(figures)} figure files to "
          f"{figures[0].parent if figures else '-'}")


if __name__ == "__main__":
    main()
