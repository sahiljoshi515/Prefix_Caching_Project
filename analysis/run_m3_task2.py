"""Milestone 3 Task 2: replay mixed-workload plans (spec 007).

One run replays a plan built by ``analysis/mix.py`` on a fresh simulator
server with a given cache configuration and eviction policy, with the cache
event log on, and records a manifest::

    python -m analysis.run_m3_task2 replay \\
        --plan data/processed/m3/balanced-s1 --policy lru \\
        --capacity 262144 --out results/m3-task2/balanced-s1-lru

The run folder is laid out like Task 2's (``client.jsonl``, its metric
snapshots, ``events.jsonl``, ``server.log``, ``run.json``), so
``analysis/run_m2_task2.analyze_run`` reads it.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from analysis.run_m2_task2 import git_state
from analysis.server import SimulatorServer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def replay_plan(plan_dir: Path,
                run_dir: Path,
                *,
                policy: str,
                capacity_tokens: int,
                block_size: int = 16,
                model: str = DEFAULT_MODEL,
                max_model_len: int = 32768,
                port: int = 8000,
                labels: Optional[str] = None,
                event_log: bool = True,
                max_concurrency: int = 64,
                extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Replay `plan_dir`'s plan into `run_dir` (which must not exist)."""
    if capacity_tokens % block_size:
        raise ValueError("The capacity must be a whole number of blocks")
    plan = json.loads((plan_dir / "manifest.json").read_text())
    run_dir.mkdir(parents=True)
    num_blocks = capacity_tokens // block_size
    server_args = [
        "--enable-prefix-caching", "--enable-prompt-tokens-details",
        "--max-model-len",
        str(max_model_len), "--block-size",
        str(block_size), "--num-gpu-blocks-override",
        str(num_blocks)
    ]
    env = {"VLLM_EVICTION_POLICY": policy, **(extra_env or {})}
    if labels is not None:
        env["VLLM_WA_LABELS"] = labels
    client_output = run_dir / "client.jsonl"
    manifest: Dict[str, Any] = {
        "experiment": "m3-task2",
        **git_state(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_dir": str(plan_dir),
        "plan": plan,
        "model": model,
        "server_args": server_args,
        "eviction_policy": policy,
        "wa_labels": labels,
        "block_size": block_size,
        "capacity_tokens": capacity_tokens,
        "num_gpu_blocks": num_blocks,
        "env": env,
        "max_concurrency": max_concurrency,
    }
    with SimulatorServer(model,
                         plan_dir / "responses.jsonl",
                         log_path=run_dir / "server.log",
                         event_log=run_dir /
                         "events.jsonl" if event_log else None,
                         args=server_args,
                         env=env,
                         port=port) as server:
        manifest["kv_cache"] = server.kv_cache_blocks()
        client = subprocess.run([
            sys.executable, "-m", "client_simulator.plan_replay", "--plan",
            str(plan_dir / "plan.jsonl.gz"), "--model", model, "--base-url",
            server.base_url, "--max-concurrency",
            str(max_concurrency), "--output",
            str(client_output), "--capture-metrics"
        ],
                                capture_output=True,
                                text=True,
                                check=False)
    (run_dir / "client.out").write_text(client.stdout + client.stderr)
    client_run = json.loads(Path(str(client_output) + ".run.json").read_text())
    # Exit code 1 only means some requests were skipped or failed; those are
    # counted in the summary. Anything else is a broken run.
    if client.returncode not in (0, 1) or client_run["status"] != "finished":
        raise RuntimeError(f"client failed ({client.returncode}); see "
                           f"{run_dir / 'client.out'}")
    manifest["client_summary"] = client_run.get("summary")
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="Replay one plan once")
    replay.add_argument("--plan", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--policy", default="lru")
    replay.add_argument("--labels")
    replay.add_argument("--capacity", type=int, required=True)
    replay.add_argument("--block-size", type=int, default=16)
    replay.add_argument("--max-model-len", type=int, default=32768)
    replay.add_argument("--port", type=int, default=8000)
    replay.add_argument("--model", default=DEFAULT_MODEL)
    replay.add_argument("--no-event-log", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "replay":
        manifest = replay_plan(args.plan,
                               args.out,
                               policy=args.policy,
                               labels=args.labels,
                               capacity_tokens=args.capacity,
                               block_size=args.block_size,
                               model=args.model,
                               max_model_len=args.max_model_len,
                               port=args.port,
                               event_log=not args.no_event_log)
        print(json.dumps(manifest["client_summary"], indent=2))


if __name__ == "__main__":
    main()
