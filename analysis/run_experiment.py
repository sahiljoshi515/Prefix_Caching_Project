"""Shared policy runner for prepared chat, token or mixed workloads.

python -m analysis.run_experiment --config examples/bailian_experiment.json \
    --out results/bailian-smoke
Paths in the config are relative to that config file. All streams share one
server, clock and concurrency limit; each policy starts with an empty cache.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from analysis import events
from analysis import run_m2_task2 as task2
from analysis import sweep
from analysis.server import SimulatorServer
from client_simulator.client_sim import (iter_requests, load_conversations,
                                         load_tokens)


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_config(path):
    supplied = json.loads(path.read_text())
    config = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "capacity_tokens": 65536,
        "block_size": 16,
        "max_model_len": 32768,
        "max_concurrency": 32,
        "seed": 0,
        **supplied
    }
    for key in ("capacity_tokens", "block_size", "max_model_len",
                "max_concurrency"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["capacity_tokens"] < config["max_model_len"]:
        raise ValueError("Cache must hold at least max_model_len tokens")
    sweep.expand_grid(["lru"], [config["block_size"]],
                      [config["capacity_tokens"]])
    names = set()
    for i, work in enumerate(config["workloads"]):
        work.setdefault("format", "chat")
        work.setdefault("mode", "multi")
        work.setdefault("conversation_rate", 1)
        work.setdefault("think_time", 0)
        work.setdefault("offset_s", 0)
        work.setdefault("seed", config["seed"] + i)
        name = work["name"]
        if not name or not name.replace("_", "").replace(
                "-", "").isalnum() or name in names:
            raise ValueError("Workload names must be unique simple names")
        names.add(name)
        if work["format"] not in ("chat",
                                  "tokens") or work["mode"] not in ("single",
                                                                    "multi"):
            raise ValueError(
                "Workload format/mode must be chat/tokens and single/multi")
        if work.get("limit") is not None and (type(work["limit"]) is not int
                                              or work["limit"] <= 0):
            raise ValueError("Workload limit must be a positive integer")
        for key in ("think_time", "offset_s"):
            if not math.isfinite(work[key]) or work[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not math.isfinite(
                work["conversation_rate"]) or work["conversation_rate"] <= 0:
            raise ValueError("conversation_rate must be finite and positive")
        for key in ("trace", "responses"):
            work[key] = str((path.parent / work[key]).resolve())
    if not names:
        raise ValueError("At least one workload is required")
    return config


def copy_tokens(work, block_size, client_path, server_path, merged):
    """Copy one paired request at a time, including an uncapped trace."""
    ids = []
    with client_path.open("w") as client, server_path.open("w") as server:
        for row in load_tokens(Path(work["trace"]), Path(work["responses"]),
                               work.get("limit")):
            if row["block_size"] != block_size:
                raise ValueError("Token trace and server block sizes differ")
            answer = row.pop("response_token_ids")
            row["output_length"] = len(answer)
            client.write(json.dumps(row) + "\n")
            line = json.dumps({
                "request_id": row["request_id"],
                "response_token_ids": answer
            }) + "\n"
            server.write(line)
            merged.write(line)
            ids.append(row["request_id"])
    return ids


def prepare(config, out):
    """Freeze selected inputs and merge response IDs for the shared server."""
    recorded = out / "experiment.json"
    if recorded.exists():
        sample = json.loads(recorded.read_text())
        if sample["config"] != config:
            raise ValueError(
                "Output has different settings; choose a new --out")
        for path, expected in sample["checksums"].items():
            if checksum(path) != expected:
                raise ValueError(f"Frozen sample changed: {path}")
        return sample
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Output is not an experiment folder; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    all_ids = {}
    workloads, token_ids, checksums = [], set(), {}
    combined = out / "sample.responses.jsonl"
    with combined.open("w") as merged:
        for work in config["workloads"]:
            trace = Path(work["trace"])
            sample_trace = out / f"{work['name']}.requests.jsonl"
            sample_answers = out / f"{work['name']}.responses.jsonl"
            if work["format"] == "tokens":
                ids = copy_tokens(work, config["block_size"], sample_trace,
                                  sample_answers, merged)
                count = len(ids)
                token_ids.update(ids)
            else:
                conversations = list(
                    islice(load_conversations(trace), work.get("limit")))
                rows = [asdict(c) for c in conversations]
                ids = [
                    r.request_id for c in conversations
                    for r in iter_requests(c, work["mode"])
                ]
                wanted, answers = set(ids), {}
                with Path(work["responses"]).open() as source:
                    for line in source:
                        if not line.strip():
                            continue
                        response = json.loads(line)
                        rid = response["request_id"]
                        if rid in wanted:
                            if rid in answers:
                                raise ValueError(
                                    f"Duplicate response ID: {rid}")
                            answers[rid] = response
                if set(answers) != wanted:
                    raise ValueError(f"Missing responses for {work['name']}")
                count = len(rows)
                sample_trace.write_text("".join(
                    json.dumps(r) + "\n" for r in rows))
                with sample_answers.open("w") as target:
                    for rid in ids:
                        line = json.dumps(answers[rid]) + "\n"
                        target.write(line)
                        merged.write(line)
            if (not count or len(ids) != len(set(ids))
                    or any(rid in all_ids for rid in ids)):
                raise ValueError("Empty workload or repeated IDs")
            for rid in ids:
                all_ids[rid] = work["name"]
            workloads.append({
                **work, "trace": str(sample_trace),
                "responses": str(sample_answers),
                "limit": count
            })
            for path in (sample_trace, sample_answers):
                checksums[str(path)] = checksum(path)
    checksums[str(combined)] = checksum(combined)
    sample = {
        "config": config,
        "workloads": workloads,
        "responses": str(combined),
        "request_ids": sorted(all_ids),
        "checksums": checksums,
        "token_request_ids": sorted(token_ids),
        **task2.git_state()
    }
    recorded.write_text(json.dumps(sample, indent=2))
    return sample


def replay(config, sample, settings, directory, port):
    directory.mkdir(parents=True)
    server_args = [
        *config.server_args(), "--max-model-len",
        str(settings["max_model_len"]), "--max-num-seqs",
        str(settings["max_concurrency"]), "--enable-prefix-caching",
        "--enable-prompt-tokens-details", "--return-tokens-as-token-ids",
        "--enable-chunked-prefill=False", "--disable-log-requests"
    ]
    manifest = {
        "sample": sample,
        "model": settings["model"],
        "server_args": server_args,
        "eviction_policy": config.policy,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        **task2.git_state()
    }
    output = directory / "client.jsonl"
    with SimulatorServer(settings["model"],
                         Path(sample["responses"]),
                         directory / "server.log",
                         event_log=directory / "events.jsonl",
                         args=server_args,
                         env=config.env(),
                         port=port) as server:
        manifest["kv_cache"] = server.kv_cache_blocks()
        if manifest["kv_cache"] != {
                "num_gpu_blocks": config.num_blocks,
                "block_size": config.block_size
        }:
            raise ValueError(
                "Actual server cache does not match configuration")
        command = [
            sys.executable, "-m", "client_simulator.client_sim", "--workloads",
            str(directory.parent / "experiment.json"), "--model",
            settings["model"], "--base-url", server.base_url,
            "--max-concurrency",
            str(settings["max_concurrency"]), "--output",
            str(output), "--capture-metrics"
        ]
        manifest["client_command"] = command
        (directory / "run.json").write_text(json.dumps(manifest, indent=2))
        result = subprocess.run(command, capture_output=True, text=True)
    (directory / "client.out").write_text(result.stdout + result.stderr)
    client_run = json.loads(Path(str(output) + ".run.json").read_text())
    if result.returncode not in (0, 1) or client_run["status"] != "finished":
        raise RuntimeError(f"Client failed; see {directory / 'client.out'}")
    manifest["client_summary"] = {
        "failed": 0,
        "skipped_context": 0,
        **client_run["summary"]
    }
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (directory / "run.json").write_text(json.dumps(manifest, indent=2))


def analyze(directory):
    return task2.analyze_run(directory, 10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policies",
                        nargs="+",
                        choices=sweep.POLICIES,
                        default=list(sweep.POLICIES))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    settings = read_config(args.config.resolve())
    out = args.out.resolve()
    if args.analyze_only and not (out / "experiment.json").exists():
        parser.error("No existing experiment to analyze")
    sample = prepare(settings, out)
    configs = sweep.expand_grid(args.policies, [settings["block_size"]],
                                [settings["capacity_tokens"]])
    rows = sweep.run_sweep(
        configs,
        out,
        lambda c, d: replay(c, sample, settings, d, args.port),
        analyze,
        settings["max_model_len"],
        reanalyze=args.analyze_only,
        replay_if=lambda c: not args.analyze_only)
    task2.write_csv(out / "summary.csv", rows)
    populations = []
    for row in rows:
        if row["status"] != "finished":
            continue
        raw = events.load_results(out / row["name"] / "client.jsonl")
        populations.append({r["request_id"]: r["status"] for r in raw})
        if len(populations[-1]) != len(raw) or not set(populations[-1]) <= set(
                sample["request_ids"]):
            raise ValueError("Unexpected or duplicate replay request IDs")
        if not set(sample["token_request_ids"]) <= set(populations[-1]):
            raise ValueError(
                "Some timestamped token requests were not replayed")
        measured = json.loads((out / row["name"] / "summary.json").read_text())
        if row["requests_failed"] or not measured["valid_measurement"]:
            raise ValueError(
                "Run failed measurement checks; inspect saved results")
    if any(p != populations[0] for p in populations[1:]):
        raise ValueError(
            "Policies completed/skipped different request populations")
    print(f"Saved comparison to {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
