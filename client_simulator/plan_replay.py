"""Replay a token-level plan against a simulator server (spec 006 FR-003).

    python -m client_simulator.plan_replay --plan plan.jsonl.gz \\
        --model Qwen/Qwen2.5-0.5B-Instruct --output results/run.jsonl \\
        --capture-metrics

Each request is sent to ``/v1/completions`` with its prompt as token ids (no
chat template: text workloads were templated when the plan was built) and
``max_tokens`` = its recorded output plus the simulator's final EOS, with
``X-Request-Id`` set to the plan's request id. A session's requests go in
order:

- the first at its ``arrival_s``;
- a closed-loop request (``after_s``) that long after the previous one
  completes;
- an open-loop request (``arrival_s``, recorded timestamps) at its time, but
  not before the previous one completes, so a turn can read the blocks its
  parent wrote. The wait is recorded as ``dispatch_delay_s``.

A request that fails or exceeds the context ends a closed-loop session (as
the chat client does); an open-loop session goes on. A request completes if
it finishes with ``stop`` after exactly its recorded output. Results have the
chat client's fields, so ``analysis/`` reads them unchanged, plus
``workload``. The ShareGPT chat client (``client_sim.py``) is unchanged.
"""
import argparse
import asyncio
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import httpx

from client_simulator.client_sim import server_urls
from client_simulator.schema import TokenRequest
from client_simulator.token_plan import read_plan


def group_sessions(requests: List[TokenRequest]) -> List[List[TokenRequest]]:
    """Sessions in order of their first arrival, turns in order."""
    sessions: Dict[str, List[TokenRequest]] = {}
    for request in requests:
        sessions.setdefault(request.session_id, []).append(request)
    out = []
    for session in sessions.values():
        session.sort(key=lambda r: r.turn_index)
        if session[0].arrival_s is None:
            raise ValueError(f"{session[0].request_id}: a session's first "
                             "request needs arrival_s")
        for r in session[1:]:
            if (r.arrival_s is None) == (r.after_s is None):
                raise ValueError(f"{r.request_id}: set exactly one of "
                                 "arrival_s and after_s")
        out.append(session)
    out.sort(key=lambda s: (s[0].arrival_s, s[0].session_id))
    return out


class PlanReplay:

    def __init__(self, http, base_url, model, results, max_concurrency=64):
        self.http, self.model, self.results = http, model, results
        self.root, self.api = server_urls(base_url)
        self.slots = asyncio.Semaphore(max_concurrency)
        self.counts: Counter = Counter()
        self.started = 0.0
        self.max_model_len = None

    def elapsed(self):
        return time.perf_counter() - self.started

    async def check_server(self):
        health = await self.http.get(self.root + "/health")
        health.raise_for_status()
        models = await self.http.get(self.api + "/models")
        models.raise_for_status()
        cards = {m["id"]: m for m in models.json()["data"]}
        if self.model not in cards:
            raise ValueError(
                f"{self.model!r} is not served; available: {list(cards)}")
        self.max_model_len = cards[self.model].get("max_model_len")

    async def send(self, request: TokenRequest, result):
        budget = len(request.output_token_ids) + 1  # +1 for the final EOS.
        prompt_len = len(request.prompt_token_ids)
        result.update(prompt_tokens=prompt_len,
                      max_tokens=budget,
                      max_model_len=self.max_model_len)
        if (self.max_model_len is not None
                and prompt_len + budget > self.max_model_len):
            result.update(status="skipped_context",
                          error="Context limit exceeded")
            return
        result["sent_s"] = self.elapsed()
        result["dispatch_delay_s"] = result["sent_s"] - result["ready_s"]
        response = await self.http.post(
            self.api + "/completions",
            headers={"X-Request-Id": request.request_id},
            json={
                "model": self.model,
                "prompt": request.prompt_token_ids,
                "max_tokens": budget,
                "temperature": 0,
                "stream": False,
            })
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or 0
        count = usage.get("prompt_tokens", prompt_len)
        generated = usage.get("completion_tokens")
        result.update(prompt_tokens=count,
                      completion_tokens=generated,
                      cached_tokens=cached,
                      cached_fraction=cached / count if count else None,
                      finish_reason=choice.get("finish_reason"))
        # The simulator replays the recorded tokens, then EOS, which vLLM
        # counts as a generated token.
        if choice.get("finish_reason") != "stop":
            result["error"] = "Response did not finish with stop"
        elif generated != budget:
            result["error"] = (f"Generated {generated} tokens, expected "
                               f"{budget - 1} and EOS")
        else:
            result["status"] = "completed"

    async def replay_session(self, session: List[TokenRequest]):
        self.counts["sessions"] += 1
        for index, request in enumerate(session):
            if request.arrival_s is not None:
                await asyncio.sleep(
                    max(0.0, request.arrival_s - self.elapsed()))
            else:
                await asyncio.sleep(request.after_s)
            result = {
                "request_id": request.request_id,
                "conversation_id": request.session_id,
                "turn_index": request.turn_index,
                "workload": request.workload,
                "planned_s": request.arrival_s,
                "after_s": request.after_s,
                "ready_s": self.elapsed(),
                "sent_s": None,
                "latency_s": None,
                "status": "failed",
                "error": None,
            }
            if request.arrival_s is not None:
                result["start_drift_s"] = max(
                    0.0, result["ready_s"] - request.arrival_s)
            try:
                async with self.slots:
                    await self.send(request, result)
            except (httpx.HTTPError, ValueError, KeyError, IndexError,
                    TypeError) as exc:
                result["error"] = str(exc) or type(exc).__name__
                if isinstance(exc, httpx.HTTPStatusError):
                    result["server_error"] = exc.response.text[:1000]
            result["completed_s"] = self.elapsed()
            if result["sent_s"] is not None:
                result["latency_s"] = result["completed_s"] - result["sent_s"]
            self.results.write(json.dumps(result) + "\n")
            self.results.flush()
            self.counts[result["status"]] += 1
            closed_loop = any(r.after_s is not None
                              for r in session[index + 1:])
            if result["status"] != "completed" and closed_loop:
                self.counts["abandoned_turns"] += len(session) - index - 1
                return

    async def run(self, sessions: List[List[TokenRequest]], time_scale=1.0):
        await self.check_server()
        self.started = time.perf_counter()
        if time_scale != 1.0:
            for session in sessions:
                for r in session:
                    if r.arrival_s is not None:
                        r.arrival_s *= time_scale
        tasks = []
        try:
            for session in sessions:
                await asyncio.sleep(
                    max(0.0, session[0].arrival_s - self.elapsed()))
                tasks.append(asyncio.create_task(self.replay_session(session)))
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return {**self.counts, "elapsed_s": self.elapsed()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--time-scale",
                        type=float,
                        default=1.0,
                        help="Multiply every arrival time (pacing studies)")
    parser.add_argument("--output",
                        type=Path,
                        default=Path("results/plan-replay.jsonl"))
    parser.add_argument("--capture-metrics", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not math.isfinite(args.time_scale) or args.time_scale <= 0:
        parser.error("--time-scale must be finite and positive")
    if args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    try:
        server_urls(args.base_url)
    except ValueError as exc:
        parser.error(str(exc))
    return args


async def run_cli(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = Path(str(args.output) + ".run.json")
    metric_paths = {}
    if args.capture_metrics:
        for phase in ("before", "after"):
            metric_paths[phase] = Path(
                str(args.output) + f".metrics.{phase}.txt")
    outputs = [args.output, manifest, *metric_paths.values()]
    if any(p.exists() for p in outputs):
        raise ValueError("Output already exists; choose a new output path")
    sessions = group_sessions(list(read_plan(args.plan)))
    metadata = {
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "sessions": len(sessions),
        "requests": sum(len(s) for s in sessions),
    }
    limits = httpx.Limits(max_connections=args.max_concurrency + 4)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as http:
        with args.output.open("x", encoding="utf-8") as output:
            replay = PlanReplay(http, args.base_url, args.model, output,
                                args.max_concurrency)
            manifest.write_text(json.dumps(metadata, indent=2),
                                encoding="utf-8")
            try:
                if args.capture_metrics:
                    before = await http.get(replay.root + "/metrics")
                    before.raise_for_status()
                    metric_paths["before"].write_text(before.text,
                                                      encoding="utf-8")
                summary = await replay.run(sessions, args.time_scale)
                metadata.update(status="finished", summary=summary)
                if args.capture_metrics:
                    after = await http.get(replay.root + "/metrics")
                    after.raise_for_status()
                    metric_paths["after"].write_text(after.text,
                                                     encoding="utf-8")
            except BaseException as exc:
                interrupted = isinstance(
                    exc, (KeyboardInterrupt, asyncio.CancelledError))
                metadata.update(
                    status="interrupted" if interrupted else "failed",
                    error=str(exc),
                    summary=dict(replay.counts))
                raise
            finally:
                metadata["finished_at_utc"] = datetime.now(
                    timezone.utc).isoformat()
                manifest.write_text(json.dumps(metadata, indent=2),
                                    encoding="utf-8")
    return metadata


def main(argv=None):
    args = parse_args(argv)
    metadata = asyncio.run(run_cli(args))
    print(json.dumps(metadata["summary"], indent=2))
    counts = metadata["summary"]
    if counts.get("failed") or counts.get("skipped_context"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
