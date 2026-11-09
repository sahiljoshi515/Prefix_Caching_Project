import argparse
import asyncio
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import islice, zip_longest
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

import httpx

from client_simulator.schema import (Conversation, Message, ReplayRequest,
                                     ScheduledConversation, request_id)


def iter_requests(conversation: Conversation,
                  mode="multi") -> Iterator[ReplayRequest]:
    """Each request contains the history before its recorded answer"""
    if mode not in {"single", "multi"}:
        raise ValueError("mode must be 'single' or 'multi'")
    for i in range(2 * conversation.first_turn + 1, len(conversation.messages),
                   2):
        turn = i // 2
        rid = request_id(conversation.conversation_id, turn)
        yield ReplayRequest(rid, conversation.conversation_id, turn,
                            conversation.messages[:i],
                            conversation.messages[i].content)
        if mode == "single":
            break


def load_conversations(path):
    """Read the processed JSONL file one conversation at a time"""
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                cid = row["conversation_id"]
                if not isinstance(cid, str) or not cid or cid in seen:
                    raise ValueError("missing, invalid, or duplicate ID")
                messages = [Message(**m) for m in row["messages"]]
                if not messages or len(messages) % 2:
                    raise ValueError("expected complete user/assistant pairs")
                for i, message in enumerate(messages):
                    role = "user" if i % 2 == 0 else "assistant"
                    if (message.role != role
                            or not isinstance(message.content, str)):
                        raise ValueError(f"invalid message {i}")
                first_turn = row.get("first_turn", 0)
                if (not isinstance(first_turn, int)
                        or not 0 <= first_turn < len(messages) // 2):
                    raise ValueError("invalid first_turn")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen.add(cid)
            yield Conversation(cid, messages, first_turn)


def schedule_conversations(conversations, rate, seed, limit=None):
    """Poisson conversation starts"""
    if math.isnan(rate) or rate <= 0:
        raise ValueError("rate must be positive (or inf)")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rng = random.Random(seed)
    arrival = 0.0
    for conversation in islice(conversations, limit):
        if math.isfinite(rate):
            arrival += rng.expovariate(rate)
        yield ScheduledConversation(conversation, arrival)


def load_tokens(path, responses, limit=None):
    """Read paired files and reject mismatched IDs."""
    seen, previous = set(), -1.0
    with path.open() as client, responses.open() as server:
        for raw, answer in islice(zip_longest(client, server), limit):
            if raw is None or answer is None:
                raise ValueError("Token request and response counts differ")
            row, reply = json.loads(raw), json.loads(answer)
            rid = row["request_id"]
            timestamp = row["timestamp_s"]
            prompt, output = row["prompt_token_ids"], reply[
                "response_token_ids"]
            if (rid != reply["request_id"] or rid in seen or not prompt
                    or len(output) != row.pop("output_length") or any(
                        type(t) is not int or t < 0 for t in prompt + output)
                    or not math.isfinite(timestamp) or timestamp < previous):
                raise ValueError(f"Invalid paired token request: {rid}")
            seen.add(rid)
            previous = timestamp
            row["response_token_ids"] = output
            yield row


def server_urls(base_url):
    parts = urlsplit(base_url)
    if (parts.scheme not in {"http", "https"} or not parts.netloc
            or parts.query or parts.fragment):
        raise ValueError("expected an HTTP(S) URL without query or fragment")
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    root = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    return root, root + "/v1"


class ReplayClient:

    def __init__(self,
                 http,
                 base_url,
                 model,
                 results,
                 mode="multi",
                 think_time=0,
                 max_concurrency=32):
        self.http, self.model, self.results = http, model, results
        self.root, self.api = server_urls(base_url)
        self.mode, self.think_time = mode, think_time
        self.slots = asyncio.Semaphore(max_concurrency)
        self.counts = Counter({
            "completed": 0,
            "failed": 0,
            "skipped_context": 0
        })
        self.started = 0.0
        self.workload = "sharegpt"

    def elapsed(self):
        return time.perf_counter() - self.started

    async def check_server(self):
        health = await self.http.get(self.root + "/health")
        health.raise_for_status()
        models = await self.http.get(self.api + "/models")
        models.raise_for_status()
        available = [m["id"] for m in models.json()["data"]]
        if self.model not in available:
            raise ValueError(
                f"{self.model!r} is not served; available: {available}")

    async def tokenize(self, **payload):
        # vLLM closes idle connections after 5 s, the usual think time, so
        # the first call of a turn can reuse a connection as it closes.
        # Tokenizing has no side effects, so one retry is safe (spec 003).
        for attempt in range(2):
            try:
                response = await self.http.post(self.root + "/tokenize",
                                                json={
                                                    "model": self.model,
                                                    **payload
                                                })
                break
            except httpx.TransportError:
                if attempt:
                    raise
        response.raise_for_status()
        return response.json()

    async def send_turn(self, request: ReplayRequest, result):
        """Check context size, send one request, then validate the answer"""
        messages = [asdict(m) for m in request.messages]
        prompt = await self.tokenize(messages=messages,
                                     add_generation_prompt=True,
                                     add_special_tokens=False)
        answer = await self.tokenize(prompt=request.response,
                                     add_special_tokens=False)
        budget = answer["count"] + 1  # +1 for the simulator final EOS.
        result.update(prompt_tokens=prompt["count"],
                      max_tokens=budget,
                      max_model_len=prompt["max_model_len"])
        if prompt["count"] + budget > prompt["max_model_len"]:
            result.update(status="skipped_context",
                          error="Context limit exceeded")
            return

        result["sent_s"] = self.elapsed()
        result["dispatch_delay_s"] = result["sent_s"] - result["ready_s"]
        response = await self.http.post(
            self.api + "/chat/completions",
            headers={"X-Request-Id": request.request_id},
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": budget,
                "temperature": 0,
                "stream": False,
                "add_generation_prompt": True,
                "add_special_tokens": False
            })
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        actual = choice["message"].get("content") or ""
        usage = data.get("usage") or {}
        # vLLM omits prompt_tokens_details when no prompt token was cached.
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or 0
        count = usage.get("prompt_tokens", prompt["count"])
        result.update(prompt_tokens=count,
                      completion_tokens=usage.get("completion_tokens"),
                      cached_tokens=cached,
                      cached_fraction=cached / count if count else None,
                      finish_reason=choice.get("finish_reason"),
                      answer_match=actual == request.response)
        if choice.get("finish_reason") != "stop":
            result["error"] = "Response did not finish with stop"
        elif actual != request.response:
            result["error"] = "Answer differs from recording"
        else:
            result["status"] = "completed"
        if result["status"] != "completed":
            result.update(actual_response_preview=actual[:500],
                          expected_response_preview=request.response[:500])

    async def replay_conversation(self, conversation, arrival_s):
        actual_start = self.elapsed()
        turns = (conversation.first_turn +
                 1 if self.mode == "single" else len(conversation.messages) //
                 2)
        self.counts["conversations"] += 1
        for request in iter_requests(conversation, self.mode):
            result = {
                "workload": self.workload,
                "request_id": request.request_id,
                "conversation_id": request.conversation_id,
                "turn_index": request.turn_index,
                "planned_conversation_start_s": arrival_s,
                "actual_conversation_start_s": actual_start,
                "conversation_start_drift_s": max(0, actual_start - arrival_s),
                "ready_s": self.elapsed(),
                "sent_s": None,
                "latency_s": None,
                "status": "failed",
                "error": None,
            }
            try:
                async with self.slots:
                    await self.send_turn(request, result)
            except (httpx.HTTPError, ValueError, KeyError, IndexError,
                    TypeError) as exc:
                result["error"] = str(exc)
                if isinstance(exc, httpx.HTTPStatusError):
                    result["server_error"] = exc.response.text[:1000]
            result["completed_s"] = self.elapsed()
            if result["sent_s"] is not None:
                result["latency_s"] = result["completed_s"] - result["sent_s"]
            self.results.write(json.dumps(result, ensure_ascii=False) + "\n")
            self.results.flush()
            self.counts[result["status"]] += 1
            if result["status"] != "completed":
                remaining = turns - request.turn_index - 1
                self.counts["abandoned_turns"] += remaining
                break
            if request.turn_index + 1 < turns:
                await asyncio.sleep(self.think_time)

    async def run(self, conversations, rate, seed, limit=None, offset_s=0):
        if not self.started:
            await self.check_server()
            self.started = time.perf_counter()
        tasks = []
        try:
            schedule = schedule_conversations(conversations, rate, seed, limit)
            for item in schedule:
                arrival = item.arrival_s + offset_s
                await asyncio.sleep(max(0, arrival - self.elapsed()))
                tasks.append(
                    asyncio.create_task(
                        self.replay_conversation(item.conversation, arrival)))
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return {**self.counts, "elapsed_s": self.elapsed()}

    async def replay_token(self, request, arrival, max_model_len):
        row = {
            "request_id": request["request_id"],
            "conversation_id": request["conversation_id"],
            "turn_index": request["turn_index"],
            "workload": self.workload,
            "request_type": request["request_type"],
            "timestamp_s": request["timestamp_s"],
            "scheduled_s": arrival,
            "ready_s": self.elapsed(),
            "sent_s": None,
            "prompt_tokens": len(request["prompt_token_ids"]),
            "max_tokens": len(request["response_token_ids"]) + 1,
            "cached_tokens": None,
            "cached_fraction": None,
            "status": "failed",
            "error": None
        }
        try:
            if row["prompt_tokens"] + row["max_tokens"] > max_model_len:
                row.update(status="skipped_context",
                           error="Context limit exceeded")
            else:
                async with self.slots:
                    row["sent_s"] = self.elapsed()
                    row["dispatch_delay_s"] = row["sent_s"] - row["ready_s"]
                    row["arrival_drift_s"] = row["sent_s"] - arrival
                    response = await self.http.post(
                        self.api + "/completions",
                        headers={"X-Request-Id": request["request_id"]},
                        json={
                            "model": self.model,
                            "prompt": request["prompt_token_ids"],
                            "max_tokens": row["max_tokens"],
                            "add_special_tokens": False,
                            "temperature": 0,
                            "logprobs": 0
                        })
                    response.raise_for_status()
                    data = response.json()
                    choice = data["choices"][0]
                    # Exact IDs come from --return-tokens-as-token-ids.
                    actual = [
                        int(t.removeprefix("token_id:"))
                        for t in choice["logprobs"]["tokens"]
                    ]
                    expected = request["response_token_ids"]
                    match = (actual == expected
                             or actual == expected + [request["eos_token_id"]])
                    row.update(
                        answer_match=match,
                        finish_reason=choice["finish_reason"],
                        completion_tokens=data["usage"]["completion_tokens"])
                    if match and choice["finish_reason"] == "stop":
                        row["status"] = "completed"
                    else:
                        row["error"] = "Token mismatch or abnormal finish"
        except (httpx.HTTPError, ValueError, KeyError, IndexError,
                TypeError) as exc:
            row["error"] = str(exc)
        row["completed_s"] = self.elapsed()
        row["latency_s"] = (row["completed_s"] - row["sent_s"]
                            if row["sent_s"] is not None else None)
        self.results.write(json.dumps(row) + "\n")
        self.results.flush()
        self.counts[row["status"]] += 1

    async def run_tokens(self, requests, offset_s=0):
        if not self.started:
            await self.check_server()
            self.started = time.perf_counter()
        # Query context size once; do not re-tokenize synthetic inputs.
        config = await self.tokenize(prompt="", add_special_tokens=False)
        tasks, first = [], None
        try:
            for request in requests:
                if first is None:
                    first = request["timestamp_s"]
                arrival = request["timestamp_s"] - first + offset_s
                await asyncio.sleep(max(0, arrival - self.elapsed()))
                tasks.append(
                    asyncio.create_task(
                        self.replay_token(request, arrival,
                                          config["max_model_len"])))
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return {**self.counts, "elapsed_s": self.elapsed()}


async def run_workloads(replay, workloads):
    """Overlay streams on one clock with one concurrency limit."""
    if not workloads:
        raise ValueError("At least one workload is required")
    for work in workloads:
        if work["format"] not in ("chat", "tokens"):
            raise ValueError("Workload format must be chat or tokens")
        offset = work.get("offset_s", 0)
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("Workload offset must be finite and nonnegative")
    await replay.check_server()
    started = time.perf_counter()
    clients, jobs = [], []
    for work in workloads:
        client = ReplayClient(replay.http, replay.api, replay.model,
                              replay.results, work.get("mode", "multi"),
                              work.get("think_time", 0))
        client.started, client.slots = started, replay.slots
        client.workload = work["name"]
        offset = work.get("offset_s", 0)
        if work["format"] == "tokens":
            requests = load_tokens(Path(work["trace"]),
                                   Path(work["responses"]), work.get("limit"))
            jobs.append(client.run_tokens(requests, offset))
        else:
            jobs.append(
                client.run(load_conversations(Path(work["trace"])),
                           work.get("conversation_rate", 1),
                           work.get("seed", 0), work.get("limit"), offset))
        clients.append(client)
    tasks = [asyncio.create_task(job) for job in jobs]
    try:
        summaries = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Preserve partial counters if an input or transport task fails.
        replay.counts = sum((c.counts for c in clients), Counter())
    return {
        **replay.counts, "elapsed_s": time.perf_counter() - started,
        "workloads": {w["name"]: s
                      for w, s in zip(workloads, summaries)}
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--trace", type=Path)
    inputs.add_argument("--workloads",
                        type=Path,
                        help="JSON file of prepared workloads")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument(
        "--conversation-rate",
        type=float,
        default=1,
        help="New conversations/second; inf starts immediately")
    parser.add_argument("--think-time", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output",
                        type=Path,
                        default=Path("results/replay.jsonl"))
    parser.add_argument("--capture-metrics", action="store_true")
    args = parser.parse_args(argv)
    if math.isnan(args.conversation_rate) or args.conversation_rate <= 0:
        parser.error("--conversation-rate must be positive (or inf)")
    if not math.isfinite(args.think_time) or args.think_time < 0:
        parser.error("--think-time must be finite and nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.max_concurrency <= 0 or (args.limit is not None
                                     and args.limit <= 0):
        parser.error("--max-concurrency and --limit must be positive")
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
    workloads, inputs = [], set()
    if args.workloads:
        workloads = json.loads(args.workloads.read_text())["workloads"]
        inputs.add(args.workloads.resolve())
        for work in workloads:
            for key in ("trace", "responses"):
                work[key] = str((args.workloads.parent / work[key]).resolve())
                inputs.add(Path(work[key]))
    else:
        inputs.add(args.trace.resolve())
    if any(p.exists() or p.resolve() in inputs for p in outputs):
        raise ValueError("Output already exists; choose a new output path")
    metadata = {
        "config": {
            k: str(v) if isinstance(v, Path) or v == math.inf else v
            for k, v in vars(args).items()
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    limits = httpx.Limits(max_connections=args.max_concurrency + 4)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as http:
        with args.output.open("x", encoding="utf-8") as output:
            replay = ReplayClient(http, args.base_url, args.model, output,
                                  args.mode, args.think_time,
                                  args.max_concurrency)
            manifest.write_text(json.dumps(metadata, indent=2),
                                encoding="utf-8")
            try:
                if args.capture_metrics:
                    before = await http.get(replay.root + "/metrics")
                    before.raise_for_status()
                    metric_paths["before"].write_text(before.text,
                                                      encoding="utf-8")
                if not args.workloads:
                    summary = await replay.run(load_conversations(args.trace),
                                               args.conversation_rate,
                                               args.seed, args.limit)
                else:
                    summary = await run_workloads(replay, workloads)
                metadata.update(status="finished", summary=summary)
                if args.capture_metrics:
                    after = await http.get(replay.root + "/metrics")
                    after.raise_for_status()
                    metric_paths["after"].write_text(after.text,
                                                     encoding="utf-8")
            except BaseException as exc:
                interrupted = isinstance(
                    exc, (KeyboardInterrupt, asyncio.CancelledError))
                status = "interrupted" if interrupted else "failed"
                metadata.update(status=status,
                                error=str(exc),
                                summary=dict(replay.counts))
                raise
            finally:
                manifest.write_text(json.dumps(metadata, indent=2),
                                    encoding="utf-8")
    return summary


def main():
    try:
        summary = asyncio.run(run_cli(parse_args()))
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise SystemExit(f"Replay failed: {exc}") from exc
    print(json.dumps(summary, indent=2))
    if summary.get("failed") or summary.get("skipped_context"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
