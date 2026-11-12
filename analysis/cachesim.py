"""Fast trace-driven model of vLLM's prefix cache (spec 007 design tool).

Replays a token-level plan (``analysis/mix.py``) through vLLM's own
``PrefixCachingBlockAllocator``, evictors and ``BlockTable``, the calls the
block manager makes, on a simulated clock and without a server or scheduler:

- a request allocates its prompt's blocks at its start (lookups, hits,
  misses, evictions), as ``BlockSpaceManager.allocate`` does;
- its output fills blocks as it decodes, each full block being inserted or
  deduplicated, as ``append_slots`` does;
- at its finish its blocks get their last-access time and are freed, as the
  block manager's ``free`` does.

A run takes seconds instead of the plan's real-time duration, so policy,
capacity and parameter sweeps run here first. It is a design and tuning
tool: reported results come from ``vllm serve`` runs (constitution I), and
the simulator's agreement with them is checked (spec 007 plan, validation).

Timing comes from either:

- ``model``: arrivals from the plan (closed loop: the previous turn's finish
  plus its think time; open loop: the recorded time, but not before the
  previous turn finishes) and a latency of ``prefill_s + step_s`` per
  generated token; or
- ``engine:<run dir>``: each request's start, decode-block and finish times
  and the last-access time vLLM stamped on its blocks, read from that run's
  cache event log. Request timing does not depend on the policy in the
  simulator (compute is skipped), so one engine run times every policy.

Not modelled: the scheduler's batching and its free-block watermark, and
preemption (a decode block that finds no free block is counted and the
request stops growing). A request whose prompt does not fit waits until a
block is freed, like ``AllocStatus.LATER``; such waits are counted.

The policy is one of vLLM's (``lru``, ``lfu``, ``fifo``, ``ttdf``,
``race_rp``, set through ``VLLM_EVICTION_POLICY``; text after a colon, as in
``ttdf:tau_short=2``, becomes ``VLLM_EVICTION_PARAMS``) or ``belady``, the
offline optimum of
``analysis/oracle.py``, which reads every future prompt lookup. Belady
bounds what any policy could reuse at a capacity. It is a reference that
only the simulator can run, never a vLLM policy, and its summary says so
(``offline_oracle``).
"""
import argparse
import contextlib
import gzip
import heapq
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Dict, Iterator, List, Optional,
                    Sequence, Tuple)

from client_simulator.plan_replay import group_sessions, planned_s
from client_simulator.schema import TokenRequest
from client_simulator.token_plan import read_plan

if TYPE_CHECKING:
    from analysis.oracle import BeladyEvictor

# The engine's request id for a completions request (`analysis/events.py`).
ENGINE_ID = "cmpl-{}-0"

# The offline optimum (`analysis/oracle.py`); not a vLLM policy.
BELADY = "belady"


def block_hashes(tokens: List[int], block_size: int) -> List[int]:
    """The content hash of each full block of `tokens`, chained as vLLM's
    prefix-caching allocator chains them (no extra hash)."""
    from vllm.core.block.prefix_caching_block import PrefixCachingBlock
    out: List[int] = []
    prev = None
    for i in range(len(tokens) // block_size):
        prev = PrefixCachingBlock.hash_block_tokens(
            i == 0, prev, tokens[i * block_size:(i + 1) * block_size])
        out.append(prev)
    return out


@dataclass
class Timing:
    """One request's times. `start`, `finish` and `decode` are on the
    simulated clock; `la` is the last-access time stamped on its blocks."""
    start: float
    finish: float
    la: float
    decode: Optional[List[float]] = None


def model_timing(sessions: List[List[TokenRequest]],
                 prefill_s: float = 0.005,
                 step_s: float = 0.0002) -> Dict[str, Timing]:
    """Arrivals as `plan_replay` sends them, with a simple latency model
    (the simulator's measured latency is about 0.2 ms per token)."""
    timing = {}
    for session in sessions:
        previous: Optional[float] = None
        for r in session:
            if r.arrival_s is not None:
                start = r.arrival_s if previous is None else max(
                    r.arrival_s, previous)
            else:
                assert previous is not None and r.after_s is not None
                start = previous + r.after_s
            finish = start + prefill_s + step_s * (len(r.output_token_ids) + 1)
            timing[r.request_id] = Timing(start, finish, la=finish)
            previous = finish
    return timing


def _events(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            if line.strip():
                yield json.loads(line)


def engine_timing(run_dir: Path,
                  requests: List[TokenRequest]) -> Dict[str, Timing]:
    """Times of each request in an engine run's event log.

    A request with no event (no full prompt block, no full decode block)
    falls back to the client's times, shifted onto the log's clock by the
    median offset of the requests that have both.
    """
    from analysis.events import client_request_id, log_path
    path = log_path(run_dir)
    first: Dict[str, float] = {}
    free_t: Dict[str, float] = {}
    la: Dict[str, float] = {}
    decode: Dict[str, List[float]] = defaultdict(list)
    for e in _events(path):
        req = e.get("req")
        if req is None or e["ev"] in ("header", "end"):
            continue
        rid = client_request_id(req)
        first.setdefault(rid, e["t"])
        if e["op"] == "append_slots" and e["ev"] in ("insert", "dedupe"):
            decode[rid].append(e["t"])
        elif e["op"] == "free" and e["ev"] == "free":
            free_t.setdefault(rid, e["t"])
            if e.get("la") is not None:  # logs from before spec 007 lack it
                la.setdefault(rid, e["la"])
    client = {}
    with (run_dir / "client.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("sent_s") is not None:
                client[row["request_id"]] = (row["sent_s"], row["completed_s"])
    offsets = sorted(first[r] - client[r][0] for r in first if r in client)
    offset = offsets[len(offsets) // 2] if offsets else 0.0
    # The wall clock vLLM stamps, relative to the log's monotonic clock.
    walls = sorted(la[r] - free_t[r] for r in la)
    wall = walls[len(walls) // 2] if walls else 0.0
    timing = {}
    for r in requests:
        rid = r.request_id
        if rid not in client:
            continue  # never sent (skipped or abandoned in that run)
        sent, done = client[rid]
        start = first.get(rid, sent + offset)
        finish = free_t.get(rid, done + offset)
        timing[rid] = Timing(start,
                             max(finish, start),
                             la=la.get(rid, finish + wall),
                             decode=decode.get(rid))
    return timing


class _SimClock:
    """Stands in for the `time` module of the event recorder, so logged
    times are simulated times."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now


@dataclass
class _Active:
    request: TokenRequest
    table: Any
    workload: str
    appended: int = 0  # output tokens appended so far


@dataclass
class SimResult:
    summary: Dict[str, Any]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    # (simulated time, workload of the block's last request) per eviction.
    evictions: List[Tuple[float, Optional[str]]] = field(default_factory=list)


@contextlib.contextmanager
def _env(values: Dict[str, Optional[str]]):
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _quiet(logger_name: str):
    """Hide a logger's info lines while the block runs."""
    logger = logging.getLogger(logger_name)
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(level)


class CacheSim:
    """One replay of one plan at one configuration."""

    def __init__(self,
                 num_blocks: int,
                 block_size: int = 16,
                 policy: str = "lru",
                 env: Optional[Dict[str, str]] = None,
                 event_log: Optional[Path] = None):
        from vllm.core.block import cache_events
        from vllm.core.block.cpu_gpu_block_allocator import (
            CpuGpuBlockAllocator)
        from vllm.utils import Device
        self.block_size = block_size
        self.clock = _SimClock()
        self._cache_events = cache_events
        # Belady is not a vLLM policy: the allocator is built with LRU (not
        # logged, as it is not the policy) and the oracle's evictor replaces
        # LRU's before anything is cached. `run` hands it the lookups.
        name, _, params = policy.partition(":")
        belady = name.strip().lower() == BELADY
        settings: Dict[str, Optional[str]] = {
            "VLLM_EVICTION_POLICY": "lru" if belady else name,
            "VLLM_EVICTION_PARAMS": params or None,
            "VLLM_CACHE_EVENT_LOG": str(event_log) if event_log else None,
            **(env or {}),
        }
        quiet = (_quiet("vllm.core.eviction_policies")
                 if belady else contextlib.nullcontext())
        with _env(settings), self._sim_time(), quiet:
            self.alloc = CpuGpuBlockAllocator.create("prefix_caching",
                                                     num_gpu_blocks=num_blocks,
                                                     num_cpu_blocks=0,
                                                     block_size=block_size)
        self.gpu = self.alloc._allocators[Device.GPU]
        self.belady: Optional[BeladyEvictor] = None
        if belady:
            from analysis import oracle
            self.belady = oracle.BeladyEvictor()
            self.gpu.evictor = self.belady
        self.recorder = self.gpu._events
        self.counts: Counter = Counter()
        self._hits = 0
        # Content hash -> workload of the last request that used the block.
        self.owner: Dict[int, str] = {}
        self.evictions: List[Tuple[float, Optional[str]]] = []
        record_hit, evict = self.gpu.evictor.record_hit, self.gpu.evictor.evict

        def counting_hit(block_id: int) -> None:
            self._hits += 1
            record_hit(block_id)

        def counting_evict():
            self.counts["evictions"] += 1
            start = time.perf_counter()
            block_id, content_hash = evict()
            self.counts["evict_seconds"] += time.perf_counter() - start
            self.evictions.append(
                (self.clock.now, self.owner.get(content_hash)))
            return block_id, content_hash

        self.gpu.evictor.record_hit = counting_hit
        self.gpu.evictor.evict = counting_evict

    @contextlib.contextmanager
    def _sim_time(self):
        saved = self._cache_events.time
        self._cache_events.time = self.clock
        try:
            yield
        finally:
            self._cache_events.time = saved

    def _attribute(self, request: TokenRequest, op: str) -> None:
        # As the block manager's `attributed` methods do, for the event log
        # and for evictors that read the request's workload.
        self._cache_events.set_current_request(
            ENGINE_ID.format(request.request_id), op)

    def _hashes(self, tokens: List[int]) -> List[int]:
        return block_hashes(tokens, self.block_size)

    def _decode_chunks(self, request: TokenRequest) -> List[int]:
        """Output tokens appended at each decode-block fill: the first chunk
        completes the prompt's partial block."""
        bs = self.block_size
        first = (bs - len(request.prompt_token_ids) % bs) % bs or bs
        sizes, left = [], len(request.output_token_ids)
        size = first
        while left >= size:
            sizes.append(size)
            left -= size
            size = bs
        return sizes

    def run(self, requests: List[TokenRequest],
            timing: Dict[str, Timing]) -> SimResult:
        from vllm.core.block.block_table import BlockTable
        from vllm.core.block.interfaces import BlockAllocator
        from vllm.utils import Device

        # (time, order, seq, kind, request id, payload); at equal times a
        # request's decode fills go before its free, and frees before new
        # prompts (the engine frees finished requests before it schedules).
        queue: List[Tuple[float, int, int, str, str, Any]] = []
        seq = 0
        by_id = {r.request_id: r for r in requests}
        for r in requests:
            t = timing.get(r.request_id)
            if t is None:
                self.counts["not_sent"] += 1
                continue
            heapq.heappush(queue,
                           (t.start, 2, seq, "alloc", r.request_id, None))
            seq += 1
        active: Dict[str, _Active] = {}
        waiting: List[str] = []  # prompts that did not fit, in order
        rows: List[Dict[str, Any]] = []
        per: Dict[str, Counter] = defaultdict(Counter)
        lookups = None
        if self.belady is not None:
            # Read before `schedule_decodes` shifts the times of a request
            # that waited (the oracle keys on allocation order, which waits
            # do not change).
            from analysis.oracle import future_lookups
            lookups = future_lookups(requests, timing, self.block_size)
            self.belady.lookups = lookups

        def schedule_decodes(rid: str, start: float) -> None:
            """Queue a request's decode fills and free. A request that waited
            for space keeps its timeline, shifted by the wait."""
            nonlocal seq
            r, t = by_id[rid], timing[rid]
            delay = max(start - t.start, 0.0)
            if delay:
                timing[rid] = t = Timing(
                    t.start + delay, t.finish + delay, t.la + delay,
                    [d + delay
                     for d in t.decode] if t.decode is not None else None)
            sizes = self._decode_chunks(r)
            times = t.decode
            if times is None or len(times) != len(sizes):
                span = max(t.finish - start, 0.0)
                times = [
                    start + span * (i + 1) / (len(sizes) + 1)
                    for i in range(len(sizes))
                ]
                if t.decode is not None:
                    self.counts["decode_times_interpolated"] += 1
            last = start
            for when, size in zip(times, sizes):
                last = max(when, last)
                heapq.heappush(queue, (last, 0, seq, "decode", rid, size))
                seq += 1
            heapq.heappush(queue,
                           (max(t.finish, last), 1, seq, "free", rid, None))
            seq += 1

        def try_allocate(rid: str, now: float) -> bool:
            r = by_id[rid]
            needed = -(-len(r.prompt_token_ids) // self.block_size)
            if needed > self.alloc.get_num_free_blocks(Device.GPU):
                return False
            hashes = self._hashes(r.prompt_token_ids)
            cached = self.gpu._cached_blocks
            usable = 0
            for h in hashes:
                if h not in cached:
                    break
                usable += 1
            vllm_cached = len(self.alloc.find_cached_blocks_prefix(hashes))
            table = BlockTable(self.block_size, self.alloc)
            self._attribute(r, "allocate")
            hits_before = self._hits
            table.allocate(r.prompt_token_ids, device=Device.GPU)
            if lookups is not None:
                lookups.allocated(rid)  # raises if out of the oracle's order
            self.alloc.mark_blocks_as_computed([])
            hits = self._hits - hits_before
            workload = r.workload
            for h in hashes:
                self.owner[h] = workload
            prompt = len(r.prompt_token_ids)
            usable_tokens = min(usable * self.block_size, prompt)
            row = {
                "request_id": r.request_id,
                "conversation_id": r.session_id,
                "turn_index": r.turn_index,
                "workload": workload,
                "status": "completed",
                "prompt_tokens": prompt,
                "cached_tokens": vllm_cached * self.block_size,
                "usable_cached_tokens": usable_tokens,
                "hits": hits,
                "lookups": len(hashes),
                "hits_after_miss": hits - usable,
                "start_s": now,
            }
            rows.append(row)
            c = per[workload]
            c["requests"] += 1
            c["prompt_tokens"] += prompt
            c["usable_tokens"] += usable_tokens
            c["vllm_cached_tokens"] += min(vllm_cached * self.block_size,
                                           prompt)
            c["hits"] += hits
            c["lookups"] += len(hashes)
            c["requests_with_hits_after_miss"] += hits > usable
            active[rid] = _Active(r, table, workload)
            schedule_decodes(rid, now)
            return True

        with self._sim_time():
            while queue:
                now, _, _, kind, rid, payload = heapq.heappop(queue)
                self.clock.now = now
                if kind == "alloc":
                    if waiting or not try_allocate(rid, now):
                        waiting.append(rid)
                        self.counts["deferred"] += 1
                    continue
                state = active[rid]
                if kind == "decode":
                    if state.appended < 0:
                        continue  # stopped after a failed decode block
                    chunk = state.request.output_token_ids[state.appended:state
                                                           .appended + payload]
                    self._attribute(state.request, "append_slots")
                    try:
                        state.table.append_token_ids(chunk)
                        self.alloc.mark_blocks_as_computed([])
                        state.appended += payload
                    except BlockAllocator.NoFreeBlocksError:
                        self.counts["decode_out_of_blocks"] += 1
                        state.appended = -1
                    continue
                # free
                r = state.request
                if state.appended >= 0:
                    rest = r.output_token_ids[state.appended:]
                    if rest:
                        # Output that fills no block is still this request's
                        # decode: name it, or a new block's eviction would be
                        # logged under whichever request ran last.
                        self._attribute(r, "append_slots")
                        try:
                            state.table.append_token_ids(rest)
                        except BlockAllocator.NoFreeBlocksError:
                            self.counts["decode_out_of_blocks"] += 1
                for block in state.table.blocks:
                    if block.content_hash is not None:
                        self.owner[block.content_hash] = state.workload
                self.alloc.mark_blocks_as_accessed(
                    state.table.physical_block_ids, timing[rid].la)
                self._attribute(r, "free")
                state.table.free()
                del active[rid]
                while waiting and try_allocate(waiting[0], now):
                    waiting.pop(0)
            if self.recorder is not None:
                self.recorder.close()
        self.counts["never_allocated"] += len(waiting)
        return SimResult(self._summary(per, rows), rows, self.evictions)

    def _summary(self, per: Dict[str, Counter],
                 rows: List[Dict[str, Any]]) -> Dict[str, Any]:

        def fractions(c: Counter) -> Dict[str, Any]:
            prompt = c["prompt_tokens"]
            return {
                **dict(c),
                "usable_reused_fraction":
                c["usable_tokens"] / prompt if prompt else None,
                "vllm_cached_share":
                c["vllm_cached_tokens"] / prompt if prompt else None,
                "block_hit_rate":
                c["hits"] / c["lookups"] if c["lookups"] else None,
            }

        total: Counter = Counter()
        for c in per.values():
            total.update(c)
        return {
            **fractions(total),
            "hit_rate_metric":
            self.gpu.get_prefix_cache_hit_rate(),
            **dict(self.counts),
            # Belady reads the future: a bound, not a deployable policy.
            **({
                "offline_oracle": True
            } if self.belady is not None else {}),
            # TTDF's and RACE-RP's own counters (scans, metadata size).
            **({
                "evictor": self.gpu.evictor.summary()
            } if hasattr(self.gpu.evictor, "summary") else {}),
            "workloads": {w: fractions(c)
                          for w, c in sorted(per.items())},
        }


def window_summary(
    rows: Sequence[Dict[str, Any]],
    requests: Sequence[TokenRequest],
    w0: float,
    stop: float,
    evictions: Sequence[Tuple[float, Optional[str]]] = ()
) -> Dict[str, Any]:
    """Prompt and usable tokens of the requests planned in [w0, stop),
    overall and per workload, as the study's engine numbers count them
    (``client_simulator.plan_replay.planned_s``), and the evictions in
    [w0, stop) on the simulated clock, by the workload of the evicted
    block's last request (``evictions``, total and ``workloads``)."""
    planned = {r.request_id: planned_s(r) for r in requests}
    per: Dict[str, Counter] = {}
    for row in rows:
        t = planned.get(row["request_id"])
        if t is None or not w0 <= t < stop:
            continue
        c = per.setdefault(row["workload"], Counter())
        c["requests"] += 1
        c["prompt_tokens"] += row["prompt_tokens"]
        c["usable_tokens"] += row["usable_cached_tokens"]

    def fractions(c: Counter) -> Dict[str, Any]:
        prompt = c["prompt_tokens"]
        return {
            **dict(c), "usable_reused_fraction":
            c["usable_tokens"] / prompt if prompt else None
        }

    evicted: Counter = Counter(w for t, w in evictions if w0 <= t < stop)
    for w, n in evicted.items():
        if w is not None:
            per.setdefault(w, Counter())["evictions"] = n
    total: Counter = Counter()
    for c in per.values():
        total.update(c)
    if evicted:
        total["evictions"] = sum(evicted.values())
    return {
        "w0_s": w0,
        "stop_s": stop,
        **fractions(total),
        "workloads": {w: fractions(c)
                      for w, c in sorted(per.items())},
    }


def simulate(plan_dir: Path,
             policy: str,
             capacity_tokens: int,
             block_size: int = 16,
             timing: str = "model",
             env: Optional[Dict[str, str]] = None,
             out: Optional[Path] = None,
             window: Optional[Tuple[float, float]] = None) -> SimResult:
    """Replay `plan_dir`'s plan once; with `out`, also write a run folder
    that `analysis/run_m2_task2.analyze_run` reads. With `window`
    (w0, stop), the summary's ``window`` holds the same counts over the
    requests planned in [w0, stop) (`window_summary`)."""
    requests = list(read_plan(plan_dir / "plan.jsonl.gz"))
    sessions = group_sessions(requests)
    if timing == "model":
        times = model_timing(sessions)
    elif timing.startswith("engine:"):
        times = engine_timing(Path(timing[len("engine:"):]), requests)
    else:
        raise ValueError(f"Unknown timing {timing!r}")
    event_log = None
    if out is not None:
        out.mkdir(parents=True)
        event_log = out / "events.jsonl.gz"
    sim = CacheSim(capacity_tokens // block_size, block_size, policy, env,
                   event_log)
    result = sim.run(requests, times)
    if window is not None:
        result.summary["window"] = window_summary(result.rows, requests,
                                                  *window, result.evictions)
    if out is not None:
        with (out / "client.jsonl").open("w") as f:
            for row in result.rows:
                f.write(json.dumps(row) + "\n")
        (out / "client.jsonl.metrics.after.txt").write_text(
            "vllm:gpu_prefix_cache_hit_rate{model_name=\"sim\"} "
            f"{result.summary['hit_rate_metric']}\n")
        (out / "sim.json").write_text(
            json.dumps(
                {
                    "plan_dir": str(plan_dir),
                    "policy": policy,
                    "capacity_tokens": capacity_tokens,
                    "block_size": block_size,
                    "timing": timing,
                    "env": env or {},
                    "summary": result.summary,
                },
                indent=2))
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--policy",
                        default="lru",
                        help="lru, lfu, fifo, ttdf or race_rp (vLLM's "
                        "evictors; settings after a colon, e.g. "
                        "ttdf:tau_short=2), or belady (the offline optimum; "
                        "simulator only)")
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--timing",
                        default="model",
                        help="model, or engine:<run dir>")
    parser.add_argument("--env",
                        nargs="*",
                        default=[],
                        help="Extra VAR=value settings for the policy")
    parser.add_argument("--out",
                        type=Path,
                        help="Write a run folder (event log, client rows)")
    args = parser.parse_args(argv)
    env = dict(item.split("=", 1) for item in args.env)
    result = simulate(args.plan, args.policy, args.capacity, args.block_size,
                      args.timing, env, args.out)
    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
