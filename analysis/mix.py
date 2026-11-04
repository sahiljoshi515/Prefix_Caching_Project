"""Milestone 3 Task 2: build a mixed-workload replay plan (spec 007 FR-001).

Combines whole sessions of several workloads on one timeline::

    python -m analysis.mix --mix balanced --pacing human --seed 2 \\
        --out data/processed/m3/balanced-human-s2

writes ``plan.jsonl.gz`` (token-level requests, see
``client_simulator/token_plan.py``), ``responses.jsonl`` (the simulator
trace) and ``manifest.json`` into ``--out``.

The plan covers ``--duration`` seconds (default 1,800) and stops there:
sessions start throughout the window, and no turn is planned at or after
``stop_s = duration + tail`` (``--tail``, default 0). So every policy replays
the same requests, the mix is stationary until the stop, and the analysis
can skip a warm-up. Per workload, the builder targets
``share x rate x duration`` requests, counting only the turns it keeps:

- **Text workloads** (``sharegpt``, ``agentbank``, ``ccbench``): seeded
  sessions from the processed conversations, tokenized with the model's chat
  template. Each session starts at a uniform time in the window (a Poisson
  process, given the count). Turns are closed loop: a turn is sent its pause
  after the previous one completes. A turn whose zero-latency send time (the
  start plus the pauses before it, ``meta["planned_s"]``) reaches ``stop_s``
  is dropped with the rest of its session. CC-Bench keeps at most one
  model's run per task.
- **Bailian traces**: one window of ``duration`` seconds per trace. With
  ``--bailian-window partition`` (default) seed k takes the trace's k-th
  window (modulo the windows that fit), so seeds 1 to 4 replay disjoint
  quarters of a two-hour trace; ``random`` takes a seeded random window.
  Sessions whose root arrives in the window keep their recorded times (open
  loop), shifted to the window's start, and are subsampled to the target;
  their turns at or after ``stop_s`` are dropped.

Pacing (``--pacing``) sets the closed-loop pauses; the Bailian traces keep
their recorded times either way:

- ``human`` (default): a workload without timestamps borrows real pauses
  from the closest recorded loop of its kind.

  - ShareGPT (people chatting) samples the bailian-chat trace's text
    requests: the time from the parent's arrival to the child's, less
    ``CHAT_DECODE_S`` per token of the parent's answer (production's answer
    time, which the replay produces itself), at least ``CHAT_GAP_FLOOR_S``,
    drawn from the same turn number (turns 6 and later pooled).
  - CC-Bench (an agent waiting for its tools) uses each trajectory's own
    recorded tool latencies (``Conversation.turn_gaps_s``).
  - AgentBank (an agent waiting for its environment) samples CC-Bench's
    tool latencies, pooled.

  Draws are seeded per session.
- ``fast``: the constant pauses of spec 007's Phase 0 (``THINK_S``).

ShareGPT's and AgentBank's pauses are therefore inputs, not measurements.
``--think-scale`` multiplies every closed-loop pause (spec 007 US4.3).
Session choice and start times do not depend on the pacing, so the pacings
of one seed share their first sessions. Turns that exceed
``--max-model-len`` are dropped with the rest of their session (Bailian:
only that request). The manifest records each workload's request and
prompt-token shares, its pacing and pause quantiles, what was dropped and
why, the thinning of each Bailian trace, and every input file's sha256.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from client_simulator.adapters import bailian
from client_simulator.schema import Conversation, Message, TokenRequest
from client_simulator.token_plan import tokenize_conversation, write_plan

TEXT_WORKLOADS = ("sharegpt", "agentbank", "ccbench")
WORKLOADS = TEXT_WORKLOADS + tuple(bailian.TRACES)
PACINGS = ("human", "fast")
# Phase 0's closed-loop pauses, kept as the "fast" pacing: a person reading
# and typing (ShareGPT, as in Milestone 2), an environment step (AgentBank),
# a tool run (CC-Bench). Assumptions, recorded in every manifest.
THINK_S = {"sharegpt": 5.0, "agentbank": 1.0, "ccbench": 2.0}
# Human pacing's calibration (module docstring).
CHAT_GAP_TRACE = "bailian-chat"
CHAT_GAP_TYPES = ("text", )
CHAT_DECODE_S = 0.03
CHAT_GAP_FLOOR_S = 1.0
CHAT_GAP_LAST_TURN = 6
MIXES: Dict[str, Dict[str, float]] = {
    "balanced": {w: 1 / len(WORKLOADS)
                 for w in WORKLOADS},
    "chat-heavy": {
        "sharegpt": 0.30,
        "bailian-chat": 0.25,
        "bailian-thinking": 0.10,
        "agentbank": 0.10,
        "bailian-api": 0.10,
        "bailian-coder": 0.10,
        "ccbench": 0.05,
    },
    "agent-heavy": {
        "agentbank": 0.25,
        "bailian-coder": 0.20,
        "ccbench": 0.15,
        "bailian-api": 0.15,
        "sharegpt": 0.10,
        "bailian-chat": 0.10,
        "bailian-thinking": 0.05,
    },
    **{f"only-{w}": {
        w: 1.0
    }
       for w in WORKLOADS},
}


@dataclass
class MixSpec:
    name: str
    shares: Dict[str, float]
    duration_s: float = 1800.0
    rate: float = 3.0
    max_model_len: int = 32768
    # Text sessions considered per workload, before the seeded shuffle: the
    # first N conversations of the processed file (None: all of them).
    # ShareGPT's file is 572 MB, so it is capped; AgentBank's is ordered by
    # task, so it must not be.
    pool: Dict[str, Optional[int]] = field(
        default_factory=lambda: {"sharegpt": 20000})
    # No turn is planned at or after the window's end plus this.
    tail_s: float = 0.0
    # "partition": seed k replays each Bailian trace's k-th window;
    # "random": a seeded random window.
    bailian_window: str = "partition"

    @property
    def stop_s(self) -> float:
        """No request is planned at or after this, from the plan's start."""
        return self.duration_s + self.tail_s

    def target(self, workload: str) -> int:
        return round(self.shares[workload] * self.rate * self.duration_s)


@dataclass
class Pacer:
    """The closed-loop pause before each turn of a text session."""
    kind: str = "human"
    think_scale: float = 1.0
    # Human pacing's sources: bailian-chat's pauses by turn (2 to
    # CHAT_GAP_LAST_TURN), and CC-Bench's recorded tool latencies, pooled.
    chat_gaps: Dict[int, List[float]] = field(default_factory=dict)
    tool_gaps: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.kind not in PACINGS:
            raise ValueError(f"Unknown pacing {self.kind!r}; use one of "
                             f"{', '.join(PACINGS)}")

    def source(self, workload: str) -> str:
        if self.kind == "fast":
            return f"constant {THINK_S[workload]} s (Phase 0)"
        return {
            "sharegpt": "bailian-chat text turn gaps, by turn (human)",
            "agentbank": "CC-Bench tool latencies, pooled (agent)",
            "ccbench": "own recorded tool latencies (agent)",
        }[workload]

    def pauses(self, workload: str, conversation: Conversation,
               rng: random.Random) -> List[Optional[float]]:
        """Entry k is the pause before turn k; entry 0 is None."""
        turns = len(conversation.messages) // 2
        recorded = conversation.turn_gaps_s or []
        out: List[Optional[float]] = [None]
        for turn in range(1, turns):
            if self.kind == "fast":
                pause = THINK_S[workload]
            elif workload == "sharegpt":
                # A turn index k (from 0) is the trace's turn k + 1.
                pause = rng.choice(self._chat_pool(turn + 1))
            elif (workload == "ccbench" and turn < len(recorded)
                  and recorded[turn] is not None):
                pause = recorded[turn]
            else:  # AgentBank, or a CC-Bench gap without timestamps.
                pause = rng.choice(self._tool_pool())
            out.append(round(pause * self.think_scale, 3))
        return out

    def _chat_pool(self, trace_turn: int) -> List[float]:
        pool = self.chat_gaps.get(min(trace_turn, CHAT_GAP_LAST_TURN))
        if not pool:
            raise ValueError("Human pacing needs bailian-chat's turn gaps "
                             f"(none for turn {trace_turn})")
        return pool

    def _tool_pool(self) -> List[float]:
        if not self.tool_gaps:
            raise ValueError(
                "Human pacing needs CC-Bench's recorded tool latencies; "
                "reprocess it with `python -m "
                "client_simulator.process_raw_traces --dataset ccbench`")
        return self.tool_gaps


def chat_gap_pool(records: Sequence[dict]) -> Dict[int, List[float]]:
    """bailian-chat's human pauses by turn number (module docstring)."""
    by_id = {r["chat_id"]: r for r in records}
    pools: Dict[int, List[float]] = {}
    for r in records:
        parent = by_id.get(r["parent_chat_id"])
        if parent is None or parent is r or r["type"] not in CHAT_GAP_TYPES:
            continue
        gap = (r["timestamp"] - parent["timestamp"] -
               CHAT_DECODE_S * parent["output_length"])
        pools.setdefault(min(r["turn"], CHAT_GAP_LAST_TURN),
                         []).append(round(max(gap, CHAT_GAP_FLOOR_S), 3))
    return pools


def tool_gap_pool(conversations: Sequence[Conversation]) -> List[float]:
    """Every recorded pause between CC-Bench turns, in file order."""
    return [
        g for c in conversations for g in (c.turn_gaps_s or [])[1:]
        if g is not None
    ]


def planned_times(start: float,
                  pauses: Sequence[Optional[float]]) -> List[float]:
    """Each turn's send time if requests took no time: the start plus the
    pauses before it."""
    times = [start]
    for pause in pauses[1:]:
        times.append(round(times[-1] + pause, 6))
    return times


def quantiles(values: Sequence[float]) -> Dict[str, Any]:
    """Nearest-rank summary for the manifest (empty for no values)."""
    if not values:
        return {}
    ordered = sorted(values)

    def rank(p: float) -> float:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        **{f"p{round(p * 100)}": rank(p)
           for p in (0.1, 0.5, 0.9, 0.99)},
    }


def read_conversations(path: Path,
                       limit: Optional[int] = None) -> List[Conversation]:
    out: List[Conversation] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if limit is not None and len(out) >= limit:
                break
            if line.strip():
                row = json.loads(line)
                out.append(
                    Conversation(row["conversation_id"],
                                 [Message(**m) for m in row["messages"]],
                                 row.get("turn_gaps_s")))
    return out


def ccbench_task(conversation_id: str) -> str:
    """``ccbench:<task>-<model>`` -> ``<task>``."""
    return conversation_id.split(":", 1)[1].split("-", 1)[0]


def _closer(count: int, size: int, target: int) -> bool:
    """Whether adding a session of `size` requests brings `count` closer to
    `target` (ties add)."""
    return abs(count + size - target) <= abs(count - target)


def sample_text(
        workload: str, spec: MixSpec, seed: int, rng: random.Random,
        conversations: List[Conversation], tokenizer,
        pacer: Pacer) -> Tuple[List[List[TokenRequest]], Dict[str, Any]]:
    pool = list(conversations)
    rng.shuffle(pool)
    target = spec.target(workload)
    sessions: List[List[TokenRequest]] = []
    tasks_used = set()
    count = 0
    dropped: Counter = Counter()
    pauses_kept: List[float] = []
    for conversation in pool:
        if count >= target:
            break
        # Drawn for every conversation looked at, before any choice that
        # depends on the pacing, so the pacings of a seed share starts.
        start = round(rng.uniform(0, spec.duration_s), 6)
        if workload == "ccbench":
            task = ccbench_task(conversation.conversation_id)
            if task in tasks_used:
                continue
        pause_rng = random.Random(
            f"{seed}:{workload}:{conversation.conversation_id}:pause")
        pauses = pacer.pauses(workload, conversation, pause_rng)
        planned = planned_times(start, pauses)
        # Planned times only grow, so the kept turns are a prefix.
        kept = bisect_left(planned, spec.stop_s)
        if not _closer(count, kept, target):
            continue
        truncated = Conversation(conversation.conversation_id,
                                 conversation.messages[:2 * kept],
                                 conversation.turn_gaps_s)
        requests, skipped = tokenize_conversation(truncated, tokenizer,
                                                  spec.max_model_len, pauses)
        dropped["skipped_turns_context"] += skipped
        if not requests:
            dropped["skipped_sessions_context"] += 1
            continue
        dropped["turns_after_stop"] += len(planned) - kept
        requests[0].arrival_s = start
        for request in requests:
            request.meta["planned_s"] = planned[request.turn_index]
        pauses_kept += [r.after_s for r in requests[1:]]
        sessions.append(requests)
        count += len(requests)
        if workload == "ccbench":
            tasks_used.add(task)
    sessions.sort(key=lambda s: (s[0].arrival_s, s[0].session_id))
    info = {
        "arrivals":
        "closed loop, Poisson session starts",
        "pacing":
        pacer.source(workload),
        "think_s": (THINK_S[workload] *
                    pacer.think_scale if pacer.kind == "fast" else None),
        "pause_s":
        quantiles(pauses_kept),
        "pool_conversations":
        len(pool),
        "turns_after_stop":
        dropped["turns_after_stop"],
        "skipped_turns_context":
        dropped["skipped_turns_context"],
        "skipped_sessions_context":
        dropped["skipped_sessions_context"],
    }
    return sessions, info


def bailian_window(spec: MixSpec, rng: random.Random,
                   trace_sessions: List[List[dict]],
                   index: int) -> Tuple[float, Dict[str, Any]]:
    """The trace window a seed replays: its start and how it was chosen."""
    first = min(s[0]["timestamp"] for s in trace_sessions)
    end = max(s[-1]["timestamp"] for s in trace_sessions)
    if spec.bailian_window == "partition":
        # The 1% slack lets a 7,199 s trace hold four 1,800 s windows.
        windows = max(
            1, int((end - first + 0.01 * spec.duration_s) // spec.duration_s))
        index %= windows
        start = round(first + index * spec.duration_s, 3)
        return start, {
            "window": "partition",
            "window_index": index,
            "windows": windows
        }
    if spec.bailian_window == "random":
        start = round(rng.uniform(first, max(first, end - spec.duration_s)), 3)
        return start, {"window": "random"}
    raise ValueError(f"Unknown Bailian window {spec.bailian_window!r}")


def sample_bailian(
    workload: str, spec: MixSpec, seed: int, rng: random.Random,
    trace_sessions: List[List[dict]]
) -> Tuple[List[List[TokenRequest]], Dict[str, Any]]:
    start, window_info = bailian_window(spec, rng, trace_sessions, seed - 1)
    window, late = [], 0
    for session in trace_sessions:
        if not start <= session[0]["timestamp"] < start + spec.duration_s:
            continue
        # Time order, so cutting the late turns keeps every parent.
        kept = [r for r in session if r["timestamp"] < start + spec.stop_s]
        late += len(session) - len(kept)
        window.append(kept)
    order = list(range(len(window)))
    rng.shuffle(order)
    target = spec.target(workload)
    chosen: List[List[dict]] = []
    count = 0
    for index in order:
        if count >= target:
            break
        if _closer(count, len(window[index]), target):
            chosen.append(window[index])
            count += len(window[index])
    chosen.sort(key=lambda s: (s[0]["timestamp"], s[0]["chat_id"]))
    sessions, dropped, emptied = [], 0, 0
    for records in chosen:
        kept = []
        for request in bailian.synthesize(workload, records):
            request.arrival_s = round(request.arrival_s - start, 6)
            request.meta["planned_s"] = request.arrival_s
            if (len(request.prompt_token_ids) + len(request.output_token_ids) +
                    1 > spec.max_model_len):
                dropped += 1
                continue
            kept.append(request)
        if kept:
            sessions.append(kept)
        else:
            emptied += 1
    window_requests = sum(len(s) for s in window)
    info = {
        "arrivals":
        "open loop, recorded timestamps",
        **window_info,
        "window_start_s":
        start,
        "window_sessions":
        len(window),
        "window_requests":
        window_requests,
        # The thinning: cross-session reuse in the replay is about this
        # fraction as dense as in the trace.
        "kept_session_fraction":
        round(len(chosen) / len(window), 4) if window else None,
        "kept_request_fraction":
        round(count / window_requests, 4) if window_requests else None,
        "stop_s":
        spec.stop_s,
        "window_turns_after_stop":
        late,
        # Over the chosen sessions; a session left with no request
        # (all its requests too long) is also counted on its own.
        "skipped_requests_context":
        dropped,
        "skipped_sessions_context":
        emptied,
    }
    return sessions, info


def build(
    spec: MixSpec,
    seed: int,
    conversations: Dict[str, List[Conversation]],
    trace_sessions: Dict[str, List[List[dict]]],
    tokenizer,
    *,
    pacer: Pacer,
) -> Tuple[List[TokenRequest], Dict[str, Any]]:
    """The plan's requests (sessions in start order) and its manifest."""
    all_sessions: List[List[TokenRequest]] = []
    per_workload: Dict[str, Any] = {}
    for workload in sorted(spec.shares):
        if workload not in WORKLOADS:
            raise ValueError(f"Unknown workload {workload!r}; use one of "
                             f"{', '.join(WORKLOADS)}")
        rng = random.Random(f"{seed}:{workload}")
        if workload in TEXT_WORKLOADS:
            sessions, info = sample_text(workload, spec, seed, rng,
                                         conversations[workload], tokenizer,
                                         pacer)
        else:
            sessions, info = sample_bailian(workload, spec, seed, rng,
                                            trace_sessions[workload])
        requests = [r for s in sessions for r in s]
        prompts = [len(r.prompt_token_ids) for r in requests]
        per_workload[workload] = {
            "target_requests":
            spec.target(workload),
            "sessions":
            len(sessions),
            "requests":
            len(requests),
            "prompt_tokens":
            sum(prompts),
            "output_tokens":
            sum(len(r.output_token_ids) for r in requests),
            "prompt_tokens_p50":
            statistics.median(prompts) if prompts else 0,
            "turns_per_session_mean":
            (len(requests) / len(sessions) if sessions else 0),
            **info,
        }
        all_sessions += sessions
    all_sessions.sort(key=lambda s: (s[0].arrival_s, s[0].session_id))
    requests = [r for s in all_sessions for r in s]
    total_requests = len(requests)
    total_tokens = sum(len(r.prompt_token_ids) for r in requests)
    for stats in per_workload.values():
        stats["request_share"] = (stats["requests"] /
                                  total_requests if total_requests else 0)
        stats["prompt_token_share"] = (stats["prompt_tokens"] /
                                       total_tokens if total_tokens else 0)
    pacing: Dict[str, Any] = {
        "kind": pacer.kind,
        "think_scale": pacer.think_scale
    }
    if pacer.kind == "fast":
        pacing["think_s"] = dict(THINK_S)
    else:
        pacing.update(chat_gap_trace=CHAT_GAP_TRACE,
                      chat_gap_types=list(CHAT_GAP_TYPES),
                      chat_decode_s=CHAT_DECODE_S,
                      chat_gap_floor_s=CHAT_GAP_FLOOR_S,
                      chat_gap_last_turn=CHAT_GAP_LAST_TURN,
                      chat_gap_pool={
                          str(k): len(v)
                          for k, v in sorted(pacer.chat_gaps.items())
                      },
                      tool_gap_pool=len(pacer.tool_gaps))
    manifest = {
        "mix": spec.name,
        "seed": seed,
        "shares": spec.shares,
        "duration_s": spec.duration_s,
        "tail_s": spec.tail_s,
        "stop_s": spec.stop_s,
        "rate": spec.rate,
        "max_model_len": spec.max_model_len,
        "bailian_window": spec.bailian_window,
        "pacing": pacing,
        "requests": total_requests,
        "sessions": len(all_sessions),
        "prompt_tokens": total_tokens,
        "workloads": per_workload,
    }
    return requests, manifest


def load_inputs(
    spec: MixSpec,
    data: Path,
    pacing: str = "human",
    think_scale: float = 1.0
) -> Tuple[Dict[str, List[Conversation]], Dict[str, List[List[dict]]], Pacer,
           List[Path]]:
    """The mix's conversations and trace sessions, its pacer, and every file
    read (for the manifest's checksums)."""
    need_chat = pacing == "human" and "sharegpt" in spec.shares
    need_tools = pacing == "human" and ("agentbank" in spec.shares
                                        or "ccbench" in spec.shares)
    conversations: Dict[str, List[Conversation]] = {}
    traces: Dict[str, List[List[dict]]] = {}
    inputs: List[Path] = []
    for workload in TEXT_WORKLOADS:
        if workload in spec.shares or (workload == "ccbench" and need_tools):
            path = data / "processed" / f"{workload}.conversations.jsonl"
            conversations[workload] = read_conversations(
                path, spec.pool.get(workload))
            inputs.append(path)
    chat_records: List[dict] = []
    for workload, name in bailian.TRACES.items():
        if workload in spec.shares or (workload == CHAT_GAP_TRACE
                                       and need_chat):
            path = data / "raw" / "bailian" / name
            records = bailian.load_records(path)
            inputs.append(path)
            if workload in spec.shares:
                traces[workload] = bailian.sessions(records)
            if workload == CHAT_GAP_TRACE:
                chat_records = records
    pacer = Pacer(pacing, think_scale)
    if need_chat:
        pacer.chat_gaps = chat_gap_pool(chat_records)
    if need_tools:
        pacer.tool_gaps = tool_gap_pool(conversations["ccbench"])
    if "ccbench" not in spec.shares:
        # Loaded only for AgentBank's pauses; not a workload of this mix.
        conversations.pop("ccbench", None)
    return conversations, traces, pacer, inputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mix",
                        required=True,
                        help=f"One of {', '.join(MIXES)}, or a JSON file "
                        "of {workload: share}")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--rate",
                        type=float,
                        default=3.0,
                        help="Target requests per second over the window")
    parser.add_argument("--tail",
                        type=float,
                        default=0.0,
                        help="Plan turns up to this many seconds after the "
                        "window (default 0: the plan stops with the window)")
    parser.add_argument("--pacing",
                        choices=PACINGS,
                        default="human",
                        help="Closed-loop pauses: borrowed from recorded "
                        "loops (human) or Phase 0's constants (fast)")
    parser.add_argument("--think-scale",
                        type=float,
                        default=1.0,
                        help="Multiply every closed-loop pause "
                        "(spec 007 US4.3)")
    parser.add_argument("--bailian-window",
                        choices=("partition", "random"),
                        default="partition")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.tail < 0 or args.think_scale <= 0:
        parser.error("--duration and --think-scale must be positive, and "
                     "--tail not negative")
    return args


def mix_spec(args: argparse.Namespace) -> MixSpec:
    if args.mix in MIXES:
        name, shares = args.mix, MIXES[args.mix]
    else:
        path = Path(args.mix)
        name, shares = path.stem, json.loads(path.read_text())
    total = sum(shares.values())
    if total <= 0 or any(v < 0 for v in shares.values()):
        raise SystemExit("Shares must be non-negative and sum above 0")
    return MixSpec(name=name,
                   shares={w: v / total
                           for w, v in shares.items()},
                   duration_s=args.duration,
                   rate=args.rate,
                   max_model_len=args.max_model_len,
                   tail_s=args.tail,
                   bailian_window=args.bailian_window)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    spec = mix_spec(args)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conversations, traces, pacer, inputs = load_inputs(spec, args.data,
                                                       args.pacing,
                                                       args.think_scale)
    requests, manifest = build(spec,
                               args.seed,
                               conversations,
                               traces,
                               tokenizer,
                               pacer=pacer)
    from analysis.run_m2_task2 import git_state
    manifest.update(model=args.model,
                    inputs={str(p): sha256(p)
                            for p in inputs},
                    **git_state())
    args.out.mkdir(parents=True, exist_ok=True)
    write_plan(requests, args.out / "plan.jsonl.gz",
               args.out / "responses.jsonl")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(requests)} requests in {manifest['sessions']} "
          f"sessions to {args.out} ({pacer.kind} pacing)")
    for workload, stats in sorted(manifest["workloads"].items()):
        extra = ""
        if "kept_session_fraction" in stats:
            extra = f", {stats['kept_session_fraction']:.3f} of sessions kept"
        elif stats.get("pause_s"):
            extra = f", pause p50 {stats['pause_s']['p50']} s"
        print(f"  {workload:17s} {stats['requests']:5d} requests "
              f"({stats['request_share']:.2f} of requests, "
              f"{stats['prompt_token_share']:.2f} of prompt tokens{extra})")


if __name__ == "__main__":
    main()
