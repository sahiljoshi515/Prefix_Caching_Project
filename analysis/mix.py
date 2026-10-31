"""Milestone 3 Task 2: build a mixed-workload replay plan (spec 007 FR-001).

Combines whole sessions of several workloads on one timeline::

    python -m analysis.mix --mix balanced --seed 1 \\
        --out data/processed/m3/balanced-s1

writes ``plan.jsonl.gz`` (token-level requests, see
``client_simulator/token_plan.py``), ``responses.jsonl`` (the simulator
trace) and ``manifest.json`` into ``--out``.

Per workload, the builder targets ``share x rate x duration`` requests:

- **Text workloads** (``sharegpt``, ``agentbank``, ``ccbench``): seeded
  sessions from the processed conversations, tokenized with the model's chat
  template. Turns are closed loop: a turn is sent the workload's think time
  after the previous one completes. Session starts are a Poisson process
  over the window (uniform order statistics of the session count).
  CC-Bench keeps at most one model's run per task.
- **Bailian traces**: one seeded window of ``duration`` seconds per trace.
  Sessions whose root arrives in it keep their recorded times (open loop),
  shifted to the window's start, and are subsampled to the target.

Sessions are never split. A session is taken while it brings the workload's
count closer to its target. Turns that exceed ``--max-model-len`` are
dropped with the rest of their session (Bailian: only that request). The
manifest records each workload's request and prompt-token shares, its
pacing, and what was dropped.
"""
import argparse
import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from client_simulator.adapters import bailian
from client_simulator.schema import Conversation, Message, TokenRequest
from client_simulator.token_plan import tokenize_conversation, write_plan

TEXT_WORKLOADS = ("sharegpt", "agentbank", "ccbench")
WORKLOADS = TEXT_WORKLOADS + tuple(bailian.TRACES)
# Closed-loop pause between a session's turns: a person reading and typing
# (ShareGPT, as in Milestone 2), an environment step (AgentBank), a tool run
# (CC-Bench). Assumptions of ours, recorded in every manifest; spec 007
# User Story 4 varies them.
THINK_S = {"sharegpt": 5.0, "agentbank": 1.0, "ccbench": 2.0}
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
    duration_s: float = 1200.0
    rate: float = 3.0
    think_s: Dict[str, float] = field(default_factory=lambda: dict(THINK_S))
    max_model_len: int = 32768
    # Text sessions considered per workload, before the seeded shuffle: the
    # first N conversations of the processed file (None: all of them).
    # ShareGPT's file is 572 MB, so it is capped; AgentBank's is ordered by
    # task, so it must not be.
    pool: Dict[str, Optional[int]] = field(
        default_factory=lambda: {"sharegpt": 20000})

    def target(self, workload: str) -> int:
        return round(self.shares[workload] * self.rate * self.duration_s)


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
                                 [Message(**m) for m in row["messages"]]))
    return out


def ccbench_task(conversation_id: str) -> str:
    """``ccbench:<task>-<model>`` -> ``<task>``."""
    return conversation_id.split(":", 1)[1].split("-", 1)[0]


def _closer(count: int, size: int, target: int) -> bool:
    """Whether adding a session of `size` requests brings `count` closer to
    `target` (ties add)."""
    return abs(count + size - target) <= abs(count - target)


def sample_text(workload: str, spec: MixSpec, rng: random.Random,
                conversations: List[Conversation],
                tokenizer) -> Tuple[List[List[TokenRequest]], Dict[str, Any]]:
    pool = list(conversations)
    rng.shuffle(pool)
    target = spec.target(workload)
    sessions: List[List[TokenRequest]] = []
    tasks_used = set()
    count = skipped_turns = skipped_sessions = 0
    for conversation in pool:
        if count >= target:
            break
        if workload == "ccbench":
            task = ccbench_task(conversation.conversation_id)
            if task in tasks_used:
                continue
        # The turn count bounds the requests, so decide before tokenizing.
        if not _closer(count, len(conversation.messages) // 2, target):
            continue
        requests, skipped = tokenize_conversation(conversation, tokenizer,
                                                  spec.max_model_len,
                                                  spec.think_s[workload])
        skipped_turns += skipped
        if not requests:
            skipped_sessions += 1
            continue
        sessions.append(requests)
        count += len(requests)
        if workload == "ccbench":
            tasks_used.add(task)
    starts = sorted(rng.uniform(0, spec.duration_s) for _ in sessions)
    for session, start in zip(sessions, starts):
        session[0].arrival_s = start
    info = {
        "arrivals": "closed loop, Poisson session starts",
        "think_s": spec.think_s[workload],
        "pool_conversations": len(pool),
        "skipped_turns_context": skipped_turns,
        "skipped_sessions_context": skipped_sessions,
    }
    return sessions, info


def sample_bailian(
    workload: str, spec: MixSpec, rng: random.Random,
    trace_sessions: List[List[dict]]
) -> Tuple[List[List[TokenRequest]], Dict[str, Any]]:
    end = max(s[-1]["timestamp"] for s in trace_sessions)
    # Rounded once, so the manifest's start is the one the times use.
    start = round(rng.uniform(0, max(0.0, end - spec.duration_s)), 3)
    window = [
        s for s in trace_sessions
        if start <= s[0]["timestamp"] < start + spec.duration_s
    ]
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
    sessions, dropped = [], 0
    for records in chosen:
        kept = []
        for request in bailian.synthesize(workload, records):
            request.arrival_s = round(request.arrival_s - start, 6)
            if (len(request.prompt_token_ids) + len(request.output_token_ids) +
                    1 > spec.max_model_len):
                dropped += 1
                continue
            kept.append(request)
        if kept:
            sessions.append(kept)
    info = {
        "arrivals":
        "open loop, recorded timestamps",
        "window_start_s":
        start,
        "window_sessions":
        len(window),
        "window_requests":
        sum(len(s) for s in window),
        "kept_session_fraction":
        round(len(chosen) / len(window), 4) if window else None,
        "skipped_requests_context":
        dropped,
    }
    return sessions, info


def build(
    spec: MixSpec,
    seed: int,
    conversations: Dict[str, List[Conversation]],
    trace_sessions: Dict[str, List[List[dict]]],
    tokenizer,
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
            sessions, info = sample_text(workload, spec, rng,
                                         conversations[workload], tokenizer)
        else:
            sessions, info = sample_bailian(workload, spec, rng,
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
    manifest = {
        "mix": spec.name,
        "seed": seed,
        "shares": spec.shares,
        "duration_s": spec.duration_s,
        "rate": spec.rate,
        "max_model_len": spec.max_model_len,
        "think_s":
        {w: spec.think_s[w]
         for w in spec.shares if w in TEXT_WORKLOADS},
        "requests": total_requests,
        "sessions": len(all_sessions),
        "prompt_tokens": total_tokens,
        "workloads": per_workload,
    }
    return requests, manifest


def load_inputs(
    spec: MixSpec, data: Path
) -> Tuple[Dict[str, List[Conversation]], Dict[str, List[List[dict]]]]:
    conversations, traces = {}, {}
    for workload in spec.shares:
        if workload in TEXT_WORKLOADS:
            conversations[workload] = read_conversations(
                data / "processed" / f"{workload}.conversations.jsonl",
                spec.pool.get(workload))
        elif workload in bailian.TRACES:
            records = bailian.load_records(data / "raw" / "bailian" /
                                           bailian.TRACES[workload])
            traces[workload] = bailian.sessions(records)
    return conversations, traces


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
    parser.add_argument("--duration", type=float, default=1200.0)
    parser.add_argument("--rate",
                        type=float,
                        default=3.0,
                        help="Target requests per second over the window")
    parser.add_argument("--think-scale",
                        type=float,
                        default=1.0,
                        help="Multiply every think time (spec 007 US4.3)")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    return parser.parse_args(argv)


def mix_spec(args: argparse.Namespace) -> MixSpec:
    if args.mix in MIXES:
        name, shares = args.mix, MIXES[args.mix]
    else:
        path = Path(args.mix)
        name, shares = path.stem, json.loads(path.read_text())
    total = sum(shares.values())
    if total <= 0 or any(v < 0 for v in shares.values()):
        raise SystemExit("Shares must be non-negative and sum above 0")
    return MixSpec(
        name=name,
        shares={w: v / total
                for w, v in shares.items()},
        duration_s=args.duration,
        rate=args.rate,
        think_s={w: s * args.think_scale
                 for w, s in THINK_S.items()},
        max_model_len=args.max_model_len)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    spec = mix_spec(args)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conversations, traces = load_inputs(spec, args.data)
    requests, manifest = build(spec, args.seed, conversations, traces,
                               tokenizer)
    manifest["model"] = args.model
    args.out.mkdir(parents=True, exist_ok=True)
    write_plan(requests, args.out / "plan.jsonl.gz",
               args.out / "responses.jsonl")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(requests)} requests in {manifest['sessions']} "
          f"sessions to {args.out}")
    for workload, stats in sorted(manifest["workloads"].items()):
        print(f"  {workload:17s} {stats['requests']:5d} requests "
              f"({stats['request_share']:.2f} of requests, "
              f"{stats['prompt_token_share']:.2f} of prompt tokens)")


if __name__ == "__main__":
    main()
