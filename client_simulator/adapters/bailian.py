"""Qwen-Bailian anonymized KV-cache traces as token-level requests.

Source: GitHub ``alibaba-edu/qwen-bailian-usagetraces-anon`` (Git LFS), four
two-hour traces from a production Qwen cluster, each its own workload:

========================  ==============================  ===================
workload                  file                            scenario
========================  ==============================  ===================
``bailian-chat``          ``qwen_traceA_blksz_16.jsonl``  to-C chat
``bailian-api``           ``qwen_traceB_blksz_16.jsonl``  to-B API automation
``bailian-thinking``      ``qwen_thinking_blksz_16.jsonl``  reasoning chat
``bailian-coder``         ``qwen_coder_blksz_16.jsonl``   code generation
========================  ==============================  ===================

A record is ``{chat_id, parent_chat_id (-1 for a root), timestamp (s),
input_length, output_length, type, turn, hash_ids}``; ``hash_ids`` are
salted hashes of the prompt's 16-token blocks after the chat template, the
last one possibly over a partly filled block.

Synthesis rules (spec 006 FR-005):

- **Blocks.** Each ``(workload, hash id)`` maps to 16 fixed tokens drawn
  from ordinary Qwen2.5 tokens (ids 1,000 to 149,999, below every special
  token) by a keyed hash. Equal hash-id prefixes become equal token
  prefixes, so vLLM's prefix cache sees the trace's sharing. The workload is
  part of the key because the traces were anonymized separately: the same
  id in two traces is unrelated content.
- **Prompts.** A request's prompt is its blocks' tokens, cut to
  ``input_length``.
- **Sessions.** A session is a root request and its descendants
  (``parent_chat_id`` links). A child whose full blocks repeat its parent's
  full blocks extends the parent: its prompt starts with the parent's exact
  prompt, including the parent's partly filled last block, and continues
  with tokens from its own blocks.
- **Outputs.** If such a child's prompt is at least as long as the parent's
  prompt plus output, the parent's output is the child's continuation, so
  the blocks the parent writes while decoding are the ones the child reads,
  as in production. Otherwise (no child, or a child that drops part of the
  output, such as a thinking trace's reasoning) the output is filler tokens
  of ``output_length``.
- **Ids.** Requests are numbered by time within their session:
  ``<workload>:<root chat id>:<n>``.

The trace's block size must equal the replay's block size; ``synthesize``
refuses anything else, because the sharing would otherwise be wrong.
"""
import functools
import hashlib
import json
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from client_simulator.schema import TokenRequest

TRACES = {
    "bailian-chat": "qwen_traceA_blksz_16.jsonl",
    "bailian-api": "qwen_traceB_blksz_16.jsonl",
    "bailian-thinking": "qwen_thinking_blksz_16.jsonl",
    "bailian-coder": "qwen_coder_blksz_16.jsonl",
}
URL = ("https://media.githubusercontent.com/media/alibaba-edu/"
       "qwen-bailian-usagetraces-anon/main/{name}")
TRACE_BLOCK_SIZE = 16
# Ordinary Qwen2.5 tokens: special tokens start at 151,643.
TOKEN_LOW, TOKEN_HIGH = 1000, 150000

Record = dict


def check_download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in TRACES.values():
        path = raw_dir / name
        if path.is_file():
            continue
        temporary = path.with_name(path.name + ".download")
        print(f"Downloading {name} to {path}...")
        try:
            with urllib.request.urlopen(URL.format(name=name),
                                        timeout=60) as source, \
                    temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _tokens(key: str, count: int) -> List[int]:
    digest = hashlib.blake2b(key.encode(), digest_size=4 * count).digest()
    span = TOKEN_HIGH - TOKEN_LOW
    return [
        TOKEN_LOW + int.from_bytes(digest[4 * i:4 * i + 4], "little") % span
        for i in range(count)
    ]


@functools.lru_cache(maxsize=1 << 18)
def block_tokens(workload: str, hash_id: int) -> tuple:
    """The 16 tokens of one hash id of one trace."""
    return tuple(_tokens(f"{workload}:{hash_id}", TRACE_BLOCK_SIZE))


def filler_tokens(workload: str, chat_id: int, length: int) -> List[int]:
    out: List[int] = []
    block = 0
    while len(out) < length:
        out += _tokens(f"{workload}:out:{chat_id}:{block}", 16)
        block += 1
    return out[:length]


def load_records(path: Path) -> List[Record]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sessions(records: Iterable[Record]) -> List[List[Record]]:
    """Each root with its descendants, in time order. A request whose
    parent is not in the trace starts its own session."""
    records = list(records)
    by_id = {r["chat_id"]: r for r in records}
    children: Dict[int, List[Record]] = defaultdict(list)
    roots = []
    for r in records:
        parent = r["parent_chat_id"]
        if parent in by_id and parent != r["chat_id"]:
            children[parent].append(r)
        else:
            roots.append(r)
    out = []
    for root in roots:
        members, stack = [], [root]
        while stack:
            node = stack.pop()
            members.append(node)
            stack.extend(children.get(node["chat_id"], ()))
        members.sort(key=lambda r: (r["timestamp"], r["chat_id"]))
        out.append(members)
    out.sort(key=lambda s: (s[0]["timestamp"], s[0]["chat_id"]))
    return out


def _span(workload: str, record: Record, start: int, end: int) -> List[int]:
    """Tokens at positions [start, end) of `record`'s prompt, from its own
    blocks."""
    out: List[int] = []
    ids = record["hash_ids"]
    for block in range(start // TRACE_BLOCK_SIZE, -(-end // TRACE_BLOCK_SIZE)):
        tokens = block_tokens(workload, ids[block])
        lo = max(start - block * TRACE_BLOCK_SIZE, 0)
        hi = min(end - block * TRACE_BLOCK_SIZE, TRACE_BLOCK_SIZE)
        out.extend(tokens[lo:hi])
    return out


def synthesize(workload: str,
               session: List[Record],
               block_size: int = TRACE_BLOCK_SIZE) -> List[TokenRequest]:
    """Token-level requests of one session (rules in the module docstring).

    `arrival_s` is each record's trace timestamp; the mix builder shifts it.
    """
    if block_size != TRACE_BLOCK_SIZE:
        raise ValueError(f"Bailian hash ids cover {TRACE_BLOCK_SIZE}-token "
                         f"blocks; replaying with block size {block_size} "
                         "would not preserve the trace's sharing")
    prompts: Dict[int, List[int]] = {}
    outputs: Dict[int, List[int]] = {}
    by_id = {r["chat_id"]: r for r in session}
    for r in session:
        if len(r["hash_ids"]) != -(-r["input_length"] // TRACE_BLOCK_SIZE):
            raise ValueError(f"{workload} chat {r['chat_id']}: "
                             f"{len(r['hash_ids'])} hash ids for "
                             f"{r['input_length']} input tokens")
        parent = by_id.get(r["parent_chat_id"])
        prompt = None
        if parent is not None and parent["chat_id"] in prompts:
            parent_len = parent["input_length"]
            full = parent_len // TRACE_BLOCK_SIZE
            if (r["input_length"] >= parent_len
                    and r["hash_ids"][:full] == parent["hash_ids"][:full]):
                prompt = prompts[parent["chat_id"]] + _span(
                    workload, r, parent_len, r["input_length"])
                end = parent_len + parent["output_length"]
                if (parent["chat_id"] not in outputs
                        and r["input_length"] >= end):
                    outputs[parent["chat_id"]] = prompt[parent_len:end]
        if prompt is None:
            prompt = _span(workload, r, 0, r["input_length"])
        prompts[r["chat_id"]] = prompt
    root = session[0]["chat_id"]
    requests = []
    for turn, r in enumerate(session):
        output = outputs.get(r["chat_id"])
        if output is None:
            output = filler_tokens(workload, r["chat_id"], r["output_length"])
        requests.append(
            TokenRequest(request_id=f"{workload}:{root}:{turn}",
                         session_id=f"{workload}:{root}",
                         turn_index=turn,
                         prompt_token_ids=prompts[r["chat_id"]],
                         output_token_ids=output,
                         arrival_s=float(r["timestamp"]),
                         meta={
                             "chat_id": r["chat_id"],
                             "parent_chat_id": r["parent_chat_id"],
                             "type": r["type"],
                             "output_from_child": r["chat_id"] in outputs,
                         }))
    return requests


class TraceAdapter:
    """`process_raw_traces` adapter for one of the four traces."""

    def __init__(self, workload: str):
        self.workload = workload

    def check_download(self, raw_dir: Path) -> None:
        check_download(raw_dir)

    def load_token_requests(self, raw_dir: Path) -> Iterator[TokenRequest]:
        records = load_records(raw_dir / TRACES[self.workload])
        for session in sessions(records):
            yield from synthesize(self.workload, session)


ADAPTERS = {workload: TraceAdapter(workload) for workload in TRACES}
