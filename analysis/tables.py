"""Per-run tables of every lookup, free-pool stay and request (spec 007).

M3 Task 2's questions (reuse interval, reuse portion, which workload a
request is, and what an eviction policy sees) all start from the same facts
in a run's cache event log. This module reads the log once and writes them
to ``<run_dir>/tables/`` as three parquet tables, so every analysis reads
one definition of each fact:

- ``lookups.parquet``: every prompt-block lookup (a hit or miss during
  ``allocate``), with its request, its place in the request's prefill,
  ``004``'s usable prefix, whether the block was shared with a running
  request, and the previous access of its hash (``004``'s reuse interval:
  ``metrics.AccessHistory``) with who made it;
- ``residencies.parquet``: every stay of a cached block in the free pool,
  from its ``free`` to its ``revive`` or ``evict``, or right-censored at the
  log's last event, with what each policy orders the pool by (LRU: the last
  access ``la``; FIFO: the first free; LFU: the hits);
- ``requests.parquet``: every client request, with its first prefill's
  hits and ``004``'s usable fraction.

Each table's parquet metadata holds a JSON ``run`` entry: the run's policy,
capacity, plan, block size, clocks, ``censor_t`` and analysis window
(``w0_s``, ``stop_s``), plus ``checks``: counts of what a sound log never
holds (a hit on a free block without its revive, a revive or eviction of a
block not in the pool, ...), empty for every log so far. Requests are named
by their client ids. Usage::

    python -m analysis.tables results/m3-task2/<run> [...] [--jobs N]
"""
import argparse
import gzip
import json
import math
import os
import time
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import (Any, Callable, Dict, Iterator, List, NamedTuple, Optional,
                    Tuple, Union)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis import events as ev
from analysis import metrics as m

PathLike = Union[str, Path]

TABLES = ("lookups", "residencies", "requests")
W0_S = 600.0
SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[1]

ROW_GROUP = 1 << 18  # rows per parquet row group
REQUEST_TYPES = {
    "req": pa.string(),
    "workload": pa.string(),
    "session": pa.string(),
    "turn": pa.int32(),
    "planned_s": pa.float64(),
    "sent_s": pa.float64(),
    "status": pa.string(),
    "prompt_tokens": pa.int32(),
    "prompt_blocks": pa.int32(),
    "hits": pa.int32(),
    "usable_hits": pa.int32(),
    "hits_after_miss": pa.int32(),
    "usable_fraction": pa.float64(),
    "cached_tokens": pa.int32(),
    "t_start": pa.float64(),
    "preempted": pa.bool_(),
}
SOURCES = ("first", "same_session", "other_session", "other_workload")
ENDS = ("censored", "revive", "evict")
KINDS = ("insert", "hit")
NAN = math.nan

# --------------------------------------------------------------------------
# Inputs: the run's manifest, its plan and its client rows
# --------------------------------------------------------------------------


class PlanRequest(NamedTuple):
    session: str
    turn: int
    planned_s: Optional[float]
    prompt_tokens: int


def run_manifest(run_dir: PathLike) -> Dict[str, Any]:
    """``run.json`` (engine run) or ``sim.json`` (``analysis/cachesim``)."""
    for name in ("run.json", "sim.json"):
        path = Path(run_dir) / name
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"No run.json or sim.json in {run_dir}")


def _locate(value: PathLike, run_dir: PathLike, *names: str) -> Path:
    """A folder a manifest names (absolute, or relative to the working
    folder, the repository or a folder above the run) that holds one of
    `names`."""
    path = Path(value)
    if path.is_absolute():
        candidates = [path]
    else:
        above = [parent / path for parent in Path(run_dir).resolve().parents]
        candidates = [Path.cwd() / path, REPO / path, *above]
    for candidate in candidates:
        if any((candidate / name).exists() for name in names):
            return candidate
    raise FileNotFoundError(f"{value} not found (looked in "
                            f"{', '.join(map(str, candidates))})")


def resolve_plan_dir(value: PathLike, run_dir: PathLike) -> Path:
    """A manifest's ``plan_dir``."""
    return _locate(value, run_dir, "plan.jsonl.gz")


def la_clock(manifest: Dict[str, Any],
             run_dir: PathLike) -> Tuple[Optional[Dict[str, Any]], str]:
    """The header whose (``time``, ``t``) pair puts a log's ``la`` on the
    log's clock, and what it is: None for the log's own (``"log"``).

    A simulator run timed from an engine run (``timing: engine:<run>``)
    runs on that engine's monotonic clock but stamps its wall-clock ``la``,
    while its own header pairs 0 with 0: that run's header converts it. If
    that run is missing, ``la`` is unknown (``"unknown"``, NaN).
    """
    timing = manifest.get("timing") or ""
    if not timing.startswith("engine:"):
        return None, "log"
    try:
        engine = _locate(timing[len("engine:"):], run_dir, "events.jsonl.gz",
                         "events.jsonl")
    except FileNotFoundError:
        return None, "unknown"
    return next(_rows(ev.log_path(engine))), f"engine run {engine}"


def read_plan_requests(plan_dir: PathLike) -> Dict[str, PlanRequest]:
    """Each planned request's session, turn, planned send time (as the
    client computes it: ``arrival_s``, else ``meta.planned_s``) and prompt
    length. Simulator client rows lack the planned time, so it comes from
    here."""
    from client_simulator.plan_replay import planned_s
    from client_simulator.token_plan import read_plan
    return {
        r.request_id: PlanRequest(r.session_id, r.turn_index, planned_s(r),
                                  len(r.prompt_token_ids))
        for r in read_plan(Path(plan_dir) / "plan.jsonl.gz")
    }


def la_monotonic(header: Dict[str, Any], la: Optional[float]) -> float:
    """A free's ``la`` (wall time, ``time.time()``) on the log's monotonic
    clock; NaN for a log without it."""
    if la is None:
        return NAN
    return la - header["time"] + header["t"]


def _rows(path: Path) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------
# The scan: one pass over the log
# --------------------------------------------------------------------------


class _Columns:
    """Typed column buffers (``array.array``), appended row by row."""

    def __init__(self, **typecodes: str) -> None:
        for name, code in typecodes.items():
            setattr(self, name, array(code))

    def np(self, name: str) -> np.ndarray:
        column = getattr(self, name)
        dtype = {"d": np.float64, "q": np.int64, "i": np.int32, "b": np.int8}
        return np.frombuffer(column, dtype=dtype[column.typecode])


class _Block:
    """What the log has said so far about one content hash."""
    __slots__ = ("hits", "first_free", "inserter", "index", "evicted", "pool")

    def __init__(self) -> None:
        self.hits = 0  # hits since its last insert (LFU's count)
        self.first_free = -1  # residency row of its first free since then
        self.inserter = -1  # request code of its last insert
        self.index = -1  # its place in its token chain
        self.evicted = False  # evicted since its last access
        self.pool = -1  # its residency row while it is in the free pool


class _Scanner:
    """Feeds every event to ``004``'s `AccessHistory` and `Prefills` and
    fills the lookup and residency columns. Requests are numbered (codes)
    in order of appearance; a code names a client request id. `la_header`
    converts ``la`` in place of the log's header (see `la_clock`)."""

    def __init__(self,
                 la_header: Optional[Dict[str, Any]] = None,
                 la_known: bool = True) -> None:
        self.la_header, self.la_known = la_header, la_known
        self.header: Dict[str, Any] = {}
        self.complete = False
        self.censor_t = NAN
        self.seq = 0
        self.history = m.AccessHistory()
        self.prefills = m.Prefills()
        self.engine: Dict[str, tuple] = {}  # engine id -> (code, id)
        self.client_codes: Dict[str, int] = {}
        self.clients: List[str] = []
        self.next_index: List[int] = []  # next decode block's chain index
        self.prompt_blocks: List[int] = []  # first-prefill counts
        self.hits: List[int] = []
        self.usable_hits: List[int] = []
        self.blocks: Dict[int, _Block] = {}
        self.checks: Counter = Counter()
        self.expect_revive: Optional[tuple] = None
        self.previous_event: tuple = (None, None)
        self.lookups = _Columns(t="d",
                                req="i",
                                prefill="i",
                                index="i",
                                hash="q",
                                hit="b",
                                usable="b",
                                in_use="b",
                                prev_t="d",
                                prev_req="i",
                                prev_kind="b",
                                requests_between="i",
                                evicted_before="b",
                                seq="q")
        self.residencies = _Columns(hash="q",
                                    block="q",
                                    t_free="d",
                                    la="d",
                                    first="q",
                                    owner="i",
                                    inserter="i",
                                    index="i",
                                    hits_before="i",
                                    end="b",
                                    t_end="d",
                                    evictor="i",
                                    seq="q")

    def _request(self, req: Optional[str]) -> tuple:
        """(code, canonical engine id) of an engine request id."""
        if req is None:
            return -1, None
        entry = self.engine.get(req)
        if entry is None:
            client = ev.client_request_id(req)
            code = self.client_codes.get(client)
            if code is None:
                code = self.client_codes[client] = len(self.clients)
                self.clients.append(client)
                for column in (self.next_index, self.prompt_blocks, self.hits,
                               self.usable_hits):
                    column.append(0)
            entry = self.engine[req] = (code, req)
        return entry

    def feed(self, row: Dict[str, Any]) -> None:
        kind = row["ev"]
        if kind == "header":
            self.header = row
            return
        if kind == "end":
            self.complete = True
            return
        t, h, block, op = row["t"], row["hash"], row["block"], row["op"]
        code, req = self._request(row["req"])
        if t < self.censor_t:
            self.checks["non_monotonic_events"] += 1
        self.censor_t = t
        if self.expect_revive is not None:
            if kind != "revive" or self.expect_revive != (h, block):
                self.checks["free_hits_without_revive"] += 1
            self.expect_revive = None
        previous = self.history.step(t, kind, h, req, op)
        lookup = self.prefills.step(kind, h, req, op)
        state = self.blocks.get(h)
        if state is None:
            state = self.blocks[h] = _Block()
        if kind == "free":
            self._free(t, h, block, code, state, row.get("la"))
        elif kind == "insert":
            state.hits, state.first_free = 0, -1
            state.inserter, state.evicted = code, False
            if op == "append_slots":
                state.index = self._decode_index(code)
        elif kind == "evict":
            self._end(state, 2, t, code, block)
            state.evicted = True
        elif kind == "hit" or kind == "miss":
            self._lookup(t, kind, h, block, code, req, state, previous, lookup)
        elif kind == "revive":
            if self.previous_event not in (("hit", h), ("dedupe", h)):
                self.checks["revives_not_after_hit_or_dedupe"] += 1
            self._end(state, 1, t, -1, block)
        elif kind == "dedupe" and op == "append_slots":
            state.index = self._decode_index(code)
        self.previous_event = (kind, h)
        self.seq += 1

    def _decode_index(self, code: int) -> int:
        """A block filled while decoding comes after the prompt's blocks."""
        if code < 0:
            return -1
        index = self.next_index[code]
        self.next_index[code] = index + 1
        return index

    def _lookup(self, t: float, kind: str, h: int, block: Optional[int],
                code: int, req: Optional[str], state: _Block,
                previous: Optional[m.Access],
                lookup: Optional[m.Lookup]) -> None:
        hit = kind == "hit"
        in_use = hit and state.pool < 0
        if hit and not in_use:
            self.expect_revive = (h, block)
        if previous is not None and hit == state.evicted:
            # A hit right after an eviction, or a miss on a cached hash.
            self.checks["lookups_contradicting_evictions"] += 1
        if lookup is None:
            self.checks["lookups_outside_allocate"] += 1
        else:
            c = self.lookups
            c.t.append(t)
            c.req.append(code)
            c.prefill.append(lookup.prefill)
            c.index.append(lookup.index)
            c.hash.append(h)
            c.hit.append(hit)
            c.usable.append(lookup.usable)
            c.in_use.append(in_use)
            if previous is None:
                c.prev_t.append(NAN)
                c.prev_req.append(-1)
                c.prev_kind.append(-1)
                c.requests_between.append(-1)
                c.evicted_before.append(False)
            else:
                c.prev_t.append(previous.t)
                c.prev_req.append(self._request(previous.req)[0])
                c.prev_kind.append(KINDS.index(previous.kind))
                c.requests_between.append(
                    self.history.requests_between(previous, t, req))
                c.evicted_before.append(state.evicted)
            c.seq.append(self.seq)
            state.index = lookup.index
            if code >= 0:
                self.next_index[code] = lookup.index + 1
                if lookup.prefill == 0:
                    self.prompt_blocks[code] += 1
                    self.hits[code] += hit
                    self.usable_hits[code] += lookup.usable
        if hit:
            state.hits += 1
            state.evicted = False

    def _free(self, t: float, h: int, block: Optional[int], code: int,
              state: _Block, la: Optional[float]) -> None:
        c = self.residencies
        row = len(c.t_free)
        if state.pool >= 0:
            self.checks["frees_while_in_pool"] += 1
        if state.first_free < 0:
            state.first_free = row
        c.hash.append(h)
        c.block.append(-1 if block is None else block)
        c.t_free.append(t)
        c.la.append(self._la(la))
        c.first.append(state.first_free)
        c.owner.append(code)
        c.inserter.append(state.inserter)
        c.index.append(state.index)
        c.hits_before.append(state.hits)
        c.end.append(0)
        c.t_end.append(NAN)
        c.evictor.append(-1)
        c.seq.append(self.seq)
        state.pool = row

    def _la(self, la: Optional[float]) -> float:
        if not self.la_known:
            return NAN
        return la_monotonic(self.la_header or self.header, la)

    def _end(self, state: _Block, end: int, t: float, evictor: int,
             block: Optional[int]) -> None:
        """Close the block's stay in the free pool (revive or evict)."""
        row = state.pool
        if row < 0:
            self.checks[f"{ENDS[end]}s_without_free"] += 1
            return
        c = self.residencies
        if c.block[row] != (-1 if block is None else block):
            self.checks["block_id_mismatches"] += 1
        c.end[row] = end
        c.t_end[row] = t
        c.evictor[row] = evictor
        state.pool = -1


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


class _Requests:
    """Per request code: client id, workload, session, turn and planned
    time, looked up per row by code (-1 gives null)."""

    def __init__(self, clients: List[str], plan: Dict[str, PlanRequest],
                 client_rows: Dict[str, Dict[str, Any]]) -> None:
        sessions, turns, planned = [], [], []
        for client in clients:
            info = plan.get(client)
            parsed = ev.parse_request_id(client)
            if info is not None:
                session, turn = info.session, info.turn
            elif parsed is not None:
                session, turn = parsed
            else:
                session, turn = client, -1
            sessions.append(session)
            turns.append(turn)
            planned.append(_planned(info, client_rows.get(client)))
        workloads = [ev.workload_of(c) for c in clients]
        self.req = pa.array(clients, pa.string())
        self.workload = pa.array(workloads, pa.string())
        self.session = pa.array(sessions, pa.string())
        self.turn = np.asarray(turns, dtype=np.int32)
        self.planned_s = np.asarray(planned, dtype=np.float64)
        self.session_code = _codes(sessions)
        self.workload_code = _codes(workloads)

    def text(self, column: str, codes: np.ndarray) -> pa.Array:
        """`column` (req, workload or session) of each code's request."""
        return getattr(self, column).take(pa.array(codes, mask=codes < 0))

    def turns(self, codes: np.ndarray) -> pa.Array:
        values = self.turn[np.maximum(codes, 0)]
        return pa.array(values, mask=(codes < 0) | (values < 0))

    def planned(self, codes: np.ndarray) -> np.ndarray:
        return np.where(codes < 0, NAN, self.planned_s[np.maximum(codes, 0)])


def _codes(values: List[str]) -> np.ndarray:
    index: Dict[str, int] = {}
    return np.asarray([index.setdefault(v, len(index)) for v in values],
                      dtype=np.int64)


def _planned(info: Optional[PlanRequest],
             client_row: Optional[Dict[str, Any]]) -> float:
    if info is not None and info.planned_s is not None:
        return float(info.planned_s)
    if client_row is not None and client_row.get("planned_s") is not None:
        return float(client_row["planned_s"])
    return NAN


def _labels(names: tuple, codes: np.ndarray) -> pa.Array:
    """`names[code]` per row; null where the code is -1."""
    labels = pa.array(list(names), pa.string())
    codes = codes.astype(np.int64)
    return labels.take(pa.array(codes, mask=codes < 0))


def _lookup_rows(scan: _Scanner, info: _Requests,
                 rows: slice) -> Dict[str, Any]:
    c = scan.lookups
    req, prev = c.np("req")[rows], c.np("prev_req")[rows]
    t, prev_t = c.np("t")[rows], c.np("prev_t")[rows]
    prev_kind = c.np("prev_kind")[rows]
    safe_req, safe_prev = np.maximum(req, 0), np.maximum(prev, 0)
    same_session = info.session_code[safe_req] == info.session_code[safe_prev]
    same_workload = (
        info.workload_code[safe_req] == info.workload_code[safe_prev])
    source = np.select([prev < 0, same_session, same_workload], [0, 1, 2], 3)
    return {
        "t": t,
        "req": info.text("req", req),
        "workload": info.text("workload", req),
        "session": info.text("session", req),
        "turn": info.turns(req),
        "planned_s": info.planned(req),
        "prefill": c.np("prefill")[rows],
        "index": c.np("index")[rows],
        "hash": c.np("hash")[rows],
        "hit": c.np("hit")[rows].astype(bool),
        "usable": c.np("usable")[rows].astype(bool),
        "in_use": c.np("in_use")[rows].astype(bool),
        "prev_t": prev_t,
        "prev_req": info.text("req", prev),
        "prev_kind": _labels(KINDS, prev_kind),
        "source": _labels(SOURCES, source),
        "interval_s": t - prev_t,
        "requests_between": c.np("requests_between")[rows],
        "evicted_before": c.np("evicted_before")[rows].astype(bool),
    }


def next_lookups(lookup_hash: np.ndarray, lookup_seq: np.ndarray,
                 lookup_t: np.ndarray, hashes: np.ndarray,
                 seqs: np.ndarray) -> np.ndarray:
    """For each (hash, log position), the time of the first lookup of that
    hash later in the log; NaN if there is none."""
    n_lookups = len(lookup_hash)
    keys = np.concatenate([lookup_hash, hashes])
    order = np.lexsort((np.concatenate([lookup_seq, seqs]), keys))
    keys = keys[order]
    is_lookup = order < n_lookups
    n = len(order)
    # The first lookup at or after each position, in (hash, position) order.
    ahead = np.where(is_lookup, np.arange(n), n)
    ahead = np.minimum.accumulate(ahead[::-1])[::-1]
    positions = np.flatnonzero(~is_lookup)
    found = ahead[positions]
    ok = found < n
    ok[ok] = keys[found[ok]] == keys[positions[ok]]
    out = np.full(len(hashes), NAN)
    out[order[positions[ok]] - n_lookups] = lookup_t[order[found[ok]]]
    return out


def _residency_rows(scan: _Scanner, info: _Requests, next_lookup_t: np.ndarray,
                    rows: slice) -> Dict[str, Any]:
    c = scan.residencies
    first = c.np("first")[rows]
    t_free, la = c.np("t_free")[rows], c.np("la")[rows]
    end = c.np("end")[rows]
    t_end = np.where(end == 0, scan.censor_t, c.np("t_end")[rows])
    owner, evictor = c.np("owner")[rows], c.np("evictor")[rows]
    inserter = c.np("inserter")[rows]
    return {
        "hash": c.np("hash")[rows],
        "block": c.np("block")[rows],
        "t_free": t_free,
        "la": la,
        "first_free_t": c.np("t_free")[first],
        "owner_req": info.text("req", owner),
        "owner_workload": info.text("workload", owner),
        "owner_session": info.text("session", owner),
        "owner_turn": info.turns(owner),
        "owner_planned_s": info.planned(owner),
        "inserter_workload": info.text("workload", inserter),
        "index": c.np("index")[rows],
        "hits_before": c.np("hits_before")[rows],
        "end": _labels(ENDS, end),
        "t_end": t_end,
        "idle_s": t_end - t_free,
        "evictor_req": info.text("req", evictor),
        "evictor_workload": info.text("workload", evictor),
        "next_lookup_t": next_lookup_t[rows],
        # FIFO's exact key: the evictor's last access at that first free.
        "first_free_la": c.np("la")[first],
    }


def _request_columns(scan: _Scanner, client_rows: List[Dict[str, Any]],
                     plan: Dict[str, PlanRequest],
                     block_size: int) -> Dict[str, pa.Array]:
    starts: Dict[int, float] = {}
    preempted = set()
    for engine_id, (code, _) in scan.engine.items():
        start = scan.history.starts.get(engine_id)
        if start is not None:
            starts[code] = min(start, starts.get(code, start))
        if scan.prefills.count(engine_id) > 1:
            preempted.add(code)
    cols: Dict[str, List[Any]] = {name: [] for name in REQUEST_TYPES}
    for row in client_rows:
        rid = row["request_id"]
        code = scan.client_codes.get(rid, -1)
        info = plan.get(rid)
        prompt = row.get("prompt_tokens")
        if prompt is None and info is not None:
            prompt = info.prompt_tokens
        if code >= 0:
            blocks, hits = scan.prompt_blocks[code], scan.hits[code]
            usable = scan.usable_hits[code]
        else:
            blocks = hits = usable = 0
        session, turn = row.get("conversation_id"), row.get("turn_index")
        if info is not None:
            session = info.session if session is None else session
            turn = info.turn if turn is None else turn
        sent = row.get("sent_s")
        usable_fraction = NAN
        if prompt:
            usable_fraction = min(prompt, block_size * usable) / prompt
        values = {
            "req": rid,
            "workload": row.get("workload") or ev.workload_of(rid),
            "session": session,
            "turn": turn,
            "planned_s": _planned(info, row),
            "sent_s": NAN if sent is None else sent,
            "status": row.get("status"),
            "prompt_tokens": prompt,
            "prompt_blocks": blocks,
            "hits": hits,
            "usable_hits": usable,
            "hits_after_miss": hits - usable,
            "usable_fraction": usable_fraction,
            "cached_tokens": row.get("cached_tokens"),
            "t_start": starts.get(code, NAN),
            "preempted": code in preempted,
        }
        for name, value in values.items():
            cols[name].append(value)
    return {
        name: pa.array(values, REQUEST_TYPES[name])
        for name, values in cols.items()
    }


def _slicer(columns: Dict[str, pa.Array]) -> Callable[[slice], Dict]:
    return lambda rows: {name: c[rows] for name, c in columns.items()}


def _write(path: Path, n_rows: int, build: Callable[[slice], Dict[str, Any]],
           meta: Dict[str, Any]) -> None:
    """Write a table a row group at a time (`build` makes the columns of a
    slice of rows, which bounds memory), with the ``run`` entry."""
    tmp = path.with_name(path.name + ".tmp")
    writer = None
    try:
        for start in range(0, max(n_rows, 1), ROW_GROUP):
            table = pa.table(build(slice(start, start + ROW_GROUP)))
            if writer is None:
                schema = table.schema.with_metadata({"run": json.dumps(meta)})
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            writer.write_table(table.cast(schema))
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def extract_run(run_dir: PathLike,
                plan_dir: Optional[PathLike] = None,
                w0_s: float = W0_S) -> Dict[str, Path]:
    """Write a run's lookups, residencies and requests tables to
    ``<run_dir>/tables/``; return their paths by name.

    `plan_dir` defaults to the manifest's (``run.json`` or ``sim.json``).
    The log is read once.
    """
    run_dir = Path(run_dir)
    manifest = run_manifest(run_dir)
    if plan_dir is None:
        if not manifest.get("plan_dir"):
            raise ValueError(f"{run_dir}'s manifest names no plan_dir")
        plan_dir = resolve_plan_dir(manifest["plan_dir"], run_dir)
    plan_dir = Path(plan_dir)
    plan = read_plan_requests(plan_dir)
    stop_s = (manifest.get("plan") or {}).get("stop_s")
    if (plan_dir / "manifest.json").exists():
        stop_s = json.loads(
            (plan_dir / "manifest.json").read_text()).get("stop_s")
    client_rows = ev.load_results(run_dir / "client.jsonl")
    by_id = {r["request_id"]: r for r in client_rows}

    la_header, la_source = la_clock(manifest, run_dir)
    scan = _Scanner(la_header, la_known=la_source != "unknown")
    for row in _rows(ev.log_path(run_dir)):
        scan.feed(row)
    # Per-hash state is done with; the tables need only the columns.
    scan.blocks.clear()
    scan.history.last.clear()
    missing = sum(c not in by_id for c in scan.clients)
    if missing:
        scan.checks["log_requests_without_client_row"] = missing
    header = scan.header
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "policy": manifest.get("eviction_policy", manifest.get("policy")),
        "capacity_tokens": manifest.get("capacity_tokens"),
        "ceiling": manifest.get("ceiling"),
        "timing": manifest.get("timing"),
        "la_clock": la_source,
        "plan_dir": str(plan_dir),
        "block_size": header.get("block_size"),
        "header_time": header.get("time"),
        "header_t": header.get("t"),
        "censor_t": scan.censor_t,
        "w0_s": w0_s,
        "stop_s": stop_s,
        "complete": scan.complete,
        "checks": dict(scan.checks),
    }
    info = _Requests(scan.clients, plan, by_id)
    out = run_dir / "tables"
    out.mkdir(exist_ok=True)
    paths = {name: out / f"{name}.parquet" for name in TABLES}
    lk, res = scan.lookups, scan.residencies
    next_lookup_t = next_lookups(lk.np("hash"), lk.np("seq"), lk.np("t"),
                                 res.np("hash"), res.np("seq"))
    _write(paths["residencies"], len(res.t_free),
           lambda rows: _residency_rows(scan, info, next_lookup_t, rows), meta)
    _write(paths["lookups"], len(lk.t),
           lambda rows: _lookup_rows(scan, info, rows), meta)
    requests = _request_columns(scan, client_rows, plan,
                                header.get("block_size", 16))
    _write(paths["requests"], len(client_rows), _slicer(requests), meta)
    return paths


def load_table(run_dir: PathLike, name: str) -> pa.Table:
    """One of a run's tables (`extract_run` writes them)."""
    if name not in TABLES:
        raise ValueError(f"Unknown table {name!r}; use one of {TABLES}")
    return pq.read_table(Path(run_dir) / "tables" / f"{name}.parquet")


def run_info(table: pa.Table) -> Dict[str, Any]:
    """The ``run`` metadata of a table from `load_table`."""
    return json.loads(table.schema.metadata[b"run"])


def _extract(args: tuple) -> str:
    run_dir, plan_dir, w0_s = args
    start = time.perf_counter()
    paths = extract_run(run_dir, plan_dir, w0_s)
    rows = {
        name: pq.read_metadata(path).num_rows
        for name, path in paths.items()
    }
    return (f"{run_dir}: {rows} in {time.perf_counter() - start:.0f} s")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--plan-dir",
                        type=Path,
                        help="Plan folder (default: the run's manifest's)")
    parser.add_argument("--w0",
                        type=float,
                        default=W0_S,
                        help="Start of the analysis window, in seconds")
    parser.add_argument("--jobs",
                        type=int,
                        default=1,
                        help="Runs to extract in parallel")
    args = parser.parse_args(argv)
    tasks = [(d, args.plan_dir, args.w0) for d in args.run_dirs]
    if args.jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(min(args.jobs, len(tasks))) as pool:
            for line in pool.map(_extract, tasks):
                print(line, flush=True)
    else:
        for task in tasks:
            print(_extract(task), flush=True)


if __name__ == "__main__":
    main()
