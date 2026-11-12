"""The project's prefix-sharing metrics, one implementation each.

Definitions: ``specs/004-m2-prefix-sharing-analysis/spec.md`` (Metric
Definitions). Inputs are `analysis.events.Event` lists in log order and the
client simulator's result rows. No vLLM import.
"""
import bisect
import math
from collections import Counter, defaultdict
from typing import (Any, Dict, Iterable, List, NamedTuple, Optional, Sequence,
                    Tuple, TypeVar)

from analysis.events import Event, client_request_id, parse_request_id

Row = Dict[str, Any]
_T = TypeVar("_T", int, float)

LOOKUPS = ("hit", "miss")
ACCESSES = ("insert", "hit")

# --------------------------------------------------------------------------
# Per-request metrics (client results)
# --------------------------------------------------------------------------


def cached_fractions(results: Iterable[Row]) -> Tuple[List[Row], Counter]:
    """Per-request cached fraction for completed requests.

    Known zero hits give fraction 0.0; unavailable API counts stay None.
    Requests that did not complete are left out and counted by status.
    """
    rows: List[Row] = []
    excluded: Counter = Counter()
    for result in results:
        if result.get("status") != "completed":
            excluded[result.get("status")] += 1
            continue
        prompt = result["prompt_tokens"]
        cached = result.get("cached_tokens")
        rows.append({
            "request_id":
            result["request_id"],
            "conversation_id":
            result["conversation_id"],
            "turn_index":
            result["turn_index"],
            "prompt_tokens":
            prompt,
            "cached_tokens":
            cached,
            "cached_fraction":
            cached / prompt if cached is not None else None,
            **({
                "workload": result["workload"]
            } if "workload" in result else {}),
        })
    return rows, excluded


def token_reused_fraction(rows: Sequence[Row],
                          key: str = "cached_tokens") -> Optional[float]:
    """Sum of cached tokens over sum of prompt tokens."""
    prompt = sum(r["prompt_tokens"] for r in rows)
    if any(r[key] is None for r in rows):
        return None
    return sum(r[key] for r in rows) / prompt if prompt else None


def add_usable_fractions(rows: List[Row], prefixes: Dict[str, Row],
                         block_size: int) -> None:
    """Add each request's usable cached tokens and fraction to its row.

    Usable is the contiguous cached prefix of the first prefill
    (`prefix_hits`), the tokens vLLM can skip. vLLM's own `cached_tokens`
    comes from a binary search that assumes cached blocks form a prefix;
    LRU keeps that true, FIFO does not, and then it overcounts.
    """
    for r in rows:
        prefix = prefixes.get(r["request_id"], {})
        usable = min(block_size * prefix.get("prefix_hits", 0),
                     r["prompt_tokens"])
        r["usable_cached_tokens"] = usable
        r["usable_cached_fraction"] = usable / r["prompt_tokens"]
        r["hits_after_miss"] = prefix.get("hits_after_miss", 0)


def mean_by_turn(rows: Sequence[Row],
                 key: str = "cached_fraction") -> Dict[int, float]:
    by_turn: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        if r[key] is not None:
            by_turn[r["turn_index"]].append(r[key])
    return {t: sum(v) / len(v) for t, v in sorted(by_turn.items())}


# --------------------------------------------------------------------------
# Block metrics (event log)
# --------------------------------------------------------------------------


def hits_per_block(events: Iterable[Event]) -> Dict[int, int]:
    """Hits per content hash over the run; blocks never hit count 0."""
    hits: Dict[int, int] = {}
    for e in events:
        if e.ev == "insert":
            hits.setdefault(e.hash, 0)
        elif e.ev == "hit":
            hits[e.hash] = hits.get(e.hash, 0) + 1
    return hits


def hits_per_residency(events: Iterable[Event]) -> List[int]:
    """Hits between each insert of a hash and its next evict (or the end)."""
    current: Dict[int, int] = {}
    done: List[int] = []
    for e in events:
        if e.ev == "insert":
            if e.hash in current:  # Not evicted in the log; close it anyway.
                done.append(current[e.hash])
            current[e.hash] = 0
        elif e.ev == "hit" and e.hash in current:
            current[e.hash] += 1
        elif e.ev == "evict" and e.hash in current:
            done.append(current.pop(e.hash))
    return done + list(current.values())


def request_starts(events: Iterable[Event]) -> Dict[str, float]:
    """Time of each request's first event during `allocate`."""
    starts: Dict[str, float] = {}
    for e in events:
        if e.op == "allocate" and e.req is not None:
            starts.setdefault(e.req, e.t)
    return starts


class Access(NamedTuple):
    """One access of a hash (``004``): an ``insert`` or a ``hit``."""
    t: float
    req: Optional[str]  # engine request id, as in the log
    kind: str


class AccessHistory:
    """The previous access of every hash, fed the log's events in order.

    The one implementation of ``004``'s reuse interval: `reuse_intervals`
    and ``analysis/tables.py`` both read it. It also keeps each request's
    start (its first ``allocate`` event, as `request_starts`) for
    `requests_between`.
    """

    def __init__(self) -> None:
        self.last: Dict[int, Access] = {}
        self.starts: Dict[str, float] = {}
        self._start_times: List[float] = []

    def step(self, t: float, kind: str, content_hash: int, req: Optional[str],
             op: Optional[str]) -> Optional[Access]:
        """Record one event; return the previous access of its hash, or
        None for the first. An insert or hit becomes the hash's last
        access."""
        if op == "allocate" and req is not None and req not in self.starts:
            self.starts[req] = t
            bisect.insort(self._start_times, t)
        previous = self.last.get(content_hash)
        if kind in ACCESSES:
            self.last[content_hash] = Access(t, req, kind)
        return previous

    def requests_between(self, previous: Access, t: float,
                         req: Optional[str]) -> int:
        """Requests other than `req` that started strictly between
        `previous` and `t`: a rate-independent distance."""
        times = self._start_times
        between = (bisect.bisect_left(times, t) -
                   bisect.bisect_right(times, previous.t))
        start = self.starts.get(req) if req is not None else None
        if start is not None and previous.t < start < t:
            between -= 1
        return between


def reuse_intervals(events: Iterable[Event]) -> List[Row]:
    """Gaps between consecutive accesses (insert or hit) of each hash.

    ``requests_between`` counts the other requests whose first allocate
    event falls strictly inside the gap: a rate-independent distance. Each
    row also names both accesses: ``t``/``req`` and ``prev_t``/``prev_req``
    /``prev_kind`` (engine request ids, as in the log).
    """
    history = AccessHistory()
    rows: List[Row] = []
    for e in events:
        previous = history.step(e.t, e.ev, e.hash, e.req, e.op)
        if previous is None or e.ev not in ACCESSES:
            continue
        between = history.requests_between(previous, e.t, e.req)
        rows.append({
            "hash": e.hash,
            "interval_s": e.t - previous.t,
            "requests_between": between,
            "kind": e.ev,
            "t": e.t,
            "prev_t": previous.t,
            "req": e.req,
            "prev_req": previous.req,
            "prev_kind": previous.kind,
        })
    return rows


class Lookup(NamedTuple):
    """Where one lookup falls in its request's prefills."""
    prefill: int  # 0 for the first prefill, k for the k-th re-prefill
    index: int  # the block's place in the prefill's chain (0 = first)
    usable: bool  # a hit with no miss earlier in this prefill


class _Prefill:
    __slots__ = ("number", "lookups", "missed", "first_hash", "open")

    def __init__(self, number: int, is_open: bool) -> None:
        self.number = number
        self.lookups = 0
        self.missed = False
        self.first_hash: Optional[int] = None
        self.open = is_open


class Prefills:
    """Each request's prefills, fed the log's events in order.

    A prefill is a run of a request's ``allocate`` events. The next run, a
    re-prefill after a preemption, starts after any event of another op or
    when the request looks up its prefill's first block again (a
    preemption that frees only blocks other requests still hold logs
    nothing). The one implementation of ``004``'s usable prefix:
    `prefix_hits` and ``analysis/tables.py`` both read it.
    """

    def __init__(self) -> None:
        self._current: Dict[str, _Prefill] = {}

    def count(self, req: str) -> int:
        """Prefills of `req` so far; a request whose first event is not an
        ``allocate`` one had a first prefill without events."""
        prefill = self._current.get(req)
        return prefill.number + 1 if prefill is not None else 0

    def step(self, kind: str, content_hash: int, req: Optional[str],
             op: Optional[str]) -> Optional[Lookup]:
        """Record one event; for a lookup (hit or miss) during
        ``allocate``, say where it falls."""
        if req is None:
            return None
        prefill = self._current.get(req)
        if op != "allocate":
            if prefill is None:
                self._current[req] = _Prefill(0, is_open=False)
            else:
                prefill.open = False
            return None
        if prefill is None:
            prefill = self._current[req] = _Prefill(0, is_open=True)
        elif not prefill.open:
            prefill = self._current[req] = _Prefill(prefill.number + 1, True)
        if kind not in LOOKUPS:
            return None
        if prefill.lookups and content_hash == prefill.first_hash:
            prefill = self._current[req] = _Prefill(prefill.number + 1, True)
        index = prefill.lookups
        prefill.lookups += 1
        if not index:
            prefill.first_hash = content_hash
        if kind == "miss":
            prefill.missed = True
        return Lookup(prefill.number, index, kind == "hit"
                      and not prefill.missed)


def prefix_hits(events: Iterable[Event]) -> Dict[str, Row]:
    """Per client request id, the lookups of its first prefill: hits before
    the first miss (``prefix_hits``) and hits after it
    (``hits_after_miss``).

    The first prefill is the request's first run of `allocate` events
    (`Prefills`); a re-prefill after preemption is not counted. Only the
    prefix can be skipped: the scheduler computes everything from the
    first uncached block on, so hits behind a miss save no compute.
    """
    prefills = Prefills()
    counts: Dict[str, Row] = {}
    for e in events:
        lookup = prefills.step(e.ev, e.hash, e.req, e.op)
        if lookup is None or lookup.prefill:
            continue
        row = counts.setdefault(client_request_id(e.req), {
            "prefix_hits": 0,
            "hits_after_miss": 0
        })
        if lookup.usable:
            row["prefix_hits"] += 1
        elif e.ev == "hit":
            row["hits_after_miss"] += 1
    return counts


def block_hit_rate(events: Iterable[Event]) -> Optional[float]:
    """hits / (hits + misses); vLLM's gpu_prefix_cache_hit_rate."""
    counts = Counter(e.ev for e in events if e.ev in LOOKUPS)
    lookups = counts["hit"] + counts["miss"]
    return counts["hit"] / lookups if lookups else None


def eviction_count(events: Iterable[Event]) -> int:
    """Number of `evict` events (spec 005, FR-007); 0 if the cache never
    filled."""
    return sum(e.ev == "evict" for e in events)


def eviction_rate(events: Iterable[Event]) -> Optional[float]:
    """Evictions per 1,000 lookups, ``1000 × evictions / (hits + misses)``
    (spec 005, FR-007)."""
    counts = Counter(e.ev for e in events)
    lookups = counts["hit"] + counts["miss"]
    return 1000 * counts["evict"] / lookups if lookups else None


def windowed_hit_rate(events: Iterable[Event],
                      window_s: float = 10.0) -> List[Row]:
    """Hit rate per window of `window_s`, from the first event."""
    t0 = None
    bins: Dict[int, Counter] = defaultdict(Counter)
    for e in events:
        if t0 is None:
            t0 = e.t
        if e.ev in LOOKUPS:
            bins[math.floor((e.t - t0) / window_s)][e.ev] += 1
    rows = []
    for index in sorted(bins):
        c = bins[index]
        total = c["hit"] + c["miss"]
        rows.append({
            "start_s": index * window_s,
            "lookups": total,
            "hit_rate": c["hit"] / total,
        })
    return rows


def hit_sources(events: Iterable[Event],
                conversations=None) -> Dict[Tuple[str, Any], int]:
    """Count hits by who inserted the block in its current residency.

    Keys are (``intra`` | ``cross`` | ``unknown``, insert op): ``intra`` if
    the inserting request belongs to the hitting request's conversation;
    op ``append_slots`` means the block holds generated tokens.
    """
    inserted_by: Dict[int, Event] = {}
    counts: Counter = Counter()
    for e in events:
        if e.ev == "insert":
            inserted_by[e.hash] = e
        elif e.ev == "evict":
            inserted_by.pop(e.hash, None)
        elif e.ev == "hit":
            insert = inserted_by.get(e.hash)
            if conversations is None:
                hitter_id = parse_request_id(e.req)
                owner_id = parse_request_id(insert.req) if insert else None
                hitter = hitter_id[0] if hitter_id else None
                owner = owner_id[0] if owner_id else None
            else:
                hitter = conversations.get(client_request_id(
                    e.req)) if e.req else None
                owner = conversations.get(client_request_id(
                    insert.req)) if insert and insert.req else None
            if insert is None or hitter is None or owner is None:
                counts[("unknown", insert.op if insert else None)] += 1
            else:
                same = hitter == owner
                counts[("intra" if same else "cross", insert.op)] += 1
    return dict(counts)


# --------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------


def cached_token_gaps(rows: Sequence[Row], events: Iterable[Event],
                      block_size: int) -> List[Row]:
    """SC-004: cached_tokens - block_size * allocate-op hits, per request."""
    hits: Counter = Counter(
        client_request_id(e.req) for e in events
        if e.ev == "hit" and e.op == "allocate" and e.req is not None)
    return [{
        "request_id":
        r["request_id"],
        "cached_tokens":
        r["cached_tokens"],
        "logged_hits":
        hits[r["request_id"]],
        "gap_tokens": (r["cached_tokens"] - block_size * hits[r["request_id"]]
                       if r["cached_tokens"] is not None else None),
    } for r in rows]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def cdf(values: Iterable[_T]) -> Tuple[List[_T], List[float]]:
    xs = sorted(values)
    n = len(xs)
    return xs, [(i + 1) / n for i in range(n)]
