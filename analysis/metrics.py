"""The project's prefix-sharing metrics, one implementation each.

Definitions: ``specs/004-m2-prefix-sharing-analysis/spec.md`` (Metric
Definitions). Inputs are `analysis.events.Event` lists in log order and the
client simulator's result rows. No vLLM import.
"""
import bisect
import math
from collections import Counter, defaultdict
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Tuple,
                    TypeVar)

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

    A request without a hit has fraction 0.0 (never missing). Requests that
    did not complete are left out and counted by status.
    """
    rows: List[Row] = []
    excluded: Counter = Counter()
    for result in results:
        if result.get("status") != "completed":
            excluded[result.get("status")] += 1
            continue
        prompt = result["prompt_tokens"]
        cached = result.get("cached_tokens") or 0
        rows.append({
            "request_id": result["request_id"],
            "conversation_id": result["conversation_id"],
            "turn_index": result["turn_index"],
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "cached_fraction": cached / prompt,
        })
    return rows, excluded


def token_reused_fraction(rows: Sequence[Row],
                          key: str = "cached_tokens") -> Optional[float]:
    """Sum of cached tokens over sum of prompt tokens."""
    prompt = sum(r["prompt_tokens"] for r in rows)
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


def reuse_intervals(events: Sequence[Event]) -> List[Row]:
    """Gaps between consecutive accesses (insert or hit) of each hash.

    ``requests_between`` counts the other requests whose first allocate
    event falls strictly inside the gap: a rate-independent distance.
    """
    starts = request_starts(events)
    start_times = sorted(starts.values())
    last: Dict[int, Event] = {}
    rows: List[Row] = []
    for e in events:
        if e.ev not in ACCESSES:
            continue
        previous = last.get(e.hash)
        last[e.hash] = e
        if previous is None:
            continue
        lo = bisect.bisect_right(start_times, previous.t)
        hi = bisect.bisect_left(start_times, e.t)
        between = hi - lo
        if e.req in starts and previous.t < starts[e.req] < e.t:
            between -= 1
        rows.append({
            "hash": e.hash,
            "interval_s": e.t - previous.t,
            "requests_between": between,
            "kind": e.ev,
        })
    return rows


def prefix_hits(events: Iterable[Event]) -> Dict[str, Row]:
    """Per client request id, the lookups of its first prefill: hits before
    the first miss (``prefix_hits``) and hits after it
    (``hits_after_miss``).

    The first prefill is the request's `allocate` events up to its first
    event of another op; a re-prefill after preemption is not counted.
    Only the prefix can be skipped: the scheduler computes everything from
    the first uncached block on, so hits behind a miss save no compute.
    """
    counts: Dict[str, Row] = {}
    missed, closed = set(), set()
    for e in events:
        if e.req is None or e.req in closed:
            continue
        if e.op != "allocate":
            closed.add(e.req)
            continue
        if e.ev not in LOOKUPS:
            continue
        row = counts.setdefault(client_request_id(e.req), {
            "prefix_hits": 0,
            "hits_after_miss": 0
        })
        if e.ev == "miss":
            missed.add(e.req)
        elif e.req in missed:
            row["hits_after_miss"] += 1
        else:
            row["prefix_hits"] += 1
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


def windowed_hit_rate(events: Sequence[Event],
                      window_s: float = 10.0) -> List[Row]:
    """Hit rate per window of `window_s`, from the first event."""
    lookups = [e for e in events if e.ev in LOOKUPS]
    if not lookups:
        return []
    t0 = events[0].t
    bins: Dict[int, Counter] = defaultdict(Counter)
    for e in lookups:
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


def hit_sources(events: Iterable[Event]) -> Dict[Tuple[str, Any], int]:
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
            hitter = parse_request_id(e.req)
            owner = parse_request_id(insert.req) if insert else None
            if insert is None or hitter is None or owner is None:
                counts[("unknown", insert.op if insert else None)] += 1
            else:
                same = hitter[0] == owner[0]
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
        "gap_tokens":
        r["cached_tokens"] - block_size * hits[r["request_id"]],
    } for r in rows]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def cdf(values: Iterable[_T]) -> Tuple[List[_T], List[float]]:
    xs = sorted(values)
    n = len(xs)
    return xs, [(i + 1) / n for i in range(n)]
