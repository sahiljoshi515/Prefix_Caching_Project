"""Test the ATC'25 paper's claims on the Bailian traces and on our mix.

Wang et al., "KVCache Cache in the Wild: Characterizing and Optimizing
KVCache Cache at a Large Cloud Provider" (USENIX ATC '25) characterized the
traces we replay. Its Trace A (to-C: chat, file, multimodal and search) is
our ``bailian-chat``, and its Trace B (to-B API calls) is our
``bailian-api``. The thinking and coder traces came later; the paper does
not cover them.

`scorecard` checks ten of the paper's claims, each on two inputs:

- **Raw**: each released trace alone, at its full rate, with a cache that
  never evicts. A block is a full 16-token unit of a prompt (vLLM caches
  nothing smaller), keyed by its whole prefix of hash ids as vLLM's chained
  hash keys it, so two blocks are equal only if everything before them is
  too. An *access* is any prompt block, and a *reuse* (or hit) is an access
  to a block seen before. The traces have no user id, so a *session* (a
  root request and its descendants by ``parent_chat_id``) stands in for a
  user. Metrics skip the first `W0_S` seconds as a warm-up, as the mix's
  analysis window does; forward-looking ones ("reused later") also skip the
  last `HORIZON_S` seconds, which cannot show a reuse yet.
- **Mix**: our mixed workload's no-eviction ("ceiling") run, from the
  tables of ``analysis/tables.py`` (lookups, residencies, requests), over
  the analysis window. A simulator run (``sim.json``) only guides.

Each row holds the claim, the paper's figure and value, our metric, the
raw and mix values, a verdict (``holds``, ``partly``, ``no`` or ``not
testable here``) and the rule that decided it. ``verdict`` is the rule on
Traces A and B (the paper's own data), ``all_traces_verdict`` on all four
traces, and ``mix_verdict`` on our mix's workloads. Two limits apply to
every row: the released traces cover two hours where the paper used days,
and their hash ids cover prompts only, so the raw cache cannot hit blocks
written while decoding (our replay can).

`thinning_effect` measures how much our mix's session sampling dilutes the
Bailian traces' reuse: the ideal reuse portion of a plan's window with all
of the window's sessions (full rate), and with only the sessions the plan
kept::

    python -m analysis.paper_checks --raw-dir data/raw/bailian \\
        --ceiling results/dev/sim-s1-ceiling \\
        --plan data/processed/m3/balanced-human-s1 --out paper_checks.json
"""
import argparse
import gzip
import json
import math
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis import survival as sv
from client_simulator.adapters import bailian

BLOCK_SIZE = bailian.TRACE_BLOCK_SIZE
W0_S = 600.0  # the mix's analysis window starts here
HORIZON_S = 600.0  # follow-up a "not reused later" or "no next turn" needs
SPLIT_S = 3600.0  # claim 6: fit on the first hour, test on the second
GRID_S = 10.0  # claim 10: the live working set, sampled every 10 s
CAPACITY_BLOCKS = 131072 // BLOCK_SIZE  # our evaluation cache
MIN_CATEGORY = 1000  # accesses a category needs to be compared
PAPER_TRACES = ("bailian-chat", "bailian-api")  # its Trace A and Trace B
TURN_BUCKETS = ("1", "2", "3-5", "6+")  # by the 1-based turn number
INDEX_EDGES = (0, 1, 4, 16, 64, 256)
FREQUENCY_BUCKETS = ("0", "1", "2", "3+")
NOISE = 0.01  # identical runs differ by about this (spec 007 Background 5)
HOLDS, PARTLY, NO, UNTESTABLE = "holds", "partly", "no", "not testable here"

Row = Dict[str, Any]

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _mean(values: np.ndarray) -> Optional[float]:
    return float(np.mean(values)) if len(values) else None


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return float(a / b) if a is not None and b else None


def quantiles(
    values: np.ndarray, ps: Sequence[float] = (0.5, 0.8, 0.9, 0.99)
) -> Optional[Row]:
    """Count, mean and quantiles; the p-quantile is the value at sorted
    index ``int(p * n)``, as in `analysis.characterize.quantiles`."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    index = np.minimum(len(v) - 1, (np.asarray(ps) * len(v)).astype(int))
    part = np.partition(v, np.unique(index))
    out: Row = {"n": int(len(v)), "mean": float(v.mean())}
    for p, i in zip(ps, index):
        out[f"p{round(p * 100)}"] = float(part[i])
    return out


def top_share(counts: np.ndarray, fraction: float = 0.10) -> Optional[float]:
    """Share of the total held by the top `fraction` of items (at least
    one item)."""
    c = np.sort(np.asarray(counts))[::-1]
    total = c.sum()
    if not total:
        return None
    k = max(1, math.ceil(fraction * len(c)))
    return float(c[:k].sum() / total)


def turn_bucket(turn: np.ndarray) -> np.ndarray:
    """Index into `TURN_BUCKETS` of each 1-based turn number."""
    turn = np.asarray(turn)
    return np.select([turn <= 1, turn == 2, turn <= 5], [0, 1, 2], 3)


def live_counts(starts: np.ndarray, ends: np.ndarray,
                grid: np.ndarray) -> np.ndarray:
    """At each grid time, the number of intervals ``[start, end)`` that
    cover it."""
    return (np.searchsorted(np.sort(starts), grid, side="right") -
            np.searchsorted(np.sort(ends), grid, side="right"))


def live_summary(counts: np.ndarray, capacity: int) -> Optional[Row]:
    if not len(counts):
        return None
    return {
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "p50": float(np.percentile(counts, 50)),
        "p95": float(np.percentile(counts, 95)),
        "capacity_blocks": capacity,
        "share_of_time_above_capacity": float((counts > capacity).mean()),
    }


def _index_name(lo: int, hi: float) -> str:
    if hi == math.inf:
        return f"{lo}+"
    return str(lo) if hi == lo + 1 else f"{lo}-{int(hi) - 1}"


def position_profile(position: np.ndarray, blocks: np.ndarray,
                     outcome: np.ndarray) -> Row:
    """Mean of `outcome` by the block's tenth of its prompt (0 = head) and
    by its index in the prompt, with the access counts per tenth."""
    position = np.asarray(position)
    outcome = np.asarray(outcome, dtype=float)
    decile = np.minimum(9, position * 10 // np.maximum(blocks, 1))
    n = np.bincount(decile, minlength=10)
    total = np.bincount(decile, weights=outcome, minlength=10)
    edges = list(INDEX_EDGES) + [math.inf]
    return {
        "by_decile": [float(s / c) if c else None for s, c in zip(total, n)],
        "decile_accesses": n.tolist(),
        "by_index": {
            _index_name(lo, hi):
            _mean(outcome[(position >= lo) & (position < hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        },
    }


def frequency_profile(past: np.ndarray, outcome: np.ndarray) -> Row:
    """Mean of `outcome` by the past access count (0, 1, 2, 3+)."""
    k = np.minimum(np.asarray(past), 3)
    outcome = np.asarray(outcome, dtype=float)
    return {
        name: _mean(outcome[k == i])
        for i, name in enumerate(FREQUENCY_BUCKETS)
    }


@dataclass
class Neighbours:
    """Per access: its key's previous and next access (-1 if none), the
    number of earlier accesses, and its key's group. Per group (key): the
    first and last access and the access count."""
    prev: np.ndarray
    next: np.ndarray
    rank: np.ndarray
    group: np.ndarray
    first: np.ndarray
    last: np.ndarray
    count: np.ndarray


def neighbours(keys: np.ndarray) -> Neighbours:
    """`keys` lists accesses in time order, so a stable sort by key lists
    each key's accesses in time order."""
    n = len(keys)
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    new = np.ones(n, dtype=bool)
    new[1:] = ordered[1:] != ordered[:-1]
    prev = np.full(n, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    same = ~new[1:]
    prev[order[1:][same]] = order[:-1][same]
    nxt[order[:-1][same]] = order[1:][same]
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, n))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n) - np.repeat(starts, sizes)
    group = np.empty(n, dtype=np.int64)
    group[order] = np.repeat(np.arange(len(starts)), sizes)
    return Neighbours(prev, nxt, rank, group, order[starts],
                      order[starts + sizes - 1], sizes)


def _spread(values: Dict[str, float]) -> Optional[float]:
    v = [x for x in values.values() if 0 < x < math.inf]
    return max(v) / min(v) if v else None


def _weighted(fits: Row, key: str) -> Optional[float]:
    """Access-weighted mean of `key` over the categories' fits."""
    if not fits:
        return None
    n = np.array([f["n"] for f in fits.values()], dtype=float)
    v = np.array([f[key] for f in fits.values()], dtype=float)
    return float(v @ n / n.sum())


def category_fits(durations: np.ndarray,
                  observed: np.ndarray,
                  category: np.ndarray,
                  names: Sequence[str],
                  other: Optional[np.ndarray] = None,
                  min_n: int = MIN_CATEGORY) -> Row:
    """`survival.fit_summary` for each category with at least `min_n`
    accesses; the ratios of their plain rates (each pair, and the largest
    over the smallest) and of their cure rates; and the access-weighted KS
    of each model.

    `other` flags reuses made by another session: each category then also
    gets that share and the median reuse time of each kind, to show
    whether two processes mix (shared prompts reused within seconds, a
    conversation's history within minutes).
    """
    fits: Row = {}
    for code in np.unique(category):
        inside = category == code
        if inside.sum() < min_n:
            continue
        fit = sv.fit_summary(durations[inside], observed[inside])
        if other is not None:
            back = inside & observed
            mixed = other[back]
            fit["reuse_by_other_session_share"] = _mean(mixed)
            fit["median_s_same_session"] = _median(durations[back][~mixed])
            fit["median_s_other_session"] = _median(durations[back][mixed])
        fits[names[code]] = fit
    rates = {k: v["rate"] for k, v in fits.items()}
    return {
        "categories":
        fits,
        "rate_ratio_max":
        _spread(rates),
        "cure_rate_ratio_max":
        _spread({k: v["cure_rate"]
                 for k, v in fits.items()}),
        "rate_ratio": {
            f"{a}/{b}": rates[a] / rates[b]
            for a in rates for b in rates if a != b and rates[b] > 0
        },
        "ks_weighted":
        _weighted(fits, "ks"),
        "cure_ks_weighted":
        _weighted(fits, "cure_ks"),
    }


def _median(values: np.ndarray) -> Optional[float]:
    return float(np.median(values)) if len(values) else None


def history_check(t: np.ndarray,
                  next_t: np.ndarray,
                  has_next: np.ndarray,
                  category: np.ndarray,
                  names: Sequence[str],
                  split: float,
                  min_n: int = MIN_CATEGORY) -> Row:
    """Claim 6. Fit each category's cure model on the accesses before
    `split` (censored at `split`); then compare the later accesses'
    Kaplan-Meier curve with that fit, with one fit pooled over all
    categories, and with the later accesses' own fit (KS distances), and
    take the ratio of the two periods' plain rates."""
    train = t < split
    d_train = np.minimum(next_t, split) - t
    e_train = has_next & (next_t < split)
    d_test = next_t - t
    pooled = sv.fit_cure(d_train[train], e_train[train])
    rows: Row = {}
    for code in np.unique(category):
        a = train & (category == code)
        b = ~train & (category == code)
        if a.sum() < min_n or b.sum() < min_n:
            continue
        p, rate = sv.fit_cure(d_train[a], e_train[a])
        test_p, test_rate = sv.fit_cure(d_test[b], has_next[b])
        train_rate = sv.fit_exponential(d_train[a], e_train[a])
        test_rate_plain = sv.fit_exponential(d_test[b], has_next[b])
        rows[names[code]] = {
            "train_n":
            int(a.sum()),
            "test_n":
            int(b.sum()),
            "train_cure": [p, rate],
            "test_cure": [test_p, test_rate],
            "rate_ratio":
            _ratio(test_rate_plain, train_rate),
            "ks_own_fit":
            sv.ks_distance(d_test[b], has_next[b], rate, p),
            "ks_pooled_fit":
            sv.ks_distance(d_test[b], has_next[b], pooled[1], pooled[0]),
            "ks_refit":
            sv.ks_distance(d_test[b], has_next[b], test_rate, test_p),
        }
    weights = np.array([v["test_n"] for v in rows.values()], dtype=float)

    def weighted(key: str) -> Optional[float]:
        if not rows:
            return None
        values = np.array([v[key] for v in rows.values()], dtype=float)
        return float(values @ weights / weights.sum())

    steady = sum(v["test_n"] for v in rows.values()
                 if v["rate_ratio"] and 2 / 3 <= v["rate_ratio"] <= 1.5)
    return {
        "split_s": split,
        "pooled_cure": list(pooled),
        "categories": rows,
        "ks_own_fit_weighted": weighted("ks_own_fit"),
        "ks_pooled_fit_weighted": weighted("ks_pooled_fit"),
        "ks_refit_weighted": weighted("ks_refit"),
        "rate_within_1_5x_share": (steady / weights.sum() if rows else None),
    }


# --------------------------------------------------------------------------
# Raw traces
# --------------------------------------------------------------------------


@dataclass
class Trace:
    """A raw Bailian trace in time order: one entry per request, and the
    hash ids of every request's full blocks, concatenated."""
    workload: str
    t: np.ndarray  # seconds
    chat_id: np.ndarray
    parent: np.ndarray
    root: np.ndarray  # index of the session's root request
    turn: np.ndarray  # the trace's 1-based turn number
    kind: np.ndarray  # index into `kinds` (the record's type)
    kinds: List[str]
    input_length: np.ndarray
    output_length: np.ndarray
    has_child: np.ndarray
    starts: np.ndarray  # request r's blocks: hashes[starts[r]:starts[r+1]]
    hashes: np.ndarray

    @property
    def end_s(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.starts)


def _starts(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)


def _roots(chat: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """Index of each request's session root. A request whose parent is not
    in the trace starts its own session, as in `bailian.sessions`."""
    n = len(chat)
    if not n:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(chat, kind="stable")
    ordered = chat[order]
    if np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("Duplicate chat_id in the trace")
    at = np.minimum(np.searchsorted(ordered, parent), n - 1)
    found = (ordered[at] == parent) & (parent != chat)
    up = np.where(found, order[at], np.arange(n))
    for _ in range(64):
        higher = up[up]
        if np.array_equal(higher, up):
            return up
        up = higher
    raise ValueError("parent_chat_id links form a cycle")


def _columns(records) -> tuple:
    t, chat, parent, turn, kind, inp, out, lengths = ([] for _ in range(8))
    hashes = array("q")
    kinds: Dict[str, int] = {}
    for r in records:
        ids = r["hash_ids"][:r["input_length"] // BLOCK_SIZE]
        hashes.extend(ids)
        lengths.append(len(ids))
        t.append(r["timestamp"])
        chat.append(r["chat_id"])
        parent.append(r["parent_chat_id"])
        turn.append(r["turn"])
        kind.append(kinds.setdefault(r["type"], len(kinds)))
        inp.append(r["input_length"])
        out.append(r["output_length"])
    return t, chat, parent, turn, kind, list(kinds), inp, out, lengths, hashes


def _build_trace(workload: str, t, chat, parent, turn, kind, kinds, inp, out,
                 lengths, hashes) -> Trace:
    t = np.asarray(t, dtype=float)
    chat = np.asarray(chat, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    hashes = np.frombuffer(hashes, dtype=np.int64)
    starts = _starts(lengths)
    order = np.lexsort((chat, t))  # time order, ties by chat id
    if not np.array_equal(order, np.arange(len(t))):
        new_starts = _starts(lengths[order])
        shift = np.repeat(starts[:-1][order] - new_starts[:-1], lengths[order])
        hashes = hashes[shift + np.arange(len(hashes))]
        starts = new_starts
    chat = chat[order]
    parent = np.asarray(parent, dtype=np.int64)[order]
    return Trace(workload=workload,
                 t=t[order],
                 chat_id=chat,
                 parent=parent,
                 root=_roots(chat, parent),
                 turn=np.asarray(turn, dtype=np.int32)[order],
                 kind=np.asarray(kind, dtype=np.int64)[order],
                 kinds=kinds,
                 input_length=np.asarray(inp, dtype=np.int64)[order],
                 output_length=np.asarray(out, dtype=np.int64)[order],
                 has_child=np.isin(chat, parent[parent != chat]),
                 starts=starts,
                 hashes=hashes)


def trace_from_records(records: Sequence[Row], workload: str) -> Trace:
    """A `Trace` from record dicts (the tests build traces this way)."""
    return _build_trace(workload, *_columns(records))


def load_trace(path: Path, workload: str) -> Trace:
    """Stream a trace file, keeping each request's scalars and its full
    blocks' hash ids only."""

    def records():
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)

    return _build_trace(workload, *_columns(records()))


def prefix_keys(hashes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """A number per block, equal for two blocks exactly when their whole
    prefixes of hash ids are (as vLLM's chained hash).

    Builds the prefix tree one depth at a time: at depth d a block is the
    pair (its parent's number, its hash id), numbered with `np.unique`.
    Requests are sorted longest first, so the ones still open at depth d
    are always the first ones.
    """
    lengths = np.diff(starts)
    keys = np.empty(len(hashes), dtype=np.int64)
    if not len(hashes):
        return keys
    order = np.argsort(-lengths, kind="stable")
    ordered = lengths[order]
    base = starts[:-1][order]
    width = int(hashes.max()) + 1
    if (len(hashes) + 1) * width >= 2**62:
        raise ValueError("Too many blocks to number prefixes in int64")
    open_at = np.searchsorted(-ordered, -np.arange(ordered[0]), side="left")
    parent = np.zeros(len(order), dtype=np.int64)
    numbered = 0
    for depth, count in enumerate(open_at):
        at = base[:count] + depth
        pair = parent[:count] * width + hashes[at]
        unique, inverse = np.unique(pair, return_inverse=True)
        node = numbered + 1 + inverse  # 0 is the empty prefix
        numbered += len(unique)
        keys[at] = node
        parent[:count] = node
    return keys


class RawAnalysis:
    """A trace's accesses keyed two ways: by prefix (what vLLM can reuse)
    and by bare hash id (the same 16 tokens anywhere in a prompt, a
    content view)."""

    def __init__(self, trace: Trace):
        self.trace = trace
        lengths = trace.lengths
        self.req = np.repeat(np.arange(len(trace.t)), lengths)
        self.pos = np.arange(len(self.req)) - np.repeat(
            trace.starts[:-1], lengths)
        self.blocks = lengths[self.req]
        self.t = trace.t[self.req]
        self.key = prefix_keys(trace.hashes, trace.starts)
        self.near = neighbours(self.key)
        content = neighbours(trace.hashes)
        self.content_hit = content.prev >= 0
        self.content_later = content.next >= 0
        self.next_t = np.where(self.later,
                               self.t[np.maximum(self.near.next,
                                                 0)], trace.end_s)

    @property
    def hit(self) -> np.ndarray:
        return self.near.prev >= 0

    @property
    def later(self) -> np.ndarray:
        return self.near.next >= 0

    def category(self) -> Tuple[np.ndarray, List[str]]:
        """Each access's request category: its type and turn bucket."""
        tr = self.trace
        code = tr.kind * len(TURN_BUCKETS) + turn_bucket(tr.turn)
        names = [f"{k}-{b}" for k in tr.kinds for b in TURN_BUCKETS]
        return code[self.req], names


def next_turn(has_child: np.ndarray, turn: np.ndarray) -> Row:
    """P(a request has a child) by 1-based turn number (6 and later
    pooled), and for single-turn (turn 1) and later requests."""
    turns = np.minimum(turn, 6)
    return {
        "requests": int(len(turn)),
        "single_turn": _mean(has_child[turn == 1]),
        "multi_turn": _mean(has_child[turn >= 2]),
        "by_turn": {("6+" if k == 6 else str(k)): _mean(has_child[turns == k])
                    for k in range(1, 7) if (turns == k).any()},
    }


def raw_metrics(a: RawAnalysis,
                w0: float = W0_S,
                horizon: float = HORIZON_S,
                split: float = SPLIT_S,
                capacity: int = CAPACITY_BLOCKS) -> Row:
    """Every claim's value on one raw trace, keyed by claim number."""
    tr = a.trace
    hit, later = a.hit, a.later
    warm = a.t >= w0
    reuse = hit & warm
    before = a.near.prev[reuse]
    same = tr.root[a.req[reuse]] == tr.root[a.req[before]]
    followed = warm & (a.t <= tr.end_s - horizon)
    category, names = a.category()
    after = a.req[np.maximum(a.near.next, 0)]
    other = later & (tr.root[after] != tr.root[a.req])
    sessions = np.bincount(tr.root, minlength=len(tr.t))
    eligible = tr.t <= tr.end_s - horizon
    lifespan = a.t[a.near.last] - a.t[a.near.first]
    grid = np.arange(w0, tr.end_s - horizon, GRID_S)
    return {
        "requests":
        int(len(tr.t)),
        "accesses":
        int(len(a.t)),
        "end_s":
        tr.end_s,
        1: {
            "prompt_blocks": int(warm.sum()),
            "ideal_hit_ratio": _mean(hit[warm]),
            "ideal_hit_ratio_from_cold": _mean(hit),
            "content_hit_ratio": _mean(a.content_hit[warm]),
        },
        2: {
            "reuses": int(reuse.sum()),
            "same_session_share": _mean(same),
            "top10_block_share_of_reuses": top_share(a.near.count - 1),
            "multi_request_sessions": bool((sessions > 1).any()),
        },
        3: {
            "turn1_share_of_hits": _mean(tr.turn[a.req[reuse]] == 1),
            "turn1_share_of_requests": _mean(tr.turn == 1),
            "all_requests_turn_1": bool((tr.turn == 1).all()),
        },
        4:
        quantiles(a.t[reuse] - a.t[before]),
        5:
        category_fits(a.next_t[warm] - a.t[warm], later[warm], category[warm],
                      names, other[warm]),
        6:
        history_check(a.t[warm], a.next_t[warm], later[warm], category[warm],
                      names, split),
        7:
        next_turn(tr.has_child[eligible], tr.turn[eligible]),
        8: {
            "prefix":
            position_profile(a.pos[followed], a.blocks[followed],
                             later[followed]),
            "content":
            position_profile(a.pos[followed], a.blocks[followed],
                             a.content_later[followed]),
        },
        9: {
            "lifespan_s":
            quantiles(lifespan, (0.5, 0.9, 0.99)),
            "blocks_never_reused":
            _mean(a.near.count == 1),
            "future_reuse_by_past_accesses":
            frequency_profile(a.near.rank[followed], later[followed]),
        },
        10: {
            "live_blocks":
            live_summary(live_counts(a.t[later], a.next_t[later], grid),
                         capacity),
        },
    }


# --------------------------------------------------------------------------
# Thinning
# --------------------------------------------------------------------------


def plan_chat_ids(plan_dir: Path) -> Dict[str, np.ndarray]:
    """The Bailian chat ids of a plan's requests, per workload."""
    out: Dict[str, List[int]] = {}
    with gzip.open(Path(plan_dir) / "plan.jsonl.gz", "rt",
                   encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if "chat_id" in row.get("meta", {}):
                workload = row["request_id"].split(":", 1)[0]
                out.setdefault(workload, []).append(row["meta"]["chat_id"])
    return {w: np.asarray(ids, dtype=np.int64) for w, ids in out.items()}


def subset_reuse(a: RawAnalysis, keep: np.ndarray, measure_from: float) -> Row:
    """Ideal reuse among only the requests in `keep` (a mask), as if they
    were the whole trace: a cache that never evicts and starts empty. Hits
    are measured on requests at or after `measure_from`, and split by
    whether the block's previous access was in the same session."""
    tr = a.trace
    chosen = keep[a.req]
    req = a.req[chosen]
    near = neighbours(a.key[chosen])
    hit = near.prev >= 0
    same = np.zeros(len(req), dtype=bool)
    same[hit] = tr.root[req[hit]] == tr.root[req[near.prev[hit]]]
    measured = tr.t[req] >= measure_from
    return {
        "requests": int(keep.sum()),
        "sessions": int(len(np.unique(tr.root[keep]))),
        "prompt_blocks": int(measured.sum()),
        "ideal_hit_ratio": _mean(hit[measured]),
        "same_session_hit_ratio": _mean((hit & same)[measured]),
        "cross_session_hit_ratio": _mean((hit & ~same)[measured]),
        "ideal_hit_ratio_whole_window": _mean(hit),
    }


def window_mask(trace: Trace, start: float, duration: float, stop: float,
                max_model_len: int) -> np.ndarray:
    """The plan's sampling frame (`analysis.mix.sample_bailian`): sessions
    whose root arrives in ``[start, start + duration)``, less their turns
    at or after ``start + stop`` and requests too long for the model."""
    root_t = trace.t[trace.root]
    fits = trace.input_length + trace.output_length + 1 <= max_model_len
    return ((root_t >= start) & (root_t < start + duration) &
            (trace.t < start + stop) & fits)


def thinning_for(a: RawAnalysis,
                 manifest: Row,
                 chat_ids: np.ndarray,
                 w0: float = W0_S) -> Row:
    """One workload of one plan: (a) its window at full rate, (b) only the
    plan's sessions, and how much of (a)'s reuse (b) keeps. Hits count
    from ``w0`` into the window, as the mix's analysis window."""
    info = manifest["workloads"][a.trace.workload]
    start = float(info["window_start_s"])
    full = window_mask(a.trace, start, float(manifest["duration_s"]),
                       float(manifest["stop_s"]),
                       int(manifest.get("max_model_len", 2**62)))
    kept = full & np.isin(a.trace.chat_id, chat_ids)
    full_rate = subset_reuse(a, full, start + w0)
    thinned = subset_reuse(a, kept, start + w0)
    return {
        "window_start_s":
        start,
        "kept_session_fraction":
        info.get("kept_session_fraction"),
        "plan_requests":
        int(len(chat_ids)),
        "plan_requests_outside_window":
        int(len(chat_ids) - kept.sum()),
        "full_rate":
        full_rate,
        "kept":
        thinned,
        "kept_over_full_hit_ratio":
        _ratio(thinned["ideal_hit_ratio"], full_rate["ideal_hit_ratio"]),
        "kept_over_full_cross_session":
        _ratio(thinned["cross_session_hit_ratio"],
               full_rate["cross_session_hit_ratio"]),
    }


def thinning_effect(plan_dir: Path,
                    raw_dir: Path,
                    analyses: Optional[Dict[str, RawAnalysis]] = None,
                    w0: float = W0_S) -> Row:
    """For each Bailian workload of a plan: its ideal reuse portion in the
    plan's window at full rate and with only the plan's kept sessions.
    `analyses` passes traces already built (by workload)."""
    manifest = json.loads((Path(plan_dir) / "manifest.json").read_text())
    ids = plan_chat_ids(plan_dir)
    out: Row = {}
    for workload, name in bailian.TRACES.items():
        if workload not in manifest["workloads"]:
            continue
        a = (analyses or {}).get(workload)
        if a is None:
            a = RawAnalysis(load_trace(Path(raw_dir) / name, workload))
        out[workload] = thinning_for(a, manifest,
                                     ids.get(workload, np.empty(0)), w0)
    return out


# --------------------------------------------------------------------------
# Our mix: a ceiling run's tables
# --------------------------------------------------------------------------

LOOKUP_COLUMNS = ("t", "req", "workload", "session", "turn", "planned_s",
                  "prefill", "index", "hash", "hit", "prev_t", "source",
                  "interval_s")
RESIDENCY_COLUMNS = ("t_free", "t_end", "end", "hits_before", "owner_workload",
                     "next_lookup_t")
REQUEST_COLUMNS = ("workload", "session", "turn", "planned_s")


@dataclass
class Table:
    """A parquet table as numpy columns; a string column becomes integer
    codes into ``labels[column]`` (-1 for null)."""
    columns: Dict[str, np.ndarray]
    labels: Dict[str, List[str]]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    def code(self, column: str, value: str) -> int:
        """`value`'s code in a string column (-2 if absent)."""
        names = self.labels.get(column, [])
        return names.index(value) if value in names else -2

    def where(self, mask: np.ndarray) -> "Table":
        return Table({k: v[mask]
                      for k, v in self.columns.items()}, self.labels)


def read_table(path: Path, columns: Sequence[str]) -> Tuple[Table, Row]:
    """A table's columns and its ``run`` metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=list(columns))
    run = json.loads((table.schema.metadata or {}).get(b"run", b"{}"))
    out: Dict[str, np.ndarray] = {}
    labels: Dict[str, List[str]] = {}
    for name in columns:
        column = table.column(name).combine_chunks()
        if pa.types.is_string(column.type) or pa.types.is_large_string(
                column.type):
            column = column.dictionary_encode()
        if pa.types.is_dictionary(column.type):
            out[name] = np.asarray(column.indices.fill_null(-1),
                                   dtype=np.int64)
            labels[name] = [str(v) for v in column.dictionary.to_pylist()]
        else:
            out[name] = column.to_numpy(zero_copy_only=False)
    return Table(out, labels), run


@dataclass
class MixRun:
    name: str
    run: Row
    guide_only: bool  # a simulator run
    lookups: Table
    residencies: Table
    requests: Table


def load_mix(run_dir: Path) -> MixRun:
    """A run's lookups, residencies and requests tables
    (``<run_dir>/tables/*.parquet``)."""
    tables = Path(run_dir) / "tables"
    lookups, run = read_table(tables / "lookups.parquet", LOOKUP_COLUMNS)
    residencies, _ = read_table(tables / "residencies.parquet",
                                RESIDENCY_COLUMNS)
    requests, _ = read_table(tables / "requests.parquet", REQUEST_COLUMNS)
    return MixRun(name=Path(run_dir).name,
                  run=run,
                  guide_only=(Path(run_dir) / "sim.json").exists(),
                  lookups=lookups,
                  residencies=residencies,
                  requests=requests)


class MixAnalysis:
    """A ceiling run's first-prefill lookups with each block's neighbours,
    computed once. A lookup is an access; the first lookup of a block
    written while decoding is a hit, and that write was an access too."""

    def __init__(self, mix: MixRun, horizon: float = HORIZON_S):
        self.mix = mix
        lk = mix.lookups.where(mix.lookups["prefill"] == 0)
        self.lk = lk
        self.w0 = float(mix.run.get("w0_s", W0_S))
        self.stop = float(mix.run.get("stop_s", math.inf))
        t = lk["t"]
        self.censor = float(mix.run.get("censor_t",
                                        t.max() if len(t) else 0.0))
        self.window = (lk["planned_s"] >= self.w0) & (lk["planned_s"] <
                                                      self.stop)
        self.hit = lk["hit"].astype(bool)
        self.near = neighbours(lk["hash"])
        self.later = self.near.next >= 0
        self.next_t = np.where(self.later, t[np.maximum(self.near.next, 0)],
                               self.censor)
        self.followed = self.window & (t <= self.censor - horizon)
        decoded = self.hit[self.near.first]
        self.past = self.near.rank + decoded[self.near.group]
        self.blocks = np.bincount(lk["req"])[lk["req"]]
        self.workloads = lk.labels["workload"]

    def lifespan(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per block: first access to last lookup, and the workload of its
        first lookup."""
        lk, near = self.lk, self.near
        prev = np.nan_to_num(lk["prev_t"][near.first], nan=math.inf)
        first = np.minimum(lk["t"][near.first], prev)
        return lk["t"][near.last] - first, lk["workload"][near.first]

    def category(self) -> Tuple[np.ndarray, List[str]]:
        """Each lookup's category: its workload and turn bucket."""
        code = self.lk["workload"] * len(TURN_BUCKETS) + turn_bucket(
            self.lk["turn"] + 1)
        return code, [f"{w}-{b}" for w in self.workloads for b in TURN_BUCKETS]


def mix_workload_metrics(m: MixAnalysis, mine: np.ndarray, life: np.ndarray,
                         owner_mask: np.ndarray) -> Row:
    """Claims 1-4, 8 and 9 for the lookups in `mine` (a mask)."""
    lk = m.lk
    inside = mine & m.window
    reuse = inside & m.hit
    source = lk["source"][reuse]
    hits = np.bincount(m.near.group[reuse], minlength=len(m.near.count))
    looked_up = np.unique(m.near.group[inside])
    followed = mine & m.followed
    return {
        1: {
            "prompt_blocks": int(inside.sum()),
            "ideal_hit_ratio": _mean(m.hit[inside]),
        },
        2: {
            "reuses":
            int(reuse.sum()),
            "same_session_share":
            _mean(source == lk.code("source", "same_session")),
            "other_session_share":
            _mean(source == lk.code("source", "other_session")),
            "other_workload_share":
            _mean(source == lk.code("source", "other_workload")),
            "top10_block_share_of_reuses":
            top_share(hits[looked_up]),
        },
        3: {
            "turn1_share_of_hits": _mean(lk["turn"][reuse] == 0)
        },
        4:
        quantiles(lk["interval_s"][reuse]),
        8:
        position_profile(lk["index"][followed], m.blocks[followed],
                         m.later[followed]),
        9: {
            "lifespan_s":
            quantiles(life[owner_mask], (0.5, 0.9, 0.99)),
            "future_reuse_by_past_accesses":
            frequency_profile(m.past[followed], m.later[followed]),
        },
    }


def mix_next_turn(mix: MixRun, horizon: float = HORIZON_S) -> Row:
    """P(a request's session has its next turn) by turn number, per
    workload, for requests planned at least `horizon` before the stop."""
    rq = mix.requests
    stop = float(mix.run.get("stop_s", math.inf))
    key = rq["session"].astype(np.int64) * 1_000_000 + rq["turn"]
    has_next = np.isin(key + 1, key)
    eligible = rq["planned_s"] < stop - horizon
    sessions = np.bincount(rq["session"][rq["session"] >= 0])
    out: Row = {}
    for code, workload in enumerate(rq.labels["workload"]):
        mine = eligible & (rq["workload"] == code)
        row = next_turn(has_next[mine], rq["turn"][mine] + 1)
        own = np.unique(rq["session"][(rq["workload"] == code)
                                      & (rq["session"] >= 0)])
        row["multi_request_sessions"] = bool((sessions[own] > 1).any())
        out[workload] = row
    return out


def residency_frequency(mix: MixRun,
                        censor: float,
                        horizon: float = HORIZON_S) -> Row:
    """LFU's view when a block enters the free pool: P(it is looked up
    again) by its hits since insert (0, 1, 2, 3+), per owner workload, for
    blocks freed at least `horizon` before the log ends."""
    rs = mix.residencies
    followed = rs["t_free"] <= censor - horizon
    again = ~np.isnan(rs["next_lookup_t"].astype(float))
    groups = [("all", followed)
              ] + [(w, followed & (rs["owner_workload"] == code))
                   for code, w in enumerate(rs.labels["owner_workload"])]
    return {
        w: frequency_profile(rs["hits_before"][mask], again[mask])
        for w, mask in groups
    }


def mix_live(m: MixAnalysis, capacity: int = CAPACITY_BLOCKS) -> Row:
    """Over the analysis window: blocks with a future reuse (each hit's
    block was needed from its previous access until the hit), the free
    pool's blocks that come back (residencies ending in a revive), and the
    free pool's size."""
    lk = m.lk
    t = lk["t"][m.window]
    grid = np.arange(t.min(), t.max(), GRID_S) if len(t) else np.empty(0)
    reuse = m.hit & ~np.isnan(lk["prev_t"])
    rs = m.mix.residencies
    back = rs["end"] == rs.code("end", "revive")
    return {
        "live_blocks":
        live_summary(live_counts(lk["prev_t"][reuse], lk["t"][reuse], grid),
                     capacity),
        "free_pool_live_blocks":
        live_summary(live_counts(rs["t_free"][back], rs["t_end"][back], grid),
                     capacity),
        "free_pool_blocks":
        live_summary(live_counts(rs["t_free"], rs["t_end"], grid), capacity),
    }


def mix_metrics(mix: MixRun,
                horizon: float = HORIZON_S,
                capacity: int = CAPACITY_BLOCKS) -> Row:
    """Every claim's value in our mix: per workload, and over all of them
    (``"all"``, which also holds claims 5, 6, 7 and 10)."""
    m = MixAnalysis(mix, horizon)
    life, owner = m.lifespan()
    out: Row = {
        "run":
        mix.name,
        "guide_only":
        mix.guide_only,
        "w0_s":
        m.w0,
        "stop_s":
        m.stop,
        "all":
        mix_workload_metrics(m, np.ones(len(m.hit), dtype=bool), life,
                             np.ones(len(life), dtype=bool)),
    }
    for code, workload in enumerate(m.workloads):
        out[workload] = mix_workload_metrics(m, m.lk["workload"] == code, life,
                                             owner == code)
    category, names = m.category()
    w = m.window
    t = m.lk["t"]
    session = m.lk["session"]
    other = m.later & (session[np.maximum(m.near.next, 0)] != session)
    split = float(np.median(t[w])) if w.any() else 0.0
    out["all"][5] = category_fits(m.next_t[w] - t[w], m.later[w], category[w],
                                  names, other[w])
    out["all"][6] = history_check(t[w], m.next_t[w], m.later[w], category[w],
                                  names, split)
    out["all"][7] = mix_next_turn(mix, horizon)
    out["all"][9]["revived_by_hits_before"] = residency_frequency(
        mix, m.censor, horizon)
    out["all"][10] = mix_live(m, capacity)
    return out


# --------------------------------------------------------------------------
# Verdicts: one rule per claim, applied to each source and then combined
# --------------------------------------------------------------------------

HIT_CEILING = 0.80  # claim 1: "well below 80%"
SPREAD = 0.05  # claim 1: sources differ by at least 5 points
SKEW = 0.50  # claim 2: the top 10% of blocks make half of the reuses
LOCAL = 0.50  # claim 2: half of the reuses stay in their session
A_P80_S, B_P80_S = 600.0, 10.0  # claim 4: the paper's thresholds
TIMESCALE = 10.0  # claim 4: Trace A's P80 at least 10x Trace B's
FIT_KS = 0.10  # claim 5: a model fits if its weighted KS is this or less
DIFFER = 2.0  # claim 5: categories differ if their rates differ 2x
STEADY = 0.75  # claim 6: share of later accesses whose rate moved < 1.5x
RISE = 0.10  # claim 7: P(next turn) rises by 0.10 after turn 1
HEAD_DROP, HEAD_BUMP = 0.10, 0.02  # claim 8: fall head to tail; no rise
LIFESPAN_P90_S = 2 * 612.0  # claim 9: twice the paper's Trace A value
LIFT = 0.05  # claim 9: frequency "uninformative" if it lifts reuse <= this

RULES = {
    1: (f"holds if every source's ideal hit ratio is below "
        f"{HIT_CEILING:.0%} and the sources differ by at least "
        f"{SPREAD * 100:.0f} points; partly if only one of the two"),
    2: (f"per source: holds if the top 10% of blocks make at least "
        f"{SKEW:.0%} of the reuses and (where sessions have several "
        f"requests) at least {LOCAL:.0%} of reuses follow an access in the "
        "same session; partly if only one of the two. Sources combine as "
        "holds or no only when they agree. A session is stricter than a "
        "user, so a low same-session share cannot refute the paper's "
        "same-user claim"),
    3: ("not testable: the released Trace B (bailian-api) is all turn 1, "
        "so its turn-1 share of hits is 1 by construction"),
    4: (f"holds if Trace A's P80 reuse time is at most {A_P80_S:.0f} s and "
        f"Trace B's at most {B_P80_S:.0f} s; partly if A's P80 is at least "
        f"{TIMESCALE:.0f}x B's but a threshold is missed; no otherwise"),
    5: (f"per source: holds if a plain exponential fits (access-weighted "
        f"KS <= {FIT_KS}) and categories differ (largest plain or cure rate "
        f"ratio >= {DIFFER:.0f}); no if no model fits and categories do not "
        "differ; partly otherwise. Sources combine as holds or no only "
        "when they agree"),
    6: (f"per source: holds if (i) at least {STEADY:.0%} of the second "
        "hour's accesses are in categories whose plain rate moved less "
        "than 1.5x from the first hour, and (ii) the first hour's "
        "per-category fits beat one pooled fit on the second hour "
        "(access-weighted KS); partly if one of the two"),
    7: (f"per source with multi-turn sessions: holds if P(next turn) is "
        f"at least {RISE} higher after turn 1 than at turn 1, partly if "
        "higher by less, no if not higher"),
    8: (f"per source: holds if the prefix-keyed P(reused later) falls by "
        f"at least {HEAD_DROP} from the prompt's first tenth to its last "
        f"and never rises by more than {HEAD_BUMP} between adjacent tenths; "
        "partly if one of the two. A prefix cache makes the fall within "
        "one prompt certain, so the content view (bare hash ids) is shown "
        "as well"),
    9: (f"per source: holds if 90% of blocks are dead (no later reuse) "
        f"within {LIFESPAN_P90_S:.0f} s (twice the paper's Trace A value, "
        f"for our two-hour sample) and frequency adds at most {LIFT} to "
        "P(reused later) (3+ past accesses against 1); partly if one of "
        "the two"),
    10: (f"needs a capacity sweep: holds if the policies agree (within "
         f"{NOISE}) at every capacity at or above the mix's live working "
         "set (its P95) and differ at some capacity below it; partly if "
         "one of the two"),
}


def _combine(*parts: Optional[bool]) -> str:
    """holds if every testable part does, no if none does."""
    known = [p for p in parts if p is not None]
    if not known:
        return UNTESTABLE
    if all(known):
        return HOLDS
    return PARTLY if any(known) else NO


def _across(verdicts) -> str:
    """One verdict over several sources: holds or no only if all agree."""
    known = [v for v in verdicts if v != UNTESTABLE]
    if not known:
        return UNTESTABLE
    if all(v == HOLDS for v in known):
        return HOLDS
    return NO if all(v == NO for v in known) else PARTLY


def rule_1(values: Dict[str, Row]) -> str:
    ratios = [
        v["ideal_hit_ratio"] for v in values.values()
        if v and v.get("ideal_hit_ratio") is not None
    ]
    if len(ratios) < 2:
        return UNTESTABLE
    return _combine(
        max(ratios) < HIT_CEILING,
        max(ratios) - min(ratios) >= SPREAD)


def rule_2(v: Row) -> str:
    skew, local = v.get("top10_block_share_of_reuses"), None
    if v.get("multi_request_sessions"):
        local = v.get("same_session_share")
    return _combine(None if skew is None else skew >= SKEW,
                    None if local is None else local >= LOCAL)


def rule_4(values: Dict[str, Row]) -> str:
    a, b = values.get("bailian-chat"), values.get("bailian-api")
    if not a or not b:
        return UNTESTABLE
    if a["p80"] <= A_P80_S and b["p80"] <= B_P80_S:
        return HOLDS
    return PARTLY if a["p80"] >= TIMESCALE * b["p80"] else NO


def rule_5(v: Row) -> str:
    if len(v["categories"]) < 2:
        return UNTESTABLE
    plain = v["ks_weighted"] <= FIT_KS
    cure = v["cure_ks_weighted"] <= FIT_KS
    differ = max(v["rate_ratio_max"] or 0, v["cure_rate_ratio_max"]
                 or 0) >= DIFFER
    if plain and differ:
        return HOLDS
    return PARTLY if plain or cure or differ else NO


def rule_6(v: Row) -> str:
    if not v["categories"]:
        return UNTESTABLE
    return _combine(v["rate_within_1_5x_share"] >= STEADY,
                    v["ks_own_fit_weighted"] < v["ks_pooled_fit_weighted"])


def rule_7(v: Row) -> str:
    if v.get("multi_turn") is None or v.get("single_turn") is None:
        return UNTESTABLE
    rise = v["multi_turn"] - v["single_turn"]
    if rise >= RISE:
        return HOLDS
    return PARTLY if rise > 0 else NO


def rule_8(profile: Row) -> str:
    deciles = [x for x in profile["by_decile"] if x is not None]
    if len(deciles) < 2:
        return UNTESTABLE
    return _combine(
        deciles[0] - deciles[-1] >= HEAD_DROP,
        all(b - a <= HEAD_BUMP for a, b in zip(deciles, deciles[1:])))


def rule_9(v: Row) -> str:
    life, freq = v.get("lifespan_s"), v["future_reuse_by_past_accesses"]
    short = life["p90"] <= LIFESPAN_P90_S if life else None
    lift = (freq["3+"] - freq["1"]
            if freq["3+"] is not None and freq["1"] is not None else None)
    return _combine(short, lift <= LIFT if lift is not None else None)


def policy_spread(sweep: Sequence[Row],
                  live_p95: Optional[float],
                  metric: str = "hit_ratio") -> Row:
    """Claim 10's capacity half, from a capacity sweep: rows with
    ``capacity_blocks`` (or ``capacity_tokens``), ``policy`` and `metric`.
    The spread of `metric` across policies at each capacity, and the
    verdict of `RULES[10]` against the live working set's P95."""
    by_capacity: Dict[int, List[float]] = {}
    for r in sweep:
        blocks = r.get("capacity_blocks") or r["capacity_tokens"] // BLOCK_SIZE
        by_capacity.setdefault(int(blocks), []).append(float(r[metric]))
    spread = {
        c: max(v) - min(v)
        for c, v in sorted(by_capacity.items()) if len(v) >= 2
    }
    if live_p95 is None or not spread:
        return {"spread_by_capacity_blocks": spread, "verdict": UNTESTABLE}
    above = [s for c, s in spread.items() if c >= live_p95]
    below = [s for c, s in spread.items() if c < live_p95]
    return {
        "spread_by_capacity_blocks":
        spread,
        "live_p95_blocks":
        live_p95,
        "verdict":
        _combine(
            all(s <= NOISE for s in above) if above else None,
            any(s > NOISE for s in below) if below else None),
    }


# --------------------------------------------------------------------------
# The scorecard
# --------------------------------------------------------------------------

CLAIMS = {
    1: ("The ideal (never-evicting) hit ratio is well below 80% and varies "
        "by source", "Fig. 4, Sec. 3.2", "Trace A 62%, Trace B 54%",
        "hit blocks / prompt blocks with a cache that never evicts, for "
        "requests from 600 s on (mix: first-prefill lookups planned in "
        "the analysis window)"),
    2: ("Reuse stays within a user's own requests, and it is skewed",
        "Figs. 6-7, Sec. 1",
        "most hits come from the same user; 10% of blocks give 77% of "
        "the reuses",
        "share of reuses whose previous access was in the same session "
        "(raw: same root by parent_chat_id); share of reuses on the top "
        "10% of blocks (all blocks counted)"),
    3:
    ("Single-turn requests dominate the hits in the API trace", "Sec. 3.2",
     "97% of Trace B's hits", "share of hit blocks read by turn-1 requests"),
    4: ("The reuse timescale differs by source", "Fig. 12",
        "Trace A: 80% of reuse times under 10 min; Trace B: under 10 s",
        "time between consecutive accesses of a block (004's interval), "
        "for reuses from 600 s on: P50, P80, P90, P99"),
    5: ("Per-category reuse times are about exponential, and categories "
        "differ", "Fig. 15",
        "exponentials fit each category; text-1 differs from text-2",
        "per category (type x turn 1, 2, 3-5, 6+; mix: workload x turn): "
        "time from each access to the block's next access, right-censored "
        "at the end; plain exponential MLE and cure model (share p ever "
        "reused, at a rate), their KS distances, and the ratio of rates"),
    6: ("Recent history predicts the next window", "Fig. 15, Sec. 3.4",
        "a category's 9-10 am distribution matches 10-11 am",
        "per category: fit on the first hour (censored at 3,600 s), test "
        "on the second: rate ratio, KS against the first hour's fit, "
        "against one pooled fit, and against the second hour's own fit "
        "(mix: split at the window's median lookup time)"),
    7: ("The chance of a next turn rises with turn number",
        "Fig. 11, Sec. 3.3", "single-turn mean 0.5, multi-turn 0.8",
        "P(a request has a child) by turn number, for requests at least "
        "600 s before the end (mix: P(the session's next turn is in the "
        "plan))"),
    8:
    ("Hits concentrate at the prompt head (spatial locality)", "Figs. 16-18",
     "caching from the beginning of a prompt gives the best locality",
     "P(an access is reused later) by its tenth of the prompt and by "
     "block index, for accesses at least 600 s before the end"),
    9: ("Short lifespans make frequency misleading (so LFU does worst)",
        "Fig. 19, Sec. 4.2",
        "Trace A: 90% of blocks dead after 612 s; Trace B: after 0.3 s; "
        "frequency 'can not provide useful information'",
        "lifespan (first access to last reuse) quantiles; P(reused later) "
        "by past access count 0, 1, 2, 3+ at each access (mix: also LFU's "
        "view, P(looked up again) by hits_before when a block is freed)"),
    10: ("Eviction policy matters only below the live working set",
         "Fig. 22, Sec. 3.5; Figs. 25-27",
         "a small cache reaches the ideal hit ratio; policies differ at "
         "limited capacity",
         "blocks with a future reuse over time (the live working set: what "
         "a never-wrong policy must keep) against our 8,192-block cache; "
         "the policy spread comes from a capacity sweep (see "
         "policy_spread)"),
}


def _mix_values(m: Row, claim: int) -> Row:
    """One claim's values from `mix_metrics`."""
    if claim in (5, 6, 10):
        return m["all"][claim]
    if claim == 7:
        return m["all"][7]
    workloads = [k for k in m if isinstance(m[k], dict) and claim in m[k]]
    out = {w: dict(m[w][claim]) for w in workloads}
    if claim == 2:
        for w, v in out.items():
            v["multi_request_sessions"] = m["all"][7].get(w, {}).get(
                "multi_request_sessions", w == "all")
    return out


def _source_verdicts(claim: int, values: Dict[str, Row]) -> str:
    """`RULES[claim]` on the sources in `values`."""
    if not values:
        return UNTESTABLE
    if claim == 1:
        return rule_1(values)
    if claim == 3:
        return UNTESTABLE
    if claim == 4:
        return rule_4(values)
    one = {
        2: rule_2,
        5: rule_5,
        6: rule_6,
        7: rule_7,
        8: lambda v: rule_8(v.get("prefix", v)),  # raw: the prefix view
        9: rule_9,
    }.get(claim)
    if one is None:
        return UNTESTABLE
    return _across(one(v) for v in values.values() if v)


def _mix_verdict(claim: int, values: Row, sweep: Optional[Sequence[Row]]):
    if claim == 3:
        return UNTESTABLE, None
    if claim in (5, 6):
        return {5: rule_5, 6: rule_6}[claim](values), None
    if claim == 10:
        live = (values.get("live_blocks") or {}).get("p95")
        spread = policy_spread(sweep or [], live)
        return spread["verdict"], spread
    per_workload = {w: v for w, v in values.items() if w != "all"}
    return _source_verdicts(claim, per_workload), None


def build_rows(raw: Dict[str, Row],
               mix: Dict[str, Row],
               sweep: Optional[Sequence[Row]] = None) -> List[Row]:
    """The ten scorecard rows from `raw_metrics` (by workload) and
    `mix_metrics` (by run)."""
    rows = []
    for claim, (statement, figure, paper, metric) in CLAIMS.items():
        values = {w: r[claim] for w, r in raw.items()}
        paper_values = {w: v for w, v in values.items() if w in PAPER_TRACES}
        row: Row = {
            "claim": claim,
            "statement": statement,
            "figure": figure,
            "paper": paper,
            "metric": metric,
            "rule": RULES[claim],
            "raw": values,
            "verdict": _source_verdicts(claim, paper_values),
            "all_traces_verdict": _source_verdicts(claim, values),
            "mix": {},
            "mix_verdict": {},
        }
        for name, m in mix.items():
            mixed = _mix_values(m, claim)
            row["mix"][name] = mixed
            verdict, extra = _mix_verdict(claim, mixed, sweep)
            row["mix_verdict"][name] = verdict
            if extra is not None:
                row.setdefault("policy_spread", {})[name] = extra
        rows.append(row)
    return rows


def scorecard(raw_dir: Path,
              ceiling_run_dirs: Sequence[Path] = (),
              plan_dirs: Sequence[Path] = (),
              sweep: Optional[Sequence[Row]] = None,
              workloads: Sequence[str] = tuple(bailian.TRACES),
              capacity: int = CAPACITY_BLOCKS) -> Row:
    """The ten rows on the raw traces and on each ceiling run, and the
    thinning of each plan's Bailian workloads, with runtimes."""
    started = time.perf_counter()
    runtimes: Row = {}
    plans = []
    for plan_dir in plan_dirs:
        t0 = time.perf_counter()
        manifest = json.loads((Path(plan_dir) / "manifest.json").read_text())
        plans.append((Path(plan_dir), manifest, plan_chat_ids(plan_dir)))
        runtimes[f"plan {Path(plan_dir).name}"] = time.perf_counter() - t0
    raw: Dict[str, Row] = {}
    thinning: Dict[str, Row] = {p.name: {} for p, _, _ in plans}
    for workload in workloads:
        t0 = time.perf_counter()
        trace = load_trace(Path(raw_dir) / bailian.TRACES[workload], workload)
        t1 = time.perf_counter()
        analysis = RawAnalysis(trace)
        t2 = time.perf_counter()
        raw[workload] = raw_metrics(analysis, capacity=capacity)
        t3 = time.perf_counter()
        for plan_dir, manifest, ids in plans:
            if workload in manifest["workloads"]:
                thinning[plan_dir.name][workload] = thinning_for(
                    analysis, manifest, ids.get(workload, np.empty(0)))
        runtimes[workload] = {
            "load_s": t1 - t0,
            "keys_and_neighbours_s": t2 - t1,
            "metrics_s": t3 - t2,
            "thinning_s": time.perf_counter() - t3,
        }
        del analysis, trace
    mix: Dict[str, Row] = {}
    for run_dir in ceiling_run_dirs:
        t0 = time.perf_counter()
        run = load_mix(run_dir)
        t1 = time.perf_counter()
        mix[run.name] = mix_metrics(run, capacity=capacity)
        runtimes[f"mix {run.name}"] = {
            "load_s": t1 - t0,
            "metrics_s": time.perf_counter() - t1,
        }
        plan_name = Path(str(run.run.get("plan_dir", ""))).name
        for workload, row in thinning.get(plan_name, {}).items():
            row.setdefault("mix_ceiling_hit_ratio",
                           {})[run.name] = (mix[run.name].get(
                               workload, {}).get(1, {}).get("ideal_hit_ratio"))
    runtimes["total_s"] = time.perf_counter() - started
    return {
        "paper": "Wang et al., KVCache Cache in the Wild, USENIX ATC '25",
        "inputs": {
            "raw_dir": str(raw_dir),
            "ceiling_runs": [str(p) for p in ceiling_run_dirs],
            "plans": [str(p) for p, _, _ in plans],
            "guide_only_runs": [n for n, m in mix.items() if m["guide_only"]],
        },
        "settings": {
            "block_size": BLOCK_SIZE,
            "w0_s": W0_S,
            "horizon_s": HORIZON_S,
            "split_s": SPLIT_S,
            "capacity_blocks": capacity,
            "min_category_accesses": MIN_CATEGORY,
        },
        "rows": build_rows(raw, mix, sweep),
        "thinning": thinning,
        "runtimes_s": runtimes,
    }


def jsonable(value: Any) -> Any:
    """Numpy scalars to Python, NaN and infinities to None, keys to str."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def summary_lines(card: Row) -> List[str]:
    """The verdicts and the thinning, one line each."""
    lines = [f"{'claim':<5} {'traces A, B':<18} {'all four traces':<18} mix"]
    for row in card["rows"]:
        mix = ", ".join(f"{k}: {v}" for k, v in row["mix_verdict"].items())
        lines.append(f"{row['claim']:<5} {row['verdict']:<18} "
                     f"{row['all_traces_verdict']:<18} {mix or '-'}")
    for plan, workloads in card["thinning"].items():
        for workload, t in workloads.items():
            lines.append(
                f"thinning {plan} {workload}: kept "
                f"{t['kept_session_fraction']} of sessions; ideal hit ratio "
                f"{_fmt(t['full_rate']['ideal_hit_ratio'])} full rate, "
                f"{_fmt(t['kept']['ideal_hit_ratio'])} kept")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir",
                        type=Path,
                        default=Path("data/raw/bailian"))
    parser.add_argument("--ceiling",
                        type=Path,
                        action="append",
                        default=[],
                        help="A no-eviction run with tables/ (repeatable)")
    parser.add_argument("--plan",
                        type=Path,
                        action="append",
                        default=[],
                        help="A plan folder, for the thinning (repeatable)")
    parser.add_argument("--sweep",
                        type=Path,
                        help="JSON list of capacity-sweep rows (claim 10)")
    parser.add_argument("--traces",
                        default=",".join(bailian.TRACES),
                        help="Comma-separated Bailian workloads")
    parser.add_argument("--capacity-blocks", type=int, default=CAPACITY_BLOCKS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sweep = json.loads(args.sweep.read_text()) if args.sweep else None
    card = scorecard(args.raw_dir,
                     args.ceiling,
                     args.plan,
                     sweep,
                     workloads=[w for w in args.traces.split(",") if w],
                     capacity=args.capacity_blocks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(card), indent=1))
    print("\n".join(summary_lines(card)))
    print(f"Wrote {args.out} in {card['runtimes_s']['total_s']:.1f} s")


if __name__ == "__main__":
    main()
