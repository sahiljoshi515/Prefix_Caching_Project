"""TTDF and RACE-RP prefix-cache evictors (COMP 536 project, Milestone 3).

Two policies for a cache shared by several workloads that rank idle blocks
by a score of their estimated future reuse and evict the lowest. Neither
reads workload labels.

- **TTDF** (two-timescale decayed frequency): each resident block keeps
  ``c_short``, ``c_long`` and ``last_update``. On every access at time t
  both counters decay by ``exp(-(t - last_update) / tau)`` and grow by 1.
  The score is ``w_short / tau_short * C_s + w_long / tau_long * C_l``,
  with both counters decayed to the eviction time.
- **RACE-RP** (RACE-based reuse probability): the score of a block is the
  smoothed ratio ``(W + alpha * r_bar) / (N + alpha)`` read from two RACE
  sketches (``RaceSketch``) at the block's feature vector: accesses in the
  last 10 s and 120 s, seconds since its last access, its depth, and the
  requests in the last 120 s that share its first two blocks, each as
  ``log(1 + value)``. The sketches are trained offline on seed 1
  (``analysis/race_train.py``) and frozen.

What both share (``LeafScoreEvictor``):

- An *access* is a request using the block: a lookup hit, or the block's
  insertion or deduplication once full (the allocator's ``record_access``),
  so a block computed again after an eviction counts as accessed.
- Times are the block manager's event times
  (``cache_events.event_time``): every access and eviction of one
  ``allocate`` or ``append_slots`` call has the time the call began, and
  the cache simulator's calls have its simulated times.
- Only *leaves* can be evicted: idle blocks (refcount 0, in the evictor)
  with no cached child. A block's prefix ancestors are therefore never
  evicted before it, and blocks in use are never candidates.
- Victim: the lowest score, then the deepest block, then the oldest last
  access, then the lowest block id.
- At eviction time every candidate is scored (a full scan). The scan is
  kept, as a heap, for the rest of the call: scores at a fixed time change
  only when a candidate is accessed (it then leaves the evictor), when a
  block becomes a leaf (it is scored and pushed) or, for RACE-RP, when a
  second block is accessed (a family count changes; the next eviction
  rescans). The heap order is that of a fresh scan.

Settings come from ``VLLM_EVICTION_PARAMS`` (``key=value,...``):
``tau_short``, ``tau_long``, ``w_short``, ``w_long`` (TTDF); ``sketch``
(the trained sketch's JSON file, required) and ``alpha`` (RACE-RP); and
``stats``, a file both write their counters to at exit. See
``specs/007-m3-mixed-workload-policy``.
"""
import atexit
import heapq
import json
import math
import time
from bisect import bisect_right
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)

import numpy as np

from vllm.core.block import cache_events
from vllm.core.evictor import Evictor

TTDF_DEFAULTS = {
    "tau_short": 5.0,
    "tau_long": 120.0,
    "w_short": 0.7,
    "w_long": 0.3
}
RACE_DEFAULTS = {"alpha": 5.0}
# Feature windows of RACE-RP and how long an evicted block's access times
# are kept.
SHORT_S = 10.0
LONG_S = 120.0
GHOST_S = 180.0
NUM_FEATURES = 5
# Standard deviations at or below this count as zero (replaced by 1).
STD_FLOOR = 1e-9
FEATURES = ("log1p_f10", "log1p_f120", "log1p_idle_s", "log1p_depth",
            "log1p_family_120")


def parse_params(text: Optional[str],
                 allowed: Iterable[str]) -> Dict[str, str]:
    """``key=value,...`` as a dict; a key outside `allowed` (and ``stats``)
    is an error, so a misspelled setting stops the server."""
    allowed = set(allowed) | {"stats"}
    params: Dict[str, str] = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or key not in allowed:
            raise ValueError(f"VLLM_EVICTION_PARAMS: {item.strip()!r} is not "
                             f"one of {', '.join(sorted(allowed))} "
                             "given as key=value")
        params[key] = value.strip()
    return params


class _Block:
    """A cached block, in use or idle, keyed by its content hash."""
    __slots__ = ("depth", "parent", "family", "children", "last", "block_id",
                 "idle", "version", "c_short", "c_long")

    def __init__(self, depth: int, parent: Optional[int],
                 family: Optional[int]) -> None:
        self.depth = depth
        self.parent = parent
        # The block among its first two that names its family (RACE-RP).
        self.family = family
        self.children = 0  # cached blocks whose parent this is
        self.last = 0.0  # time of the last access
        self.block_id: Optional[int] = None
        self.idle = False
        # Set from the evictor's counter whenever anything a heap entry
        # depends on changes, so no two states of any block share a value.
        self.version = 0
        self.c_short = 0.0
        self.c_long = 0.0


_Entry = Tuple[float, int, float, int, int, int]


class LeafScoreEvictor(Evictor):
    """Evicts the idle leaf block with the lowest ``_scores`` value."""

    def __init__(self,
                 clock: Optional[Callable[[], float]] = None,
                 stats_path: Optional[str] = None) -> None:
        # The block manager then sets each call's time and request.
        cache_events.enable_request_context()
        self.clock = clock or cache_events.event_time
        self.blocks: Dict[int, _Block] = {}  # content hash -> block
        self.free_table: Dict[int, int] = {}  # idle block id -> content hash
        # The candidates: idle blocks with no cached child, by content hash.
        self.leaves: Dict[int, _Block] = {}
        self._heap: List[_Entry] = []
        self._heap_time: Optional[float] = None  # None: rescan first
        self._tick = 0  # the last version handed out
        self.stats: Dict[str, float] = {
            "evictions": 0,
            "scans": 0,
            "candidates_scored": 0,
            "max_candidates": 0,
            "evict_seconds": 0.0,
            "score_seconds": 0.0,
            "max_resident_blocks": 0,
        }
        if stats_path:
            atexit.register(self.write_stats, stats_path)

    # --- Scores (subclasses) ---------------------------------------------

    def _scores(self, hashes: Sequence[int], blocks: Sequence[_Block],
                now: float) -> List[float]:
        raise NotImplementedError

    def _on_access(self, content_hash: int, block: _Block, now: float) -> None:
        """Update the block's own counters for an access at `now`."""

    def _on_evict(self, content_hash: int, block: _Block, now: float) -> None:
        """The block left the cache."""

    def _rescan_after(self, block: _Block) -> bool:
        """Whether an access to `block` changes other candidates' scores."""
        return False

    def metadata_words(self) -> Dict[str, int]:
        """Values the policy stores (8 bytes each in a compact layout)."""
        return {"dependency": 4 * len(self.blocks)}

    def _bump(self, block: _Block) -> None:
        """Invalidate `block`'s heap entries. Versions come from one counter
        for the whole evictor, so a block evicted and cached again at the
        same time cannot match an entry of its earlier self."""
        self._tick += 1
        block.version = self._tick

    # --- Evictor interface -------------------------------------------------

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def record_access(self, block_id: int, content_hash: int, depth: int,
                      prev_hash: Optional[int]) -> None:
        now = self.clock()
        block = self.blocks.get(content_hash)
        if block is None:
            parent = None if prev_hash is None else self.blocks.get(prev_hash)
            if depth <= 2:
                family: Optional[int] = content_hash
            else:
                family = None if parent is None else parent.family
            # Linked only to a cached parent (the allocator always has one:
            # a request holds a block's whole prefix).
            block = self.blocks[content_hash] = _Block(
                depth, None if parent is None else prev_hash, family)
            block.last = now
            if parent is not None:
                parent.children += 1
                self._bump(parent)
                self.leaves.pop(prev_hash, None)  # type: ignore[arg-type]
            self.stats["max_resident_blocks"] = max(
                self.stats["max_resident_blocks"], len(self.blocks))
        self._on_access(content_hash, block, now)
        block.last = now
        block.block_id = block_id
        self._bump(block)
        if self._rescan_after(block):
            self._heap_time = None
        elif block.idle and block.children == 0:
            # Not from the allocator, which removes a block before using
            # it; keep the heap whole anyway.
            self._keep_heap(content_hash, block, now)

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        block = self.blocks.get(content_hash)
        if block is None:
            raise ValueError(f"Block {block_id} was freed without an access "
                             "(record_access) since it was cached")
        self.free_table[block_id] = content_hash
        block.block_id = block_id
        block.idle = True
        self._bump(block)
        if block.children == 0:
            self.leaves[content_hash] = block
            self._keep_heap(content_hash, block)

    def update(self, block_id: int, last_accessed: float):
        """The scheduler's access stamp; neither score uses it (an access
        is a request using the block, `record_access`)."""
        if block_id not in self.free_table:
            raise ValueError("Updating a block that is not in the evictor")

    def remove(self, block_id: int):
        content_hash = self.free_table.pop(block_id, None)
        if content_hash is None:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block = self.blocks[content_hash]
        block.idle = False
        self._bump(block)
        self.leaves.pop(content_hash, None)

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")
        start = time.perf_counter()
        now = self.clock()
        if self._heap_time != now:
            self._rescan(now)
        entry = self._pop_live()
        if entry is None:
            # Unreachable while the cache is prefix-closed (every idle block
            # has an idle leaf below it); rescan once before giving up.
            self._rescan(now)
            entry = self._pop_live()
            if entry is None:
                raise ValueError("No evictable leaf block among the "
                                 f"{len(self.free_table)} idle blocks")
        block_id, content_hash = entry[3], entry[5]
        block = self.blocks.pop(content_hash)
        del self.free_table[block_id]
        del self.leaves[content_hash]
        self._on_evict(content_hash, block, now)
        parent = (None
                  if block.parent is None else self.blocks.get(block.parent))
        if parent is not None:
            parent.children -= 1
            self._bump(parent)
            if parent.idle and parent.children == 0:
                self.leaves[block.parent] = parent  # type: ignore[index]
                self._push(block.parent, parent, now)  # type: ignore[arg-type]
        self.stats["evictions"] += 1
        self.stats["evict_seconds"] += time.perf_counter() - start
        return block_id, content_hash

    # --- Scanning ------------------------------------------------------------

    def _entry(self, content_hash: int, block: _Block, score: float) -> _Entry:
        assert block.block_id is not None
        return (score, -block.depth, block.last, block.block_id, block.version,
                content_hash)

    def _keep_heap(self,
                   content_hash: int,
                   block: _Block,
                   now: Optional[float] = None) -> None:
        """A new candidate: pushed onto the kept heap if that heap is for
        this time; otherwise the heap is dropped, so a clock that steps back
        to the heap's time cannot reuse a heap without this block."""
        if self._heap_time is None:
            return
        now = self.clock() if now is None else now
        if now == self._heap_time:
            self._push(content_hash, block, now)
        else:
            self._heap_time = None

    def _push(self, content_hash: int, block: _Block, now: float) -> None:
        score = self._scores([content_hash], [block], now)[0]
        self.stats["candidates_scored"] += 1
        heapq.heappush(self._heap, self._entry(content_hash, block, score))

    def _rescan(self, now: float) -> None:
        start = time.perf_counter()
        hashes = list(self.leaves)
        blocks = list(self.leaves.values())
        scores = self._scores(hashes, blocks, now) if hashes else []
        self._heap = [
            self._entry(h, b, s) for h, b, s in zip(hashes, blocks, scores)
        ]
        heapq.heapify(self._heap)
        self._heap_time = now
        self.stats["scans"] += 1
        self.stats["candidates_scored"] += len(hashes)
        self.stats["max_candidates"] = max(self.stats["max_candidates"],
                                           len(hashes))
        self.stats["score_seconds"] += time.perf_counter() - start

    def _pop_live(self) -> Optional[_Entry]:
        while self._heap:
            entry = heapq.heappop(self._heap)
            block = self.leaves.get(entry[5])
            if (block is not None and block.version == entry[4]
                    and self.free_table.get(entry[3]) == entry[5]):
                return entry
        return None

    # --- Overhead ---------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Counters, mean costs and metadata size, for reports."""
        stats: Dict[str, Any] = dict(self.stats)
        evictions = stats["evictions"]
        stats["evict_us_mean"] = (1e6 * stats["evict_seconds"] /
                                  evictions if evictions else None)
        stats["candidates_per_scan"] = (stats["candidates_scored"] /
                                        stats["scans"]
                                        if stats["scans"] else None)
        words = self.metadata_words()
        stats["metadata_words"] = words
        stats["metadata_bytes"] = 8 * sum(words.values())
        return stats

    def write_stats(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2))


# --- TTDF --------------------------------------------------------------------


class TTDFEvictor(LeafScoreEvictor):
    """Two-timescale decayed frequency."""

    def __init__(self,
                 tau_short: float = TTDF_DEFAULTS["tau_short"],
                 tau_long: float = TTDF_DEFAULTS["tau_long"],
                 w_short: float = TTDF_DEFAULTS["w_short"],
                 w_long: float = TTDF_DEFAULTS["w_long"],
                 clock: Optional[Callable[[], float]] = None,
                 stats_path: Optional[str] = None) -> None:
        if tau_short <= 0 or tau_long <= 0:
            raise ValueError("TTDF time constants must be positive")
        super().__init__(clock, stats_path)
        self.tau_short = tau_short
        self.tau_long = tau_long
        self.w_short = w_short
        self.w_long = w_long

    @classmethod
    def from_params(cls, text: Optional[str]) -> "TTDFEvictor":
        params = parse_params(text, TTDF_DEFAULTS)
        values = {k: float(params.get(k, v)) for k, v in TTDF_DEFAULTS.items()}
        return cls(**values, stats_path=params.get("stats"))

    def _on_access(self, content_hash: int, block: _Block, now: float) -> None:
        dt = now - block.last
        block.c_short = block.c_short * math.exp(-dt / self.tau_short) + 1.0
        block.c_long = block.c_long * math.exp(-dt / self.tau_long) + 1.0

    def score(self, block: _Block, now: float) -> float:
        """S_TTDF of `block` at `now`, its counters decayed lazily."""
        dt = now - block.last
        return (self.w_short / self.tau_short * block.c_short *
                math.exp(-dt / self.tau_short) + self.w_long / self.tau_long *
                block.c_long * math.exp(-dt / self.tau_long))

    def _scores(self, hashes: Sequence[int], blocks: Sequence[_Block],
                now: float) -> List[float]:
        return [self.score(b, now) for b in blocks]

    def metadata_words(self) -> Dict[str, int]:
        # c_short, c_long, last_update per resident block.
        return {**super().metadata_words(), "counters": 3 * len(self.blocks)}


# --- Leaf-LRU control --------------------------------------------------------


class LeafLRUEvictor(LeafScoreEvictor):
    """``lru_leaf``: LRU over the same idle leaves as TTDF and RACE-RP, by
    the same access clock (``record_access``). Only the score differs from
    them, so a policy's gain over this control is its score's (spec 007
    FR-012)."""

    @classmethod
    def from_params(cls, text: Optional[str]) -> "LeafLRUEvictor":
        params = parse_params(text, ())
        return cls(stats_path=params.get("stats"))

    def _scores(self, hashes: Sequence[int], blocks: Sequence[_Block],
                now: float) -> List[float]:
        return [b.last for b in blocks]


# --- RACE-RP -----------------------------------------------------------------


class AccessHistory:
    """Access times of prefix-block identities, cached or recently evicted.

    Each identity keeps its access times of the last `window_s` seconds
    before its latest access, and that latest time. An identity that left
    the cache is forgotten once it has not been accessed for `ghost_s`
    seconds; a cached one never is.
    """

    def __init__(self,
                 window_s: float = LONG_S,
                 ghost_s: float = GHOST_S) -> None:
        self.window_s = window_s
        self.ghost_s = ghost_s
        self.times: Dict[int, List[float]] = {}
        self.last: Dict[int, float] = {}
        self._ghosts: List[Tuple[float, int]] = []  # (forget after, hash)
        self.num_times = 0
        self.max_times = 0
        self.max_identities = 0

    def record(self, content_hash: int, now: float) -> None:
        times = self.times.get(content_hash)
        if times is None:
            times = self.times[content_hash] = []
            self.max_identities = max(self.max_identities, len(self.times))
        times.append(now)
        old = bisect_right(times, now - self.window_s)
        if old:
            del times[:old]
        self.num_times += 1 - old
        self.max_times = max(self.max_times, self.num_times)
        self.last[content_hash] = now

    def count(self, content_hash: int, now: float, span: float) -> int:
        """Accesses in (now - span, now]."""
        times = self.times.get(content_hash)
        if not times:
            return 0
        return bisect_right(times, now) - bisect_right(times, now - span)

    def evicted(self, content_hash: int) -> None:
        last = self.last.get(content_hash)
        if last is not None:
            heapq.heappush(self._ghosts, (last + self.ghost_s, content_hash))

    def expire(self, now: float, cached: Dict[int, Any]) -> None:
        """Forget evicted identities idle for more than `ghost_s`."""
        while self._ghosts and self._ghosts[0][0] < now:
            _, content_hash = heapq.heappop(self._ghosts)
            last = self.last.get(content_hash)
            if (content_hash not in cached and last is not None
                    and last + self.ghost_s < now):
                self.num_times -= len(self.times.pop(content_hash))
                del self.last[content_hash]

    def features(self, content_hash: int, depth: int, family: Optional[int],
                 now: float) -> Tuple[float, ...]:
        """x_b(now). The family count is that of the block among its first
        two (`family`; the block itself at depth 1 or 2): each request uses
        a block once, so its accesses are the requests sharing those
        blocks."""
        last = self.last.get(content_hash, now)
        fam = 0 if family is None else self.count(family, now, LONG_S)
        return (math.log1p(self.count(content_hash, now, SHORT_S)),
                math.log1p(self.count(content_hash, now, LONG_S)),
                math.log1p(max(now - last,
                               0.0)), math.log1p(depth), math.log1p(fam))


class RaceSketch:
    """Random-hyperplane hashes and the two RACE count arrays.

    `planes` is (L, P, 5): table l's bucket for a feature vector x is
    ``sum_j 2**j * [planes[l, j] . z >= 0]`` (j from 0), where z is x
    standardized by `mean` and `std` and scaled to unit length (zero stays
    zero). `counts` (N) and `rewards` (W) are (L, 2**P).
    """

    def __init__(self,
                 planes: np.ndarray,
                 mean: np.ndarray,
                 std: np.ndarray,
                 counts: Optional[np.ndarray] = None,
                 rewards: Optional[np.ndarray] = None,
                 r_bar: float = 0.0,
                 alpha: float = RACE_DEFAULTS["alpha"],
                 meta: Optional[Dict[str, Any]] = None) -> None:
        self.planes = np.asarray(planes, dtype=np.float64)
        if self.planes.ndim != 3 or self.planes.shape[2] != NUM_FEATURES:
            raise ValueError("planes must have shape (L, P, 5)")
        self.L, self.P = self.planes.shape[:2]
        self.mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        # A constant feature's computed std can be ~1e-15 instead of 0.
        self.std = np.where(std <= STD_FLOOR, 1.0, std)
        shape = (self.L, 2**self.P)
        self.counts = (np.zeros(shape) if counts is None else np.asarray(
            counts, dtype=np.float64))
        self.rewards = (np.zeros(shape) if rewards is None else np.asarray(
            rewards, dtype=np.float64))
        if self.counts.shape != shape or self.rewards.shape != shape:
            raise ValueError(f"count arrays must have shape {shape}")
        self.r_bar = float(r_bar)
        self.alpha = float(alpha)
        if not self.alpha > 0:
            raise ValueError("RACE-RP's alpha must be positive (an empty "
                             "bucket's score would be 0/0)")
        self.meta = dict(meta or {})
        self._weights = 2**np.arange(self.P)
        self._flat = self.planes.reshape(self.L * self.P, NUM_FEATURES)
        self._offsets = np.arange(self.L) * 2**self.P

    @classmethod
    def random(cls, L: int, P: int, seed: int, mean: np.ndarray,
               std: np.ndarray, **kwargs: Any) -> "RaceSketch":
        """Empty sketch with R_l,ij ~ N(0, 1) drawn from `seed`."""
        planes = np.random.default_rng(seed).standard_normal(
            (L, P, NUM_FEATURES))
        return cls(planes, mean, std, **kwargs)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(np.asarray(x, dtype=np.float64)) -
             self.mean) / self.std
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        return np.divide(z, norms, out=np.zeros_like(z), where=norms > 0)

    def buckets(self, x: np.ndarray) -> np.ndarray:
        """(n, L) bucket of each row of `x` in each table."""
        z = self.normalize(x)
        bits = (z @ self._flat.T >= 0).reshape(len(z), self.L, self.P)
        return bits @ self._weights

    def add(self, x: np.ndarray, r: np.ndarray) -> None:
        """Add training observations: N += 1 and W += r at their buckets."""
        cells = (self.buckets(x) + self._offsets).ravel()
        r = np.repeat(np.asarray(r, dtype=np.float64), self.L)
        size = self.L * 2**self.P
        self.counts += np.bincount(cells,
                                   minlength=size).reshape(self.counts.shape)
        self.rewards += np.bincount(cells, weights=r,
                                    minlength=size).reshape(self.rewards.shape)

    def query(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(N_hat, W_hat): the mean over tables of each row's counts."""
        cells = self.buckets(x) + self._offsets
        return (self.counts.ravel()[cells].mean(axis=1),
                self.rewards.ravel()[cells].mean(axis=1))

    def score(self, x: np.ndarray) -> np.ndarray:
        n, w = self.query(x)
        return (w + self.alpha * self.r_bar) / (n + self.alpha)

    def to_json(self) -> Dict[str, Any]:
        return {
            "L": self.L,
            "P": self.P,
            "features": list(FEATURES),
            "planes": self.planes.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "counts": self.counts.tolist(),
            "rewards": self.rewards.tolist(),
            "r_bar": self.r_bar,
            "alpha": self.alpha,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "RaceSketch":
        return cls(np.array(data["planes"]), np.array(data["mean"]),
                   np.array(data["std"]), np.array(data["counts"]),
                   np.array(data["rewards"]), data["r_bar"], data["alpha"],
                   data.get("meta"))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path: Path) -> "RaceSketch":
        return cls.from_json(json.loads(Path(path).read_text()))


class RaceRPEvictor(LeafScoreEvictor):
    """RACE-based reuse probability, from frozen training sketches."""

    def __init__(self,
                 sketch: RaceSketch,
                 clock: Optional[Callable[[], float]] = None,
                 stats_path: Optional[str] = None) -> None:
        super().__init__(clock, stats_path)
        self.sketch = sketch
        self.history = AccessHistory()

    @classmethod
    def from_params(cls, text: Optional[str]) -> "RaceRPEvictor":
        params = parse_params(text, ("sketch", "alpha"))
        if not params.get("sketch"):
            raise ValueError(
                "race_rp needs VLLM_EVICTION_PARAMS=sketch=<file> "
                "(analysis/race_train.py writes it)")
        sketch = RaceSketch.load(Path(params["sketch"]))
        if "alpha" in params:
            sketch.alpha = float(params["alpha"])
            if not sketch.alpha > 0:
                raise ValueError("race_rp: alpha must be positive")
        return cls(sketch, stats_path=params.get("stats"))

    def _on_access(self, content_hash: int, block: _Block, now: float) -> None:
        self.history.expire(now, self.blocks)
        self.history.record(content_hash, now)

    def _on_evict(self, content_hash: int, block: _Block, now: float) -> None:
        self.history.evicted(content_hash)

    def _rescan_after(self, block: _Block) -> bool:
        # A second block's access changes its family's counts.
        return block.depth == 2

    def features(self, content_hash: int, block: _Block,
                 now: float) -> Tuple[float, ...]:
        return self.history.features(content_hash, block.depth, block.family,
                                     now)

    def _scores(self, hashes: Sequence[int], blocks: Sequence[_Block],
                now: float) -> List[float]:
        x = np.array(
            [self.features(h, b, now) for h, b in zip(hashes, blocks)])
        return self.sketch.score(x).tolist()

    def metadata_words(self) -> Dict[str, int]:
        s = self.sketch
        return {
            **super().metadata_words(),
            # family hash per block; per identity, its last access time and
            # its access times.
            "family":
            len(self.blocks),
            "history":
            len(self.history.last) + self.history.num_times,
            "sketch": (2 * s.L * 2**s.P + s.L * s.P * NUM_FEATURES +
                       2 * NUM_FEATURES + 2),
        }

    def summary(self) -> Dict[str, Any]:
        stats = super().summary()
        stats["max_history_times"] = self.history.max_times
        stats["max_history_identities"] = self.history.max_identities
        return stats
