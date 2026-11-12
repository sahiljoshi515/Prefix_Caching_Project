"""Sampled LRFU prefix-cache evictor (COMP 536 project, Milestone 3).

LRFU fuses LRU and LFU: each cached block i keeps a reuse score s_i and
the logical time tau_i of its last access. On an access at logical time t,

    s_i <- 2 ** (-(t - tau_i) / H) * s_i + 1,    tau_i <- t,

and a newly cached block starts with s_i = 1, tau_i = t. Its score at
time T is S_i(T) = 2 ** (-(T - tau_i) / H) * s_i: every past access counts,
halving every H accesses. A very large half-life H gives LFU (counts); a
very small one gives LRU (the latest access dominates).

- **Logical clock.** One step per block-access event (the allocator's
  ``record_access``: a lookup hit, an insertion, a deduplication), in the
  order the block manager makes them. A request counts each block at most
  once (a recomputed request is not counted twice): each of the last
  `REQUEST_MEMORY` requests remembers the blocks it counted, by block id
  and the time the block was cached. An access not counted does not move
  the clock.
- **Eviction.** Up to k blocks are drawn uniformly, without replacement,
  from the blocks no request is using (refcount 0); the one with the lowest
  S_i(T) goes, ties to the lower block id. The draw is ranked by
  log2(s_i) + tau_i / H, which orders blocks as S_i(T) does (the factor
  2 ** (-T / H) is common to all) but cannot underflow for a long-idle
  block. The candidates sit in an array
  with a block-id-to-position map and swap-removal, so adding, removing and
  sampling are O(1) and nothing is scanned, sorted or aged in bulk.
- **Prefix sharing** is vLLM's, as for LRU, LFU and FIFO: any idle block
  can be drawn.

Settings come from ``VLLM_EVICTION_PARAMS``: ``half_life`` (default 512
accesses), ``sample_size`` (4), ``seed`` (0, the policy's own random
generator) and ``stats`` (a file its counters are written to at exit).
"""
import atexit
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from vllm.core.block import cache_events
from vllm.core.evictor import Evictor
from vllm.core.reuse_evictors import parse_params

DEFAULTS = {"half_life": 512.0, "sample_size": 4, "seed": 0}
# How many recent requests remember the blocks they counted (a preempted
# request comes back within a few hundred requests).
REQUEST_MEMORY = 4096


class SampledLRFUEvictor(Evictor):
    """Evicts the lowest-scoring of up to `sample_size` random idle blocks.
    """

    def __init__(self,
                 half_life: float = DEFAULTS["half_life"],
                 sample_size: int = DEFAULTS["sample_size"],
                 seed: int = DEFAULTS["seed"],
                 stats_path: Optional[str] = None,
                 request: Optional[Callable[[], Optional[str]]] = None):
        if not half_life > 0:
            raise ValueError("Sampled LRFU's half_life must be positive")
        if sample_size < 1:
            raise ValueError("Sampled LRFU's sample_size must be at least 1")
        # The block manager then names each call's request.
        cache_events.enable_request_context()
        self.half_life = float(half_life)
        self.sample_size = int(sample_size)
        self.rng = random.Random(seed)
        self.request = request or cache_events.current_request
        self.now = 0  # the logical clock: block-access events so far
        # Per cached block id: s_i, tau_i, and the time it was cached.
        self.score: Dict[int, float] = {}
        self.stamp: Dict[int, int] = {}
        self._born: Dict[int, int] = {}
        # Per recent request: the (block id, time cached) it counted.
        self._counted: OrderedDict[str, Set[Tuple[int, int]]] = (OrderedDict())
        self._counted_entries = 0
        # The candidates (refcount 0): an array, each block's position in
        # it, and its content hash.
        self._pool: List[int] = []
        self._pos: Dict[int, int] = {}
        self._hash: Dict[int, int] = {}
        self.stats: Dict[str, float] = {
            "evictions": 0,
            "scans": 0,
            "candidates_scored": 0,
            "max_candidates": 0,
            "evict_seconds": 0.0,
            "max_resident_blocks": 0,
        }
        if stats_path:
            atexit.register(self.write_stats, stats_path)

    @classmethod
    def from_params(cls, text: Optional[str]) -> "SampledLRFUEvictor":
        params = parse_params(text, DEFAULTS)
        return cls(
            half_life=float(params.get("half_life", DEFAULTS["half_life"])),
            sample_size=int(params.get("sample_size",
                                       DEFAULTS["sample_size"])),
            seed=int(params.get("seed", DEFAULTS["seed"])),
            stats_path=params.get("stats"))

    def current(self, block_id: int, now: Optional[int] = None) -> float:
        """S_i(now), decayed lazily from the block's last access."""
        now = self.now if now is None else now
        return (2.0**(-(now - self.stamp[block_id]) / self.half_life) *
                self.score[block_id])

    def rank(self, block_id: int) -> float:
        """log2(S_i(T)) + T / H, the same for every T: orders blocks as
        S_i(T) does without underflowing (s_i >= 1 after any access)."""
        return (math.log2(self.score[block_id]) +
                self.stamp[block_id] / self.half_life)

    # --- Evictor interface -------------------------------------------------

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._pos

    @property
    def num_blocks(self) -> int:
        return len(self._pool)

    def record_access(self, block_id: int, content_hash: int, depth: int,
                      prev_hash: Optional[int]) -> None:
        request = self.request()
        known = block_id in self.score
        if known and request is not None:
            counted = self._counted.get(request)
            if counted is not None and (block_id,
                                        self._born[block_id]) in counted:
                return  # this request already counted the block
        self.now += 1
        if known:
            self.score[block_id] = self.current(block_id) + 1.0
        else:
            self.score[block_id] = 1.0
            self._born[block_id] = self.now
            self.stats["max_resident_blocks"] = max(
                self.stats["max_resident_blocks"], len(self.score))
        self.stamp[block_id] = self.now
        if request is not None:
            self._remember(request, (block_id, self._born[block_id]))

    def _remember(self, request: str, block: Tuple[int, int]) -> None:
        counted = self._counted.get(request)
        if counted is None:
            counted = self._counted[request] = set()
            if len(self._counted) > REQUEST_MEMORY:
                _, oldest = self._counted.popitem(last=False)
                self._counted_entries -= len(oldest)
        else:
            self._counted.move_to_end(request)
        counted.add(block)
        self._counted_entries += 1

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        if block_id not in self.score:
            raise ValueError(f"Block {block_id} was freed without an access "
                             "(record_access) since it was cached")
        if block_id not in self._pos:
            self._pos[block_id] = len(self._pool)
            self._pool.append(block_id)
        self._hash[block_id] = content_hash

    def update(self, block_id: int, last_accessed: float):
        """The scheduler's access stamp; the score does not use it."""
        if block_id not in self._pos:
            raise ValueError("Updating a block that is not in the evictor")

    def remove(self, block_id: int):
        if block_id not in self._pos:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self._take(block_id)

    def evict(self) -> Tuple[int, int]:
        if not self._pool:
            raise ValueError("No usable cache memory left")
        start = time.perf_counter()
        n = len(self._pool)
        m = min(self.sample_size, n)
        candidates = [self._pool[i] for i in self.rng.sample(range(n), m)]
        victim = min(candidates, key=lambda b: (self.rank(b), b))
        content_hash = self._take(victim)
        del self.score[victim], self.stamp[victim], self._born[victim]
        self.stats["evictions"] += 1
        self.stats["scans"] += 1
        self.stats["candidates_scored"] += m
        self.stats["max_candidates"] = max(self.stats["max_candidates"], m)
        self.stats["evict_seconds"] += time.perf_counter() - start
        return victim, content_hash

    def _take(self, block_id: int) -> int:
        """Swap-remove `block_id` from the candidates; its content hash."""
        i = self._pos.pop(block_id)
        last = self._pool.pop()
        if last != block_id:
            self._pool[i] = last
            self._pos[last] = i
        return self._hash.pop(block_id)

    # --- Overhead ---------------------------------------------------------

    def metadata_words(self) -> Dict[str, int]:
        """Values the policy stores (8 bytes each in a compact layout)."""
        return {
            # s_i and tau_i per cached block.
            "lrfu":
            2 * len(self.score),
            # Pool slot, position and content hash per candidate, each
            # cached block's caching time, and the recent requests' counted
            # (block id, time) pairs.
            "bookkeeping": (3 * len(self._pool) + len(self._born) +
                            2 * self._counted_entries),
        }

    def summary(self) -> Dict[str, Any]:
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
