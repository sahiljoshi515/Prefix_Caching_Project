"""S3-FIFO over the idle leaf blocks (``s3fifo_leaf``, spec 007 Phase 2).

S3-FIFO (Yang et al., "FIFO queues are all you need for cache eviction",
SOSP 2023) keeps every cached object in one of two FIFO queues: a small
queue S that new objects enter, and a main queue M. A hit does not move an
object; it only raises the object's counter. When the cache needs room:

- while S holds at least ``small_ratio`` of the cached objects, S's oldest
  object leaves S: to M if it was hit at least ``threshold`` times while
  in S, otherwise out of the cache, its key going to a ghost queue;
- otherwise M's oldest object leaves: back to M's head if it was hit since
  it was last examined (its counter, capped at 3, drops by one: a 2-bit
  clock), otherwise out of the cache.

An object inserted while its key is in the ghost queue (S evicted it
recently) goes straight to M. The ghost queue keeps the keys of
``ghost_ratio`` times as many objects as the cache holds. The defaults are
those of the authors' libCacheSim implementation (``small_ratio=0.1``,
``ghost_ratio=0.9``, ``threshold=2``); like TTDF's and RACE-RP's, they are
specified, not tuned on seed 1.

Adapted to vLLM's prefix cache through ``LeafScoreEvictor``:

- The queues hold every cached block, in use or idle, by content hash, but
  only idle leaves can be evicted, as for TTDF, RACE-RP, ``lru_leaf`` and
  ``wa``. The two hands look only at those candidates: a block in use, or
  one with a cached block after it, is passed over and keeps its place and
  counter.
- Every access to a cached block (``record_access``: a lookup hit or a
  dedupe) raises its counter. A block computed again after an eviction is
  a new object.
- The evictor does not know the cache's capacity in blocks, so it uses
  the most blocks it has seen cached at once (``max_resident_blocks``):
  S's share and the ghost queue's size (rounded up) are of that.

Settings (``VLLM_EVICTION_PARAMS``): ``small_ratio``, ``ghost_ratio``,
``threshold`` and ``stats``.
"""
import heapq
import math
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence

from vllm.core.reuse_evictors import (LeafScoreEvictor, _Block, _Entry,
                                      parse_params)

DEFAULTS = {"small_ratio": "0.1", "ghost_ratio": "0.9", "threshold": "2"}
SMALL, MAIN = 0, 1
MAX_FREQ = 3  # the counter's cap in M (2 bits)


class _State:
    """A cached block's queue, its place there (lower is older) and its
    counter."""
    __slots__ = ("queue", "seq", "freq")

    def __init__(self, queue: int, seq: int) -> None:
        self.queue = queue
        self.seq = seq
        self.freq = 0


class S3FIFOLeafEvictor(LeafScoreEvictor):
    """S3-FIFO's queues, evicting only idle leaves."""

    def __init__(self,
                 small_ratio: float = 0.1,
                 ghost_ratio: float = 0.9,
                 threshold: int = 2,
                 clock: Optional[Callable[[], float]] = None,
                 stats_path: Optional[str] = None) -> None:
        if not 0 < small_ratio < 1 or ghost_ratio < 0 or threshold < 1:
            raise ValueError("s3fifo_leaf needs 0 < small_ratio < 1, "
                             "ghost_ratio >= 0 and threshold >= 1")
        super().__init__(clock, stats_path)
        self.small_ratio = small_ratio
        self.ghost_ratio = ghost_ratio
        self.threshold = threshold
        self.state: Dict[int, _State] = {}  # content hash -> queue state
        self.small_count = 0
        self.ghost: OrderedDict[int, None] = OrderedDict()
        self._seq = 0
        self.stats.update({
            "moved_to_main": 0,
            "reinserted": 0,
            "ghost_hits": 0
        })

    @classmethod
    def from_params(cls, text: Optional[str]) -> "S3FIFOLeafEvictor":
        params = parse_params(text, DEFAULTS)
        return cls(small_ratio=float(
            params.get("small_ratio", DEFAULTS["small_ratio"])),
                   ghost_ratio=float(
                       params.get("ghost_ratio", DEFAULTS["ghost_ratio"])),
                   threshold=int(params.get("threshold",
                                            DEFAULTS["threshold"])),
                   stats_path=params.get("stats"))

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def capacity(self) -> int:
        """The cache's size in blocks, as far as the evictor has seen."""
        return max(int(self.stats["max_resident_blocks"]), len(self.blocks))

    # --- Queues -------------------------------------------------------------

    def _on_access(self, content_hash: int, block: _Block, now: float) -> None:
        state = self.state.get(content_hash)
        if state is not None:
            state.freq += 1
        elif content_hash in self.ghost:
            del self.ghost[content_hash]
            self.state[content_hash] = _State(MAIN, self._next_seq())
            self.stats["ghost_hits"] += 1
        else:
            self.state[content_hash] = _State(SMALL, self._next_seq())
            self.small_count += 1

    def _on_evict(self, content_hash: int, block: _Block, now: float) -> None:
        state = self.state.pop(content_hash)
        if state.queue == SMALL:
            self.small_count -= 1
            self.ghost[content_hash] = None
            limit = math.ceil(self.ghost_ratio * self.capacity())
            while len(self.ghost) > limit:
                self.ghost.popitem(last=False)

    def _hand(self, queue: int) -> Optional[int]:
        """Run `queue`'s hand over its candidates, oldest first; the
        victim's content hash, or None if every candidate moved on."""
        heap = [(self.state[h].seq, h) for h in self.leaves
                if self.state[h].queue == queue]
        heapq.heapify(heap)
        self.stats["scans"] += 1
        self.stats["candidates_scored"] += len(heap)
        self.stats["max_candidates"] = max(self.stats["max_candidates"],
                                           len(heap))
        while heap:
            _, content_hash = heapq.heappop(heap)
            state = self.state[content_hash]
            if queue == SMALL:
                if state.freq < self.threshold:
                    return content_hash
                state.queue, state.seq, state.freq = MAIN, self._next_seq(), 0
                self.small_count -= 1
                self.stats["moved_to_main"] += 1
            else:
                if state.freq < 1:
                    return content_hash
                state.freq = min(state.freq, MAX_FREQ) - 1
                state.seq = self._next_seq()
                heapq.heappush(heap, (state.seq, content_hash))
                self.stats["reinserted"] += 1
        return None

    # --- LeafScoreEvictor hooks: the hands replace the score heap ------------

    def _rescan(self, now: float) -> None:
        self._heap_time = now

    def _push(self, content_hash: int, block: _Block, now: float) -> None:
        """Candidates are read from ``leaves`` at each eviction."""

    def _pop_live(self) -> Optional[_Entry]:
        small_first = self.small_count >= self.small_ratio * self.capacity()
        for queue in ((SMALL, MAIN) if small_first else (MAIN, SMALL)):
            content_hash = self._hand(queue)
            if content_hash is not None:
                block = self.blocks[content_hash]
                assert block.block_id is not None
                return (0.0, -block.depth, block.last, block.block_id,
                        block.version, content_hash)
        return None

    def _scores(self, hashes: Sequence[int], blocks: Sequence[_Block],
                now: float) -> List[float]:
        return [0.0] * len(hashes)

    def metadata_words(self) -> Dict[str, int]:
        return {
            **super().metadata_words(), "queues": 3 * len(self.state),
            "ghost": len(self.ghost)
        }
