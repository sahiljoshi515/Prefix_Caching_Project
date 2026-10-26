"""FIFO and LFU prefix-cache evictors (COMP 536 project).

vLLM v0 ships only ``LRUEvictor`` (``vllm/core/evictor.py``). This module adds
two more policies for the eviction study and lets ``VLLM_EVICTION_POLICY``
(``lru``, ``lfu`` or ``fifo``; unset means ``lru``) choose the evictor of the
GPU prefix-caching allocator. See ``specs/005-m2-cache-tuning``.

Both evictors order the blocks in the free pool (refcount 0) by a key, lowest
first:

- **FIFO**: ``(inserted, -num_hashed_tokens, block_id)``. ``inserted`` is the
  block's ``last_accessed`` time the first time it entered the free pool since
  it was computed. Reuse does not refresh it; eviction clears it, so a block
  computed again starts over.
- **LFU**: ``(hits, last_accessed, -num_hashed_tokens, block_id)``. ``hits``
  counts the allocator's lookup hits on the block (``record_hit``), including
  hits while it is in use, and is cleared on eviction. Ties go to the least
  recently used block.

The ``-num_hashed_tokens, block_id`` tail is LRU's own tie-break: among blocks
with equal keys the deepest one (most tokens before it) goes first, which keeps
the shared roots of prefix chains.

A physical block id keeps the same content from the time it is cached until it
is evicted, so per-block counters keyed by block id are per content hash.

Unlike ``LRUEvictor.update``, which changes a block's time without pushing a
new heap entry (the block then cannot be evicted until the next cleanup),
every key change here pushes a fresh entry. Stale entries are skipped when
popped and dropped when they outnumber live ones, so ``evict`` is O(log n)
amortized.
"""
import heapq
from typing import Dict, List, Optional, Tuple

import vllm.envs as envs
from vllm.core.evictor import EvictionPolicy, Evictor
from vllm.logger import init_logger

logger = init_logger(__name__)

POLICY_NAMES = {
    "lru": EvictionPolicy.LRU,
    "fifo": EvictionPolicy.FIFO,
    "lfu": EvictionPolicy.LFU,
}

_Key = Tuple[float, ...]


def eviction_policy_from_env() -> EvictionPolicy:
    """The policy named by ``VLLM_EVICTION_POLICY``; LRU when unset.

    Raises ``ValueError`` naming the valid values for anything else, so a
    misspelled policy stops the server at startup instead of silently
    running LRU.
    """
    value = envs.VLLM_EVICTION_POLICY
    name = "lru" if value is None or not value.strip() else value.strip()
    policy = POLICY_NAMES.get(name.lower())
    if policy is None:
        raise ValueError(f"VLLM_EVICTION_POLICY={value!r} is not a known "
                         f"eviction policy; use one of "
                         f"{', '.join(POLICY_NAMES)}.")
    logger.info("Prefix-cache eviction policy: %s", policy.name)
    return policy


class _FreeBlock:
    __slots__ = ("content_hash", "num_hashed_tokens", "last_accessed", "key")

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.key: _Key = ()


class _KeyedEvictor(Evictor):
    """Evicts the free block with the lowest ``_key``.

    The heap holds ``(key, block_id)`` entries. An entry is live while its
    block is in the free table with that same key; any other entry is stale.
    """

    # Rebuild the heap when it holds more than this many entries per free
    # block (plus a small floor so tiny pools do not rebuild constantly).
    REBUILD_FACTOR = 2
    REBUILD_MIN = 64

    def __init__(self) -> None:
        self.free_table: Dict[int, _FreeBlock] = {}
        self._heap: List[Tuple[_Key, int]] = []

    def _key(self, block_id: int, block: _FreeBlock) -> _Key:
        raise NotImplementedError

    def _on_add(self, block_id: int, last_accessed: float) -> None:
        """Hook run before a block's key is computed in ``add``."""

    def _on_evict(self, block_id: int) -> None:
        """Hook run when a block leaves the cache for good."""

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def _push(self, block_id: int) -> None:
        block = self.free_table[block_id]
        key = self._key(block_id, block)
        if key == block.key:
            return  # The existing entry is still live.
        block.key = key
        heapq.heappush(self._heap, (key, block_id))
        if len(self._heap) > max(self.REBUILD_MIN,
                                 self.REBUILD_FACTOR * len(self.free_table)):
            self._rebuild()

    def _rebuild(self) -> None:
        self._heap = [(block.key, block_id)
                      for block_id, block in self.free_table.items()]
        heapq.heapify(self._heap)

    def evict(self) -> Tuple[int, int]:
        while self._heap:
            key, block_id = heapq.heappop(self._heap)
            block = self.free_table.get(block_id)
            if block is not None and block.key == key:
                del self.free_table[block_id]
                self._on_evict(block_id)
                return block_id, block.content_hash
        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self._on_add(block_id, last_accessed)
        self.free_table[block_id] = _FreeBlock(content_hash, num_hashed_tokens,
                                               last_accessed)
        self._push(block_id)

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed
        self._push(block_id)

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        # Its heap entry goes stale; counters survive, since the block keeps
        # its content until it is evicted.
        del self.free_table[block_id]

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class FIFOEvictor(_KeyedEvictor):
    """Evicts the block that first entered the free pool earliest."""

    def __init__(self) -> None:
        super().__init__()
        # block id -> last_accessed when it first entered the free pool.
        self._inserted: Dict[int, float] = {}

    def _on_add(self, block_id: int, last_accessed: float) -> None:
        self._inserted.setdefault(block_id, last_accessed)

    def _on_evict(self, block_id: int) -> None:
        self._inserted.pop(block_id, None)

    def _key(self, block_id: int, block: _FreeBlock) -> _Key:
        return (self._inserted[block_id], -block.num_hashed_tokens, block_id)


class LFUEvictor(_KeyedEvictor):
    """Evicts the block with the fewest lookup hits, then the LRU one."""

    def __init__(self) -> None:
        super().__init__()
        # block id -> lookup hits since the block was computed.
        self._hits: Dict[int, int] = {}

    def record_hit(self, block_id: int) -> None:
        self._hits[block_id] = self._hits.get(block_id, 0) + 1
        if block_id in self.free_table:
            self._push(block_id)

    def hits(self, block_id: int) -> int:
        return self._hits.get(block_id, 0)

    def _on_evict(self, block_id: int) -> None:
        self._hits.pop(block_id, None)

    def _key(self, block_id: int, block: _FreeBlock) -> _Key:
        return (self._hits.get(block_id, 0), block.last_accessed,
                -block.num_hashed_tokens, block_id)


def make_project_evictor(policy: EvictionPolicy) -> Optional[Evictor]:
    """The evictor this module implements for ``policy``, or None."""
    if policy == EvictionPolicy.FIFO:
        return FIFOEvictor()
    if policy == EvictionPolicy.LFU:
        return LFUEvictor()
    return None
