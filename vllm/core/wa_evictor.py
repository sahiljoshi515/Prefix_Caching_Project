"""Workload-aware eviction (``wa``), first version for spec 007 Phase 2.

A prototype of ``plan.md``'s class hit-density design, before its review
(spec 007 FR-004, T005). Among the idle leaf blocks (``LeafScoreEvictor``,
so the cache stays prefix-closed, FR-005) it evicts the block with the
lowest *hit density*: the hits a block of its class is expected to get per
second of cache time it still holds, at its current age (Beckmann et al.,
LHD, NSDI 2018).

- **Age**: seconds since the block's last access, as in TTDF and RACE-RP.
- **Class**: ``(label, hits)``. The label is the workload of the last
  request that accessed the block (``labels=oracle``: the request id's
  workload prefix, ``cache_events.workload_label``) or one label for every
  block (``labels=none``). ``hits`` is the number of lookup hits on the
  block since it was computed, capped at 2 (``buckets=0`` turns it off).
- **Statistics**, learned online (FR-010): every idle period ends in a hit
  (a lookup takes the idle block out to reuse it) or an eviction. A dedupe,
  where a request computed the block itself and the allocator then swaps
  in the cached copy, ends the period without a hit: the cache saved no
  work. The allocator tells the two apart by calling ``record_hit`` just
  before a lookup takes the block. For each class and log-spaced age bin
  the evictor keeps the number of ends, the sum of their ages and the
  number of hits, all decayed with half-life ``half_life`` seconds.
- **Hit density** of a class at age ``a``, over the ends at ages ``>= a``::

      HD(a) = hits / (sum of end ages - a * ends)

  The bins after ``a``'s bin hold only older ends. The ends in ``a``'s own
  bin are taken together at their mean age: counted if that mean is at
  least ``a``, left out otherwise, so the remaining lifetime is never
  negative. A class with few ends is shrunk toward the pooled statistics,
  as if it had ``kappa`` more ends distributed like all of them.
- **Fresh scores**: every recorded end changes the statistics, so the
  candidates are scored again before the next eviction, even within one
  scheduler step.

Settings (``VLLM_EVICTION_PARAMS``): ``labels`` (``oracle`` or ``none``),
``half_life`` (seconds, default 600), ``kappa`` (default 20), ``buckets``
(1 or 0) and ``stats``.
"""
import math
from bisect import bisect_right
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vllm.core.block import cache_events
from vllm.core.reuse_evictors import LeafScoreEvictor, _Block, parse_params

DEFAULTS = {
    "labels": "oracle",
    "half_life": "600",
    "kappa": "20",
    "buckets": "1"
}
LABEL_MODES = ("oracle", "none")
# Upper edges of the age bins: 50 ms to 1 h, log-spaced; one more bin above.
EDGES: List[float] = [
    float(e) for e in np.logspace(math.log10(0.05), math.log10(3600.0), 48)
]
NUM_BINS = len(EDGES) + 1
# Rows of a statistics array: the ends, end ages and hits in each bin, then
# the same summed over the bins after it.
ENDS, AGES, HITS = 0, 1, 2
LATER = 3
MAX_HITS_BUCKET = 2
# Below this many seconds of remaining lifetime the density is capped.
MIN_LIFETIME_S = 1e-3

ClassKey = Tuple[str, int]


def new_statistics() -> np.ndarray:
    return np.zeros((6, NUM_BINS))


class WorkloadAwareEvictor(LeafScoreEvictor):
    """Evicts the idle leaf with the lowest class hit density at its age."""

    def __init__(self,
                 labels: str = "oracle",
                 half_life: float = 600.0,
                 kappa: float = 20.0,
                 buckets: bool = True,
                 clock: Optional[Callable[[], float]] = None,
                 stats_path: Optional[str] = None) -> None:
        if labels not in LABEL_MODES:
            raise ValueError(f"wa labels must be one of "
                             f"{', '.join(LABEL_MODES)}, not {labels!r}")
        if half_life <= 0 or kappa < 0:
            raise ValueError("wa needs half_life > 0 and kappa >= 0")
        super().__init__(clock, stats_path)
        self.use_labels = labels == "oracle"
        self.half_life = half_life
        self.kappa = kappa
        self.buckets = buckets
        self.labels: Dict[int, str] = {}  # content hash -> workload
        self.hits: Dict[int, int] = {}  # content hash -> hits since computed
        self.classes: Dict[ClassKey, np.ndarray] = {}
        self.pooled = new_statistics()
        self._last_decay: Optional[float] = None
        # The block id of the lookup hit being served (`record_hit`).
        self._hit_block: Optional[int] = None

    @classmethod
    def from_params(cls, text: Optional[str]) -> "WorkloadAwareEvictor":
        params = parse_params(text, DEFAULTS)
        return cls(labels=params.get("labels", DEFAULTS["labels"]),
                   half_life=float(
                       params.get("half_life", DEFAULTS["half_life"])),
                   kappa=float(params.get("kappa", DEFAULTS["kappa"])),
                   buckets=params.get("buckets", DEFAULTS["buckets"]) != "0",
                   stats_path=params.get("stats"))

    # --- Classes and statistics -------------------------------------------

    def class_of(self, content_hash: int) -> ClassKey:
        label = self.labels.get(content_hash, "") if self.use_labels else ""
        hits = (min(self.hits.get(content_hash, 0), MAX_HITS_BUCKET)
                if self.buckets else 0)
        return label, hits

    def _decay(self, now: float) -> None:
        """Decay the statistics to `now`, in steps of 1/16 half-life; for a
        given `now` it does the same whenever it is called."""
        if self._last_decay is None:
            self._last_decay = now
            return
        dt = now - self._last_decay
        if dt < self.half_life / 16:
            return
        factor = 0.5**(dt / self.half_life)
        for stats in self.classes.values():
            stats *= factor
        self.pooled *= factor
        self._last_decay = now

    def _record_end(self, content_hash: int, block: _Block, now: float,
                    hit: bool) -> None:
        """An idle period ended at its age: in a hit or an eviction."""
        age = max(0.0, now - block.last)
        self._decay(now)
        i = bisect_right(EDGES, age)
        key = self.class_of(content_hash)
        stats = self.classes.get(key)
        if stats is None:
            stats = self.classes[key] = new_statistics()
        end = np.array([[1.0], [age], [1.0 if hit else 0.0]])
        for target in (stats, self.pooled):
            target[:LATER, i:i + 1] += end
            target[LATER:, :i] += end
        # Every candidate's score may have changed: score them again.
        self._heap_time = None

    def density(self, key: ClassKey, age: float) -> float:
        """Expected hits per second of remaining cache time, for a block of
        class `key` at `age`."""
        i = bisect_right(EDGES, age)
        own = self.classes.get(key)
        if own is None:
            ends, ages, hits, later_ends, later_ages, later_hits = (0.0, ) * 6
        else:
            (ends, ages, hits, later_ends, later_ages,
             later_hits) = own[:, i].tolist()
        pooled_total = float(self.pooled[ENDS, 0] +
                             self.pooled[LATER + ENDS, 0])
        if pooled_total > 0 and self.kappa > 0:
            weight = self.kappa / pooled_total
            pooled = self.pooled[:, i].tolist()
            ends += weight * pooled[ENDS]
            ages += weight * pooled[AGES]
            hits += weight * pooled[HITS]
            later_ends += weight * pooled[LATER + ENDS]
            later_ages += weight * pooled[LATER + AGES]
            later_hits += weight * pooled[LATER + HITS]
        # The bin's own ends count only if, at their mean age, they are at
        # least as old as the block.
        if ends > 0.0 and ages >= age * ends:
            later_ends += ends
            later_ages += ages
            later_hits += hits
        if later_hits <= 0.0:
            return 0.0
        return later_hits / max(later_ages - age * later_ends, MIN_LIFETIME_S)

    # --- Evictor hooks -----------------------------------------------------

    def record_hit(self, block_id: int) -> None:
        # The allocator calls this on a lookup hit, before it takes the
        # block out of the evictor (`remove`) and records the access.
        self._hit_block = block_id

    def _on_access(self, content_hash: int, block: _Block, now: float) -> None:
        hit = self._hit_block is not None and self._hit_block == block.block_id
        self._hit_block = None
        if content_hash not in self.hits:
            self.hits[content_hash] = 0
        elif hit:
            self.hits[content_hash] += 1
        if self.use_labels:
            label = cache_events.workload_label(cache_events.current_request())
            if label is not None:
                self.labels[content_hash] = label

    def remove(self, block_id: int):
        # The allocator takes an idle block out to use it: a lookup hit, or
        # a dedupe of a block the request computed itself. Either ends the
        # idle period (recorded before the access changes the class).
        content_hash = self.free_table.get(block_id)
        if content_hash is not None:
            self._record_end(content_hash,
                             self.blocks[content_hash],
                             self.clock(),
                             hit=self._hit_block == block_id)
        super().remove(block_id)

    def _on_evict(self, content_hash: int, block: _Block, now: float) -> None:
        self._record_end(content_hash, block, now, hit=False)
        self.labels.pop(content_hash, None)
        self.hits.pop(content_hash, None)

    def _scores(self, hashes: Sequence[int], blocks: Sequence[_Block],
                now: float) -> List[float]:
        self._decay(now)
        return [
            self.density(self.class_of(h), now - b.last)
            for h, b in zip(hashes, blocks)
        ]

    def metadata_words(self) -> Dict[str, int]:
        return {
            **super().metadata_words(), "labels": len(self.labels),
            "hits": len(self.hits),
            "statistics": 6 * NUM_BINS * (len(self.classes) + 1)
        }
