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
  workload prefix) or one label for every block (``labels=none``). ``hits``
  is the number of accesses since the block was computed, capped at 2
  (``buckets=0`` turns it off).
- **Statistics**, learned online (FR-010): every idle period ends in a hit
  (the block is reused, ``remove``) or an eviction. For each class and
  log-spaced age bin the evictor keeps the number of ends, the sum of their
  ages and the number of hits, all decayed with half-life ``half_life``
  seconds.
- **Hit density** of a class at age ``a``, over the ends at ages ``>= a``::

      HD(a) = hits / (sum of end ages - a * ends)

  A class with few ends is shrunk toward the pooled statistics, as if it
  had ``kappa`` more ends distributed like all of them.

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
from vllm.core.wa_probe import workload_of

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
ENDS, AGES, HITS = 0, 1, 2
MAX_HITS_BUCKET = 2
# Below this many seconds of remaining lifetime the density is capped.
MIN_LIFETIME_S = 1e-3

ClassKey = Tuple[str, int]


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
        self.pooled = np.zeros((3, NUM_BINS))
        self._last_decay: Optional[float] = None
        self._version = 0
        self._suffix_version = -1
        self._suffix: Dict[ClassKey, np.ndarray] = {}
        self._suffix_pooled = np.zeros((3, NUM_BINS))

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
            stats = self.classes[key] = np.zeros((3, NUM_BINS))
        for target in (stats, self.pooled):
            target[ENDS, i] += 1.0
            target[AGES, i] += age
            if hit:
                target[HITS, i] += 1.0
        self._version += 1

    def _suffixes(self) -> None:
        if self._suffix_version == self._version:
            return
        self._suffix = {
            key: np.cumsum(stats[:, ::-1], axis=1)[:, ::-1]
            for key, stats in self.classes.items()
        }
        self._suffix_pooled = np.cumsum(self.pooled[:, ::-1], axis=1)[:, ::-1]
        self._suffix_version = self._version

    def density(self, key: ClassKey, age: float) -> float:
        """Expected hits per second of remaining cache time, for a block of
        class `key` at `age`."""
        self._suffixes()
        i = bisect_right(EDGES, age)
        own = self._suffix.get(key)
        ends, ages, hits = (0.0, 0.0, 0.0) if own is None else own[:, i]
        pooled_total = self._suffix_pooled[ENDS, 0]
        if pooled_total > 0 and self.kappa > 0:
            weight = self.kappa / pooled_total
            ends += weight * self._suffix_pooled[ENDS, i]
            ages += weight * self._suffix_pooled[AGES, i]
            hits += weight * self._suffix_pooled[HITS, i]
        if hits <= 0.0:
            return 0.0
        return hits / max(ages - age * ends, MIN_LIFETIME_S)

    # --- Evictor hooks -----------------------------------------------------

    def _on_access(self, content_hash: int, block: _Block, now: float) -> None:
        if content_hash in self.hits:
            self.hits[content_hash] += 1
        else:
            self.hits[content_hash] = 0
        if self.use_labels:
            label = workload_of(cache_events.current_request())
            if label is not None:
                self.labels[content_hash] = label

    def remove(self, block_id: int):
        # The allocator takes an idle block out to reuse it: a hit ends the
        # idle period (recorded before the new access changes the class).
        content_hash = self.free_table.get(block_id)
        if content_hash is not None:
            self._record_end(content_hash,
                             self.blocks[content_hash],
                             self.clock(),
                             hit=True)
        super().remove(block_id)

    def _on_evict(self, content_hash: int, block: _Block, now: float) -> None:
        self._record_end(content_hash, block, now, hit=False)
        self.labels.pop(content_hash, None)
        self.hits.pop(content_hash, None)

    def _scores(self, hashes: Sequence[int], blocks: Sequence[_Block],
                now: float) -> List[float]:
        return [
            self.density(self.class_of(h), now - b.last)
            for h, b in zip(hashes, blocks)
        ]

    def metadata_words(self) -> Dict[str, int]:
        return {
            **super().metadata_words(), "labels": len(self.labels),
            "hits": len(self.hits),
            "statistics": 3 * NUM_BINS * (len(self.classes) + 1)
        }
