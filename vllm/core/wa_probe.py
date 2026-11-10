"""Exploratory probe for spec 007 Phase 2: do workload labels help eviction?

Not the ``wa`` policy (spec 007 FR-003); a quick experiment run before its
design is reviewed (T005). ``wa_probe`` evicts, among the same idle leaf
blocks as TTDF and RACE-RP (``LeafScoreEvictor``), the block least likely
to be accessed again within the next ``horizon`` seconds, given its label
and how long it has gone without an access (its age ``a``):

    P(access within (a, a + horizon] | none by a) = 1 - S(a + horizon) / S(a)

``S`` is a survival curve of the time between a block's accesses, read
from a table built offline on the training seed
(``analysis/hazard_table.py``). With ``labels=oracle`` each block uses the
curve of the workload of the last request that accessed it
(``cache_events.workload_label``); with ``labels=none`` every block uses
the pooled curve. Comparing the two on held-out seeds says whether the
label carries signal beyond the age.

Settings (``VLLM_EVICTION_PARAMS``): ``table`` (required), ``horizon``
(seconds, default 30), ``labels`` (``oracle`` or ``none``), ``stats``.
"""
import json
from bisect import bisect_right
from typing import Callable, Dict, List, Optional, Sequence

from vllm.core.block import cache_events
from vllm.core.reuse_evictors import LeafScoreEvictor, _Block, parse_params

DEFAULTS = {"table": "", "horizon": "30", "labels": "oracle"}
LABEL_MODES = ("oracle", "none")


class SurvivalTable:
    """Step survival curves on shared age bins, per label and pooled."""

    def __init__(self, ages: Sequence[float], curves: Dict[str,
                                                           Sequence[float]],
                 pooled: Sequence[float]) -> None:
        if not ages or any(
                len(c) != len(ages) for c in [pooled, *curves.values()]):
            raise ValueError("Every survival curve needs one value per age")
        self.ages = list(ages)
        self.curves = {k: list(v) for k, v in curves.items()}
        self.pooled = list(pooled)

    @classmethod
    def load(cls, path: str) -> "SurvivalTable":
        with open(path) as f:
            data = json.load(f)
        return cls(data["ages"], data["survival"], data["pooled"])

    def survival(self, label: Optional[str], age: float) -> float:
        curve = self.curves.get(label, self.pooled) if label else self.pooled
        i = bisect_right(self.ages, age) - 1
        return 1.0 if i < 0 else curve[i]

    def reuse_within(self, label: Optional[str], age: float,
                     horizon: float) -> float:
        now = self.survival(label, age)
        if now <= 0.0:
            return 0.0
        return 1.0 - self.survival(label, age + horizon) / now


class HazardProbeEvictor(LeafScoreEvictor):
    """Evicts the idle leaf least likely to be accessed within `horizon`."""

    def __init__(self,
                 table: SurvivalTable,
                 horizon: float = 30.0,
                 labels: str = "oracle",
                 clock: Optional[Callable[[], float]] = None,
                 stats_path: Optional[str] = None) -> None:
        if labels not in LABEL_MODES:
            raise ValueError(f"wa_probe labels must be one of "
                             f"{', '.join(LABEL_MODES)}, not {labels!r}")
        if horizon <= 0:
            raise ValueError("wa_probe horizon must be positive")
        super().__init__(clock, stats_path)
        self.table = table
        self.horizon = horizon
        self.use_labels = labels == "oracle"
        self.labels: Dict[int, Optional[str]] = {}  # content hash -> label

    @classmethod
    def from_params(cls, text: Optional[str]) -> "HazardProbeEvictor":
        params = parse_params(text, DEFAULTS)
        if not params.get("table"):
            raise ValueError("wa_probe needs table=<file> in "
                             "VLLM_EVICTION_PARAMS")
        return cls(SurvivalTable.load(params["table"]),
                   float(params.get("horizon", DEFAULTS["horizon"])),
                   params.get("labels", DEFAULTS["labels"]),
                   stats_path=params.get("stats"))

    def _on_access(self, content_hash: int, block: _Block, now: float) -> None:
        if self.use_labels:
            label = cache_events.workload_label(cache_events.current_request())
            if label is not None:
                self.labels[content_hash] = label

    def _on_evict(self, content_hash: int, block: _Block, now: float) -> None:
        self.labels.pop(content_hash, None)

    def _scores(self, hashes: Sequence[int], blocks: Sequence[_Block],
                now: float) -> List[float]:
        return [
            self.table.reuse_within(self.labels.get(h), now - b.last,
                                    self.horizon)
            for h, b in zip(hashes, blocks)
        ]

    def metadata_words(self) -> Dict[str, int]:
        return {**super().metadata_words(), "labels": len(self.labels)}
