"""The exploratory ``wa_probe`` evictor and its survival table."""
import json

import pytest

from analysis.events import load_events
from analysis.hazard_table import access_gaps
from vllm.core.block import cache_events
from vllm.core.wa_probe import HazardProbeEvictor, SurvivalTable, workload_of

pytestmark = pytest.mark.m3


@pytest.fixture(autouse=True)
def _restore_request_context(monkeypatch):
    monkeypatch.setattr(cache_events._context, "enabled",
                        cache_events._context.enabled)
    monkeypatch.setattr(cache_events._context, "request_id",
                        cache_events._context.request_id)


class Clock:

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def h(block_id):
    return 1000 + block_id


# Ages 0, 10, 100 s. "fast" blocks come back within 10 s or never; "slow"
# blocks come back between 10 and 100 s.
TABLE = SurvivalTable([0.0, 10.0, 100.0], {
    "fast": [1.0, 0.2, 0.2],
    "slow": [1.0, 1.0, 0.1],
},
                      pooled=[1.0, 0.6, 0.15])


def test_workload_of_reads_the_request_id_prefix():
    assert workload_of("chatcmpl-sharegpt:abc:3") == "sharegpt"
    assert workload_of("cmpl-bailian-chat:x-0") == "bailian-chat"
    assert workload_of("no-colon") is None
    assert workload_of(None) is None


def test_reuse_within_is_conditional_on_the_age():
    # A fast block idle 10 s has had its chance: none of the rest return.
    assert TABLE.reuse_within("fast", 10.0, 50.0) == pytest.approx(0.0)
    # A slow block idle 10 s returns within 90 s with probability 0.9.
    assert TABLE.reuse_within("slow", 10.0, 90.0) == pytest.approx(0.9)
    # An unknown label uses the pooled curve.
    assert TABLE.reuse_within("other", 0.0, 10.0) == pytest.approx(0.4)


def access(evictor, clock, t, block_id, request):
    clock.t = t
    cache_events.set_current_request(request)
    evictor.record_access(block_id, h(block_id), 1, None)


def two_blocks(labels):
    clock = Clock()
    evictor = HazardProbeEvictor(TABLE,
                                 horizon=100.0,
                                 labels=labels,
                                 clock=clock)
    access(evictor, clock, 0.0, 2, "chatcmpl-slow:b:0")  # idle 20 s
    access(evictor, clock, 15.0, 1, "chatcmpl-fast:a:0")  # idle 5 s
    evictor.add(2, h(2), 16, 0.0)
    evictor.add(1, h(1), 16, 15.0)
    clock.t = 20.0
    return evictor


def test_oracle_labels_evict_the_block_that_will_not_return():
    # The fast block, idle 5 s, still has its 10 s chance in the table but
    # nothing after; within 100 s: 1 - S(105)/S(5) = 1 - 0.2 = 0.8. The
    # slow block, idle 20 s: 1 - S(120)/S(20) = 1 - 0.1 = 0.9. The fast
    # block goes although it is the more recent one, so LRU would keep it.
    assert two_blocks("oracle").evict()[0] == 1


def test_no_labels_use_the_pooled_curve_for_every_block():
    evictor = two_blocks("none")
    assert evictor.labels == {}
    scores = evictor._scores([h(1), h(2)],
                             [evictor.blocks[h(1)], evictor.blocks[h(2)]],
                             20.0)
    assert scores == [
        TABLE.reuse_within(None, 5.0, 100.0),
        TABLE.reuse_within(None, 20.0, 100.0)
    ]


def test_access_gaps_censor_blocks_that_never_return(tmp_path):
    # Through the real reader, which yields Event tuples, not dicts.
    lines = [
        {
            "ev": "header",
            "t": 0.0,
            "time": 0.0,
            "block_size": 16
        },
        {
            "t": 0.0,
            "ev": "insert",
            "hash": 1,
            "block": 0,
            "req": "cmpl-a:x-0",
            "op": "allocate"
        },
        {
            "t": 4.0,
            "ev": "hit",
            "hash": 1,
            "block": 0,
            "req": "cmpl-b:y-0",
            "op": "allocate"
        },
        {
            "t": 5.0,
            "ev": "insert",
            "hash": 2,
            "block": 1,
            "req": "cmpl-a:z-0",
            "op": "allocate"
        },
        {
            "t": 9.0,
            "ev": "free",
            "hash": 2,
            "block": 1,
            "req": None,
            "op": None
        },
        {
            "ev": "end"
        },
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    durations, observed = access_gaps(load_events(path).events)
    assert durations["a"] == [4.0, 4.0] and observed["a"] == [True, False]
    assert durations["b"] == [5.0] and observed["b"] == [False]


def test_table_file_round_trip(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps({
            "ages": TABLE.ages,
            "survival": TABLE.curves,
            "pooled": TABLE.pooled
        }))
    assert SurvivalTable.load(str(path)).survival("fast", 50.0) == 0.2
