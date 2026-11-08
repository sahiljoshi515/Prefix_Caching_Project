"""First version of the ``wa`` evictor (spec 007 Phase 2, prototype).

A hand-made trace teaches it two workloads: "fast" blocks are reused 1 s
after their last access, "slow" blocks are evicted at 30 s without reuse.
With oracle labels a young slow block must go before a young fast one; with
no labels both share one class and the age decides.
"""
import pytest

from vllm.core.block import cache_events
from vllm.core.eviction_policies import POLICY_NAMES, make_project_evictor
from vllm.core.evictor import EvictionPolicy
from vllm.core.wa_evictor import WorkloadAwareEvictor

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


def use(evictor, block_id, workload):
    """A request of `workload` accesses the block, then frees it."""
    cache_events.set_current_request(f"cmpl-{workload}:r{block_id}-0")
    evictor.record_access(block_id, h(block_id), 1, None)
    evictor.add(block_id, h(block_id), 16, evictor.clock())


def trained(labels):
    clock = Clock()
    evictor = WorkloadAwareEvictor(labels=labels,
                                   kappa=0.0,
                                   buckets=False,
                                   clock=clock)
    for k in range(20):  # fast: reused 1 s later (then kept in use)
        use(evictor, k, "fast")
        clock.t += 1.0
        evictor.remove(k)
    for k in range(100, 120):  # slow: evicted 30 s later, never reused
        use(evictor, k, "slow")
    clock.t += 30.0
    for _ in range(20):
        evictor.evict()
    return evictor, clock


@pytest.mark.parametrize("labels, victim", [("oracle", 200), ("none", 201)])
def test_labels_decide_between_two_young_blocks(labels, victim):
    evictor, clock = trained(labels)
    use(evictor, 200, "slow")  # idle 0.6 s at eviction
    clock.t += 0.1
    use(evictor, 201, "fast")  # idle 0.5 s at eviction
    clock.t += 0.5
    # Oracle: slow blocks never earned a hit, so the slow block goes first
    # although it is older only by 0.1 s. None: one class, whose density
    # falls as the remaining lifetime grows, so the younger block goes.
    assert evictor.evict()[0] == victim


def test_learned_densities():
    evictor, _ = trained("oracle")
    assert evictor.density(("slow", 0), 0.5) == 0.0
    # 20 hits at age 1 over 20 ends: 20 / (20 * 1 - 0.5 * 20) = 2 per s.
    assert evictor.density(("fast", 0), 0.5) == pytest.approx(2.0)
    # A fast block idle past its return time has no hits left to expect.
    assert evictor.density(("fast", 0), 5.0) == 0.0


def test_settings_and_registration(monkeypatch):
    assert POLICY_NAMES["wa"] == EvictionPolicy.WA
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", "labels=none,half_life=60")
    evictor = make_project_evictor(EvictionPolicy.WA)
    assert isinstance(evictor, WorkloadAwareEvictor)
    assert not evictor.use_labels and evictor.half_life == 60.0
    with pytest.raises(ValueError, match="labels"):
        WorkloadAwareEvictor(labels="inferred")
    with pytest.raises(ValueError, match="tau"):
        WorkloadAwareEvictor.from_params("tau=1")
