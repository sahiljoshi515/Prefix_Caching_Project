"""Leaf-LRU controls of spec 007 FR-012 (``lru_leaf``, ``lru_leaf_free``).

Both evict an idle block with no cached block after it in a prompt (a
leaf), as TTDF and RACE-RP do. ``lru_leaf`` takes the leaf whose last
access (``record_access``) is oldest; ``lru_leaf_free`` the leaf whose
scheduler stamp (``last_accessed`` at ``add`` or ``update``) is oldest.
Ties go to the deeper block, then the lower block id.
"""
import pytest

from vllm.core.eviction_policies import POLICY_NAMES, make_project_evictor
from vllm.core.evictor import EvictionPolicy
from vllm.core.reuse_evictors import LeafLRUEvictor, LeafLRUFreeEvictor

pytestmark = [pytest.mark.m3, pytest.mark.usefixtures("request_context")]


class Clock:

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def h(block_id):
    return 1000 + block_id


def chain(evictor, *block_ids):
    """One request uses `block_ids` as a prompt, root to tip."""
    parent = None
    for depth, block_id in enumerate(block_ids, start=1):
        evictor.record_access(block_id, h(block_id), depth,
                              None if parent is None else h(parent))
        parent = block_id


def free(evictor, *block_ids, stamp=0.0):
    for block_id in block_ids:
        evictor.add(block_id, h(block_id), 16, stamp)


def test_evicts_the_least_recently_accessed_leaf():
    clock = Clock(0.0)
    evictor = LeafLRUEvictor(clock=clock)
    chain(evictor, 1, 2)  # prompt A at t=0
    clock.t = 5.0
    chain(evictor, 3, 4)  # prompt B at t=5
    free(evictor, 1, 2, 3, 4)
    clock.t = 10.0
    assert evictor.evict() == (2, h(2))  # A's leaf, oldest access


def test_never_evicts_a_block_before_the_blocks_after_it():
    clock = Clock(0.0)
    evictor = LeafLRUEvictor(clock=clock)
    chain(evictor, 1, 2, 3)
    clock.t = 5.0
    evictor.record_access(1, h(1), 1, None)  # a later hit on the root only
    free(evictor, 1, 2, 3)
    clock.t = 10.0
    # Plain LRU by access time would take 2 (t=0) and strand 3; the
    # control only ever takes a leaf, so the prompt is cut from its end.
    assert [evictor.evict()[0] for _ in range(3)] == [3, 2, 1]


def test_ties_go_to_the_deeper_block():
    clock = Clock(0.0)
    evictor = LeafLRUEvictor(clock=clock)
    chain(evictor, 1)
    chain(evictor, 2, 3)
    free(evictor, 1, 2, 3)
    assert evictor.evict()[0] == 3  # depth 2 before depth 1, same time


@pytest.mark.parametrize("cls, victim", [(LeafLRUEvictor, 1),
                                         (LeafLRUFreeEvictor, 2)])
def test_the_two_controls_differ_only_in_their_clock(cls, victim):
    clock = Clock(0.0)
    evictor = cls(clock=clock)
    chain(evictor, 1)
    clock.t = 5.0
    chain(evictor, 2)
    # Accessed in the order 1, 2, but freed with stamps in the order 2, 1.
    free(evictor, 1, stamp=9.0)
    free(evictor, 2, stamp=7.0)
    clock.t = 10.0
    assert evictor.evict()[0] == victim


def test_free_variant_follows_update():
    clock = Clock(0.0)
    evictor = LeafLRUFreeEvictor(clock=clock)
    chain(evictor, 1)
    chain(evictor, 2)
    free(evictor, 1, stamp=1.0)
    free(evictor, 2, stamp=2.0)
    evictor.update(1, 3.0)
    assert evictor.evict()[0] == 2


def test_registered_and_built_from_the_environment(monkeypatch):
    assert POLICY_NAMES["lru_leaf"] == EvictionPolicy.LRU_LEAF
    assert POLICY_NAMES["lru_leaf_free"] == EvictionPolicy.LRU_LEAF_FREE
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", "")
    assert isinstance(make_project_evictor(EvictionPolicy.LRU_LEAF),
                      LeafLRUEvictor)
    assert isinstance(make_project_evictor(EvictionPolicy.LRU_LEAF_FREE),
                      LeafLRUFreeEvictor)


def test_unknown_settings_are_refused():
    with pytest.raises(ValueError, match="tau_short"):
        LeafLRUEvictor.from_params("tau_short=5")
