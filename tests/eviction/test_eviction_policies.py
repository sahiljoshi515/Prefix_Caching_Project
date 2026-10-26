"""FIFO and LFU evictors (specs/005-m2-cache-tuning, US1).

Pure evictor tests: hand-built ``add`` / ``remove`` / ``update`` /
``record_hit`` / ``evict`` sequences with victims worked out by hand, plus a
randomized check against a brute-force reference of each policy's
definition (FR-003).
"""
import random

import pytest

from vllm.core.eviction_policies import (FIFOEvictor, LFUEvictor,
                                         eviction_policy_from_env)
from vllm.core.evictor import EvictionPolicy, LRUEvictor, make_evictor

pytestmark = pytest.mark.m2

A, B, C, D = 10, 11, 12, 13
TOKENS = 16  # num_hashed_tokens of a first block


def h(block_id):
    """A distinct content hash per block id."""
    return 1000 + block_id


def free(evictor, block_id, t, tokens=TOKENS):
    evictor.add(block_id, h(block_id), tokens, t)


def evict_order(evictor):
    order = []
    while evictor.num_blocks:
        block_id, content_hash = evictor.evict()
        assert content_hash == h(block_id)
        order.append(block_id)
    return order


def scenario_1(evictor):
    """Spec US1 scenario 1: A, B, C freed at t=1, 2, 3; A reused at t=4 and
    freed again at t=5."""
    free(evictor, A, 1)
    free(evictor, B, 2)
    free(evictor, C, 3)
    evictor.remove(A)  # A is used again...
    evictor.record_hit(A)
    free(evictor, A, 5)  # ...and freed with its new access time.
    return evictor


# --- The spec's acceptance scenarios ----------------------------------------


def test_scenario_1_lru_evicts_b():
    assert scenario_1(LRUEvictor()).evict()[0] == B


def test_scenario_1_fifo_evicts_a():
    assert scenario_1(FIFOEvictor()).evict()[0] == A


def test_scenario_2_lfu_evicts_the_older_of_the_least_hit():
    lfu = LFUEvictor()
    for _ in range(3):
        lfu.record_hit(A)
    lfu.record_hit(B)
    lfu.record_hit(C)
    free(lfu, A, 1)
    free(lfu, B, 2)
    free(lfu, C, 3)
    assert lfu.evict()[0] == B
    assert evict_order(lfu) == [C, A]


# --- FIFO -------------------------------------------------------------------


def test_fifo_order_is_first_free_time():
    fifo = FIFOEvictor()
    free(fifo, C, 1)
    free(fifo, A, 2)
    free(fifo, B, 3)
    assert evict_order(fifo) == [C, A, B]


def test_fifo_ignores_access_updates_and_reuse():
    fifo = FIFOEvictor()
    free(fifo, A, 1)
    free(fifo, B, 2)
    fifo.update(A, 9)
    fifo.remove(A)
    free(fifo, A, 10)
    assert evict_order(fifo) == [A, B]


def test_fifo_evict_then_recompute_starts_over():
    fifo = FIFOEvictor()
    free(fifo, A, 1)
    free(fifo, B, 2)
    assert fifo.evict()[0] == A
    # Block id A is reused for new content, computed and freed at t=3.
    free(fifo, A, 3)
    assert evict_order(fifo) == [B, A]


# --- LFU --------------------------------------------------------------------


def test_lfu_counts_hits_while_the_block_is_in_use():
    lfu = LFUEvictor()
    lfu.record_hit(A)  # A is shared by a running request, not yet freed.
    free(lfu, A, 1)
    free(lfu, B, 2)
    assert lfu.hits(A) == 1
    assert evict_order(lfu) == [B, A]


def test_lfu_hit_on_a_free_block_reorders_it():
    lfu = LFUEvictor()
    free(lfu, A, 1)
    free(lfu, B, 2)
    lfu.record_hit(A)
    assert evict_order(lfu) == [B, A]


def test_lfu_count_survives_reuse_and_resets_on_eviction():
    lfu = LFUEvictor()
    free(lfu, A, 1)
    lfu.record_hit(A)
    lfu.remove(A)
    free(lfu, A, 2)
    assert lfu.hits(A) == 1
    assert lfu.evict()[0] == A
    assert lfu.hits(A) == 0
    # Recomputed content in block A competes from zero hits.
    free(lfu, A, 3)
    free(lfu, B, 4)
    lfu.record_hit(B)
    assert evict_order(lfu) == [A, B]


def test_lfu_update_changes_the_lru_tie_break():
    lfu = LFUEvictor()
    free(lfu, A, 1)
    free(lfu, B, 2)
    lfu.update(A, 3)
    assert evict_order(lfu) == [B, A]


# --- Tie-breaks (FR-003) ----------------------------------------------------


@pytest.mark.parametrize("make", [LRUEvictor, FIFOEvictor, LFUEvictor])
def test_ties_go_to_the_deeper_block_then_the_lower_id(make):
    evictor = make()
    free(evictor, A, 1, tokens=16)
    free(evictor, D, 1, tokens=48)
    free(evictor, C, 1, tokens=32)
    free(evictor, B, 1, tokens=32)
    assert evict_order(evictor) == [D, B, C, A]


def test_lfu_ties_on_hits_go_to_the_least_recently_used():
    lfu = LFUEvictor()
    free(lfu, A, 2)
    free(lfu, B, 1)
    free(lfu, C, 3)
    assert evict_order(lfu) == [B, A, C]


# --- Evictor interface ------------------------------------------------------


@pytest.mark.parametrize("make", [FIFOEvictor, LFUEvictor])
def test_removed_blocks_are_not_evicted(make):
    evictor = make()
    free(evictor, A, 1)
    free(evictor, B, 2)
    evictor.remove(A)
    assert A not in evictor and B in evictor
    assert evictor.num_blocks == 1
    assert evictor.evict()[0] == B
    with pytest.raises(ValueError, match="No usable cache memory left"):
        evictor.evict()
    with pytest.raises(ValueError):
        evictor.remove(A)


@pytest.mark.parametrize("make", [FIFOEvictor, LFUEvictor])
def test_updated_blocks_stay_evictable(make):
    """FR-004: LRUEvictor.update leaves a block with no live heap entry."""
    evictor = make()
    free(evictor, A, 1)
    evictor.update(A, 5)
    evictor.record_hit(A)
    assert evictor.evict()[0] == A


@pytest.mark.parametrize("make", [FIFOEvictor, LFUEvictor])
def test_heap_stays_proportional_to_the_free_pool(make):
    """FR-005: stale entries are dropped, so evict stays O(log n)."""
    evictor = make()
    for block_id in range(100):
        free(evictor, block_id, block_id)
    for step in range(10_000):
        block_id = step % 100
        evictor.update(block_id, 100 + step)
        evictor.record_hit(block_id)
    assert len(evictor._heap) <= max(evictor.REBUILD_MIN,
                                     evictor.REBUILD_FACTOR * 100)
    assert len(evict_order(evictor)) == 100


def test_lru_record_hit_is_a_no_op():
    lru = LRUEvictor()
    free(lru, A, 1)
    free(lru, B, 2)
    lru.record_hit(A)
    assert evict_order(lru) == [A, B]


# --- Randomized check against the definitions -------------------------------


class Reference:
    """Brute-force FR-003: scan the free pool for the smallest key."""

    def __init__(self, policy):
        self.policy = policy
        self.free = {}  # block id -> [tokens, last_accessed]
        self.inserted = {}
        self.hits = {}

    def add(self, block_id, tokens, t):
        self.inserted.setdefault(block_id, t)
        self.free[block_id] = [tokens, t]

    def key(self, block_id):
        tokens, t = self.free[block_id]
        tail = (-tokens, block_id)
        if self.policy == "fifo":
            return (self.inserted[block_id], ) + tail
        if self.policy == "lfu":
            return (self.hits.get(block_id, 0), t) + tail
        return (t, ) + tail

    def evict(self):
        victim = min(self.free, key=self.key)
        del self.free[victim]
        self.inserted.pop(victim, None)
        self.hits.pop(victim, None)
        return victim


@pytest.mark.parametrize("policy,make", [("fifo", FIFOEvictor),
                                         ("lfu", LFUEvictor)])
@pytest.mark.parametrize("seed", range(5))
def test_matches_the_reference_on_random_operations(policy, make, seed):
    rng = random.Random(seed)
    evictor, ref = make(), Reference(policy)
    in_use = set(range(40))
    t = 0.0
    for _ in range(3000):
        t += rng.choice([0.0, 1.0])  # Equal times exercise the tie-breaks.
        op = rng.random()
        if op < 0.3 and in_use:
            block_id = rng.choice(sorted(in_use))
            tokens = 16 * rng.randint(1, 4)
            in_use.discard(block_id)
            evictor.add(block_id, h(block_id), tokens, t)
            ref.add(block_id, tokens, t)
        elif op < 0.45 and ref.free:
            block_id = rng.choice(sorted(ref.free))
            evictor.remove(block_id)
            del ref.free[block_id]
            in_use.add(block_id)
        elif op < 0.6 and ref.free:
            block_id = rng.choice(sorted(ref.free))
            evictor.update(block_id, t)
            ref.free[block_id][1] = t
        elif op < 0.8:
            block_id = rng.randrange(40)
            evictor.record_hit(block_id)
            ref.hits[block_id] = ref.hits.get(block_id, 0) + 1
        elif ref.free:
            victim = ref.evict()
            assert evictor.evict()[0] == victim
            in_use.add(victim)
        assert evictor.num_blocks == len(ref.free)


# --- Choosing the policy ----------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (None, EvictionPolicy.LRU),
    ("", EvictionPolicy.LRU),
    ("lru", EvictionPolicy.LRU),
    ("fifo", EvictionPolicy.FIFO),
    ("LFU", EvictionPolicy.LFU),
    (" lfu ", EvictionPolicy.LFU),
])
def test_policy_from_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VLLM_EVICTION_POLICY", raising=False)
    else:
        monkeypatch.setenv("VLLM_EVICTION_POLICY", value)
    assert eviction_policy_from_env() == expected


def test_unknown_policy_names_the_valid_values(monkeypatch):
    monkeypatch.setenv("VLLM_EVICTION_POLICY", "mru")
    with pytest.raises(ValueError, match="'mru'.*lru, fifo, lfu"):
        eviction_policy_from_env()


@pytest.mark.parametrize("policy,cls", [(EvictionPolicy.LRU, LRUEvictor),
                                        (EvictionPolicy.FIFO, FIFOEvictor),
                                        (EvictionPolicy.LFU, LFUEvictor)])
def test_make_evictor_dispatches(policy, cls):
    assert type(make_evictor(policy)) is cls
