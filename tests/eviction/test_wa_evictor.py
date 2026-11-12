"""First version of the ``wa`` evictor (spec 007 Phase 2, prototype).

A hand-made trace teaches it two workloads: "fast" blocks are reused 1 s
after their last access, "slow" blocks are evicted at 30 s without reuse.
With oracle labels a young slow block must go before a young fast one; with
no labels both share one class and the age decides. Hits go through
``record_hit`` before ``remove``, as the allocator calls them.
"""
from bisect import bisect_right

import pytest

from vllm.core.block import cache_events
from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.eviction_policies import POLICY_NAMES, make_project_evictor
from vllm.core.evictor import EvictionPolicy
from vllm.core.wa_evictor import (EDGES, ENDS, HITS, WorkloadAwareEvictor,
                                  new_statistics)
from vllm.utils import Device

pytestmark = [pytest.mark.m3, pytest.mark.usefixtures("request_context")]


class Clock:

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def h(block_id):
    return 1000 + block_id


def access(evictor, block_id, workload):
    cache_events.set_current_request(f"cmpl-{workload}:r{block_id}-0")
    evictor.record_access(block_id, h(block_id), 1, None)


def use(evictor, block_id, workload):
    """A request of `workload` computes the block, then frees it."""
    access(evictor, block_id, workload)
    evictor.add(block_id, h(block_id), 16, evictor.clock())


def hit(evictor, block_id, workload, free=False):
    """A lookup of `workload` finds the cached block, as the allocator
    reports one: `record_hit`, `remove` if idle, then the access."""
    evictor.record_hit(block_id)
    if block_id in evictor:
        evictor.remove(block_id)
    access(evictor, block_id, workload)
    if free:
        evictor.add(block_id, h(block_id), 16, evictor.clock())


def dedupe(evictor, block_id, workload):
    """A request computed the block itself; the allocator swaps in the
    idle cached copy without a lookup hit."""
    evictor.remove(block_id)
    access(evictor, block_id, workload)
    evictor.add(block_id, h(block_id), 16, evictor.clock())


def trained(labels, **settings):
    clock = Clock()
    settings = {"kappa": 0.0, "buckets": False, "half_life": 1e9, **settings}
    evictor = WorkloadAwareEvictor(labels=labels, clock=clock, **settings)
    for k in range(20):  # fast: reused 1 s later (then kept in use)
        use(evictor, k, "fast")
        clock.t += 1.0
        hit(evictor, k, "fast")
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


def test_a_block_past_the_ends_of_its_bin_scores_below_a_younger_one():
    """The fast ends are all at 1 s. A block idle a little longer, in the
    same age bin, has outlived them and must not count them: its density is
    0, not the cap of a negative remaining lifetime."""
    evictor, clock = trained("oracle")
    i = bisect_right(EDGES, 1.0)
    old_age = (1.0 + EDGES[i]) / 2
    assert bisect_right(EDGES, old_age) == i
    assert bisect_right(EDGES, 0.5) < i
    assert evictor.density(("fast", 0), old_age) == 0.0
    assert evictor.density(("fast", 0), 0.5) == pytest.approx(2.0)

    use(evictor, 300, "fast")
    clock.t += old_age - 0.5
    use(evictor, 301, "fast")
    clock.t += 0.5
    assert evictor.evict()[0] == 300


def test_a_bins_ends_count_while_the_block_is_younger_than_their_mean():
    evictor, _ = trained("oracle")
    i = bisect_right(EDGES, 1.0)
    young_age = (EDGES[i - 1] + 1.0) / 2  # same bin, below the ends
    assert bisect_right(EDGES, young_age) == i
    assert evictor.density(("fast", 0),
                           young_age) == pytest.approx(1.0 / (1.0 - young_age))


def test_scores_are_fresh_within_one_step():
    """Class "a" has one hit at 10 s, class "b" one at 20 s, so at 0.5 s an
    "a" block is the denser. Evicting an old "a" block adds an end without
    a hit to "a", which then falls below "b". The next eviction, in the
    same step, must see that."""
    clock = Clock()
    evictor = WorkloadAwareEvictor(kappa=0.0,
                                   buckets=False,
                                   half_life=1e9,
                                   clock=clock)
    use(evictor, 1, "a")
    use(evictor, 2, "b")
    clock.t = 10.0
    hit(evictor, 1, "a")
    clock.t = 20.0
    hit(evictor, 2, "b")
    use(evictor, 3, "a")  # idle 100 s at eviction: past every "a" end
    clock.t = 119.5
    use(evictor, 4, "a")
    use(evictor, 5, "b")
    clock.t = 120.0
    assert evictor.density(("a", 0), 0.5) > evictor.density(("b", 0), 0.5)
    assert evictor.evict()[0] == 3
    assert evictor.density(("a", 0), 0.5) < evictor.density(("b", 0), 0.5)
    assert evictor.evict()[0] == 4


def test_a_class_with_few_ends_leans_on_the_pooled_statistics():
    """With kappa, a class has kappa extra ends shaped like all of them."""
    oracle, _ = trained("oracle", kappa=20.0)
    pooled, _ = trained("none")
    # An unseen label has only the pooled ends: 20 hits at 1 s and 20
    # evictions at 30 s, 20 / (620 - 0.5 * 40) per s.
    expected = pooled.density(("", 0), 0.5)
    assert expected == pytest.approx(20 / 600)
    assert oracle.density(("unseen", 0), 0.5) == pytest.approx(expected)
    # "fast": its own 20 ends plus half of each pooled kind (kappa 20 of
    # 40): 30 hits over 330 s of end ages, at age 0.5 over 40 ends.
    fast = oracle.density(("fast", 0), 0.5)
    assert fast == pytest.approx(30 / (330 - 0.5 * 40))
    assert expected < fast < 2.0


def test_hit_buckets_count_lookup_hits_and_not_dedupes():
    clock = Clock()
    evictor = WorkloadAwareEvictor(kappa=0.0, half_life=1e9, clock=clock)
    use(evictor, 1, "w")
    assert evictor.class_of(h(1)) == ("w", 0)
    clock.t = 1.0
    dedupe(evictor, 1, "w")
    assert evictor.class_of(h(1)) == ("w", 0)
    # The dedupe ended an idle period without a hit.
    assert evictor.classes[("w", 0)][[ENDS, HITS
                                      ]].sum(axis=1).tolist() == [1.0, 0.0]
    for expected in (1, 2, 2):
        clock.t += 1.0
        hit(evictor, 1, "w", free=True)
        assert evictor.class_of(h(1)) == ("w", expected)
    # The idle periods end in the class the block had during them.
    assert evictor.classes[("w", 0)][HITS].sum() == 1.0
    assert evictor.classes[("w", 1)][HITS].sum() == 1.0
    assert evictor.classes[("w", 2)][HITS].sum() == 1.0

    flat = WorkloadAwareEvictor(buckets=False, clock=clock)
    use(flat, 1, "w")
    hit(flat, 1, "w", free=True)
    assert flat.class_of(h(1)) == ("w", 0)


def test_a_hit_on_a_block_in_use_counts_too():
    evictor = WorkloadAwareEvictor(clock=Clock())
    access(evictor, 1, "w")  # computed, still in use
    hit(evictor, 1, "w")  # a second request reads it
    assert evictor.class_of(h(1)) == ("w", 1)


def test_statistics_halve_every_half_life():
    clock = Clock()
    evictor = WorkloadAwareEvictor(labels="none",
                                   half_life=10.0,
                                   kappa=0.0,
                                   buckets=False,
                                   clock=clock)
    use(evictor, 1, "w")
    clock.t = 1.0
    evictor.evict()  # one end at age 1
    use(evictor, 2, "w")
    clock.t = 21.0
    evictor.evict()  # two half-lives later, another at age 20
    assert evictor.pooled[ENDS].sum() == pytest.approx(0.25 + 1.0)
    assert evictor.classes[("", 0)][ENDS].sum() == pytest.approx(1.25)
    # Within a sixteenth of a half-life nothing decays.
    use(evictor, 3, "w")
    clock.t = 21.5
    evictor.evict()
    assert evictor.pooled[ENDS].sum() == pytest.approx(2.25)


def test_the_running_sums_match_the_bins():
    evictor, _ = trained("oracle", half_life=7.0)
    for stats in [*evictor.classes.values(), evictor.pooled]:
        rebuilt = new_statistics()
        rebuilt[:3] = stats[:3]
        for i in range(stats.shape[1]):
            rebuilt[3:, i] = stats[:3, i + 1:].sum(axis=1)
        assert stats == pytest.approx(rebuilt)


def test_the_allocator_tells_a_hit_from_a_dedupe():
    """Through vLLM's prefix-caching allocator (block size 4): a lookup of
    an idle block is a hit; a decode block that fills with the content of
    an idle cached block is a dedupe, and saved no work."""
    alloc = CpuGpuBlockAllocator.create("prefix_caching",
                                        num_gpu_blocks=16,
                                        num_cpu_blocks=0,
                                        block_size=4)
    gpu = alloc._allocators[Device.GPU]
    evictor = gpu.evictor = WorkloadAwareEvictor(labels="none", kappa=0.0)

    def table(tokens):
        t = BlockTable(4, alloc)
        t.allocate(list(tokens), device=Device.GPU)
        alloc.mark_blocks_as_computed([])
        return t

    def ends_and_hits():
        return evictor.pooled[[ENDS, HITS]].sum(axis=1).tolist()

    # A and B share a full block and two tokens; A decodes two more, which
    # fills its second block, and frees both.
    a, b = table([1, 2, 3, 4, 5, 6]), table([1, 2, 3, 4, 5, 6])
    a.append_token_ids([7, 8])
    a.free()
    assert ends_and_hits() == [0.0, 0.0]
    # B decodes the same two tokens: its block is deduplicated to A's idle
    # one.
    b.append_token_ids([7, 8])
    assert ends_and_hits() == [1.0, 0.0]
    b.free()
    # C's prompt finds both blocks idle in the cache: two hits.
    table([1, 2, 3, 4, 5, 6, 7, 8])
    assert ends_and_hits() == [3.0, 2.0]


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
