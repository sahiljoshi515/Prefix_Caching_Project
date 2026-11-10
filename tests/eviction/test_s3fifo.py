"""S3-FIFO over the idle leaf blocks (``s3fifo_leaf``, spec 007 Phase 2).

New blocks enter the small queue S; a block hit twice there moves to the
main queue M when S's hand reaches it, and one that was not leaves the
cache, its key going to the ghost queue. M's hand gives a hit block
another round (a 2-bit clock). Only idle leaves are evicted. Victims are
worked out by hand.
"""
import pytest

from tests.eviction.test_reuse_wiring import allocate
from tests.eviction.test_reuse_wiring import chain as hash_chain
from tests.eviction.test_reuse_wiring import make_allocator
from vllm.core.block import cache_events
from vllm.core.eviction_policies import POLICY_NAMES, make_project_evictor
from vllm.core.evictor import EvictionPolicy
from vllm.core.s3fifo_evictor import MAIN, SMALL, S3FIFOLeafEvictor

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


def free(evictor, *block_ids):
    for block_id in block_ids:
        evictor.add(block_id, h(block_id), 16, 0.0)


def queue_of(evictor, block_id):
    return evictor.state[h(block_id)].queue


def test_a_block_not_hit_in_the_small_queue_leaves_first():
    evictor = S3FIFOLeafEvictor(clock=Clock())
    for block_id in (1, 2, 3):
        chain(evictor, block_id)
    free(evictor, 1, 2, 3)
    assert [queue_of(evictor, b) for b in (1, 2, 3)] == [SMALL] * 3
    assert evictor.evict()[0] == 1  # the oldest in S
    assert list(evictor.ghost) == [h(1)]


def test_a_block_hit_twice_in_the_small_queue_moves_to_main():
    evictor = S3FIFOLeafEvictor(clock=Clock())
    chain(evictor, 1)
    chain(evictor, 2)
    chain(evictor, 1)  # two more requests read block 1
    chain(evictor, 1)
    free(evictor, 1, 2)
    # S's hand reaches 1 first, moves it to M, and evicts 2.
    assert evictor.evict()[0] == 2
    assert queue_of(evictor, 1) == MAIN
    assert evictor.state[h(1)].freq == 0
    assert evictor.stats["moved_to_main"] == 1


def test_one_hit_is_not_enough_by_default():
    evictor = S3FIFOLeafEvictor(clock=Clock())
    chain(evictor, 1)
    chain(evictor, 2)
    chain(evictor, 1)
    free(evictor, 1, 2)
    assert evictor.evict()[0] == 1
    once = S3FIFOLeafEvictor(threshold=1, clock=Clock())
    chain(once, 1)
    chain(once, 2)
    chain(once, 1)
    free(once, 1, 2)
    assert once.evict()[0] == 2


def evicted_from_small_and_back(evictor):
    """Blocks 1 and 2 are evicted from S, then computed again."""
    chain(evictor, 1)
    chain(evictor, 2)
    free(evictor, 1, 2)
    assert [evictor.evict()[0] for _ in range(2)] == [1, 2]
    chain(evictor, 1)
    chain(evictor, 2)


def test_a_block_back_from_the_ghost_queue_enters_main():
    evictor = S3FIFOLeafEvictor(clock=Clock())
    evicted_from_small_and_back(evictor)
    assert [queue_of(evictor, b) for b in (1, 2)] == [MAIN, MAIN]
    assert evictor.stats["ghost_hits"] == 2
    assert not evictor.ghost


def test_without_a_ghost_queue_it_enters_small_again():
    evictor = S3FIFOLeafEvictor(ghost_ratio=0.0, clock=Clock())
    evicted_from_small_and_back(evictor)
    assert [queue_of(evictor, b) for b in (1, 2)] == [SMALL, SMALL]


def test_the_main_queue_gives_a_hit_block_another_round():
    evictor = S3FIFOLeafEvictor(clock=Clock())
    evicted_from_small_and_back(evictor)  # 1 and 2 in M, S empty
    chain(evictor, 1)  # a hit on 1
    free(evictor, 1, 2)
    # M's hand: 1 was hit, so it goes back to M's head with its counter
    # down by one; 2 was not, and leaves.
    assert evictor.evict()[0] == 2
    assert evictor.state[h(1)].freq == 0
    assert evictor.stats["reinserted"] == 1
    assert evictor.evict()[0] == 1


def test_the_counter_is_capped_in_main():
    evictor = S3FIFOLeafEvictor(clock=Clock())
    evicted_from_small_and_back(evictor)
    for _ in range(5):
        chain(evictor, 1)
    free(evictor, 1, 2)
    assert evictor.evict()[0] == 2
    assert evictor.state[h(1)].freq == 2  # min(5, 3) - 1


@pytest.mark.parametrize("small_ratio, victim", [(0.1, 2), (0.6, 1)])
def test_small_is_emptied_while_it_holds_its_share(small_ratio, victim):
    """1 in M, 2 in S: S holds half the cached blocks, which is at least
    10% (evict from S) but less than 60% (evict from M)."""
    evictor = S3FIFOLeafEvictor(small_ratio=small_ratio, clock=Clock())
    chain(evictor, 1)
    free(evictor, 1)
    assert evictor.evict()[0] == 1  # to the ghost queue
    chain(evictor, 1)  # back, into M
    chain(evictor, 2)
    free(evictor, 1, 2)
    assert evictor.evict()[0] == victim


def test_only_idle_leaves_are_evicted():
    evictor = S3FIFOLeafEvictor(clock=Clock())
    chain(evictor, 1, 2, 3)  # one prompt: 1 is S's oldest, 3 its tip
    chain(evictor, 4)  # in use, never freed
    free(evictor, 1, 2, 3)
    # The prompt is cut from its end; the block in use stays.
    assert [evictor.evict()[0] for _ in range(3)] == [3, 2, 1]
    with pytest.raises(ValueError, match="No usable cache memory"):
        evictor.evict()
    assert set(evictor.blocks) == {h(4)}


def test_the_allocator_keeps_a_prefix_closed_cache(monkeypatch):
    """Requests that share prefixes, one after another in an 8-block cache
    of vLLM's prefix-caching allocator (block size 4), with evictions."""
    monkeypatch.setenv("VLLM_EVICTION_POLICY", "s3fifo_leaf")
    monkeypatch.delenv("VLLM_EVICTION_PARAMS", raising=False)
    monkeypatch.delenv("VLLM_CACHE_EVENT_LOG", raising=False)
    alloc, gpu = make_allocator(num_blocks=8, spy=False)
    evictor = gpu.evictor
    assert type(evictor) is S3FIFOLeafEvictor
    prompts = [
        list(range(1, 13)),
        list(range(1, 9)) + [50, 51, 52, 53, 54],
        list(range(200, 217)),
        list(range(1, 13)) + [60],
        list(range(300, 313)),
        list(range(1, 9)),
    ]
    for step, prompt in enumerate(prompts):
        cache_events._context.time = 10.0 * (step + 1)
        table = allocate(alloc, prompt)
        table.append_token_ids([70, 71, 72, 73])
        assert set(hash_chain(prompt + [70, 71, 72, 73])) <= set(evictor.state)
        table.free()
        cached = gpu._cached_blocks
        assert set(evictor.blocks) == set(cached) == set(evictor.state)
        assert all(b.parent is None or b.parent in cached
                   for b in evictor.blocks.values())
        assert evictor.small_count == sum(s.queue == SMALL
                                          for s in evictor.state.values())
    assert evictor.stats["evictions"] > 0


def test_settings_and_registration(monkeypatch):
    assert POLICY_NAMES["s3fifo_leaf"] == EvictionPolicy.S3FIFO_LEAF
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", "small_ratio=0.2,threshold=1")
    evictor = make_project_evictor(EvictionPolicy.S3FIFO_LEAF)
    assert isinstance(evictor, S3FIFOLeafEvictor)
    assert (evictor.small_ratio, evictor.ghost_ratio,
            evictor.threshold) == (0.2, 0.9, 1)
    defaults = S3FIFOLeafEvictor.from_params(None)
    assert (defaults.small_ratio, defaults.ghost_ratio,
            defaults.threshold) == (0.1, 0.9, 2)
    with pytest.raises(ValueError, match="tau_short"):
        S3FIFOLeafEvictor.from_params("tau_short=1")
    with pytest.raises(ValueError, match="small_ratio"):
        S3FIFOLeafEvictor(small_ratio=1.0)
