"""How Sampled LRFU is wired into vLLM (spec 007, Milestone 3).

The evictor itself (``vllm/core/sampled_lrfu.py``) has its own tests. These
check what connects it to vLLM's prefix cache:

- ``VLLM_EVICTION_POLICY=sampled_lrfu`` picks ``SampledLRFUEvictor`` for
  the GPU prefix-caching allocator only (the swap-space allocator keeps
  LRU), its ``VLLM_EVICTION_PARAMS`` reach it, and bad settings stop the
  allocator at startup;
- through a real ``PrefixCachingBlockAllocator``, the logical clock moves
  one step per block access (a lookup hit, an insertion, a deduplication),
  a request counts each block once, and the candidate pool is always
  exactly the cached blocks no request holds (refcount 0), so a block in
  use is never evicted.

Block size 4, so every block, score and victim is worked out by hand.
"""
import json
import random

import pytest

from vllm.core import sampled_lrfu
from vllm.core.block import cache_events
from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import BlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlock
from vllm.core.eviction_policies import (POLICY_NAMES,
                                         eviction_policy_from_env,
                                         make_project_evictor)
from vllm.core.evictor import EvictionPolicy, LRUEvictor, make_evictor
from vllm.core.sampled_lrfu import DEFAULTS, SampledLRFUEvictor
from vllm.utils import Device

pytestmark = pytest.mark.m3

BS = 4
ENV = ("VLLM_EVICTION_POLICY", "VLLM_EVICTION_PARAMS", "VLLM_CACHE_EVENT_LOG")


@pytest.fixture(autouse=True)
def clean_env_and_context(monkeypatch):
    """No policy settings from the caller's shell, and the module's request
    context (which Sampled LRFU switches on) as it was before the test."""
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    ctx = cache_events._context
    saved = ctx.enabled, ctx.request_id, ctx.op, ctx.time
    ctx.request_id, ctx.op, ctx.time = None, None, None
    yield
    ctx.enabled, ctx.request_id, ctx.op, ctx.time = saved


@pytest.fixture
def no_atexit(monkeypatch):
    """The `stats` setting registers an exit hook; record it instead."""
    registered = []
    monkeypatch.setattr(sampled_lrfu.atexit, "register",
                        lambda fn, *args: registered.append((fn, args)))
    return registered


def chain(tokens):
    """The content hash of each full block of `tokens`, chained as the
    allocator chains them."""
    hashes, prev = [], None
    for i in range(len(tokens) // BS):
        prev = PrefixCachingBlock.hash_block_tokens(
            i == 0, prev, tokens[i * BS:(i + 1) * BS])
        hashes.append(prev)
    return hashes


def make_allocator(monkeypatch, num_blocks=16, params=None, num_cpu_blocks=0):
    """The block manager's allocator and its GPU allocator, built as the
    engine builds them with ``VLLM_EVICTION_POLICY=sampled_lrfu``."""
    monkeypatch.setenv("VLLM_EVICTION_POLICY", "sampled_lrfu")
    if params is not None:
        monkeypatch.setenv("VLLM_EVICTION_PARAMS", params)
    alloc = CpuGpuBlockAllocator.create("prefix_caching",
                                        num_gpu_blocks=num_blocks,
                                        num_cpu_blocks=num_cpu_blocks,
                                        block_size=BS)
    gpu = alloc._allocators[Device.GPU]
    assert type(gpu.evictor) is SampledLRFUEvictor
    return alloc, gpu


def allocate(alloc, tokens, request=None, op="allocate"):
    """A prompt's blocks, as `BlockSpaceManager.allocate` builds them, with
    `request` named in the request context as the block manager names it."""
    if request is not None:
        cache_events.set_current_request(request, op)
    table = BlockTable(BS, alloc)
    table.allocate(list(tokens), device=Device.GPU)
    alloc.mark_blocks_as_computed([])
    return table


def append(alloc, table, tokens, request=None):
    if request is not None:
        cache_events.set_current_request(request, "append_slots")
    table.append_token_ids(list(tokens))
    alloc.mark_blocks_as_computed([])


def idle_cached(gpu):
    """The allocator's cached blocks no request holds: {block id: hash}."""
    return {
        b: h
        for h, b in gpu._cached_blocks.items() if gpu._refcounter.get(b) == 0
    }


def check_pool(gpu, tables=()):
    """The evictor's state matches the allocator's: its candidates are
    exactly the idle cached blocks (with their hashes), every cached block
    has a score and nothing else does, and no candidate is in a live
    table."""
    lrfu = gpu.evictor
    idle = idle_cached(gpu)
    assert set(lrfu._pool) == set(idle)
    assert len(lrfu._pool) == len(set(lrfu._pool)) == lrfu.num_blocks
    assert {b: lrfu._pool[i]
            for b, i in lrfu._pos.items()} == {b: b
                                               for b in lrfu._pool}
    assert lrfu._hash == idle
    assert all(gpu._refcounter.get(b) == 0 for b in lrfu._pool)
    assert all(b in lrfu for b in idle)
    cached = set(gpu._cached_blocks.values())
    # Per-block state lives exactly as long as the block is cached.
    assert set(lrfu.score) == set(lrfu.stamp) == set(lrfu._born) == cached
    held = {b for t in tables for b in t.physical_block_ids}
    assert not held & set(lrfu._pool)
    assert gpu.get_num_free_blocks() == (
        gpu._hashless_allocator.get_num_free_blocks() + len(idle))


# --- Choosing Sampled LRFU ---------------------------------------------------


@pytest.mark.parametrize("value",
                         ["sampled_lrfu", "SAMPLED_LRFU", " Sampled_LRFU "])
def test_policy_from_env_knows_sampled_lrfu(monkeypatch, value):
    monkeypatch.setenv("VLLM_EVICTION_POLICY", value)
    assert eviction_policy_from_env() is EvictionPolicy.SAMPLED_LRFU
    assert POLICY_NAMES["sampled_lrfu"] is EvictionPolicy.SAMPLED_LRFU


def test_unknown_policy_lists_sampled_lrfu(monkeypatch):
    monkeypatch.setenv("VLLM_EVICTION_POLICY", "lrfu")
    with pytest.raises(
            ValueError,
            match="'lrfu'.*lru, fifo, lfu, ttdf, race_rp, sampled_lrfu"):
        eviction_policy_from_env()


def test_sampled_lrfu_has_its_own_enum_value():
    members = EvictionPolicy.__members__
    assert "SAMPLED_LRFU" in members
    # No alias: every name has its own value, and every name is reachable.
    assert len({p.value for p in members.values()}) == len(members)
    assert set(POLICY_NAMES.values()) == set(members.values())
    # vLLM's own LRU is not the project module's to build.
    assert make_project_evictor(EvictionPolicy.LRU) is None


def test_without_settings_it_uses_its_defaults():
    assert DEFAULTS == {"half_life": 512.0, "sample_size": 4, "seed": 0}
    for evictor in (make_evictor(EvictionPolicy.SAMPLED_LRFU),
                    make_project_evictor(EvictionPolicy.SAMPLED_LRFU)):
        assert type(evictor) is SampledLRFUEvictor
        assert (evictor.half_life, evictor.sample_size) == (512.0, 4)
        assert evictor.rng.getstate() == random.Random(0).getstate()
        assert evictor.request is cache_events.current_request
        assert (evictor.now, evictor.num_blocks) == (0, 0)


def test_settings_come_from_the_params_variable(monkeypatch):
    monkeypatch.setenv("VLLM_EVICTION_PARAMS",
                       " half_life=8, sample_size=2 ,seed=5")
    evictor = make_evictor(EvictionPolicy.SAMPLED_LRFU)
    assert type(evictor) is SampledLRFUEvictor
    assert (evictor.half_life, evictor.sample_size) == (8.0, 2)
    assert type(evictor.half_life) is float
    assert type(evictor.sample_size) is int
    assert evictor.rng.getstate() == random.Random(5).getstate()


def test_the_variable_sets_the_gpu_evictor_only(monkeypatch):
    alloc, gpu = make_allocator(monkeypatch,
                                num_blocks=8,
                                params="half_life=16,sample_size=3,seed=2",
                                num_cpu_blocks=4)
    assert type(alloc._allocators[Device.CPU].evictor) is LRUEvictor
    assert (gpu.evictor.half_life, gpu.evictor.sample_size) == (16.0, 3)
    assert gpu.evictor.rng.getstate() == random.Random(2).getstate()


@pytest.mark.parametrize(
    "params,error",
    [
        ("half_life=0", "half_life must be positive"),
        ("half_life=-2", "half_life must be positive"),
        ("half_life=nan", "half_life must be positive"),
        ("sample_size=0", "sample_size must be at least 1"),
        ("sample_size=-1", "sample_size must be at least 1"),
        ("halflife=8", "'halflife=8' is not one of"),
        ("seed=1,sample=2", "'sample=2' is not one of"),
        ("half_life", "'half_life' is not one of"),
        # Other policies' settings are not Sampled LRFU's.
        ("tau_short=2", "'tau_short=2' is not one of"),
        ("sketch=x.json", "'sketch=x.json' is not one of"),
        ("half_life=soon", "could not convert"),
        ("sample_size=2.5", "invalid literal"),
    ])
def test_bad_settings_stop_the_allocator(monkeypatch, params, error):
    monkeypatch.setenv("VLLM_EVICTION_POLICY", "sampled_lrfu")
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", params)
    with pytest.raises(ValueError, match=error):
        CpuGpuBlockAllocator.create("prefix_caching", 8, 0, BS)
    with pytest.raises(ValueError, match=error):
        make_evictor(EvictionPolicy.SAMPLED_LRFU)


def test_the_error_names_every_setting(monkeypatch):
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", "k=1")
    with pytest.raises(ValueError,
                       match="one of half_life, sample_size, seed, stats"):
        make_evictor(EvictionPolicy.SAMPLED_LRFU)


def test_the_stats_setting_writes_its_counters_at_exit(monkeypatch, tmp_path,
                                                       no_atexit):
    path = tmp_path / "evictor-stats.json"
    alloc, gpu = make_allocator(monkeypatch,
                                num_blocks=2,
                                params=f"stats={path},sample_size=2")
    assert len(no_atexit) == 1
    fn, args = no_atexit[0]
    assert fn == gpu.evictor.write_stats and args == (str(path), )
    for i, start in enumerate((1, 11, 21)):
        allocate(alloc, range(start, start + 8), f"r{i}").free()
    fn(*args)
    written = json.loads(path.read_text())
    assert written == json.loads(json.dumps(gpu.evictor.summary()))
    assert written["evictions"] == 4
    assert written["max_resident_blocks"] == 2


def test_without_stats_nothing_is_written(monkeypatch, no_atexit):
    make_allocator(monkeypatch, params="half_life=4")
    assert no_atexit == []


def test_it_switches_the_request_context_on():
    cache_events._context.enabled = False
    make_evictor(EvictionPolicy.SAMPLED_LRFU)
    assert cache_events._context.enabled


# --- The logical clock through the allocator --------------------------------


def test_inserts_and_hits_each_move_the_clock_once(monkeypatch):
    """r1 computes two full blocks and a partial one (two inserts, one step
    each; the partial block is not cached); r2 reads them again (two
    hits)."""
    alloc, gpu = make_allocator(monkeypatch, params="half_life=2")
    lrfu = gpu.evictor
    tokens = list(range(1, 11))
    first = allocate(alloc, tokens, "r1")
    ids = first.physical_block_ids[:2]
    assert lrfu.now == 2
    assert [lrfu.score[b] for b in ids] == [1.0, 1.0]
    assert [lrfu.stamp[b] for b in ids] == [1, 2]
    second = allocate(alloc, tokens, "r2")
    assert second.physical_block_ids[:2] == ids
    assert lrfu.now == 4
    # Each block decayed for two steps (one half-life) before its hit.
    assert [lrfu.score[b] for b in ids] == [1.5, 1.5]
    assert [lrfu.stamp[b] for b in ids] == [3, 4]
    check_pool(gpu, [first, second])
    # Both requests counted both blocks: reading them again moves nothing.
    third = allocate(alloc, tokens, "r1")
    fourth = allocate(alloc, tokens, "r2")
    assert lrfu.now == 4
    assert [lrfu.score[b] for b in ids] == [1.5, 1.5]
    for table in (first, second, third, fourth):
        table.free()
    check_pool(gpu)
    assert lrfu.num_blocks == 2 and lrfu.now == 4


def test_a_deduplicated_block_moves_the_clock_once(monkeypatch):
    """A and B share a prompt of one full block and two tokens. A's decode
    fills and inserts the second block; B's fills the same content, which
    is deduplicated to A's block: one step each."""
    alloc, gpu = make_allocator(monkeypatch, params="half_life=1")
    lrfu = gpu.evictor
    prompt = [1, 2, 3, 4, 5, 6]
    a = allocate(alloc, prompt, "a")  # insert: 1
    b = allocate(alloc, prompt, "b")  # hit: 2
    assert lrfu.now == 2
    append(alloc, a, [7, 8], "a")  # insert: 3
    assert lrfu.now == 3
    shared = a.physical_block_ids[1]
    assert (lrfu.score[shared], lrfu.stamp[shared]) == (1.0, 3)
    append(alloc, b, [7, 8], "b")  # dedupe: 4
    assert b.physical_block_ids == a.physical_block_ids
    assert lrfu.now == 4
    assert lrfu.score[shared] == 2**-1 + 1
    assert lrfu.stamp[shared] == 4
    check_pool(gpu, [a, b])
    # A third request reading A's two blocks counts both.
    c = allocate(alloc, prompt + [7, 8], "c")
    assert c.physical_block_ids == a.physical_block_ids
    assert lrfu.now == 6
    check_pool(gpu, [a, b, c])


def test_the_clock_counts_every_allocator_access(monkeypatch):
    """Requests sharing prefixes, with decodes and evictions, each named:
    the clock is the number of `record_access` calls, as no request reads
    a block twice."""
    alloc, gpu = make_allocator(monkeypatch, num_blocks=8)
    lrfu = gpu.evictor
    calls = []
    record = lrfu.record_access

    def counted(block_id, content_hash, depth, prev_hash):
        calls.append((cache_events.current_request(), block_id))
        record(block_id, content_hash, depth, prev_hash)

    lrfu.record_access = counted
    prompts = [
        list(range(1, 13)),
        list(range(1, 9)) + [50, 51, 52, 53, 54],
        list(range(200, 217)),
        list(range(1, 13)) + [60],
    ]
    for i, prompt in enumerate(prompts):
        table = allocate(alloc, prompt, f"r{i}")
        append(alloc, table, [70, 71, 72, 73], f"r{i}")
        table.free()
        assert lrfu.now == len(calls)
        check_pool(gpu)
    assert len(set(calls)) == len(calls)
    assert lrfu.stats["evictions"] > 0


# --- A request counts each block once ----------------------------------------


def test_a_request_allocated_again_counts_its_blocks_once(monkeypatch):
    """r1 computes a three-block prompt, is freed (preempted) and allocates
    it again: the hits change no score and do not move the clock. r2's
    reads count."""
    alloc, gpu = make_allocator(monkeypatch, params="half_life=4")
    lrfu = gpu.evictor
    tokens = list(range(1, 13))
    table = allocate(alloc, tokens, "r1")
    ids = table.physical_block_ids
    assert lrfu.now == 3
    state = dict(lrfu.score), dict(lrfu.stamp)
    table.free()
    check_pool(gpu)
    again = allocate(alloc, tokens, "r1")
    assert again.physical_block_ids == ids
    assert lrfu.now == 3
    assert (dict(lrfu.score), dict(lrfu.stamp)) == state
    check_pool(gpu, [again])
    other = allocate(alloc, tokens, "r2")
    assert lrfu.now == 6
    assert all(lrfu.score[b] > 1.0 for b in ids)
    check_pool(gpu, [again, other])


def test_a_request_holding_its_blocks_twice_counts_them_once(monkeypatch):
    """A second sequence of the same request (n > 1, or a fork) reads the
    blocks the first one holds."""
    alloc, gpu = make_allocator(monkeypatch)
    lrfu = gpu.evictor
    tokens = list(range(1, 9))
    a = allocate(alloc, tokens, "r1")
    b = allocate(alloc, tokens, "r1")
    assert b.physical_block_ids == a.physical_block_ids
    assert lrfu.now == 2
    assert all(lrfu.score[x] == 1.0 for x in a.physical_block_ids)
    check_pool(gpu, [a, b])


def test_an_evicted_block_counts_again_when_recomputed(monkeypatch):
    """Two blocks. r1 computes A, r2's B evicts it, and r1 computes A again:
    a new cached block, so it counts (score 1, the clock moves)."""
    alloc, gpu = make_allocator(monkeypatch, num_blocks=2)
    lrfu = gpu.evictor
    a, b = list(range(1, 9)), list(range(11, 19))
    allocate(alloc, a, "r1").free()
    allocate(alloc, b, "r2").free()
    assert set(gpu._cached_blocks) == set(chain(b))
    assert lrfu.now == 4 and lrfu.stats["evictions"] == 2
    table = allocate(alloc, a, "r1")
    assert set(gpu._cached_blocks) == set(chain(a))
    assert lrfu.now == 6
    assert [lrfu.score[x] for x in table.physical_block_ids] == [1.0, 1.0]
    check_pool(gpu, [table])
    # r1 counted the recomputed blocks, so it does not count them again.
    again = allocate(alloc, a, "r1")
    assert again.physical_block_ids == table.physical_block_ids
    assert lrfu.now == 6
    assert [lrfu.score[x] for x in table.physical_block_ids] == [1.0, 1.0]
    check_pool(gpu, [table, again])


def test_a_block_id_reused_for_new_content_counts_again(monkeypatch):
    """One block. r1 computes A in it; r2's B evicts A and takes the same
    block id; r1 then reads B. Its earlier count was of A, a block cached
    at another time, so this read counts."""
    alloc, gpu = make_allocator(monkeypatch, num_blocks=1)
    lrfu = gpu.evictor
    a, b = [1, 2, 3, 4], [11, 12, 13, 14]
    first = allocate(alloc, a, "r1")
    (block, ) = first.physical_block_ids
    first.free()
    second = allocate(alloc, b, "r2")
    assert second.physical_block_ids == [block]
    assert lrfu.now == 2 and lrfu.stats["evictions"] == 1
    second.free()
    third = allocate(alloc, b, "r1")
    assert third.physical_block_ids == [block]
    assert lrfu.now == 3
    assert lrfu.score[block] == pytest.approx(2**(-1 / 512) + 1)
    check_pool(gpu, [third])


def test_without_a_named_request_every_access_counts(monkeypatch):
    """Outside a block-manager call no request is named, so nothing can be
    told apart: each access counts."""
    alloc, gpu = make_allocator(monkeypatch, params="half_life=1")
    lrfu = gpu.evictor
    assert cache_events.current_request() is None
    tokens = list(range(1, 5))
    allocate(alloc, tokens).free()
    allocate(alloc, tokens).free()
    (block, ) = lrfu._pool
    assert lrfu.now == 2
    assert lrfu.score[block] == 1.5
    # Unnamed reads are remembered for no request: r1's first read counts,
    # its second does not.
    allocate(alloc, tokens, "r1").free()
    allocate(alloc, tokens, "r1").free()
    assert lrfu.now == 3


def test_a_request_counts_a_block_once_even_if_others_read_it_between(
        monkeypatch):
    """r1 computes a prompt and is freed (preempted); r2 reads the same
    prompt; r1 is scheduled again and reads it a second time. r1 already
    counted these blocks, so only r2's reads count, however many other
    requests read them in between."""
    alloc, gpu = make_allocator(monkeypatch, params="half_life=2")
    lrfu = gpu.evictor
    tokens = list(range(1, 9))
    allocate(alloc, tokens, "r1").free()
    allocate(alloc, tokens, "r2").free()
    assert lrfu.now == 4
    scores = dict(lrfu.score)
    table = allocate(alloc, tokens, "r1")
    assert lrfu.now == 4
    assert lrfu.score == scores
    check_pool(gpu, [table])
    table.free()
    for i in range(3, 40):
        allocate(alloc, tokens, f"r{i}").free()
    assert lrfu.now == 4 + 2 * 37
    allocate(alloc, tokens, "r1")
    assert lrfu.now == 4 + 2 * 37


def test_only_the_most_recent_requests_are_remembered(monkeypatch):
    """With room for three requests, the one that counted a block longest
    ago is forgotten when a fourth counts one, and then counts its blocks
    again; a request still remembered does not. The memory stays
    bounded."""
    monkeypatch.setattr(sampled_lrfu, "REQUEST_MEMORY", 3)
    alloc, gpu = make_allocator(monkeypatch)
    lrfu = gpu.evictor
    tokens = list(range(1, 9))
    for rid in ("r1", "r2", "r3"):
        allocate(alloc, tokens, rid).free()
    assert lrfu.now == 6
    allocate(alloc, tokens, "r1").free()  # remembered: no count
    assert lrfu.now == 6
    allocate(alloc, tokens, "r4").free()  # r1 is forgotten
    assert lrfu.now == 8
    allocate(alloc, tokens, "r3").free()  # remembered
    assert lrfu.now == 8
    allocate(alloc, tokens, "r1").free()  # forgotten: counts; r2 forgotten
    assert lrfu.now == 10
    allocate(alloc, tokens, "r2").free()
    assert lrfu.now == 12
    words = lrfu.metadata_words()["bookkeeping"]
    for i in range(5, 50):
        allocate(alloc, tokens, f"r{i}").free()
    # Three requests of two counted blocks each, however many came.
    assert lrfu.metadata_words()["bookkeeping"] == words
    check_pool(gpu)


# --- Eviction through the allocator ------------------------------------------


def test_evicts_the_lowest_score_when_the_sample_covers_the_pool(monkeypatch):
    """Three blocks, half-life 4 accesses, a sample of 8 (every candidate).
    A is computed (t=1) and read again (t=3), B computed at 2, C at 4. At
    t=4 A scores 2^(-1/4)(2^(-1/2)+1) ~ 1.44, B 2^(-1/2) ~ 0.71 and C 1,
    so D's block takes B's. At t=5 (after D's insertion) A scores ~ 1.21,
    C 2^(-1/4) ~ 0.84 and D 1, so E's takes C's."""
    alloc, gpu = make_allocator(monkeypatch,
                                num_blocks=3,
                                params="half_life=4,sample_size=8")
    lrfu = gpu.evictor
    a, b, c, d, e = ([s, s + 1, s + 2, s + 3] for s in (1, 11, 21, 31, 41))
    victims = []
    evict = lrfu.evict

    def logged_evict():
        scores = {x: lrfu.current(x) for x in lrfu._pool}
        ranked = min(lrfu._pool, key=lambda x: (lrfu.rank(x), x))
        block_id, content_hash = evict()
        assert scores[block_id] == min(scores.values())
        assert block_id == ranked
        victims.append(content_hash)
        return block_id, content_hash

    lrfu.evict = logged_evict
    for i, tokens in enumerate((a, b, a, c)):
        allocate(alloc, tokens, f"r{i}").free()
    (ha, ), (hb, ), (hc, ), (hd, ), (he, ) = map(chain, (a, b, c, d, e))
    block_a = gpu._cached_blocks[ha]
    assert lrfu.score[block_a] == pytest.approx(2**-0.5 + 1)
    assert lrfu.current(block_a) == pytest.approx(2**-0.25 * (2**-0.5 + 1))
    allocate(alloc, d, "r4").free()
    assert victims == [hb]
    allocate(alloc, e, "r5").free()
    assert victims == [hb, hc]
    assert set(gpu._cached_blocks) == {ha, hd, he}
    check_pool(gpu)


def test_a_full_cache_of_blocks_in_use_evicts_nothing(monkeypatch):
    alloc, gpu = make_allocator(monkeypatch, num_blocks=2)
    held = allocate(alloc, range(1, 9), "r1")
    with pytest.raises(BlockAllocator.NoFreeBlocksError):
        allocate(alloc, range(11, 15), "r2")
    assert gpu.evictor.stats["evictions"] == 0
    assert gpu.evictor.num_blocks == 0
    assert set(gpu._cached_blocks) == set(chain(list(range(1, 9))))
    check_pool(gpu, [held])


def test_an_idle_blocks_access_time_changes_no_score(monkeypatch):
    """The scheduler stamps idle blocks (`mark_blocks_as_accessed`, which
    calls the evictor's `update`); the score does not use the stamp."""
    alloc, gpu = make_allocator(monkeypatch)
    table = allocate(alloc, range(1, 9), "r1")
    ids = table.physical_block_ids
    table.free()
    before = dict(gpu.evictor.score), dict(gpu.evictor.stamp)
    alloc.mark_blocks_as_accessed(ids, 1e9)
    assert (dict(gpu.evictor.score), dict(gpu.evictor.stamp)) == before
    check_pool(gpu)


@pytest.mark.parametrize("params", [
    "sample_size=1,seed=0", "sample_size=2,seed=1", "sample_size=3,seed=7",
    "sample_size=64,half_life=1"
])
def test_the_pool_is_always_the_idle_cached_blocks(monkeypatch, params):
    """A 12-block cache. A long request holds its prompt and decode blocks
    (up to four) throughout while eight short requests of three blocks,
    sharing prefixes, come and go two at a time (20 distinct blocks in
    all). After every allocate, append and free the evictor's pool is
    exactly the cached blocks with refcount 0, every eviction takes one of
    them (never a block a live table holds), and the long request's blocks
    survive."""
    alloc, gpu = make_allocator(monkeypatch, num_blocks=12, params=params)
    lrfu = gpu.evictor
    live = []
    victims = []
    evict = lrfu.evict

    def checked_evict():
        idle = idle_cached(gpu)
        assert set(lrfu._pool) == set(idle)
        block_id, content_hash = evict()
        assert idle.get(block_id) == content_hash
        assert gpu._refcounter.get(block_id) == 0
        assert all(block_id not in t.physical_block_ids for t in live)
        victims.append(content_hash)
        return block_id, content_hash

    lrfu.evict = checked_evict
    long = allocate(alloc, range(500, 504), "long")
    live.append(long)
    check_pool(gpu, live)
    append(alloc, long, range(504, 508), "long")
    check_pool(gpu, live)

    def long_survives():
        full = [x for x in long.blocks if x.content_hash is not None]
        assert len(full) >= 2
        for x in full:
            assert gpu._cached_blocks[x.content_hash] == x.block_id
            assert gpu._refcounter.get(x.block_id) >= 1

    starts = [1, 100, 1, 200, 1, 300, 100, 1]
    previous = None
    for i, start in enumerate(starts):
        rid = f"s{i}"
        prompt = list(range(start, start + 8)) + [start + 50 + i]
        table = allocate(alloc, prompt, rid)
        live.append(table)
        check_pool(gpu, live)
        append(alloc, table, [900 + i] * 3, rid)
        check_pool(gpu, live)
        if previous is not None:
            # The previous short request overlapped this one; now it ends.
            previous.free()
            live.remove(previous)
            check_pool(gpu, live)
        previous = table
        append(alloc, long, [600 + i], "long")
        check_pool(gpu, live)
        long_survives()
    assert len(long.blocks) == 4
    previous.free()
    live.remove(previous)
    check_pool(gpu, live)
    long.free()
    live.remove(long)
    check_pool(gpu)
    assert victims and lrfu.stats["evictions"] == len(victims)
    assert lrfu.num_blocks == len(gpu._cached_blocks)
