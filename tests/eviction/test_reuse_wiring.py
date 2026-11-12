"""How TTDF and RACE-RP are wired into vLLM (spec 007, Milestone 3).

The evictors themselves (``vllm/core/reuse_evictors.py``) have their own
tests. These check what connects them to vLLM's prefix cache:

- ``Evictor.record_access``, the allocator's new hook: called once per full
  block per request (a lookup hit, an insertion or a deduplication) with
  the block's 1-based depth and its parent's content hash, never for a
  partial block, and a no-op for LRU, FIFO and LFU, whose behaviour is
  unchanged;
- ``VLLM_EVICTION_POLICY=ttdf|race_rp`` and their ``VLLM_EVICTION_PARAMS``;
- the request context's call time (``cache_events.event_time``), which the
  new evictors stamp accesses and evictions with.

Block size 4, so every block id, hash and depth is worked out by hand.
"""
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

import vllm.envs as envs
from vllm.core.block import cache_events
from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlock
from vllm.core.eviction_policies import (POLICY_NAMES, FIFOEvictor, LFUEvictor,
                                         eviction_policy_from_env,
                                         make_project_evictor)
from vllm.core.evictor import EvictionPolicy, Evictor, LRUEvictor, make_evictor
from vllm.core.reuse_evictors import (NUM_FEATURES, TTDF_DEFAULTS,
                                      RaceRPEvictor, RaceSketch, TTDFEvictor)
from vllm.utils import Device

pytestmark = pytest.mark.m3

BS = 4
ENV = ("VLLM_EVICTION_POLICY", "VLLM_EVICTION_PARAMS", "VLLM_CACHE_EVENT_LOG")


@pytest.fixture(autouse=True)
def clean_env_and_context(monkeypatch, request_context):
    """No policy settings from the caller's shell, and the module's request
    context (which TTDF and RACE-RP switch on) as it was before the test."""
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sketch_file(tmp_path):
    """A small trained RACE sketch saved as RACE-RP loads it."""
    sketch = RaceSketch.random(4,
                               3,
                               seed=7,
                               mean=np.zeros(NUM_FEATURES),
                               std=np.ones(NUM_FEATURES),
                               r_bar=0.25,
                               alpha=5.0)
    rng = np.random.default_rng(0)
    sketch.add(rng.standard_normal((20, NUM_FEATURES)), rng.random(20))
    path = tmp_path / "sketch.json"
    sketch.save(path)
    return path, sketch


class FakeClock:
    """Stands in for `cache_events.time`, as the cache simulator's does."""

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock(100.0)
    monkeypatch.setattr(cache_events, "time", fake)
    return fake


def chain(tokens):
    """The content hash of each full block of `tokens`, chained as the
    allocator chains them."""
    hashes, prev = [], None
    for i in range(len(tokens) // BS):
        prev = PrefixCachingBlock.hash_block_tokens(
            i == 0, prev, tokens[i * BS:(i + 1) * BS])
        hashes.append(prev)
    return hashes


class SpyEvictor(LRUEvictor):
    """LRU that records the allocator's `record_hit` and `record_access`
    calls, in order."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def record_hit(self, block_id):
        self.calls.append(("hit", block_id))

    def record_access(self, block_id, content_hash, depth, prev_hash):
        self.calls.append(("access", block_id, content_hash, depth, prev_hash))
        # The base hook LRU inherits does nothing.
        assert super().record_access(block_id, content_hash, depth,
                                     prev_hash) is None

    def accesses(self):
        return [c[1:] for c in self.calls if c[0] == "access"]


def make_allocator(num_blocks=16, spy=True):
    """The block manager's allocator and its GPU allocator, whose evictor
    is a `SpyEvictor` or (``spy=False``) the one `VLLM_EVICTION_POLICY`
    names."""
    alloc = CpuGpuBlockAllocator.create("prefix_caching",
                                        num_gpu_blocks=num_blocks,
                                        num_cpu_blocks=0,
                                        block_size=BS)
    gpu = alloc._allocators[Device.GPU]
    if spy:
        gpu.evictor = SpyEvictor()
    return alloc, gpu


def allocate(alloc, tokens):
    """A prompt's blocks, as `BlockSpaceManager.allocate` builds them."""
    table = BlockTable(BS, alloc)
    table.allocate(list(tokens), device=Device.GPU)
    alloc.mark_blocks_as_computed([])
    return table


# --- record_access: the allocator's calls ------------------------------------


def test_a_computed_prompt_records_each_full_block_once():
    alloc, gpu = make_allocator()
    tokens = list(range(1, 15))  # three full blocks and two tokens
    table = allocate(alloc, tokens)
    h, ids = chain(tokens), table.physical_block_ids
    assert len(ids) == 4
    # Three misses, each block computed and inserted. The partial fourth
    # block has no content hash: it is not cached and not recorded.
    assert gpu.evictor.calls == [
        ("access", ids[0], h[0], 1, None),
        ("access", ids[1], h[1], 2, h[0]),
        ("access", ids[2], h[2], 3, h[1]),
    ]
    assert gpu._cached_blocks == dict(zip(h, ids[:3]))


@pytest.mark.parametrize("free_first", [False, True])
def test_lookup_hits_record_each_block_once(free_first):
    """A second request reads the first one's two full blocks, while the
    first still holds them or after it freed them (they are then revived
    from the evictor)."""
    alloc, gpu = make_allocator()
    tokens = list(range(1, 11))  # two full blocks and two tokens
    first = allocate(alloc, tokens)
    cached = first.physical_block_ids[:2]
    if free_first:
        first.free()
        assert gpu.evictor.num_blocks == 2
    gpu.evictor.calls.clear()
    second = allocate(alloc, tokens)
    h, ids = chain(tokens), second.physical_block_ids
    assert ids[:2] == cached
    assert gpu.evictor.calls == [
        ("hit", ids[0]),
        ("access", ids[0], h[0], 1, None),
        ("hit", ids[1]),
        ("access", ids[1], h[1], 2, h[0]),
    ]
    assert gpu.evictor.num_blocks == 0


def test_a_block_is_recorded_once_it_is_full():
    alloc, gpu = make_allocator()
    table = allocate(alloc, [1, 2, 3])  # one partial block
    assert gpu.evictor.calls == []
    table.append_token_ids([4])  # full: inserted
    h = chain(list(range(1, 9)))
    assert gpu.evictor.accesses() == [(table.physical_block_ids[0], h[0], 1,
                                       None)]
    table.append_token_ids([5, 6, 7])  # a new partial block
    assert len(gpu.evictor.calls) == 1
    table.append_token_ids([8])
    assert gpu.evictor.accesses()[1:] == [(table.physical_block_ids[1], h[1],
                                           2, h[0])]


def test_a_deduplicated_decode_block_records_the_cached_block():
    """A and B share a prompt of one full block and two tokens, so each has
    its own partial second block. Both decode the same two tokens: A's fill
    inserts the second block, B's finds it cached and is deduplicated to
    A's block. Each records the cached block once."""
    alloc, gpu = make_allocator()
    prompt = [1, 2, 3, 4, 5, 6]
    a, b = allocate(alloc, prompt), allocate(alloc, prompt)
    a_ids, b_ids = a.physical_block_ids, b.physical_block_ids
    assert a_ids[0] == b_ids[0] and a_ids[1] != b_ids[1]
    gpu.evictor.calls.clear()
    a.append_token_ids([7, 8])
    b.append_token_ids([7, 8])
    h = chain(prompt + [7, 8])
    assert b.physical_block_ids == a_ids
    assert gpu.evictor.calls == [("access", a_ids[1], h[1], 2, h[0])] * 2
    # B's own second block went back to the free pool.
    assert alloc.get_num_free_blocks(Device.GPU) == 16 - 2


def test_every_request_records_each_of_its_full_blocks_once():
    """Prompts that share a prefix and decode four tokens each, one after
    another in an 8-block cache: hits, insertions and evictions mixed. Each
    request records exactly its own full blocks, in order, with depths 1, 2,
    ... and each block's parent."""
    alloc, gpu = make_allocator(num_blocks=8)
    prompts = [
        list(range(1, 13)),
        list(range(1, 9)) + [50, 51, 52, 53, 54],
        list(range(1, 13)) + [60],
        list(range(100, 117)),
    ]
    hits = []
    for prompt in prompts:
        gpu.evictor.calls.clear()
        table = allocate(alloc, prompt)
        table.append_token_ids([70, 71, 72, 73])
        h = chain(prompt + [70, 71, 72, 73])
        accesses = gpu.evictor.accesses()
        assert [a[1] for a in accesses] == h
        assert [a[2] for a in accesses] == list(range(1, len(h) + 1))
        assert [a[3] for a in accesses] == [None] + h[:-1]
        assert [a[0] for a in accesses] == table.physical_block_ids[:len(h)]
        hits.append(sum(c[0] == "hit" for c in gpu.evictor.calls))
        table.free()
    assert hits == [0, 2, 3, 0]


# --- LRU, FIFO and LFU are unchanged -----------------------------------------


@pytest.mark.parametrize("make", [LRUEvictor, FIFOEvictor, LFUEvictor])
def test_existing_evictors_ignore_record_access(make):
    assert make.record_access is Evictor.record_access
    evictor = make()
    evictor.add(1, 101, BS, 1.0)
    evictor.add(2, 102, BS, 2.0)
    # Recording block 1 as just used changes nothing for these policies.
    assert evictor.record_access(1, 101, 1, None) is None
    evictor.record_access(2, 102, 2, 101)
    assert [evictor.evict()[0], evictor.evict()[0]] == [1, 2]


@pytest.mark.parametrize("policy,kept", [("lru", "a"), ("lfu", "a"),
                                         ("fifo", "b")])
def test_existing_policies_still_evict_what_they_did(monkeypatch, policy,
                                                     kept):
    """Four blocks. A ([1-8]) is computed and freed at t=1, B ([101-108]) at
    t=2, and A is read again and freed at t=3. C ([201-208]) then needs two
    blocks: LRU and LFU give up B's, FIFO A's (A entered the pool first)."""
    monkeypatch.setenv("VLLM_EVICTION_POLICY", policy)
    alloc, gpu = make_allocator(num_blocks=4, spy=False)
    assert type(gpu.evictor) is {
        "lru": LRUEvictor,
        "lfu": LFUEvictor,
        "fifo": FIFOEvictor
    }[policy]
    a, b, c = range(1, 9), range(101, 109), range(201, 209)

    def use(tokens, t):
        table = allocate(alloc, tokens)
        alloc.mark_blocks_as_accessed(table.physical_block_ids, t)
        table.free()

    for tokens, t in ((a, 1.0), (b, 2.0), (a, 3.0), (c, 4.0)):
        use(tokens, t)
    survivor = a if kept == "a" else b
    assert set(
        gpu._cached_blocks) == set(chain(list(c))) | set(chain(list(survivor)))


# --- Choosing TTDF and RACE-RP -----------------------------------------------


@pytest.mark.parametrize("value,expected",
                         [("ttdf", EvictionPolicy.TTDF),
                          ("TTDF", EvictionPolicy.TTDF),
                          (" race_rp ", EvictionPolicy.RACE_RP),
                          ("RACE_RP", EvictionPolicy.RACE_RP)])
def test_policy_from_env_knows_the_new_names(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_EVICTION_POLICY", value)
    assert eviction_policy_from_env() == expected
    assert POLICY_NAMES[value.strip().lower()] is expected


def test_unknown_policy_lists_the_new_names(monkeypatch):
    monkeypatch.setenv("VLLM_EVICTION_POLICY", "race")
    with pytest.raises(ValueError,
                       match="'race'.*lru, fifo, lfu, ttdf, race_rp"):
        eviction_policy_from_env()


def test_new_policies_have_their_own_enum_values():
    members = EvictionPolicy.__members__
    assert {"TTDF", "RACE_RP"} <= set(members)
    # No alias: every name has its own value.
    assert len({p.value for p in members.values()}) == len(members)
    assert make_project_evictor(EvictionPolicy.LRU) is None


def test_eviction_params_are_read_when_asked(monkeypatch):
    assert envs.VLLM_EVICTION_PARAMS is None
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", "tau_short=2")
    assert envs.VLLM_EVICTION_PARAMS == "tau_short=2"
    assert "VLLM_EVICTION_PARAMS" in dir(envs)


def test_ttdf_without_settings_uses_its_defaults():
    evictor = make_evictor(EvictionPolicy.TTDF)
    assert type(evictor) is TTDFEvictor
    assert {k: getattr(evictor, k) for k in TTDF_DEFAULTS} == TTDF_DEFAULTS
    assert TTDF_DEFAULTS == {
        "tau_short": 5.0,
        "tau_long": 120.0,
        "w_short": 0.7,
        "w_long": 0.3
    }


def test_ttdf_settings_come_from_the_params_variable(monkeypatch):
    monkeypatch.setenv("VLLM_EVICTION_PARAMS",
                       "tau_short=2, tau_long=60,w_short=0.5,w_long=0.25")
    evictor = make_evictor(EvictionPolicy.TTDF)
    assert (evictor.tau_short, evictor.tau_long, evictor.w_short,
            evictor.w_long) == (2.0, 60.0, 0.5, 0.25)


def test_race_rp_loads_its_sketch(monkeypatch, sketch_file):
    path, saved = sketch_file
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", f"sketch={path}")
    evictor = make_evictor(EvictionPolicy.RACE_RP)
    assert type(evictor) is RaceRPEvictor
    s = evictor.sketch
    assert (s.L, s.P, s.r_bar, s.alpha) == (4, 3, 0.25, 5.0)
    for name in ("planes", "mean", "std", "counts", "rewards"):
        assert np.array_equal(getattr(s, name), getattr(saved, name)), name
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", f"alpha=2,sketch={path}")
    assert make_evictor(EvictionPolicy.RACE_RP).sketch.alpha == 2.0


@pytest.mark.parametrize("params", [None, "", "alpha=2", "sketch="])
def test_race_rp_without_a_sketch_fails(monkeypatch, params):
    if params is not None:
        monkeypatch.setenv("VLLM_EVICTION_PARAMS", params)
    with pytest.raises(ValueError,
                       match=r"race_rp needs VLLM_EVICTION_PARAMS=sketch="):
        make_evictor(EvictionPolicy.RACE_RP)


@pytest.mark.parametrize("policy,params,bad", [
    (EvictionPolicy.TTDF, "tau_shrt=2", "tau_shrt=2"),
    (EvictionPolicy.TTDF, "tau_short", "tau_short"),
    (EvictionPolicy.TTDF, "sketch=x.json", "sketch=x.json"),
    (EvictionPolicy.RACE_RP, "sketch={path},alpha_=1", "alpha_=1"),
    (EvictionPolicy.RACE_RP, "sketch={path},tau_short=2", "tau_short=2"),
])
def test_a_misspelled_setting_fails(monkeypatch, sketch_file, policy, params,
                                    bad):
    monkeypatch.setenv("VLLM_EVICTION_PARAMS",
                       params.format(path=sketch_file[0]))
    with pytest.raises(ValueError, match=f"'{bad}' is not one of"):
        make_evictor(policy)


@pytest.mark.parametrize("policy", ["ttdf", "race_rp"])
def test_the_variable_sets_the_gpu_evictor_only(monkeypatch, sketch_file,
                                                policy):
    monkeypatch.setenv("VLLM_EVICTION_POLICY", policy)
    params = ("tau_short=2,w_long=0.5"
              if policy == "ttdf" else f"sketch={sketch_file[0]},alpha=3")
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", params)
    alloc = CpuGpuBlockAllocator.create("prefix_caching",
                                        num_gpu_blocks=8,
                                        num_cpu_blocks=4,
                                        block_size=BS)
    gpu = alloc._allocators[Device.GPU].evictor
    assert type(alloc._allocators[Device.CPU].evictor) is LRUEvictor
    if policy == "ttdf":
        assert type(gpu) is TTDFEvictor
        assert (gpu.tau_short, gpu.tau_long, gpu.w_short,
                gpu.w_long) == (2.0, 120.0, 0.7, 0.5)
    else:
        assert type(gpu) is RaceRPEvictor
        assert gpu.sketch.alpha == 3.0


@pytest.mark.parametrize("policy,params,error", [
    ("race_rp", None, "race_rp needs VLLM_EVICTION_PARAMS=sketch="),
    ("race_rp", "tau_short=2", "'tau_short=2' is not one of"),
    ("ttdf", "tau_short=0", "TTDF time constants must be positive"),
])
def test_bad_settings_stop_the_allocator(monkeypatch, policy, params, error):
    monkeypatch.setenv("VLLM_EVICTION_POLICY", policy)
    if params:
        monkeypatch.setenv("VLLM_EVICTION_PARAMS", params)
    with pytest.raises(ValueError, match=error):
        CpuGpuBlockAllocator.create("prefix_caching", 8, 0, BS)


@pytest.mark.parametrize("policy", ["ttdf", "race_rp"])
def test_new_evictors_track_exactly_the_cached_blocks(monkeypatch, sketch_file,
                                                      policy):
    """Requests that share prefixes, one after another in an 8-block cache,
    with evictions. After each, the evictor knows every cached block and
    nothing else, every one is idle, every block's parent is cached, and
    the request's blocks were all accessed at its call's time."""
    monkeypatch.setenv("VLLM_EVICTION_POLICY", policy)
    if policy == "race_rp":
        monkeypatch.setenv("VLLM_EVICTION_PARAMS", f"sketch={sketch_file[0]}")
    alloc, gpu = make_allocator(num_blocks=8, spy=False)
    evictor = gpu.evictor
    assert type(evictor) is (TTDFEvictor
                             if policy == "ttdf" else RaceRPEvictor)
    prompts = [
        list(range(1, 13)),
        list(range(1, 9)) + [50, 51, 52, 53, 54],
        list(range(200, 217)),
        list(range(1, 13)) + [60],
        list(range(300, 313)),
        list(range(1, 9)),
    ]
    for step, prompt in enumerate(prompts):
        t = 10.0 * (step + 1)
        cache_events._context.time = t
        table = allocate(alloc, prompt)
        table.append_token_ids([70, 71, 72, 73])
        for h in chain(prompt + [70, 71, 72, 73]):
            assert evictor.blocks[h].last == t
        table.free()
        cached = gpu._cached_blocks
        assert set(evictor.blocks) == set(cached)
        assert evictor.free_table == {b: h for h, b in cached.items()}
        assert all(b.parent is None or b.parent in cached
                   for b in evictor.blocks.values())
    assert evictor.stats["evictions"] > 0


# --- The request context's time ----------------------------------------------


def group(request_id, *seq_ids):
    """A SequenceGroup stand-in for `RequestContext.attribute`."""
    seqs = [SimpleNamespace(seq_id=s) for s in seq_ids]
    return SimpleNamespace(request_id=request_id, get_seqs=lambda: seqs)


def seq(seq_id):
    """A Sequence stand-in (no request id: the context maps it)."""
    return SimpleNamespace(seq_id=seq_id)


def test_a_call_keeps_its_start_time_and_nested_calls_restore_it(clock):
    ctx = cache_events.RequestContext()
    assert ctx.time is None
    with ctx.attribute(group("r1", 7), "allocate"):
        assert (ctx.request_id, ctx.op, ctx.time) == ("r1", "allocate", 100.0)
        clock.now = 101.0
        assert ctx.time == 100.0  # the call's start, not the clock
        with ctx.attribute(seq(7), "append_slots"):
            assert (ctx.request_id, ctx.op, ctx.time) == ("r1", "append_slots",
                                                          101.0)
            clock.now = 102.0
            with ctx.attribute(group("r2", 8), "swap_in"):
                assert (ctx.request_id, ctx.time) == ("r2", 102.0)
            assert (ctx.request_id, ctx.op, ctx.time) == ("r1", "append_slots",
                                                          101.0)
        assert (ctx.request_id, ctx.op, ctx.time) == ("r1", "allocate", 100.0)
    assert (ctx.request_id, ctx.op, ctx.time) == (None, None, None)


def test_the_time_is_restored_when_a_call_raises(clock):
    ctx = cache_events.RequestContext()
    with ctx.attribute(group("r1", 1), "allocate"):
        clock.now = 105.0
        with pytest.raises(RuntimeError), ctx.attribute(seq(1), "free"):
            assert ctx.time == 105.0
            raise RuntimeError("the call failed")
        assert (ctx.op, ctx.time) == ("allocate", 100.0)
    assert ctx.time is None


def test_a_call_time_is_the_wall_clock_at_its_start():
    ctx = cache_events.RequestContext()
    before = time.time()
    with ctx.attribute(group("r1", 1), "allocate"):
        after = time.time()
        assert before <= ctx.time <= after
    assert ctx.time is None


def test_event_time_falls_back_to_the_clock_outside_a_call(clock):
    assert cache_events.event_time() == 100.0
    clock.now = 101.0
    assert cache_events.event_time() == 101.0


def test_event_time_without_a_patched_clock_is_the_wall_clock():
    before = time.time()
    t = cache_events.event_time()
    assert before <= t <= time.time()


def test_set_current_request_stamps_the_time(clock):
    ctx = cache_events._context
    ctx.enabled = False
    clock.now = 101.0
    cache_events.set_current_request("cmpl-w:a:0-0", "allocate")
    assert ctx.enabled
    assert (ctx.request_id, ctx.op, ctx.time) == ("cmpl-w:a:0-0", "allocate",
                                                  101.0)
    clock.now = 103.0
    assert cache_events.event_time() == 101.0  # still that request's time
    cache_events.set_current_request("cmpl-v:b:0-0")
    assert (ctx.op, cache_events.event_time()) == (None, 103.0)


class Manager:
    """A block manager stand-in whose `allocate` reads the event time twice,
    the clock moving in between."""

    def __init__(self, clock):
        self.clock = clock

    @cache_events.attributed("allocate")
    def allocate(self, seq_group):
        first = cache_events.event_time()
        self.clock.now += 5.0
        return first, cache_events.event_time()


def test_every_read_in_a_block_manager_call_gets_its_start_time(clock):
    cache_events.enable_request_context()
    assert Manager(clock).allocate(group("r1", 1)) == (100.0, 100.0)
    assert cache_events._context.time is None
    assert cache_events.event_time() == 105.0


def test_with_the_context_off_reads_get_the_clock(clock):
    cache_events._context.enabled = False
    assert Manager(clock).allocate(group("r1", 1)) == (100.0, 105.0)


@pytest.mark.parametrize("policy", ["ttdf", "race_rp"])
def test_new_evictors_switch_the_context_on(monkeypatch, sketch_file, policy):
    cache_events._context.enabled = False
    monkeypatch.setenv("VLLM_EVICTION_PARAMS",
                       "" if policy == "ttdf" else f"sketch={sketch_file[0]}")
    evictor = make_evictor(POLICY_NAMES[policy])
    assert cache_events._context.enabled
    assert evictor.clock is cache_events.event_time


def test_accesses_and_evictions_are_stamped_with_the_call_time(
        monkeypatch, clock):
    """A TTDF allocator driven as the simulator drives it: each call is
    named with `set_current_request` at a chosen clock time. Two blocks
    of 4 tokens: A is computed at 10 and read again at 20, B computed at
    30; then C needs a block at 40 and TTDF gives up the lower score."""
    monkeypatch.setenv("VLLM_EVICTION_POLICY", "ttdf")
    alloc, gpu = make_allocator(num_blocks=2, spy=False)
    ttdf = gpu.evictor
    assert type(ttdf) is TTDFEvictor
    a, b, c = [1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]
    (ha, ), (hb, ), (hc, ) = chain(a), chain(b), chain(c)

    def use(tokens, t):
        clock.now = t
        cache_events.set_current_request(f"cmpl-w:{t}:0-0", "allocate")
        clock.now = t + 0.5  # later reads within the call keep its time
        table = allocate(alloc, tokens)
        table.free()

    use(a, 10.0)
    use(a, 20.0)
    use(b, 30.0)
    blk = ttdf.blocks[ha]
    assert blk.last == 20.0
    assert blk.c_short == pytest.approx(math.exp(-10 / 5) + 1)
    assert blk.c_long == pytest.approx(math.exp(-10 / 120) + 1)
    assert (ttdf.blocks[hb].last, ttdf.blocks[hb].c_short) == (30.0, 1.0)
    # At 40: A scores 0.7/5 * c_s e^-4 + 0.3/120 * c_l e^-1/6 ~ 0.0071 and
    # B 0.7/5 * e^-2 + 0.3/120 * e^-1/12 ~ 0.0212, so A goes.
    score_a, score_b = ttdf.score(blk, 40.0), ttdf.score(ttdf.blocks[hb], 40.0)
    assert score_a < score_b
    use(c, 40.0)
    assert set(gpu._cached_blocks) == {hb, hc}
    assert ttdf._heap_time == 40.0  # the eviction's scan time
