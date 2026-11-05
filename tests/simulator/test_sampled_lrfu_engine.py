"""Sampled LRFU inside the engine (specs/007-m3-mixed-workload-policy).

Engine-level smoke tests on the simulator (gpt2's config and tokenizer,
block size 16), with the M2 policy tests' ``make_engine`` fixture:
``VLLM_EVICTION_POLICY=sampled_lrfu`` picks the evictor for the GPU
allocator only, ``VLLM_EVICTION_PARAMS`` reaches it, bad settings stop
startup, prompts sharing a prefix get prefix-cache hits, every block access
is made inside a block-manager call that names its request, and a full
cache evicts only idle blocks, without errors.
"""
import pytest

from vllm import SamplingParams, TokensPrompt
from vllm.core.block import cache_events
from vllm.core.evictor import LRUEvictor
from vllm.core.sampled_lrfu import SampledLRFUEvictor
from vllm.utils import Device

from . import test_eviction_policy_env as m2_engine
from .test_eviction_policy_env import (GREEDY, allocator, lookups,
                                       run_to_completion)

pytestmark = [pytest.mark.m3, pytest.mark.skip_global_cleanup]

# 40 ordinary gpt2 token ids: two full blocks plus half a block.
PROMPT_40 = list(range(1000, 1040))
# The M2 policy tests' engine fixture, shared with this module.
make_engine = m2_engine.make_engine


@pytest.fixture(autouse=True)
def saved_context():
    """Sampled LRFU switches the module's request context on."""
    ctx = cache_events._context
    saved = ctx.enabled, ctx.request_id, ctx.op, ctx.time
    ctx.request_id, ctx.op, ctx.time = None, None, None
    yield
    ctx.enabled, ctx.request_id, ctx.op, ctx.time = saved


@pytest.fixture
def lrfu_engine(make_engine, monkeypatch):
    """`make_engine` for Sampled LRFU with the given settings."""

    def _make(params="", trace=None, **engine_kwargs):
        monkeypatch.setenv("VLLM_EVICTION_PARAMS", params)
        return make_engine("sampled_lrfu", trace, **engine_kwargs)

    return _make


def run(engine, request_id, tokens, max_tokens=4):
    engine.add_request(request_id, TokensPrompt(prompt_token_ids=tokens),
                       SamplingParams(max_tokens=max_tokens, **GREEDY))
    return run_to_completion(engine)[request_id]


def test_variable_selects_sampled_lrfu_for_the_gpu_only(lrfu_engine):
    engine = lrfu_engine("half_life=8,sample_size=2,seed=3")
    evictor = allocator(engine).evictor
    assert type(evictor) is SampledLRFUEvictor
    assert type(allocator(engine, Device.CPU).evictor) is LRUEvictor
    assert (evictor.half_life, evictor.sample_size) == (8.0, 2)
    assert cache_events._context.enabled


def test_without_settings_it_runs_with_its_defaults(make_engine, monkeypatch):
    monkeypatch.delenv("VLLM_EVICTION_PARAMS", raising=False)
    engine = make_engine("sampled_lrfu")
    evictor = allocator(engine).evictor
    assert type(evictor) is SampledLRFUEvictor
    assert (evictor.half_life, evictor.sample_size) == (512.0, 4)


@pytest.mark.parametrize("params,error", [
    ("half_life=0", "half_life must be positive"),
    ("sample_size=0", "sample_size must be at least 1"),
    ("halflife=8", "'halflife=8' is not one of"),
])
def test_bad_settings_stop_startup(lrfu_engine, params, error):
    with pytest.raises(ValueError, match=error):
        lrfu_engine(params)


def test_prompts_sharing_a_prefix_hit(lrfu_engine):
    """r1 computes two full blocks (two steps of the clock), r2 reads them
    (two more): each block was counted by both requests, one half-life of
    2 steps apart."""
    engine = lrfu_engine("half_life=2", trace={"r1": [2000], "r2": [2000]})
    names = []
    record = allocator(engine).evictor.record_access

    def named(block_id, content_hash, depth, prev_hash):
        names.append(cache_events.current_request())
        record(block_id, content_hash, depth, prev_hash)

    allocator(engine).evictor.record_access = named
    run(engine, "r1", PROMPT_40)
    out = run(engine, "r2", PROMPT_40)
    assert lookups(engine, "r1") == ["miss", "miss"]
    assert lookups(engine, "r2") == ["hit", "hit"]
    assert out.num_cached_tokens == 32
    gpu = allocator(engine)
    lrfu = gpu.evictor
    # Both full prompt blocks are cached and idle (r2's decode filled no
    # block).
    cached = set(gpu._cached_blocks.values())
    assert set(lrfu._pool) == set(lrfu.score) == cached
    assert len(cached) == lrfu.num_blocks == 2
    assert lrfu.now == 4
    assert sorted(lrfu.stamp.values()) == [3, 4]
    assert [lrfu.score[b] for b in sorted(cached)] == [1.5, 1.5]
    # Each access is made under the engine id of its request.
    assert names == ["r1", "r1", "r2", "r2"]


def test_outputs_match_lru_when_nothing_is_evicted(make_engine, monkeypatch):
    """A cache that never fills: the policy decides nothing, so the
    engine's outputs (tokens and cached tokens) are LRU's."""
    trace = {"a": [2000, 2001], "b": [2002], "c": [2003, 2004, 2005]}
    prompts = {
        "a": PROMPT_40,
        "b": PROMPT_40[:20] + list(range(3000, 3030)),
        "c": PROMPT_40 + list(range(4000, 4010)),
    }
    results = {}
    for policy in ("lru", "sampled_lrfu"):
        monkeypatch.delenv("VLLM_EVICTION_PARAMS", raising=False)
        engine = make_engine(policy, trace)
        outs = {
            rid: run(engine, rid, tokens, max_tokens=len(trace[rid]))
            for rid, tokens in prompts.items()
        }
        results[policy] = {
            rid:
            (list(o.outputs[0].token_ids), o.num_cached_tokens, o.finished)
            for rid, o in outs.items()
        }
        cache_events.close_all()
    assert results["sampled_lrfu"] == results["lru"]
    assert results["lru"]["c"][1] == 32


def test_a_recomputed_request_counts_its_blocks_once(lrfu_engine):
    """An 8-block cache and two requests of two prompt blocks that each
    decode 60 tokens (six blocks each): the scheduler preempts one, frees
    its blocks and allocates them again (recompute), over and over. Those
    re-allocations hit blocks the request already counted, so they do not
    move the clock: each request moves it at most once per block between
    the block's evictions."""
    engine = lrfu_engine("sample_size=64",
                         num_gpu_blocks_override=8,
                         max_model_len=128)
    lrfu = allocator(engine).evictor
    steps, life = {}, {}
    record, evict = lrfu.record_access, lrfu.evict
    skipped = []

    def counted(block_id, content_hash, depth, prev_hash):
        before = lrfu.now
        record(block_id, content_hash, depth, prev_hash)
        key = (cache_events.current_request(), block_id, life.get(block_id, 0))
        steps[key] = steps.get(key, 0) + lrfu.now - before
        if lrfu.now == before:
            skipped.append(cache_events._context.op)

    def counted_evict():
        block_id, content_hash = evict()
        life[block_id] = life.get(block_id, 0) + 1
        return block_id, content_hash

    lrfu.record_access = counted
    lrfu.evict = counted_evict
    params = SamplingParams(max_tokens=60, ignore_eos=True, **GREEDY)
    for rid, start in (("a", 1000), ("b", 2000)):
        engine.add_request(
            rid, TokensPrompt(prompt_token_ids=list(range(start, start + 32))),
            params)
    finals = run_to_completion(engine)
    assert sorted(finals) == ["a", "b"]
    assert all(len(o.outputs[0].token_ids) == 60 for o in finals.values())
    assert engine.scheduler[0].num_cumulative_preemption > 0
    assert skipped and set(skipped) == {"allocate"}
    assert set(steps.values()) == {1}
    assert lrfu.now == len(steps)
    assert {request for request, _, _ in steps} == {"a", "b"}


def test_a_full_cache_evicts_only_idle_blocks(lrfu_engine):
    """An 8-block cache. A long request (one prompt block, 40 decode
    tokens) runs while six short ones (two prompt blocks, four decode
    tokens, three of them sharing a prompt) come and go one at a time. The
    short ones' cached blocks fill the cache and get evicted; the long
    request's blocks, prompt and decode, are in use the whole time.

    Every eviction gives up a refcount-0 block that no sequence holds, the
    pool stays exactly the idle cached blocks, every block access is made
    inside a block-manager call that names its request, and every request
    finishes."""
    engine = lrfu_engine("sample_size=2,seed=1",
                         num_gpu_blocks_override=8,
                         max_model_len=128)
    gpu = allocator(engine)
    lrfu = gpu.evictor
    manager = engine.scheduler[0].block_manager
    victims, held_counts, unnamed, names = [], [], [], set()
    evict, record = lrfu.evict, lrfu.record_access

    def idle():
        return {
            b: h
            for h, b in gpu._cached_blocks.items()
            if gpu._refcounter.get(b) == 0
        }

    def checked_evict():
        before = idle()
        assert set(lrfu._pool) == set(before)
        block_id, content_hash = evict()
        assert before.get(block_id) == content_hash
        held = {
            b
            for table in manager.block_tables.values()
            for b in table.physical_block_ids
        }
        assert block_id not in held
        held_counts.append(len(held))
        victims.append(content_hash)
        return block_id, content_hash

    def checked_record(block_id, content_hash, depth, prev_hash):
        if cache_events.current_request() is None:
            unnamed.append((block_id, cache_events._context.op))
        names.add(cache_events.current_request())
        record(block_id, content_hash, depth, prev_hash)

    lrfu.evict = checked_evict
    lrfu.record_access = checked_record

    engine.add_request("long",
                       TokensPrompt(prompt_token_ids=list(range(5000, 5016))),
                       SamplingParams(max_tokens=40, **GREEDY))
    finals = {}
    for _ in range(2):  # the long request's prefill and first decode
        engine.step()
    starts = [1000, 2000, 1000, 3000, 1000, 4000]
    for i, start in enumerate(starts):
        rid = f"s{i}"
        engine.add_request(
            rid, TokensPrompt(prompt_token_ids=list(range(start, start + 32))),
            SamplingParams(max_tokens=4, **GREEDY))
        for _ in range(100):
            for out in engine.step():
                if out.finished:
                    finals[out.request_id] = out
            assert set(lrfu._pool) == set(idle())
            if rid in finals:
                break
        assert rid in finals, rid
        assert "long" not in finals  # still holding its blocks
    finals.update(run_to_completion(engine))
    assert sorted(finals) == ["long"] + [f"s{i}" for i in range(6)]
    assert len(finals["long"].outputs[0].token_ids) == 40
    assert all(
        len(finals[f"s{i}"].outputs[0].token_ids) == 4 for i in range(6))
    assert victims and lrfu.stats["evictions"] == len(victims)
    # The long request held at least its prompt and first decode block.
    assert min(held_counts) >= 2
    assert unnamed == []
    # Everything is idle again, and the evictor knows exactly the cache.
    cached = set(gpu._cached_blocks.values())
    assert set(lrfu._pool) == set(lrfu.score) == cached
    assert lrfu.num_blocks == len(cached)
    # Each request's accesses carry its engine id.
    assert names == {"long"} | {f"s{i}" for i in range(6)}
    summary = lrfu.summary()
    assert summary["evictions"] == len(victims)
    assert summary["candidates_per_scan"] <= 2
