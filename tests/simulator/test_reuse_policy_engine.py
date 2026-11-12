"""TTDF and RACE-RP inside the engine (specs/007-m3-mixed-workload-policy).

Engine-level smoke tests on the simulator (gpt2's config and tokenizer,
block size 16), with the M2 policy tests' ``make_engine`` fixture:
``VLLM_EVICTION_POLICY=ttdf|race_rp`` picks the new evictor for the GPU
allocator only, ``VLLM_EVICTION_PARAMS`` reaches it, bad settings stop
startup, prompts sharing a prefix get prefix-cache hits, and a full cache
evicts only idle blocks, without errors.
"""
import numpy as np
import pytest

from vllm import SamplingParams, TokensPrompt
from vllm.core.block import cache_events
from vllm.core.evictor import LRUEvictor
from vllm.core.reuse_evictors import (LONG_S, NUM_FEATURES, RaceRPEvictor,
                                      RaceSketch, TTDFEvictor)
from vllm.utils import Device

from . import test_eviction_policy_env as m2_engine
from .test_eviction_policy_env import (GREEDY, allocator, lookups,
                                       run_to_completion)

pytestmark = [pytest.mark.m3, pytest.mark.skip_global_cleanup]

POLICIES = {"ttdf": TTDFEvictor, "race_rp": RaceRPEvictor}
# 40 ordinary gpt2 token ids: two full blocks plus half a block.
PROMPT_40 = list(range(1000, 1040))
# The M2 policy tests' engine fixture, shared with this module.
make_engine = m2_engine.make_engine


@pytest.fixture(autouse=True)
def saved_context():
    """The new evictors switch the module's request context on."""
    ctx = cache_events._context
    saved = ctx.enabled, ctx.request_id, ctx.op, ctx.time
    ctx.time = None  # outside any block-manager call
    yield
    ctx.enabled, ctx.request_id, ctx.op, ctx.time = saved


@pytest.fixture
def sketch_path(tmp_path):
    sketch = RaceSketch.random(4,
                               3,
                               seed=3,
                               mean=np.zeros(NUM_FEATURES),
                               std=np.ones(NUM_FEATURES),
                               r_bar=0.5)
    rng = np.random.default_rng(1)
    sketch.add(rng.standard_normal((32, NUM_FEATURES)), rng.random(32))
    path = tmp_path / "sketch.json"
    sketch.save(path)
    return path


@pytest.fixture
def new_engine(make_engine, monkeypatch, sketch_path):
    """`make_engine` for TTDF or RACE-RP; RACE-RP gets the test sketch
    before any other `params`."""

    def _make(policy, params="", trace=None, **engine_kwargs):
        sketch = f"sketch={sketch_path}" if policy == "race_rp" else ""
        monkeypatch.setenv("VLLM_EVICTION_PARAMS",
                           ",".join(p for p in (sketch, params) if p))
        return make_engine(policy, trace, **engine_kwargs)

    return _make


def run(engine, request_id, tokens, max_tokens=4):
    engine.add_request(request_id, TokensPrompt(prompt_token_ids=tokens),
                       SamplingParams(max_tokens=max_tokens, **GREEDY))
    return run_to_completion(engine)[request_id]


@pytest.mark.parametrize("policy,params", [("ttdf", "tau_short=2,w_long=0.5"),
                                           ("race_rp", "alpha=2")])
def test_variable_selects_the_new_gpu_evictor_only(new_engine, policy, params):
    engine = new_engine(policy, params)
    evictor = allocator(engine).evictor
    assert type(evictor) is POLICIES[policy]
    assert type(allocator(engine, Device.CPU).evictor) is LRUEvictor
    if policy == "ttdf":
        assert (evictor.tau_short, evictor.tau_long, evictor.w_short,
                evictor.w_long) == (2.0, 120.0, 0.7, 0.5)
    else:
        assert (evictor.sketch.L, evictor.sketch.P,
                evictor.sketch.alpha) == (4, 3, 2.0)


@pytest.mark.parametrize("policy,params,error", [
    ("race_rp", None, "race_rp needs VLLM_EVICTION_PARAMS=sketch="),
    ("ttdf", "tau_shrt=2", "'tau_shrt=2' is not one of"),
])
def test_bad_settings_stop_startup(make_engine, monkeypatch, policy, params,
                                   error):
    if params is None:
        monkeypatch.delenv("VLLM_EVICTION_PARAMS", raising=False)
    else:
        monkeypatch.setenv("VLLM_EVICTION_PARAMS", params)
    with pytest.raises(ValueError, match=error):
        make_engine(policy)


@pytest.mark.parametrize("policy", POLICIES)
def test_prompts_sharing_a_prefix_hit(new_engine, policy):
    engine = new_engine(policy, trace={"r1": [2000], "r2": [2000]})
    run(engine, "r1", PROMPT_40)
    out = run(engine, "r2", PROMPT_40)
    assert lookups(engine, "r1") == ["miss", "miss"]
    assert lookups(engine, "r2") == ["hit", "hit"]
    assert out.num_cached_tokens == 32
    evictor, gpu = allocator(engine).evictor, allocator(engine)
    # Both full prompt blocks are cached and idle, each used by two
    # requests (r2's decode filled no block).
    assert set(evictor.blocks) == set(gpu._cached_blocks)
    assert len(evictor.blocks) == evictor.num_blocks == 2
    for content_hash, block in evictor.blocks.items():
        if policy == "ttdf":
            assert 1.9 < block.c_long <= 2.0
        else:
            assert evictor.history.count(content_hash, block.last, LONG_S) == 2
    first, second = sorted(evictor.blocks.values(), key=lambda b: b.depth)
    assert (first.depth, first.parent) == (1, None)
    assert second.depth == 2 and evictor.blocks[second.parent] is first


@pytest.mark.parametrize("policy", POLICIES)
def test_a_full_cache_evicts_only_idle_blocks(new_engine, policy):
    """An 8-block cache. A long request (one prompt block, 40 decode
    tokens) runs while six short ones (two prompt blocks, four decode
    tokens, three of them sharing a prompt) come and go one at a time. The
    short ones' cached blocks fill the cache and get evicted; the long
    request's blocks, prompt and decode, are in use the whole time.

    Every eviction gives up an idle block that no sequence holds, the cache
    stays prefix-closed (every cached block's parent is cached), every
    clock read of the evictor happens inside a block-manager call, and
    every request finishes."""
    engine = new_engine(policy, num_gpu_blocks_override=8, max_model_len=128)
    gpu = allocator(engine)
    evictor = gpu.evictor
    manager = engine.scheduler[0].block_manager
    victims, held_counts, outside_calls = [], [], []
    evict, clock = evictor.evict, evictor.clock

    def checked_evict():
        idle = dict(evictor.free_table)
        block_id, content_hash = evict()
        assert idle.get(block_id) == content_hash
        held = {
            b
            for table in manager.block_tables.values()
            for b in table.physical_block_ids
        }
        assert block_id not in held
        held_counts.append(len(held))
        assert all(b.parent is None or b.parent in evictor.blocks
                   for b in evictor.blocks.values())
        victims.append(content_hash)
        return block_id, content_hash

    def checked_clock():
        if cache_events._context.time is None:
            outside_calls.append(cache_events._context.op)
        return clock()

    evictor.evict = checked_evict
    evictor.clock = checked_clock

    params = SamplingParams(max_tokens=40, **GREEDY)
    engine.add_request("long",
                       TokensPrompt(prompt_token_ids=list(range(5000, 5016))),
                       params)
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
            if rid in finals:
                break
        assert rid in finals, rid
        assert "long" not in finals  # still holding its blocks
    finals.update(run_to_completion(engine))
    assert sorted(finals) == ["long"] + [f"s{i}" for i in range(6)]
    assert len(finals["long"].outputs[0].token_ids) == 40
    assert all(
        len(finals[f"s{i}"].outputs[0].token_ids) == 4 for i in range(6))
    assert victims and evictor.stats["evictions"] == len(victims)
    # The long request held at least its prompt and first decode block.
    assert min(held_counts) >= 2
    assert outside_calls == []
    # Everything is idle again, and the evictor knows exactly the cache.
    assert set(evictor.blocks) == set(gpu._cached_blocks)
    assert evictor.num_blocks == len(gpu._cached_blocks)
