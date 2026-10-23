"""``VLLM_EVICTION_POLICY`` inside the engine (specs/005-m2-cache-tuning, US1).

Engine-level tests on the simulator: gpt2's config and tokenizer only, block
size 16. The variable must pick the GPU allocator's evictor, leave the
swap-space allocator on LRU, refuse unknown values at startup, and change
which block a full cache gives up.
"""
import json
from typing import Dict

import pytest

from vllm import EngineArgs, LLMEngine, SamplingParams, TokensPrompt
from vllm.core.block import cache_events
from vllm.core.eviction_policies import FIFOEvictor, LFUEvictor
from vllm.core.evictor import LRUEvictor
from vllm.outputs import RequestOutput
from vllm.utils import Device

from .conftest import write_trace

pytestmark = [pytest.mark.m2, pytest.mark.skip_global_cleanup]

BLOCK = 16
GREEDY = dict(temperature=0.0)


@pytest.fixture
def make_engine(gpt2_dir, tmp_path, monkeypatch):
    """Build an LLMEngine on the simulator with a given eviction policy."""
    log_path = tmp_path / "events.jsonl"

    def _make(policy, trace=None, **engine_kwargs):
        monkeypatch.setenv("VLLM_SIMULATOR_TRACE",
                           write_trace(tmp_path / "trace.jsonl", trace or {}))
        monkeypatch.setenv("VLLM_CACHE_EVENT_LOG", str(log_path))
        if policy is None:
            monkeypatch.delenv("VLLM_EVICTION_POLICY", raising=False)
        else:
            monkeypatch.setenv("VLLM_EVICTION_POLICY", policy)
        kwargs = dict(model=gpt2_dir,
                      dtype="float16",
                      max_model_len=512,
                      enforce_eager=True,
                      enable_prefix_caching=True,
                      block_size=BLOCK)
        kwargs.update(engine_kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs))
        engine.log_path = log_path
        return engine

    yield _make
    cache_events.close_all()


def allocator(engine, device=Device.GPU):
    return engine.scheduler[0].block_manager.block_allocator._allocators[
        device]


def run_to_completion(engine, max_steps=5000):
    finals: Dict[str, RequestOutput] = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            return finals
        for out in engine.step():
            if out.finished:
                finals[out.request_id] = out
    raise AssertionError("engine did not finish within max_steps")


@pytest.mark.parametrize("policy,cls", [(None, LRUEvictor),
                                        ("lru", LRUEvictor),
                                        ("fifo", FIFOEvictor),
                                        ("lfu", LFUEvictor)])
def test_variable_selects_the_gpu_evictor_only(make_engine, policy, cls):
    engine = make_engine(policy)
    assert type(allocator(engine).evictor) is cls
    assert type(allocator(engine, Device.CPU).evictor) is LRUEvictor


def test_unknown_policy_stops_startup(make_engine):
    with pytest.raises(ValueError, match="'mru'.*lru, fifo, lfu"):
        make_engine("mru")


def test_lookup_hits_reach_the_evictor(make_engine):
    """The allocator's hit hook feeds LFU's counters (FR-002)."""
    prompt = TokensPrompt(prompt_token_ids=list(range(1000, 1040)))
    engine = make_engine("lfu", {"r1": [2000], "r2": [2000]})
    params = SamplingParams(max_tokens=4, **GREEDY)
    for rid in ("r1", "r2"):
        engine.add_request(rid, prompt, params)
        run_to_completion(engine)
    lfu = allocator(engine).evictor
    # r2 hit both of r1's full blocks; they now sit in the free pool.
    assert lfu.num_blocks == 2
    assert sorted(lfu.hits(b) for b in lfu.free_table) == [1, 1]


def lookups(engine, rid):
    """The request's prompt lookups, in order: "hit" or "miss" per block."""
    cache_events.close_all()
    events = [json.loads(line) for line in engine.log_path.open()][1:-1]
    return [
        e["ev"] for e in events
        if e["req"] == rid and e["op"] == "allocate" and e["ev"] in ("hit",
                                                                     "miss")
    ]


@pytest.mark.parametrize("policy,kinds", [("lru", ["hit", "hit"]),
                                          ("lfu", ["hit", "hit"]),
                                          ("fifo", ["hit", "miss"])])
def test_full_cache_gives_up_the_block_the_policy_names(
        make_engine, policy, kinds):
    """Prompts A, B, A, C, A of two full blocks each, one after another, in
    a 6-block cache. The decode token of each request needs a third block.

    After the second A the free pool holds A's and B's blocks, A's touched
    later and hit once. C takes the last two free blocks, and its decode
    block needs one eviction: LRU and LFU give up B's second block, FIFO
    A's (A entered the pool first). The final A shows which survived.
    """
    prompts = {
        "a1": 1000,
        "b": 2000,
        "a2": 1000,
        "c": 3000,
        "a3": 1000,
    }
    engine = make_engine(policy, {rid: [50]
                                  for rid in prompts},
                         num_gpu_blocks_override=6,
                         max_model_len=6 * BLOCK)
    for rid, start in prompts.items():
        prompt = TokensPrompt(prompt_token_ids=list(range(start, start + 32)))
        engine.add_request(rid, prompt, SamplingParams(max_tokens=4, **GREEDY))
        run_to_completion(engine)
    assert lookups(engine, "a3") == kinds
