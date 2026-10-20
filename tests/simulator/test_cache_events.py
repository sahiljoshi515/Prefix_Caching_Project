"""Prefix-cache event log (specs/004-m2-prefix-sharing-analysis, US1).

Engine-level tests on the simulator: gpt2's config and tokenizer only, block
size 16, ``VLLM_CACHE_EVENT_LOG`` pointing into ``tmp_path``.
"""
import json

import pytest

from vllm import EngineArgs, LLMEngine, SamplingParams, TokensPrompt
from vllm.core.block import cache_events
from vllm.utils import Device

from .conftest import write_trace

pytestmark = [pytest.mark.m2, pytest.mark.skip_global_cleanup]

BLOCK = 16
# 40 ordinary gpt2 token ids: two full blocks plus half a block.
PROMPT_40 = TokensPrompt(prompt_token_ids=list(range(1000, 1040)))
GREEDY = dict(temperature=0.0)


@pytest.fixture
def make_engine(gpt2_dir, tmp_path, monkeypatch):
    """Build an LLMEngine on the simulator, logging cache events."""
    log_path = tmp_path / "events.jsonl"

    def _make(trace, log=True, **engine_kwargs):
        monkeypatch.setenv("VLLM_SIMULATOR_TRACE",
                           write_trace(tmp_path / "trace.jsonl", trace))
        if log:
            monkeypatch.setenv("VLLM_CACHE_EVENT_LOG", str(log_path))
        else:
            monkeypatch.delenv("VLLM_CACHE_EVENT_LOG", raising=False)
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


def run_to_completion(engine, max_steps=5000):
    """Step the engine until every request finished; return final outputs."""
    finals = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            return finals
        for out in engine.step():
            if out.finished:
                finals[out.request_id] = out
    raise AssertionError("engine did not finish within max_steps")


def gpu_allocator(engine):
    return engine.scheduler[0].block_manager.block_allocator._allocators[
        Device.GPU]


def read_log(engine):
    """Close the recorder and return (header, events, last line)."""
    cache_events.close_all()
    lines = [json.loads(line) for line in engine.log_path.open()]
    return lines[0], lines[1:-1], lines[-1]


def kinds(events, req=None, op=None):
    return [
        e["ev"] for e in events
        if (req is None or e["req"] == req) and (op is None or e["op"] == op)
    ]


def logged_hit_rate(events):
    hits = kinds(events).count("hit")
    return hits / (hits + kinds(events).count("miss"))


# --------------------------------------------------------------------------
# Log format and switch
# --------------------------------------------------------------------------


def test_log_starts_with_header_and_ends_with_end_line(make_engine):
    engine = make_engine({"r1": [2000]})
    engine.add_request("r1", PROMPT_40, SamplingParams(max_tokens=4, **GREEDY))
    run_to_completion(engine)
    header, events, last = read_log(engine)
    assert header["ev"] == "header"
    assert header["block_size"] == BLOCK
    assert {"time", "t", "pid"} <= header.keys()
    assert last == {"ev": "end"}
    assert events and all(
        e.keys() == {"t", "ev", "hash", "block", "req", "op"} for e in events)
    times = [e["t"] for e in events]
    assert times == sorted(times) and times[0] >= header["t"]


def test_unset_variable_means_no_recorder_and_no_file(make_engine):
    engine = make_engine({"r1": [2000]}, log=False)
    assert engine.scheduler[0].block_manager._events is None
    assert gpu_allocator(engine)._events is None
    engine.add_request("r1", PROMPT_40, SamplingParams(max_tokens=4, **GREEDY))
    run_to_completion(engine)
    assert not engine.log_path.exists()


def test_only_the_gpu_allocator_records(make_engine):
    engine = make_engine({})
    allocators = engine.scheduler[0].block_manager.block_allocator._allocators
    assert allocators[Device.GPU]._events is not None
    assert allocators[Device.CPU]._events is None


# --------------------------------------------------------------------------
# Event sequences
# --------------------------------------------------------------------------


def test_repeated_prompt_misses_then_hits(make_engine):
    engine = make_engine({"r1": [2000, 2001], "r2": [2000, 2001]})
    params = SamplingParams(max_tokens=16, **GREEDY)
    engine.add_request("r1", PROMPT_40, params)
    run_to_completion(engine)
    engine.add_request("r2", PROMPT_40, params)
    second = run_to_completion(engine)["r2"]
    hit_rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.GPU)
    _, events, _ = read_log(engine)

    # Two full blocks: looked up, missed and inserted by r1 ...
    assert kinds(events, "r1", "allocate") == ["miss", "insert"] * 2
    # ... released when r1 finished (the half block has no hash) ...
    assert kinds(events, "r1", "free") == ["free", "free"]
    # ... and found again by r2, coming back from the evictor.
    assert kinds(events, "r2", "allocate") == ["hit", "revive"] * 2
    inserted = [e["hash"] for e in events if e["ev"] == "insert"]
    hit = [e["hash"] for e in events if e["ev"] == "hit"]
    assert hit == inserted and len(set(hit)) == 2
    # SC-003: the log agrees with vLLM's own hit rate.
    assert logged_hit_rate(events) == pytest.approx(hit_rate) == 0.5
    # SC-004: d = cached_tokens - block_size * hits = 0 here.
    assert second.num_cached_tokens == BLOCK * kinds(events, "r2").count("hit")


def test_block_filled_by_decode_is_inserted_from_append_slots(make_engine):
    # 40 prompt tokens + 10 generated: token 48 completes the third block.
    engine = make_engine({"r1": list(range(2000, 2010))})
    engine.add_request("r1", PROMPT_40, SamplingParams(max_tokens=32,
                                                       **GREEDY))
    run_to_completion(engine)
    _, events, _ = read_log(engine)
    assert kinds(events, "r1", "append_slots") == ["insert"]
    assert kinds(events, "r1", "free") == ["free"] * 3


def test_tiny_cache_evicts_blocks_that_were_freed(make_engine):
    traces = {f"r{i}": [2000] for i in range(5)}
    engine = make_engine(traces,
                         num_gpu_blocks_override=6,
                         max_model_len=6 * BLOCK)
    for i, rid in enumerate(traces):
        prompt = list(range(1000 + 100 * i, 1032 + 100 * i))
        engine.add_request(rid, TokensPrompt(prompt_token_ids=prompt),
                           SamplingParams(max_tokens=4, **GREEDY))
        run_to_completion(engine)
    _, events, _ = read_log(engine)

    evictions = [(i, e) for i, e in enumerate(events) if e["ev"] == "evict"]
    assert evictions
    for i, evict in evictions:
        earlier = [e for e in events[:i] if e["hash"] == evict["hash"]]
        assert earlier[-1]["ev"] == "free"
        # The request whose allocation needed the block is recorded.
        assert evict["req"] is not None
        assert evict["op"] in ("allocate", "append_slots")
        assert evict["req"] != earlier[-1]["req"]


def test_concurrent_requests_share_a_block_without_revive(make_engine):
    engine = make_engine({"r1": [2000], "r2": [2000]})
    params = SamplingParams(max_tokens=4, **GREEDY)
    engine.add_request("r1", PROMPT_40, params)
    engine.add_request("r2", PROMPT_40, params)
    finals = run_to_completion(engine)
    _, events, _ = read_log(engine)

    assert kinds(events, "r1", "allocate") == ["miss", "insert"] * 2
    # r1's blocks are still in use, so nothing comes back from the evictor.
    assert kinds(events, "r2", "allocate") == ["hit", "hit"]
    # vLLM counts no cached tokens for r2: r1 had not computed the blocks
    # yet (same step). This is the d < 0 case of SC-004.
    assert finals["r2"].num_cached_tokens == 0


def test_swapped_requests_log_swap_in_lookups(make_engine):
    traces = {
        f"r{i}": list(range(3000 + 50 * i, 3040 + 50 * i))
        for i in range(4)
    }
    engine = make_engine(traces,
                         num_gpu_blocks_override=6,
                         max_model_len=6 * BLOCK,
                         preemption_mode="swap")
    for rid in traces:
        engine.add_request(
            rid, TokensPrompt(prompt_token_ids=list(range(1000, 1016))),
            SamplingParams(max_tokens=64, **GREEDY))
    run_to_completion(engine)
    hit_rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.GPU)
    _, events, _ = read_log(engine)

    assert engine.scheduler[0].num_cumulative_preemption >= 1
    swap_in = [e for e in events if e["op"] == "swap_in"]
    assert {"hit", "miss"} & set(kinds(swap_in))
    assert all(e["req"] in traces for e in swap_in)
    # Swap-in lookups count toward vLLM's hit rate, and so in the log.
    assert logged_hit_rate(events) == pytest.approx(hit_rate)


# --------------------------------------------------------------------------
# Recorder unit behaviour
# --------------------------------------------------------------------------


def test_recorder_flushes_by_count_and_on_close(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_events, "FLUSH_EVENTS", 3)
    monkeypatch.setattr(cache_events, "FLUSH_SECONDS", 3600.0)
    path = tmp_path / "unit.jsonl"
    recorder = cache_events.CacheEventRecorder(str(path), block_size=16)
    for block_id in range(2):
        recorder.log("miss", 11, block_id)
    assert len(path.read_text().splitlines()) == 1  # header only
    recorder.log("miss", 11, 2)
    assert len(path.read_text().splitlines()) == 4
    recorder.close()
    recorder.close()  # idempotent
    assert path.read_text().splitlines()[-1] == '{"ev": "end"}'


def test_recorder_flushes_after_flush_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_events, "FLUSH_SECONDS", 0.0)
    path = tmp_path / "unit.jsonl"
    recorder = cache_events.CacheEventRecorder(str(path), block_size=16)
    recorder.log("hit", 11, 0)
    assert len(path.read_text().splitlines()) == 2
    recorder.close()


def test_recorder_from_env_reuses_one_recorder_per_path(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_EVENT_LOG", str(tmp_path / "a.jsonl"))
    first = cache_events.recorder_from_env(16)
    assert cache_events.recorder_from_env(16) is first
    cache_events.close_all()
    assert cache_events.recorder_from_env(16) is not first
    cache_events.close_all()
    monkeypatch.delenv("VLLM_CACHE_EVENT_LOG")
    assert cache_events.recorder_from_env(16) is None
