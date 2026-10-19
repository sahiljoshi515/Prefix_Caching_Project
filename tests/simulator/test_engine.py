"""Engine-level tests: LLMEngine + scheduler + block manager + simulator.

Only gpt2's config and tokenizer are read (never any weights), so these run
offline on a CPU-only machine once the files are cached.
"""
from typing import Dict

import pytest
from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams, TokensPrompt
from vllm.engine.metrics_types import StatLoggerBase
from vllm.outputs import RequestOutput
from vllm.simulator import SimulatorExecutor
from vllm.utils import Device, GiB_bytes
from vllm.worker.cache_engine import CacheEngine

from .conftest import write_trace

pytestmark = [pytest.mark.m1, pytest.mark.skip_global_cleanup]

BLOCK = 16
# 40 ordinary gpt2 token ids: two full blocks plus half a block.
PROMPT_40 = TokensPrompt(prompt_token_ids=list(range(1000, 1040)))
GREEDY = dict(temperature=0.0)


class CaptureStats(StatLoggerBase):
    """Collects the per-step Stats the engine normally sends to loggers."""

    def __init__(self):
        self.stats = []

    def log(self, stats):
        self.stats.append(stats)

    def info(self, type, obj):
        pass


@pytest.fixture
def make_engine(gpt2_dir, tmp_path, monkeypatch):
    """Build an LLMEngine on the simulator with the given trace entries."""

    def _make(trace, env=None, **engine_kwargs):
        monkeypatch.setenv("VLLM_SIMULATOR_TRACE",
                           write_trace(tmp_path / "trace.jsonl", trace))
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        kwargs = dict(model=gpt2_dir,
                      dtype="float16",
                      max_model_len=512,
                      enforce_eager=True)
        kwargs.update(engine_kwargs)
        capture = CaptureStats()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs),
                                            stat_loggers={"capture": capture})
        engine.capture = capture
        return engine

    return _make


@pytest.fixture(scope="module")
def tokenizer(gpt2_dir):
    return AutoTokenizer.from_pretrained(gpt2_dir)


def run_to_completion(engine, max_steps=5000):
    """Step the engine until every request finished; return final outputs."""
    finals: Dict[str, RequestOutput] = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            return finals
        for out in engine.step():
            if out.finished:
                finals[out.request_id] = out
    raise AssertionError("engine did not finish within max_steps")


def record_batches(engine):
    """Log (request_id, is_prompt, do_sample, chunk) per executor call."""
    batches = []
    original = engine.model_executor.execute_model

    def wrapped(execute_model_req):
        batches.append([(m.request_id, m.is_prompt, m.do_sample,
                         m.token_chunk_size)
                        for m in execute_model_req.seq_group_metadata_list])
        return original(execute_model_req)

    engine.model_executor.execute_model = wrapped
    return batches


def record_schedules(engine):
    """Log every (metadata list, SchedulerOutputs) the scheduler produces."""
    schedules = []
    scheduler = engine.scheduler[0]
    original = scheduler.schedule

    def wrapped():
        result = original()
        schedules.append(result[:2])
        return result

    scheduler.schedule = wrapped
    return schedules


def block_bytes(engine):
    return CacheEngine.get_cache_block_size(engine.cache_config,
                                            engine.model_config,
                                            engine.parallel_config)


# --------------------------------------------------------------------------
# Wiring and KV-cache sizing
# --------------------------------------------------------------------------


def test_engine_uses_simulator_executor_on_cpu_platform(make_engine):
    engine = make_engine({})
    assert isinstance(engine.model_executor, SimulatorExecutor)
    assert engine.device_config.device_type == "cpu"
    swap_blocks = engine.cache_config.swap_space_bytes // block_bytes(engine)
    assert engine.cache_config.num_cpu_blocks == swap_blocks


@pytest.mark.parametrize("space_gib, block_size", [(1, 16), (4, 16), (2, 32)])
def test_kv_cache_blocks_follow_space_and_block_size(make_engine, space_gib,
                                                     block_size):
    engine = make_engine({},
                         env={"VLLM_CPU_KVCACHE_SPACE": str(space_gib)},
                         block_size=block_size)
    cache = engine.cache_config
    assert cache.block_size == block_size
    assert cache.cpu_kvcache_space_bytes == space_gib * GiB_bytes
    assert cache.num_gpu_blocks == space_gib * GiB_bytes // block_bytes(engine)
    if (space_gib, block_size) == (1, 16):
        # gpt2 fp16: 2 B * 12 layers * 2 (K, V) * 16 * 12 heads * 64 dims.
        assert block_bytes(engine) == 589_824 and cache.num_gpu_blocks == 1820


def test_num_gpu_blocks_override_wins(make_engine):
    engine = make_engine({}, num_gpu_blocks_override=37)
    assert engine.cache_config.num_gpu_blocks == 37


def test_cache_smaller_than_max_model_len_is_rejected(make_engine):
    with pytest.raises(ValueError, match="max_model_len"):
        make_engine({}, num_gpu_blocks_override=2, max_model_len=512)


def test_unsupported_configurations_are_rejected_early(make_engine):
    with pytest.raises(ValueError, match="parallelism"):
        make_engine({}, tensor_parallel_size=2)
    with pytest.raises(ValueError, match="multi-step"):
        make_engine({}, num_scheduler_steps=2)


# --------------------------------------------------------------------------
# Generation follows the trace and the usual stop rules
# --------------------------------------------------------------------------


def test_trace_text_round_trips_and_stops_at_eos(make_engine, tokenizer):
    response = "Hello from the trace."
    engine = make_engine({"r1": response})
    engine.add_request("r1", "Any prompt",
                       SamplingParams(max_tokens=64, **GREEDY))
    out = run_to_completion(engine)["r1"].outputs[0]
    assert out.text == response
    assert out.finish_reason == "stop"
    expected = tokenizer.encode(response, add_special_tokens=False)
    assert list(out.token_ids) == expected + [tokenizer.eos_token_id]


def test_max_tokens_shorter_than_trace_gives_length(make_engine):
    engine = make_engine({"r1": [2000, 2001, 2002, 2003, 2004]})
    engine.add_request("r1", PROMPT_40, SamplingParams(max_tokens=3, **GREEDY))
    out = run_to_completion(engine)["r1"].outputs[0]
    assert list(out.token_ids) == [2000, 2001, 2002]
    assert out.finish_reason == "length"


def test_ignore_eos_keeps_going_until_max_tokens(make_engine, tokenizer):
    trace = [2000, 2001, 2002]
    engine = make_engine({"r1": trace})
    engine.add_request("r1", PROMPT_40,
                       SamplingParams(max_tokens=8, ignore_eos=True, **GREEDY))
    out = run_to_completion(engine)["r1"].outputs[0]
    assert list(out.token_ids[:3]) == trace
    assert set(out.token_ids[3:]) == {tokenizer.eos_token_id}
    assert len(out.token_ids) == 8 and out.finish_reason == "length"


def test_unknown_request_id_gets_filler_tokens(make_engine, tokenizer):
    engine = make_engine({"r1": [2000]})
    for rid in ("nobody", "nobody-again"):
        engine.add_request(rid, PROMPT_40,
                           SamplingParams(max_tokens=7, **GREEDY))
    finals = run_to_completion(engine)
    first = finals["nobody"].outputs[0]
    assert len(first.token_ids) == 7 and first.finish_reason == "length"
    assert tokenizer.eos_token_id not in first.token_ids
    assert first.token_ids == finals["nobody-again"].outputs[0].token_ids


def test_min_tokens_pads_with_filler_before_eos(make_engine, tokenizer):
    trace = [2000, 2001, 2002]
    engine = make_engine({"r1": trace})
    engine.add_request("r1", PROMPT_40,
                       SamplingParams(max_tokens=16, min_tokens=6, **GREEDY))
    out = run_to_completion(engine)["r1"].outputs[0]
    filler = tokenizer.encode("the", add_special_tokens=False)[-1]
    assert list(
        out.token_ids) == trace + [filler] * 3 + [tokenizer.eos_token_id]
    assert out.finish_reason == "stop"


def test_stop_string_still_applies(make_engine):
    engine = make_engine({"r1": "alpha beta gamma"})
    engine.add_request("r1", "prompt",
                       SamplingParams(max_tokens=16, stop=["beta"], **GREEDY))
    out = run_to_completion(engine)["r1"].outputs[0]
    assert out.text == "alpha "
    assert out.finish_reason == "stop" and out.stop_reason == "beta"


def test_parallel_sampling_sub_requests_follow_trace(make_engine):
    engine = make_engine({"r1": "same answer twice"})
    batches = record_batches(engine)
    engine.add_request("r1", "prompt",
                       SamplingParams(n=2, temperature=1.0, max_tokens=16))
    final = run_to_completion(engine)["r1"]
    assert [o.text for o in final.outputs] == ["same answer twice"] * 2
    # n > 1 is run as two sub-requests that prefill in the same batch.
    assert sorted(rid for rid, _, _, _ in batches[0]) == [
        "r1_parallel_sample_0", "r1_parallel_sample_1"
    ]


# --------------------------------------------------------------------------
# Continuous batching, prefix caching, preemption, chunked prefill
# --------------------------------------------------------------------------


def test_simultaneous_requests_are_batched_together(make_engine):
    traces = {
        f"r{i}": list(range(3000 + 10 * i, 3008 + 10 * i))
        for i in range(4)
    }
    engine = make_engine(traces)
    batches = record_batches(engine)
    for rid in traces:
        engine.add_request(rid, PROMPT_40,
                           SamplingParams(max_tokens=64, **GREEDY))
    finals = run_to_completion(engine)
    assert [m[1] for m in batches[0]] == [True] * 4  # one prefill batch ...
    assert [m[1] for m in batches[1]] == [False] * 4  # ... then 4 decodes
    assert max(s.num_running_sys for s in engine.capture.stats) == 4
    for rid, trace in traces.items():
        assert list(finals[rid].outputs[0].token_ids[:-1]) == trace


def test_prefix_cache_hit_on_repeated_prompt(make_engine):
    engine = make_engine({
        "r1": [2000, 2001],
        "r2": [2000, 2001]
    },
                         enable_prefix_caching=True,
                         block_size=BLOCK)
    params = SamplingParams(max_tokens=16, **GREEDY)
    engine.add_request("r1", PROMPT_40, params)
    first = run_to_completion(engine)["r1"]
    assert first.num_cached_tokens == 0

    schedules = record_schedules(engine)
    engine.add_request("r2", PROMPT_40, params)
    second = run_to_completion(engine)["r2"]
    # The two full blocks (32 tokens) are reused, the half block is not.
    assert second.num_cached_tokens == 2 * BLOCK
    assert engine.scheduler[0].get_prefix_cache_hit_rate(Device.GPU) > 0
    assert engine.capture.stats[-1].gpu_prefix_cache_hit_rate > 0
    # Scheduler-level evidence that does not go through the simulator.
    metas, _ = schedules[0]  # the prefill step of r2
    assert len(metas[0].computed_block_nums) == 2
    assert metas[0].token_chunk_size == 40


def test_prefix_cache_hit_rate_is_negative_when_disabled(make_engine):
    engine = make_engine({})
    assert engine.scheduler[0].get_prefix_cache_hit_rate(Device.GPU) == -1


def test_tiny_cache_preempts_but_outputs_stay_exact(make_engine, tokenizer):
    traces = {
        f"r{i}": list(range(3000 + 50 * i, 3040 + 50 * i))
        for i in range(4)
    }
    engine = make_engine(traces,
                         num_gpu_blocks_override=6,
                         block_size=BLOCK,
                         max_model_len=6 * BLOCK)
    batches = record_batches(engine)
    for rid in traces:
        engine.add_request(
            rid, TokensPrompt(prompt_token_ids=list(range(1000, 1016))),
            SamplingParams(max_tokens=64, **GREEDY))
    finals = run_to_completion(engine)
    for rid, trace in traces.items():
        out = finals[rid].outputs[0]
        assert list(out.token_ids) == trace + [tokenizer.eos_token_id]
        assert out.finish_reason == "stop"
    assert engine.scheduler[0].num_cumulative_preemption >= 1
    assert sum(s.num_preemption_iter for s in engine.capture.stats) >= 1
    # A preempted request is re-prefilled: it shows up as a prompt twice.
    prefills = [
        rid for batch in batches for rid, is_prompt, _, _ in batch if is_prompt
    ]
    assert len(prefills) > len(traces)


def test_chunked_prefill_splits_the_prompt(make_engine):
    trace = [2000, 2001, 2002]
    engine = make_engine({"r1": trace},
                         enable_chunked_prefill=True,
                         max_num_batched_tokens=16,
                         max_num_seqs=4)
    batches = record_batches(engine)
    engine.add_request("r1", PROMPT_40, SamplingParams(max_tokens=16,
                                                       **GREEDY))
    out = run_to_completion(engine)["r1"].outputs[0]
    chunks = [(is_prompt, do_sample, size) for batch in batches
              for _, is_prompt, do_sample, size in batch]
    assert chunks[:3] == [(True, False, 16), (True, False, 16),
                          (True, True, 8)]
    assert chunks[3] == (False, True, 1)
    assert list(out.token_ids[:-1]) == trace and out.finish_reason == "stop"
