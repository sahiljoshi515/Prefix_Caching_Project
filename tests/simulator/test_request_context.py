"""The request context reaches the evictor without the event log
(specs/007-m3-mixed-workload-policy FR-002).

An in-process engine on the simulator (gpt2's config and tokenizer, block
size 16) with ``VLLM_CACHE_EVENT_LOG`` unset; a spy on the GPU evictor's
``add`` records which request the context names when each block is freed.
"""
from typing import List

import pytest

from vllm import EngineArgs, LLMEngine, SamplingParams, TokensPrompt
from vllm.core.block import cache_events
from vllm.utils import Device

from .conftest import write_trace

pytestmark = [pytest.mark.m3, pytest.mark.skip_global_cleanup]

# 40 ordinary gpt2 token ids: two full blocks plus half a block.
PROMPT = TokensPrompt(prompt_token_ids=list(range(1000, 1040)))


@pytest.fixture
def engine(gpt2_dir, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "VLLM_SIMULATOR_TRACE",
        write_trace(tmp_path / "trace.jsonl", {
            "cmpl-wa:s:0-0": [2000],
            "cmpl-wb:t:0-0": [2001]
        }))
    monkeypatch.delenv("VLLM_CACHE_EVENT_LOG", raising=False)
    engine = LLMEngine.from_engine_args(
        EngineArgs(model=gpt2_dir,
                   dtype="float16",
                   max_model_len=512,
                   enforce_eager=True,
                   enable_prefix_caching=True,
                   block_size=16))
    saved = cache_events._context.enabled
    yield engine
    cache_events._context.enabled = saved


def spy_on_adds(engine) -> List[str]:
    allocator = engine.scheduler[0].block_manager.block_allocator._allocators[
        Device.GPU]
    seen: List[str] = []
    add = allocator.evictor.add

    def spy(*args, **kwargs):
        seen.append(cache_events.current_request())
        return add(*args, **kwargs)

    allocator.evictor.add = spy
    return seen


def run(engine, request_id: str) -> None:
    engine.add_request(request_id, PROMPT,
                       SamplingParams(max_tokens=1, temperature=0.0))
    for _ in range(100):
        if not engine.has_unfinished_requests():
            return
        engine.step()
    raise AssertionError("did not finish")


def test_freed_blocks_name_their_request_with_the_log_off(engine):
    assert engine.scheduler[0].block_manager._events is None
    cache_events.enable_request_context()
    seen = spy_on_adds(engine)
    run(engine, "cmpl-wa:s:0-0")
    run(engine, "cmpl-wb:t:0-0")
    # Each request frees its two full prompt blocks (wb reuses wa's and
    # frees them again), each time under its own name.
    assert seen == ["cmpl-wa:s:0-0"] * 2 + ["cmpl-wb:t:0-0"] * 2
    assert [cache_events.workload_label(r)
            for r in seen] == ["wa", "wa", "wb", "wb"]
    assert cache_events.current_request() is None  # outside any call


def test_context_stays_off_unless_enabled(engine):
    cache_events._context.enabled = False
    seen = spy_on_adds(engine)
    run(engine, "cmpl-wa:s:0-0")
    assert seen == [None, None]


@pytest.mark.parametrize("engine_id,label", [
    ("cmpl-bailian-chat:12:0-0", "bailian-chat"),
    ("chatcmpl-sharegpt:abc:3", "sharegpt"),
    ("plain", "plain"),
    (None, None),
])
def test_workload_label_of_engine_ids(engine_id, label):
    assert cache_events.workload_label(engine_id) == label
