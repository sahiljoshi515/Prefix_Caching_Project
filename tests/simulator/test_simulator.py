"""Unit tests for vllm/simulator.py: no engine, no model, no network."""
from typing import Optional

import pytest

from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata
from vllm.simulator import Simulator, candidate_ids, load_trace

from .conftest import write_trace

pytestmark = pytest.mark.m1

EOS = 0
VOCAB = 1000
BLOCK = 16


class StubTokenizer:
    """Just enough of the HF tokenizer API for the simulator."""
    eos_token_id: Optional[int] = EOS

    def __len__(self):
        return VOCAB

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens, "traces must not get BOS/EOS added"
        return [10 + (ord(c) % 500) for c in text]


FILLER = StubTokenizer().encode("the", add_special_tokens=False)[-1]


def make_simulator(tmp_path, entries):
    return Simulator(StubTokenizer(),
                     write_trace(tmp_path / "trace.jsonl", entries), BLOCK)


def meta(request_id,
         seq_id,
         num_prompt_tokens,
         num_output_tokens=0,
         *,
         is_prompt,
         do_sample=True,
         computed_block_nums=None):
    """A SequenceGroupMetadata like the scheduler builds, for one sequence."""
    seq_data = SequenceData.from_seqs(
        list(range(100, 100 + num_prompt_tokens)),
        list(range(200, 200 + num_output_tokens)))
    return SequenceGroupMetadata(request_id=request_id,
                                 is_prompt=is_prompt,
                                 seq_data={seq_id: seq_data},
                                 sampling_params=SamplingParams(max_tokens=64),
                                 block_tables={seq_id: [0]},
                                 do_sample=do_sample,
                                 computed_block_nums=computed_block_nums)


# --------------------------------------------------------------------------
# Trace loading
# --------------------------------------------------------------------------


def test_load_trace_text_and_token_ids(tmp_path):
    path = write_trace(tmp_path / "t.jsonl", {"a": "hi", "b": [5, 6, 7]})
    traces = load_trace(path, StubTokenizer())
    assert traces == {
        "a": StubTokenizer().encode("hi", add_special_tokens=False),
        "b": [5, 6, 7],
    }


def test_load_trace_drops_trailing_eos_and_blank_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"request_id": "a", "response_token_ids": [5, 6, 0]}\n'
                    "\n"
                    '{"request_id": "b", "response": "x", "prompt": "?"}\n')
    traces = load_trace(str(path), StubTokenizer())
    assert traces["a"] == [5, 6]  # EOS at the end is not replayed twice.
    assert traces["b"] == StubTokenizer().encode("x", add_special_tokens=False)


@pytest.mark.parametrize("line", [
    '{"response": "no id"}',
    '{"request_id": "a"}',
    '{"request_id": "a", "response_token_ids": [1, 999999]}',
    '{"request_id": "a", "response_token_ids": "464"}',
    'not json at all',
])
def test_load_trace_rejects_bad_records(tmp_path, line):
    path = tmp_path / "t.jsonl"
    path.write_text(line + "\n")
    with pytest.raises(ValueError, match=":1:"):
        load_trace(str(path), StubTokenizer())


def test_load_trace_missing_file_names_the_variable(tmp_path):
    with pytest.raises(ValueError, match="VLLM_SIMULATOR_TRACE"):
        load_trace(str(tmp_path / "absent.jsonl"), StubTokenizer())


def test_load_trace_keeps_last_duplicate(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"request_id": "a", "response_token_ids": [1]}\n'
                    '{"request_id": "a", "response_token_ids": [2]}\n')
    assert load_trace(str(path), StubTokenizer()) == {"a": [2]}


# --------------------------------------------------------------------------
# Request-id matching
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine_id, trace_key",
    [
        ("abc", "abc"),
        ("cmpl-abc-0", "abc"),  # completions: prefix + prompt index
        ("cmpl-abc-3", "abc"),
        ("cmpl-req-7-0", "req-7"),  # client ids may contain "-<int>"
        ("cmpl-abc", "abc"),
        ("chatcmpl-abc", "abc"),  # chat: prefix only
        ("chatcmpl-req-7", "req-7"),
        ("cmpl-abc-0_parallel_sample_1", "abc"),  # n > 1 sub-requests
        ("chatcmpl-abc_parallel_sample_0", "abc"),
        ("abc_parallel_sample_2", "abc"),
    ])
def test_lookup_strips_engine_decorations(tmp_path, engine_id, trace_key):
    sim = make_simulator(tmp_path, {trace_key: [1, 2, 3]})
    assert sim.lookup(engine_id) == [1, 2, 3]
    assert candidate_ids(engine_id)[-1] == trace_key


def test_lookup_prefers_the_exact_id(tmp_path):
    sim = make_simulator(tmp_path, {"cmpl-x-0": [1], "x": [2]})
    assert sim.lookup("cmpl-x-0") == [1]
    assert sim.lookup("cmpl-x-1") == [2]
    assert sim.lookup("unknown") is None


# --------------------------------------------------------------------------
# Next-token rule
# --------------------------------------------------------------------------


def test_next_token_replays_trace_then_eos(tmp_path):
    sim = make_simulator(tmp_path, {"r": [7, 8, 9]})
    assert [sim.next_token("r", i) for i in range(3)] == [7, 8, 9]
    assert [sim.next_token("r", i) for i in (3, 4, 50)] == [EOS] * 3


def test_next_token_fills_until_min_tokens_before_eos(tmp_path):
    sim = make_simulator(tmp_path, {"r": [7, 8]})
    tokens = [sim.next_token("r", i, min_tokens=4) for i in range(6)]
    assert tokens == [7, 8, FILLER, FILLER, EOS, EOS]


def test_next_token_filler_for_unknown_request(tmp_path):
    sim = make_simulator(tmp_path, {"r": [7]})
    tokens = [sim.next_token("nobody", i) for i in range(200)]
    assert tokens == [FILLER] * 200
    assert EOS not in tokens and 0 <= FILLER < VOCAB


def test_simulator_requires_an_eos_token():

    class NoEos(StubTokenizer):
        eos_token_id = None

    with pytest.raises(ValueError, match="EOS"):
        Simulator(NoEos(), None, BLOCK)


# --------------------------------------------------------------------------
# Sampler-output construction (what the engine consumes)
# --------------------------------------------------------------------------


def test_make_sampler_output_single_prefill_group(tmp_path):
    sim = make_simulator(tmp_path, {"r1": [7, 8]})
    out = sim.make_sampler_output([meta("r1", 11, 5, is_prompt=True)])
    assert len(out.outputs) == 1
    (sample, ) = out.outputs[0].samples
    assert sample.parent_seq_id == 11
    assert sample.output_token == 7
    assert sample.logprobs[7].logprob == 0.0  # the token is a key: required
    assert out.outputs[0].prompt_logprobs is None


def test_make_sampler_output_keeps_batch_order_and_positions(tmp_path):
    sim = make_simulator(tmp_path, {"r1": [7, 8], "r2": [1, 2, 3, 4]})
    out = sim.make_sampler_output([
        meta("r1", 1, 5, is_prompt=True),
        meta("r2", 2, 5, 2, is_prompt=False),  # decode, 2 tokens done
        meta("new", 3, 5, is_prompt=True),  # no trace entry
    ])
    assert [g.samples[0].output_token for g in out.outputs] == [7, 3, FILLER]
    assert [g.samples[0].parent_seq_id for g in out.outputs] == [1, 2, 3]


def test_make_sampler_output_skips_partial_prefill_chunk(tmp_path):
    sim = make_simulator(tmp_path, {"r1": [7, 8]})
    out = sim.make_sampler_output(
        [meta("r1", 1, 40, is_prompt=True, do_sample=False)])
    assert len(out.outputs) == 1  # the group still gets an entry ...
    assert out.outputs[0].samples == []  # ... but no token yet


def test_make_sampler_output_records_prefix_cache_hits(tmp_path):
    sim = make_simulator(tmp_path, {"r1": [7, 8]})
    prefill = meta("r1", 1, 40, is_prompt=True, computed_block_nums=[3, 9])
    decode = meta("r1", 2, 40, 1, is_prompt=False, computed_block_nums=[3, 9])
    sim.make_sampler_output([prefill, decode])
    assert prefill.seq_data[1].get_num_cached_tokens() == 2 * BLOCK
    assert decode.seq_data[2].get_num_cached_tokens() == 0
