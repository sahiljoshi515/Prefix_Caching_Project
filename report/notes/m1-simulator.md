# Milestone 1 report: a prefix-sharing simulator inside vLLM

COMP 536 project. This report explains what was changed in vLLM, how the
simulator works, and how to test it. Setup and run instructions are in the
root [`README.md`](../../README.md).

## 1. Goal and result

The project needs vLLM to run its real serving stack (request handling,
continuous batching, the scheduler, the KV-cache block manager, prefix
caching, metrics) on CPU-only laptops, without a GPU and without a model.
Milestone 1 asked for a `Simulator` class in a separate file that returns
tokens from a pre-collected trace, wired into vLLM so that the engine
"believes" a model is running.

Result: with `VLLM_USE_SIMULATOR=1`, both online entry points of vLLM
(`vllm serve` with its default multiprocessing engine, and the in-process
`AsyncLLMEngine`) run on a Mac with the model forward pass replaced by trace
replay. Seven upstream files were edited (about 30 lines of simulator
wiring in four files plus a small macOS serving fix in three), one source
file was added, and 67 tests cover the simulator from unit level up to a
real server driven through the OpenAI API.

## 2. What was modified

| File | Change | Why |
|---|---|---|
| `vllm/simulator.py` (new) | `Simulator` + `SimulatorExecutor` | The simulation logic and the adapter that plugs it into the engine. |
| `vllm/envs.py` | +2 declarations, +2 entries | `VLLM_USE_SIMULATOR` (switch) and `VLLM_SIMULATOR_TRACE` (trace path), following vLLM's env-var registry. |
| `vllm/platforms/__init__.py` | +9 lines | When simulating, pin the CPU platform. Without it, a source install on macOS or on a GPU-less Linux box resolves to `UnspecifiedPlatform` and engine construction fails with "Failed to infer device type". |
| `vllm/engine/llm_engine.py` | +4 lines in `_get_executor_cls` | Return `SimulatorExecutor` instead of a hardware executor. This is the path used by `vllm serve` (through `MQLLMEngine`) and by the offline `LLM` class. |
| `vllm/engine/async_llm_engine.py` | +4 lines in `_get_executor_cls` | Same hook for `--disable-frontend-multiprocessing`. |
| `vllm/entrypoints/launcher.py`, `vllm/entrypoints/openai/api_server.py`, `vllm/utils.py` | ~15 lines | macOS fixes unrelated to the simulator. The API server bound its port and then uvicorn bound it a second time, which Linux tolerates and macOS rejects ("address already in use"); the pre-bound socket is now handed to uvicorn. On shutdown, a debug-only port lookup called `psutil.net_connections()`, which needs root on macOS and turned every Ctrl-C into a traceback; it is skipped there. Both are the fixes upstream adopted later. |
| `constraints-simulator.txt` (new) | version pins | PyPI today resolves to transformers 5, openai 2+, huggingface_hub 1+, which this Dec-2024 vLLM cannot import. The pins are the versions vLLM's own CI used at this commit. |
| `tests/simulator/` (new) | 3 test files and a `conftest.py`, 67 tests | See section 4. |
| `examples/simulator/` (new) | trace + client | A runnable walkthrough for the README. |
| `README.md` / `README_vllm.md` | project README; upstream README renamed | Submission requirement. |

Not modified, on purpose:

- `vllm/core/scheduler.py`: the scheduler never calls the model. It decides
  which requests run in a step, allocates blocks, and marks them computed at
  schedule time, so prefix caching is entirely scheduler-driven. The engine
  only feeds it the executor's output. Everything measured in later
  milestones therefore comes from unmodified vLLM code.
- `vllm/engine/multiprocessing/engine.py`: `MQLLMEngine` obtains its executor
  through `LLMEngine._get_executor_cls`, so the single hook covers it.

## 3. How the simulator works, in plain language

**The engine loop.** vLLM processes requests in steps. In each step the
scheduler picks a batch: new requests get a *prefill* step (all prompt
tokens at once), running requests get a *decode* step (one new token each).
It reserves KV-cache blocks (16 tokens per block by default) for the tokens
that will be produced, and with prefix caching it reuses blocks whose token
content was seen before. The engine then asks the *executor* to run the
model on that batch and returns one sampled token per sequence. Finally the
output processor appends the token, detokenizes it, checks the stop
conditions (EOS, `max_tokens`, stop strings) and frees the blocks of
finished requests.

**What the simulator replaces.** Only the executor. `SimulatorExecutor`
implements the same interface as the GPU and CPU executors but never loads
weights. For every scheduled sequence it asks `Simulator.next_token()` for
the next token and packs the answers into the exact `SamplerOutput` structure
the real sampler produces, so the rest of the engine cannot tell the
difference.

**Where tokens come from.** A trace is a JSONL file with one object per
request: a `request_id` and either `response` (text, tokenized with the
served model's tokenizer) or `response_token_ids`. The trace position of a
sequence is simply the number of tokens it has generated so far, which the
engine tracks for us (`seq_data.get_output_len()`). This stays correct when a
request is preempted and recomputed, or when a long prompt is prefilled in
chunks. When the response is exhausted the simulator emits the tokenizer's
EOS token, so the request finishes with `finish_reason="stop"` exactly like a
real model would. Requests with no trace entry receive a filler token ("the")
until `max_tokens`, which is the contract vLLM's own benchmark client relies
on (`max_tokens=output_len, ignore_eos=True`).

**How a request is matched to its trace entry.** Clients tag requests with
the trace id: the `X-Request-Id` header on `/v1/completions` and
`/v1/chat/completions`, or the `request_id` field in a chat request body.
vLLM decorates ids on the way in (`cmpl-<id>-0`, `chatcmpl-<id>`,
`<id>_parallel_sample_<i>` for `n > 1`); the simulator strips those
decorations, trying the most specific form first.

**KV-cache size.** The real executor measures free GPU memory and divides it
by the bytes one block needs. The simulator reuses vLLM's own formula
(`CacheEngine.get_cache_block_size`) and a budget given by
`VLLM_CPU_KVCACHE_SPACE` (GiB), exactly how vLLM's CPU backend sizes its
cache. Blocks are pure bookkeeping, so an 80 GB "GPU" costs nothing to
simulate. `--num-gpu-blocks-override N` sets an exact number of blocks and
`--block-size` changes the block size. Swap space (`--swap-space`) provides
"CPU" blocks so that `--preemption-mode swap` can be studied too.

**Prefix caching stays real.** Block hashing, hit/miss accounting, LRU
eviction, the hit-rate gauge in the logs and in `/metrics`, and preemption
all come from unmodified vLLM code. The simulator adds one bookkeeping step
the GPU runner normally does: it records how many prompt tokens of each
request were served from the cache, which surfaces as
`RequestOutput.num_cached_tokens` and, with
`--enable-prompt-tokens-details`, as `usage.prompt_tokens_details.cached_tokens`
in chat responses. That is the per-request "fraction that benefits from
prefix sharing" metric Milestone 2 asks for, with the same caveats as on a
GPU: after a recompute preemption the count can exceed the prompt length
(generated blocks are cached too), and with chunked prefill it also counts
blocks the request's own earlier chunks produced.

**Why the CPU platform is pinned.** vLLM picks a "platform" at import time
from the installed build. A pure-Python build (what macOS and the `empty`
target produce) has no platform, and vLLM refuses to start. Pinning the CPU
platform whenever the simulator is on gives the same engine configuration
on every machine, including GPU boxes, which keeps experiments reproducible.

**What is deliberately ignored.** The lists of blocks to swap in, swap out
or copy (there is no physical cache to move) and prompt logprobs
(`echo=True` with `logprobs` on completions is unsupported). Features the
simulator does not emulate (LoRA, prompt adapters, tensor/pipeline
parallelism, multi-step scheduling, speculative decoding, pooling models,
the mistral tokenizer mode) are rejected at startup with a clear message,
and a KV cache smaller than `max_model_len` is refused like real workers do.
`min_tokens` is honored the way a model with EOS masked out would: filler
tokens are emitted until the minimum is reached.

## 4. How to test comprehensively

All commands run from the repository root with the virtualenv active and
`VLLM_USE_SIMULATOR=1` exported (the platform is chosen when vLLM is
imported, so the variable must be set before pytest starts).

```bash
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
python -m pytest tests/simulator -v -o asyncio_default_fixture_loop_scope=function
```

The suite has three layers, from fast and isolated to slow and realistic:

1. `tests/simulator/test_simulator.py` (28 tests, well under a second): the
   `Simulator` alone with a stub tokenizer. Trace parsing (text and token
   ids, trailing EOS dropped, blank lines, bad records, missing file,
   duplicate ids), the request-id matrix (`cmpl-abc-0`, `chatcmpl-abc`,
   `..._parallel_sample_1`, exact id wins), trace-then-EOS, `min_tokens`,
   filler tokens, and the sampler-output shape for prefill, decode, batches,
   partial chunked-prefill chunks and recorded prefix-cache hits.
2. `tests/simulator/test_engine.py` (19 tests, about 7 s): an in-process
   `LLMEngine` on gpt2's config and tokenizer. Executor and platform wiring;
   KV-cache block counts for several budgets and block sizes (1 GiB and
   16-token blocks give exactly 1820 blocks for gpt2 in fp16);
   `--num-gpu-blocks-override`; a cache smaller than `max_model_len` and
   unsupported configurations rejected at startup; text round trip with
   `finish_reason="stop"` and the EOS id appended; `max_tokens`,
   `ignore_eos`, `min_tokens`, stop strings; `n=2` sub-requests; four
   requests batched in one prefill step; a prefix cache hit on a repeated
   40-token prompt (32 cached tokens, hit rate > 0, two computed blocks seen
   by the scheduler); a 6-block cache that forces preemption while outputs
   stay exact; chunked prefill splitting a 40-token prompt into 16, 16 and
   8 tokens.
3. `tests/simulator/test_online.py` (20 tests, about 17 s): a real
   `vllm serve` subprocess, launched once with the multiprocessing engine and
   once in-process, driven with the `openai` client. Completions and chat
   with `X-Request-Id` and the `request_id` body field, header precedence,
   streaming, batched prompts, 16 concurrent requests each getting its own
   response, `/metrics` showing a positive prefix-cache hit rate, `/health`,
   model listing and the echoed request id.

Two more checks show nothing else changed:

- Upstream regression: vLLM's own CPU-safe tests for the scheduler, block
  manager, prefix-caching allocator, evictor, output processor, sequences
  and CLI parsing (1071 tests) pass identically with and without
  `VLLM_USE_SIMULATOR=1`. The three failures that remain are the same in
  both runs and predate this work: `tests/test_utils.py` opens
  `./data/test_config.yaml` relative to the working directory, so they only
  pass when pytest is started from inside `tests/` as vLLM's CI does.
- Manual run: `README.md` has a `vllm serve` + `curl` walkthrough and
  `examples/simulator/send_requests.py` replays the example trace and prints
  the prefix-cache hit rate from `/metrics`.

Lint: the repository's pinned `ruff`, `isort` and `yapf` pass on every
changed file.

## 5. Knobs for Milestone 2

| Knob | Where | Effect |
|---|---|---|
| `VLLM_CPU_KVCACHE_SPACE=<GiB>` | env | Total simulated KV-cache budget (integer GiB). |
| `--num-gpu-blocks-override N` | CLI | Exact number of cache blocks, overrides the budget. |
| `--block-size {8,16,32}` | CLI | Tokens per block. |
| `--enable-prefix-caching` | CLI | Off by default in vLLM v0; turn it on for the study. |
| `--preemption-mode recompute|swap`, `--swap-space <GiB>` | CLI | What happens under cache pressure. |
| `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-chunked-prefill` | CLI | Batch shape. |
| `--enable-prompt-tokens-details` | CLI | Adds `cached_tokens` to chat `usage`. |
| `X-Request-Id` / `request_id` | client | Which trace entry a request replays. |
| `max_tokens`, `ignore_eos` | client | `/v1/completions` defaults to 16 tokens, so send `max_tokens` at least as long as the trace response, or `ignore_eos=True` with `max_tokens` for exact lengths. |

Observed during exploration, relevant for Milestones 2 and 3:

- The eviction policy is hard-wired to LRU in `vllm/core/evictor.py`; adding
  LFU/FIFO means threading a policy through `CacheConfig`, the scheduler and
  the block allocator.
- LRU timestamps use wall-clock time. The simulator finishes requests in
  microseconds, so measured block-reuse gaps reflect client pacing and
  simulator speed, not model latency; a virtual clock or a per-step latency
  model may be needed.
- Block hashes use Python's `hash()` over token ids (and `None`); the
  result is reproducible across runs on Python 3.12 but not on 3.9 to 3.11,
  so log a content hash of the token ids if hashes must be compared across
  runs on older interpreters.

## 6. Limitations

- Trace text is tokenized with `add_special_tokens=False`; token-id traces
  are exact. `--tokenizer-mode mistral` is rejected.
- Prompt logprobs are not supported; LoRA, prompt adapters, tensor and
  pipeline parallelism, multi-step scheduling, speculative decoding and
  pooling models are rejected at startup. Beam search sub-requests get fresh
  internal ids and therefore only ever receive filler tokens.
- Any chat model works, but gpt2 has no chat template, so chat requests to
  gpt2 need `--chat-template` (the tests pass a one-line template).

## 7. Note on AI assistance

The implementation and the tests were developed with an AI coding assistant
(Claude) that explored the pinned vLLM commit, proposed the executor-level
design, and wrote and verified the code and tests; the prompts and design
decisions should be summarised in the final technical report's AI-usage
section as the course requires.
