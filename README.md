# Prefix-sharing simulator for vLLM (COMP 536 project)

This repository is [vLLM](https://github.com/vllm-project/vllm) at commit
`94d545a1a18e20ea8763a6760194589b8a3c9065` plus a small **simulator** that
lets vLLM run its complete serving stack on a CPU-only laptop without a GPU
and without model weights. Request handling, continuous batching, the
scheduler, the KV-cache block manager and prefix caching all run unchanged;
only the model forward pass is replaced by replaying pre-collected responses
from a trace. Upstream's README is kept as `README_vllm.md`; the Milestone 1
design and test notes are in `report.md`.

## What changed

| Path | Purpose |
|---|---|
| `vllm/simulator.py` (new) | `Simulator` (trace replay logic) and `SimulatorExecutor` (the vLLM executor that uses it). |
| `vllm/envs.py` | Declares `VLLM_USE_SIMULATOR` and `VLLM_SIMULATOR_TRACE`. |
| `vllm/platforms/__init__.py` | Pins the CPU platform when the simulator is on, so the engine starts on any machine. |
| `vllm/engine/llm_engine.py`, `vllm/engine/async_llm_engine.py` | Select `SimulatorExecutor` (four lines each). |
| `vllm/entrypoints/launcher.py`, `vllm/entrypoints/openai/api_server.py`, `vllm/utils.py` | macOS fixes: hand the already-bound port to uvicorn instead of binding it twice, and skip a root-only port lookup during shutdown. |
| `constraints-simulator.txt` (new) | Dependency pins contemporary with this vLLM commit. |
| `tests/simulator/` (new) | 59 tests: unit, in-process engine, and a real `vllm serve`. |
| `examples/simulator/` (new) | Example trace and a client that replays it. |

## Setup

Requirements: git, [uv](https://docs.astral.sh/uv/), Python 3.9 to 3.12
(3.12 is what we use), about 1 GB of disk for the virtualenv (uv keeps a
download cache on top of that). No GPU.

```bash
git clone https://github.com/sahiljoshi515/Prefix_Caching_Project.git
cd Prefix_Caching_Project
uv venv --python 3.12 .venv
source .venv/bin/activate

# macOS (the course command, plus the version pins):
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -c constraints-simulator.txt
# Linux without a GPU (pure-Python build, no kernel download):
#   VLLM_TARGET_DEVICE=empty uv pip install --editable . -c constraints-simulator.txt

uv pip install pytest==8.3.3 pytest-asyncio==0.24.0   # only needed for the tests
```

Check the install:

```bash
export VLLM_USE_SIMULATOR=1
python -c "from vllm.platforms import current_platform; print(current_platform)"   # CpuPlatform
python -c "from vllm.engine.llm_engine import LLMEngine; print(LLMEngine._get_executor_cls(None))"  # SimulatorExecutor
```

Notes:

- On macOS, vLLM's `setup.py` builds a pure-Python package regardless of
  `VLLM_USE_PRECOMPILED`; that is exactly what the simulator needs.
- The pins matter: a fresh `pip install` today resolves transformers 5,
  openai 2+ and huggingface_hub 1+, which this vLLM version cannot import.
- The first run of a model downloads only its `config.json` and tokenizer
  files (about 3 MB for `gpt2`, 11 MB for `Qwen/Qwen2.5-0.5B-Instruct`).
  Weights are never downloaded.

## Running the simulator

Three environment variables control it:

| Variable | Meaning |
|---|---|
| `VLLM_USE_SIMULATOR=1` | Replace model execution by trace replay (and pin the CPU platform). |
| `VLLM_SIMULATOR_TRACE=/path/to/trace.jsonl` | The trace to replay (optional; without it every request gets filler tokens). |
| `VLLM_CPU_KVCACHE_SPACE=<GiB>` | Simulated KV-cache budget in GiB (default 4). `--num-gpu-blocks-override N` gives an exact block count instead. |

Start a server and replay the example trace:

```bash
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
export VLLM_SIMULATOR_TRACE=$PWD/examples/simulator/trace_example.jsonl
vllm serve gpt2 --enable-prefix-caching --max-model-len 1024 --port 8000
```

The server answers `/health` after roughly 10 to 30 seconds (the first run
fetches the tokenizer); Ctrl-C stops it cleanly.

In another terminal (same virtualenv):

```bash
python examples/simulator/send_requests.py
# demo-1: 'The capital of France is Paris.' [stop, 8 tokens, 287 ms]
# demo-2: 'Paris, of course.' [stop, 6 tokens, 3 ms]
# demo-3: 'The sea is wide and deep.' [stop, 8 tokens, 3 ms]
# prefix cache hit rate (from /metrics): 0.5
# (token counts include the final EOS token; demo-1 and demo-2 share a
#  prompt, demo-3 shares its first 16-token block with them)
```

Or with curl; the `X-Request-Id` header names the trace entry to replay
(the prompt itself is free text; prefix caching works on full 16-token
blocks, so prompts shorter than a block never produce a hit):

```bash
PROMPT="You are a helpful assistant. Answer the question briefly, accurately and in one sentence, without any preamble or extra commentary. Question: What is the capital of France? Answer:"
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -H 'X-Request-Id: demo-1' \
  -d "{\"model\": \"gpt2\", \"prompt\": \"$PROMPT\", \"max_tokens\": 64}"
curl -s http://localhost:8000/metrics | grep '^vllm:gpu_prefix_cache_hit_rate'   # > 0 once a prompt repeats
```

Chat requests work the same way with any model that has a chat template
(for chat the id can also go in the body as `"request_id"`; completions
only read the header), for example:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --enable-prompt-tokens-details --port 8000
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "request_id": "demo-1", "max_tokens": 64,
       "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

`--enable-prompt-tokens-details` adds `usage.prompt_tokens_details.cached_tokens`
to chat responses, the number of prompt tokens served from the prefix cache.

### Trace format

One JSON object per line. `response` is text (tokenized with the served
model's tokenizer) or `response_token_ids` is a list of token ids. Other
fields are ignored by the server; the example client uses `prompt` to build
its requests (abridged from `examples/simulator/trace_example.jsonl`):

```json
{"request_id": "demo-1", "prompt": "You are a helpful assistant. ... Question: What is the capital of France? Answer:", "response": "The capital of France is Paris."}
{"request_id": "demo-3", "prompt": "You are a helpful assistant. ... Question: Describe the sea in one sentence. Answer:", "response_token_ids": [464, 5417, 318, 3094, 290, 2769, 13]}
```

Rules of replay:

- The response is returned token by token, one per decode step, then the
  model's EOS token, so the request ends with `finish_reason: "stop"`.
- `max_tokens` is still honored (`/v1/completions` defaults to 16, so send a
  larger value or `ignore_eos: true` for fixed lengths). `stop` strings and
  `min_tokens` work as usual.
- A request whose id is not in the trace receives a filler token until
  `max_tokens`.
- `n > 1`, batched prompts and streaming all work; every sub-request maps to
  the same trace entry.

### Useful vLLM flags for experiments

`--enable-prefix-caching` (off by default in this vLLM version),
`--block-size {8,16,32,64,128}`, `--num-gpu-blocks-override N`,
`--swap-space GiB` with `--preemption-mode swap`, `--max-num-seqs`,
`--max-num-batched-tokens`, `--enable-chunked-prefill`,
`--disable-frontend-multiprocessing` (run the engine inside the API process;
easier to debug). See `report.md` for how each one interacts with the
simulator. The KV cache must hold at least `--max-model-len` tokens, as on a
real GPU; the server refuses to start otherwise. With prefix caching or
chunked prefill the CPU platform logs that it casts fp16 to bf16: harmless
here, both are 2 bytes per value.

## Tests

`VLLM_USE_SIMULATOR=1` must be exported before pytest starts (vLLM chooses
its platform when it is imported).

```bash
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
python -m pytest tests/simulator -v -o asyncio_default_fixture_loop_scope=function
```

| File | What it covers | Time |
|---|---|---|
| `tests/simulator/test_simulator.py` | The `Simulator` alone: trace parsing, request-id matching, EOS/filler rules, sampler-output shape. | < 1 s |
| `tests/simulator/test_engine.py` | In-process `LLMEngine` with gpt2's tokenizer: KV-cache sizing, stop rules, batching, prefix-cache hits, preemption, chunked prefill. | ~6 s |
| `tests/simulator/test_online.py` | A real `vllm serve` (both front-ends) through the OpenAI client: completions, chat, streaming, concurrency, `/metrics`. | ~20 s |

vLLM's own CPU-safe tests for the scheduler and block manager still pass
with the simulator enabled (see `report.md`, section 4).

## Known limitations

- Not simulated, rejected at startup with a clear error: LoRA, prompt
  adapters, tensor/pipeline parallelism, multi-step scheduling, speculative
  decoding, pooling/embedding models and `--tokenizer-mode mistral`.
- Prompt logprobs (`echo` + `logprobs` on completions) are not supported.
- Beam search (`use_beam_search`) runs as internal sub-requests with fresh
  ids, so it never matches a trace and only yields filler tokens.
- `gpt2` has no chat template; use `--chat-template` or a chat model for
  `/v1/chat/completions`.
