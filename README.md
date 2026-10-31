# Prefix-sharing simulator for vLLM (COMP 536 project)

This repository is [vLLM](https://github.com/vllm-project/vllm) at commit
`94d545a1a18e20ea8763a6760194589b8a3c9065` plus a small **simulator** that
lets vLLM run its complete serving stack on a CPU-only laptop without a GPU
and without model weights. Request handling, continuous batching, the
scheduler, the KV-cache block manager and prefix caching all run unchanged;
only the model forward pass is replaced by replaying pre-collected responses
from a trace. Upstream's README is kept as `README_vllm.md`; the Milestone 1
design and test notes are in
[`report/milestone1/simulator.md`](report/milestone1/simulator.md).

## Project status

| Milestone | What | Status | Where |
|---|---|---|---|
| 1 | Simulator that replaces model execution | Done | `vllm/simulator.py`, `tests/simulator/` |
| 2, Task 1 | Client simulator and ShareGPT preprocessing | Done | `client_simulator/` |
| 2, Task 2 | Cache event log, prefix-sharing metrics, single- vs multi-turn replay | In progress | `vllm/core/block/cache_events.py`, `analysis/`, `specs/004-*` |
| 2, Task 3 | Cache size, block size and eviction policies | Done: 45-run grid, [findings](report/milestone2/task3-cache-tuning.md) | `vllm/core/eviction_policies.py`, `analysis/sweep.py`, `analysis/run_m2_task3.py`, `specs/005-*` |
| 3 | Eviction policies across workloads; a workload-aware policy | Planned | `specs/006-*`, `specs/007-*` |

How the team works (branches, pull requests, tests, CI, documentation) is
in [`CONTRIBUTING.md`](CONTRIBUTING.md). Each feature has a spec, a plan and
a task list under [`specs/`](specs/README.md); what each task found goes
into the technical report under [`report/`](report/README.md).

## What changed

| Path | Purpose |
|---|---|
| `vllm/simulator.py` (new) | `Simulator` (trace replay logic) and `SimulatorExecutor` (the vLLM executor that uses it). |
| `vllm/envs.py` | Declares `VLLM_USE_SIMULATOR`, `VLLM_SIMULATOR_TRACE`, `VLLM_CACHE_EVENT_LOG` and `VLLM_EVICTION_POLICY`. |
| `vllm/core/block/cache_events.py` (new), `prefix_caching_block.py`, `cpu_gpu_block_allocator.py`, `block_manager.py` | Milestone 2: opt-in log of every prefix-cache event (hit, miss, insert, free, revive, evict) with its request; off unless `VLLM_CACHE_EVENT_LOG` is set. |
| `vllm/core/eviction_policies.py` (new), `vllm/core/evictor.py`, `prefix_caching_block.py`, `cpu_gpu_block_allocator.py` | Milestone 2, Task 3: FIFO and LFU evictors next to vLLM's LRU, chosen with `VLLM_EVICTION_POLICY`; the allocator reports each lookup hit to the evictor (LFU counts them). LRU, exactly as before, when unset. |
| `vllm/platforms/__init__.py` | Pins the CPU platform when the simulator is on, so the engine starts on any machine. |
| `vllm/engine/llm_engine.py`, `vllm/engine/async_llm_engine.py` | Select `SimulatorExecutor` (four lines each). |
| `vllm/entrypoints/launcher.py`, `vllm/entrypoints/openai/api_server.py`, `vllm/utils.py` | macOS fixes: hand the already-bound port to uvicorn instead of binding it twice, and skip a root-only port lookup during shutdown. |
| `constraints-simulator.txt` (new) | Dependency pins contemporary with this vLLM commit. |
| `tests/simulator/` (new) | 87 tests: unit, in-process engine, a real `vllm serve`, the cache event log, and the eviction-policy variable. |
| `tests/analysis/` (new) | Milestone 2: 64 tests of the metrics, the Task 2 and Task 3 experiment commands and the server launcher, without a server. |
| `tests/eviction/` (new) | Milestone 2, Task 3: 41 tests of the FIFO and LFU evictors with hand-derived victims, and a randomized check against each policy's definition. |
| `examples/simulator/` (new) | Example trace and a client that replays it. |
| `client_simulator/` (new) | Milestone 2: ShareGPT download and preprocessing, and the replay client (Poisson arrivals, single- and multi-turn, chat template). |
| `analysis/` (new) | Milestone 2: the shared metrics (cached fraction, hits per block, reuse intervals, hit sources, evictions), the server launcher, figures, the Task 2 experiment command and the Task 3 sweep. |
| `Makefile`, `tools/run_project_tests.sh` (new) | Shared commands for setup, tests per milestone, linting and PR checks. |
| `.github/workflows/project-ci.yml`, `pr-requirements.yml`, `tools/pr_check/` (new) | CI: the project's tests on every PR, and a check of each PR against `CONTRIBUTING.md`. |
| `CONTRIBUTING.md`, `.specify/`, `specs/` (new) | Team rules, the project constitution, and a spec per milestone task. |
| `report/` (new) | The technical report: one note of findings per task, the figures it uses, and later its LaTeX source. |

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
uv pip install matplotlib==3.9.2 -c constraints-simulator.txt   # analysis figures
```

Shortcut: `make setup` runs these steps on macOS or Linux (on Linux it takes
CPU-only PyTorch), then `make test` runs the project's tests. `make help`
lists every command.

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

These environment variables control it:

| Variable | Meaning |
|---|---|
| `VLLM_USE_SIMULATOR=1` | Replace model execution by trace replay (and pin the CPU platform). |
| `VLLM_SIMULATOR_TRACE=/path/to/trace.jsonl` | The trace to replay (optional; without it every request gets filler tokens). |
| `VLLM_CPU_KVCACHE_SPACE=<GiB>` | Simulated KV-cache budget in GiB (default 4). `--num-gpu-blocks-override N` gives an exact block count instead. |
| `VLLM_CACHE_EVENT_LOG=/path/to/events.jsonl` | Milestone 2: log every prefix-cache event, one JSON line each (off when unset; see [Analyzing prefix sharing](#analyzing-prefix-sharing-milestone-2-task-2)). |
| `VLLM_EVICTION_POLICY=lru\|fifo\|lfu` | Milestone 2, Task 3: eviction policy of the prefix cache (default `lru`, vLLM's own). It is read only when the prefix cache exists (`--enable-prefix-caching`, v0 engine); the server then logs the policy at startup and refuses unknown values. Definitions: [spec 005](specs/005-m2-cache-tuning/spec.md), FR-003. |

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

### How to tell the prefix cache worked

The trace decides only the *answer*: the server looks the request id up and
never compares prompts. The prefix cache works on the prompt's tokens, as in
real vLLM, so a hit needs the same leading 16-token blocks, whatever the id.
For example, send the same question twice to a fresh Qwen server started as
above (39 prompt tokens with the chat template: 2 full blocks and 7 tokens
that never fill a block):

```bash
for id in demo-1 demo-2; do
  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -H "X-Request-Id: $id" \
    -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "max_tokens": 5,
         "messages": [{"role": "user", "content": "Explain prefix caching in one paragraph, please."}]}'
  echo
done
```

Three views of the same two hits, from coarse to detailed:

1. **vLLM's counter.** `curl -s localhost:8000/metrics | grep
   '^vllm:gpu_prefix_cache_hit_rate'` gives `0.5`: 2 hits out of 4 block
   lookups since the server started.
2. **Per request** (needs `--enable-prompt-tokens-details`). `demo-2`'s
   `usage` has `"prompt_tokens_details": {"cached_tokens": 32}`; `demo-1`'s is
   `null`, because vLLM omits a zero. A different question gets 16: only the
   first block, Qwen's default system prompt, matches.
3. **Per block** (needs `VLLM_CACHE_EVENT_LOG`, see below). `demo-1` logs
   `miss` and `insert` for each block, `demo-2` logs `hit` and `revive`.

The Task 2 experiment checks on every run that 1 and 3 agree (`sc003_pass`
in `summary.json`).

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

### Replaying ShareGPT (Milestone 2)

The client simulator downloads ShareGPT, rebuilds whole conversations, and
replays them against the server with Poisson arrivals, in single-turn (first
exchange only) or multi-turn mode. Each request carries its trace id, so the
server answers with the recorded response. In short:

```bash
# 1. Preprocess once (downloads about 670 MB into data/raw/, gitignored)
python -m client_simulator.process_raw_traces --dataset sharegpt \
  --raw data/raw/sharegpt.json \
  --output data/processed/sharegpt.responses.jsonl \
  --client-output data/processed/sharegpt.conversations.jsonl

# 2. Terminal 1: the simulator server with the response trace
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
export VLLM_SIMULATOR_TRACE="$PWD/data/processed/sharegpt.responses.jsonl"
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching \
  --enable-prompt-tokens-details --max-model-len 8192 --port 8000

# 3. Terminal 2: replay 10 conversations, all turns
python -m client_simulator.client_sim \
  --trace data/processed/sharegpt.conversations.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --mode multi \
  --conversation-rate 2 --think-time 1 --seed 1 --limit 10 \
  --output results/multi.jsonl --capture-metrics
```

`results/multi.jsonl` gets one line per request, with its cached prompt
tokens, and the run's configuration and `/metrics` snapshots are saved next
to it. All options and output fields are in
[`client_simulator/README.md`](client_simulator/README.md).

### Analyzing prefix sharing (Milestone 2, Task 2)

With `VLLM_CACHE_EVENT_LOG=<file>`, the server writes one JSON line per
prefix-cache event of the KV cache (format in
`vllm/core/block/cache_events.py`). Lines are written in batches, so read the
file after stopping the server; a complete log ends with `{"ev": "end"}`.
One command replays the same sample in
both settings, each on a fresh server with the log on, and computes the
metrics and figures of
[`specs/004-m2-prefix-sharing-analysis`](specs/004-m2-prefix-sharing-analysis/spec.md):

```bash
# Needs the preprocessed files from step 1 above and a free port 8000.
python -m analysis.run_m2_task2 --limit 1000 --out results/m2-task2
# Quick check (about 20 s): 20 conversations, no pacing
python -m analysis.run_m2_task2 --limit 20 --conversation-rate inf \
  --think-time 0 --out results/smoke
# Recompute metrics and figures without replaying
python -m analysis.run_m2_task2 --out results/m2-task2 --analyze-only
```

Each setting gets `results/m2-task2/<setting>/` with the client results, the
event log, `run.json` (configuration, commit, KV-cache size), the metrics as
CSV files and `summary.json`, which also holds the self-checks: the log's
hit rate equals vLLM's `gpu_prefix_cache_hit_rate` (SC-003), and how many
requests' `cached_tokens` differ from the hits logged for them (SC-004).
Figures go to `results/m2-task2/figures/` (`--figures` to choose).

### Sweeping cache size, block size and policy (Milestone 2, Task 3)

One command replays Task 2's multi-turn sample once per configuration of
eviction policy, block size and KV-cache capacity, each on a fresh server
with the event log on, and writes one row per configuration to
`summary.csv`. Capacities are in tokens (`--num-gpu-blocks-override` is
capacity / block size), so block sizes compare at equal cache size. Grids
and definitions:
[`specs/005-m2-cache-tuning`](specs/005-m2-cache-tuning/spec.md).

```bash
# Needs the preprocessed files from step 1 above and a free port 8000.
# Quick check (about 2 min): 2 capacities x block size 16 x 3 policies
python -m analysis.run_m2_task3 --grid smoke --out results/smoke-task3
# The full grid: 5 capacities x 3 block sizes x 3 policies, 1,000
# conversations each (about 14 h). Interrupted? Run it again: finished
# configurations are kept, an unfinished one is replayed.
python -m analysis.run_m2_task3 --grid full --out results/m2-task3
# Any axis can be chosen, e.g. one policy at two capacities:
python -m analysis.run_m2_task3 --policies lfu --capacities 65536 262144 \
  --out results/m2-task3-lfu
# Recompute every run's metrics and summary.csv without replaying
python -m analysis.run_m2_task3 --out results/m2-task3 --analyze-only
```

Each configuration gets `results/m2-task3/<policy>-bs<B>-cap<T>/`, laid out
like a Task 2 run folder. `sweep.json` records the sample, the pacing and
every configuration asked for; a folder is only resumed with the same sample
and pacing, without `--grid` or axis options it reuses its recorded grid
(so `--analyze-only` needs only `--out`), and `summary.csv` always lists
every recorded configuration. A configuration whose cache
cannot hold one `--max-model-len` request is listed as `infeasible` and not
run, because vLLM would refuse to start. The command ends with
`best_by_capacity.csv`/`.md` and spec 005's figures in
`results/m2-task3/figures/` (`--figures` to choose, `--no-figures` to skip);
the results are in
[`report/milestone2/task3-cache-tuning.md`](report/milestone2/task3-cache-tuning.md).

### Useful vLLM flags for experiments

`--enable-prefix-caching` (off by default in this vLLM version),
`--block-size {8,16,32,64,128}`, `--num-gpu-blocks-override N`,
`--swap-space GiB` with `--preemption-mode swap`, `--max-num-seqs`,
`--max-num-batched-tokens`, `--enable-chunked-prefill`,
`--disable-frontend-multiprocessing` (run the engine inside the API process;
easier to debug). See `report/milestone1/simulator.md` for how each one
interacts with the simulator. The KV cache must hold at least `--max-model-len` tokens, as on a
real GPU; the server refuses to start otherwise. With prefix caching or
chunked prefill the CPU platform logs that it casts fp16 to bf16: harmless
here, both are 2 bytes per value.

## Tests

The project's tests are grouped by milestone. The `make` commands set the
required environment variables and use `.venv` directly:

```bash
make test        # every milestone
make test-m1     # one milestone: test-m1, test-m2, test-m3
make test-core   # vLLM's own scheduler and block-manager tests (CPU-safe subset)
make ci          # everything CI runs, in the same order
```

Without `make`, export `VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1` before
pytest starts (vLLM chooses its platform when it is imported) and run
`tools/run_project_tests.sh [m1|m2|m3]`. Each test module declares its
milestone with `pytestmark = pytest.mark.m1` (or `m2`, `m3`). Milestone 3
has no tests yet.

Milestone 1 (`make test-m1`, 67 tests):

| File | What it covers | Time |
|---|---|---|
| `tests/simulator/test_simulator.py` | The `Simulator` alone: trace parsing, request-id matching, EOS/filler rules, sampler-output shape. | < 1 s |
| `tests/simulator/test_engine.py` | In-process `LLMEngine` with gpt2's tokenizer: KV-cache sizing, stop rules, batching, prefix-cache hits, preemption, chunked prefill. | ~6 s |
| `tests/simulator/test_online.py` | A real `vllm serve` (both front-ends) through the OpenAI client: completions, chat, streaming, concurrency, `/metrics`. | ~20 s |

Milestone 2 (`make test-m2`, 125 tests):

| File | What it covers | Time |
|---|---|---|
| `tests/simulator/test_cache_events.py` | The cache event log in an in-process engine: miss and insert, then hit and revive on a repeated prompt; blocks written during decode; eviction in a 6-block cache; sharing between concurrent requests; swap-in lookups; the log's hit rate equal to vLLM's; flushing. | ~4 s |
| `tests/analysis/test_metrics.py` | Every metric of spec 004 on a handcrafted log whose answers are worked out by hand. | < 1 s |
| `tests/analysis/test_run_m2_task2.py` | The experiment command without a server: the sample, CSVs and self-checks, all seven figures, refusing to overwrite a run. | ~1 s |
| `tests/analysis/test_server.py` | The server launcher: a busy port is refused, it binds 127.0.0.1, it reads the KV-cache size, it never inherits the shell's cache settings. | < 1 s |
| `tests/analysis/test_sweep.py` | The Task 3 sweep without a server: eviction count and rate, grid expansion with capacity in tokens, infeasible configurations, resuming after an interruption, the summary rows, the command with a faked replay, the recorded grid, figure 4's capacity, the best-configuration table and the figures. | ~2 s |
| `tests/eviction/test_eviction_policies.py` | The FIFO and LFU evictors alone: spec 005's scenarios, every tie-break, reuse and eviction resetting the counters, a bounded heap, and random operations checked against a brute-force reference. | ~3 s |
| `tests/simulator/test_eviction_policy_env.py` | `VLLM_EVICTION_POLICY` in an in-process engine: it picks the GPU evictor only, refuses unknown values, feeds hits to LFU, and makes FIFO give up a different block than LRU and LFU in a 6-block cache. | ~8 s |

vLLM's own CPU-safe tests for the scheduler and block manager still pass
with the simulator enabled (`make test-core`; see
`report/milestone1/simulator.md`, section 4).

Every pull request runs these tests on GitHub (`project-ci`), together with a
check of the PR against `CONTRIBUTING.md` (`pr-requirements`) and the
repository's linters. Run `make lint` and `make check-pr BODY=pr.md` before
opening one.

## Troubleshooting

- **Requests reach another program (macOS).** `vllm serve` listens on
  0.0.0.0 by default, and macOS lets it start even when another program
  already listens on 127.0.0.1 at that port; requests to `localhost` then go
  to the other program. Pass `--host 127.0.0.1` so the server refuses to
  start instead, or pick a free port (`lsof -nP -iTCP:8000 -sTCP:LISTEN`
  shows who has one). `analysis/server.py` does both.
- **`FileNotFoundError` from `os.getcwd()` while importing vLLM.** The
  terminal's current folder no longer exists, typically after an external
  drive was unmounted and mounted again. `cd` back into the repository by
  its full path.
- **The cache event log looks incomplete.** Lines are written in batches;
  stop the server and check for the final `{"ev": "end"}` line.

## Known limitations

- Not simulated, rejected at startup with a clear error: LoRA, prompt
  adapters, tensor/pipeline parallelism, multi-step scheduling, speculative
  decoding, pooling/embedding models and `--tokenizer-mode mistral`.
- Prompt logprobs (`echo` + `logprobs` on completions) are not supported.
- Beam search (`use_beam_search`) runs as internal sub-requests with fresh
  ids, so it never matches a trace and only yields filler tokens.
- `gpt2` has no chat template; use `--chat-template` or a chat model for
  `/v1/chat/completions`.
