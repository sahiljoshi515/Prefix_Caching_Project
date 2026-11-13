# Prefix sharing in LLM serving: a vLLM simulator study (COMP 536)

This repository is [vLLM](https://github.com/vllm-project/vllm) at commit
`94d545a1a18e20ea8763a6760194589b8a3c9065`, turned into a **prefix-sharing
simulator** that runs on a CPU-only laptop (Milestone 1), and the
experiments built on it:
- prefix sharing on ShareGPT chat (Milestone 2);
- cache eviction across workloads and on a mixed workload (Milestone 3).

Request handling, continuous batching, the scheduler, the KV-cache block
manager and prefix caching all run unchanged. Only the model's forward pass
is replaced: responses are replayed from a trace. So no GPU and no model
weights are needed. Upstream vLLM's README is kept as
[`README_vllm.md`](README_vllm.md).

**Contents**
1. [Results at a glance](#results-at-a-glance)
2. [Setup](#1-setup) (5 min)
3. [Quick check](#2-quick-check-about-5-minutes) (5 min)
4. [Get the data](#3-get-the-data)
5. [Milestone 1: the simulator](#4-milestone-1-the-simulator)
6. [Milestone 2: prefix sharing on ShareGPT](#5-milestone-2-prefix-sharing-on-sharegpt)
7. [Milestone 3: eviction policies across workloads](#6-milestone-3-eviction-policies-across-workloads)
8. [Reference](#7-reference): environment variables, trace format, tests, what changed, troubleshooting

Every command below was run from a fresh clone. Every section has a **quick
run** (minutes) and the **full reproduction** behind the reported numbers.
Runtimes are from a 12-core Apple-silicon laptop.

## Results at a glance

| Milestone, task | Question | Answer | Details |
|---|---|---|---|
| 1 | Can vLLM serve without a GPU while keeping its scheduler and prefix cache? | Yes. Only model execution is replaced by trace replay. 67 tests, plus vLLM's own scheduler and block-manager tests. | [report/milestone1/simulator.md](report/milestone1/simulator.md) |
| 2.1 | A client that replays ShareGPT with chat templates and Poisson arrivals | `client_simulator/` | [report/milestone2/task1-client-simulator.md](report/milestone2/task1-client-simulator.md) |
| 2.2 | How much prefix sharing do single- and multi-turn chats get? | Multi-turn: 95% of block lookups hit, almost all within the same conversation. Single-turn: 13%, mostly the shared system prompt. | [report/milestone2/task2-prefix-sharing-analysis.md](report/milestone2/task2-prefix-sharing-analysis.md) |
| 2.3 | Which cache size, block size and eviction policy? | Capacity decides almost everything: 2% reuse at 16K tokens, 95% at 128K. LRU is best; FIFO strands hits behind misses; LFU collapses under pressure. | [report/milestone2/task3-cache-tuning.md](report/milestone2/task3-cache-tuning.md) |
| 3.1 | Best policy for each workload? | Bailian: LFU (15.0% vs LRU 13.3%). AgentBank: LRU (89.5% vs FIFO 55.1%, LFU 18.5%). CC-Bench: no policy helps at this load; the cache thrashes (1.5–2.1%), LFU slightly ahead. | [report/milestone3/task1-workload-eviction.md](report/milestone3/task1-workload-eviction.md) |
| 3.2 | Best policy for a mix of 4 workloads (7 traces)? | TTDF, RACE-RP and our workload-aware `wa` beat LRU on every seed; RACE-RP and `wa` by about 3 points, and they tie with each other. Engine runs at 128K tokens (Phase 2): LRU 55.8%, RACE-RP 58.7%, `wa` 59.1% with oracle workload labels and 58.2% without. Belady, the offline optimum, reaches 68.3% in the replay. | [report/milestone3/task2-mixed-workload-policy.md](report/milestone3/task2-mixed-workload-policy.md), [spec 007 plan](specs/007-m3-mixed-workload-policy/plan.md) (Phase 2) |

## 1. Setup

You need git, [uv](https://docs.astral.sh/uv/), and Python 3.12 (3.9–3.12
work; uv can install it). No GPU.

Disk: about 1 GB for the virtualenv; about 3 GB more for all datasets;
about 22 GB for M2 Task 3 and 20 GB for M3 Task 1's CC-Bench; check
each section before a full run.

```bash
git clone https://github.com/sahiljoshi515/Prefix_Caching_Project.git
cd Prefix_Caching_Project
make setup
```

`make setup` does the following:
- creates `.venv` with Python 3.12;
- installs this vLLM from source in editable mode (`VLLM_USE_PRECOMPILED=1` on macOS, as the course asks; a CPU-only build on Linux), pinned by `constraints-simulator.txt`;
- installs the test, plotting and parquet packages;
- prints `Setup OK: <...CpuPlatform...>`.

`make help` lists every command.

Every command below assumes the virtualenv is active and the simulator is
switched on:

```bash
source .venv/bin/activate
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
```

<details><summary>The same setup without make</summary>

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
# macOS:
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -c constraints-simulator.txt
# Linux without a GPU:
#   VLLM_TARGET_DEVICE=empty uv pip install --editable . -c constraints-simulator.txt \
#     --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install pytest==8.3.3 pytest-asyncio==0.24.0 matplotlib==3.9.2 pyarrow==18.0.0 \
  -c constraints-simulator.txt
```

The pins matter. An unpinned install today resolves transformers 5,
openai 2 and huggingface_hub 1, which this vLLM version cannot import.
</details>

## 2. Quick check (about 5 minutes)

```bash
make ci           # every milestone's tests + vLLM's scheduler/block-manager tests (about 3.5 min)
```

Expected: M1 67 passed; M2 136 passed; M3 1,197 passed and 1 xfailed;
`tests/core` 944 passed; PR checker 39 passed. Then see the simulator serve
requests ([Milestone 1](#4-milestone-1-the-simulator)).

## 3. Get the data

Raw datasets go in `data/raw/`; the processed files go in `data/processed/`.
Both are ignored by git. All sources are public, and none needs a Hugging
Face token. Each adapter downloads its raw file the first time, if missing.

| Dataset | Used by | Source (license) | Download |
|---|---|---|---|
| ShareGPT | M2; M3 T2 | [anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | about 670 MB |
| AgentBank | M3 T1, T2 | [Solaris99/AgentBank](https://huggingface.co/datasets/Solaris99/AgentBank) (Apache-2.0) | 30 MB |
| CC-Bench | M3 T1, T2 | [zai-org/CC-Bench-trajectories](https://huggingface.co/datasets/zai-org/CC-Bench-trajectories) (MIT) | 43 MB |
| Qwen Bailian traces A, B, thinking, coder | M3 T1 (A), T2 (all four) | [alibaba-edu/qwen-bailian-usagetraces-anon](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) (Apache-2.0) | 298 MB |

Process only what you need: M2 needs ShareGPT; M3 Task 1 needs block B;
M3 Task 2 needs blocks A and C. All three take about 3 minutes and need
about 3.5 GB free: about 1 GB of downloads, and 2.3 GB of processed files
(`df -h .` shows what is free).

```bash
# A. ShareGPT (Milestone 2, and the M3 mix): about 670 MB download
python -m client_simulator.process_raw_traces --dataset sharegpt \
  --raw data/raw/sharegpt.json \
  --output data/processed/sharegpt.responses.jsonl \
  --client-output data/processed/sharegpt.conversations.jsonl
```

```bash
# B. Milestone 3, Task 1: AgentBank, CC-Bench and Bailian Trace A
python -m client_simulator.process_raw_traces --dataset agentbank \
  --raw data/raw/agentbank --seed 0 \
  --output data/processed/agentbank.responses.jsonl \
  --client-output data/processed/agentbank.conversations.jsonl
python -m client_simulator.process_raw_traces --dataset ccbench \
  --raw data/raw/ccbench/train.parquet --seed 0 \
  --output data/processed/ccbench.responses.jsonl \
  --client-output data/processed/ccbench.conversations.jsonl
# Bailian: the whole trace (used for the reported result) and a 3,000-request sample
python -m client_simulator.process_raw_traces --dataset bailian \
  --raw data/raw/bailian/qwen_traceA_blksz_16.jsonl --seed 0 \
  --client-output data/processed/bailian-full/requests.jsonl \
  --output data/processed/bailian-full/responses.jsonl
python -m client_simulator.process_raw_traces --dataset bailian \
  --raw data/raw/bailian/qwen_traceA_blksz_16.jsonl --seed 0 --limit 3000 \
  --client-output data/processed/bailian-3000/requests.jsonl \
  --output data/processed/bailian-3000/responses.jsonl
# ... and the 500-request sample the quick run reads
python -m client_simulator.process_raw_traces --dataset bailian \
  --raw data/raw/bailian/qwen_traceA_blksz_16.jsonl --seed 0 --limit 500 \
  --client-output data/processed/bailian/requests.jsonl \
  --output data/processed/bailian/responses.jsonl
```

```bash
# C. Milestone 3, Task 2: its own processing of AgentBank and CC-Bench,
#    and the four Bailian traces (read directly from data/raw/bailian/)
python -m client_simulator.process_raw_traces --dataset agentbank-mix \
  --raw data/raw/agentbank \
  --output data/processed/agentbank-mix.responses.jsonl \
  --client-output data/processed/agentbank-mix.conversations.jsonl
python -m client_simulator.process_raw_traces --dataset ccbench-mix \
  --raw data/raw/ccbench/train.parquet \
  --output data/processed/ccbench-mix.responses.jsonl \
  --client-output data/processed/ccbench-mix.conversations.jsonl
python -c "from pathlib import Path; from client_simulator.adapters import bailian; \
bailian.check_download(Path('data/raw/bailian'))"
```

Notes:
- **Two processings of AgentBank and CC-Bench.** Task 1 (`agentbank`,
  `ccbench`) and Task 2 (`agentbank-mix`, `ccbench-mix`) process the same
  data differently. Task 2 paces agent turns by CC-Bench's recorded tool
  latencies, which only its processing keeps. See
  [client_simulator/README.md](client_simulator/README.md#task-2s-agentbank-mix-and-ccbench-mix).
- **The Bailian adapter refuses to overwrite** a prepared sample. Delete
  the output folder to rebuild it.
- **Keep the raw file names.** The Bailian adapter derives request ids from
  them.

## 4. Milestone 1: the simulator

**What it is.** `vllm/simulator.py` holds a `Simulator`, which returns each
request's next token from a trace, and a `SimulatorExecutor`, which vLLM
uses instead of the GPU executor when `VLLM_USE_SIMULATOR=1`. Everything
else is real vLLM v0 in online-serving mode: the OpenAI-compatible server,
the scheduler, the block manager and the prefix cache. Design, the changed
files and the tests: [report/milestone1/simulator.md](report/milestone1/simulator.md).

**Run it.** Terminal 1 starts a server on the example trace. It answers
`/health` after 10–30 s; the first run downloads only gpt2's tokenizer,
about 3 MB.

```bash
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
export VLLM_SIMULATOR_TRACE=$PWD/examples/simulator/trace_example.jsonl
vllm serve gpt2 --enable-prefix-caching --max-model-len 1024 --port 8000
```

Terminal 2 sends three requests. The last two reuse cached blocks of the
first:

```bash
python examples/simulator/send_requests.py
# demo-1: 'The capital of France is Paris.' [stop, 8 tokens, ...]
# demo-2: 'Paris, of course.' [stop, 6 tokens, ...]
# demo-3: 'The sea is wide and deep.' [stop, 8 tokens, ...]
# prefix cache hit rate (from /metrics): 0.5
```

For a server on another port, pass `--base-url http://localhost:<port>/v1`.
Stop the server with Ctrl-C. **Tests:** `make test-m1` runs 67 tests in
under a minute: the `Simulator` alone, an in-process engine, and a real
`vllm serve`.

## 5. Milestone 2: prefix sharing on ShareGPT

All three tasks need data block A of [Get the data](#3-get-the-data).
Each experiment command starts its own simulator servers (port 8000 must
be free) and serves `Qwen/Qwen2.5-0.5B-Instruct`. Only its config and
tokenizer, about 11 MB, are downloaded.

### Task 1: the client simulator

`client_simulator/client_sim.py` replays conversations against a running
server:
- **Arrivals:** Poisson (`--conversation-rate`), plus a think time between turns.
- **Prompts:** formatted by the model's chat template, with the server applying it via `/v1/chat/completions`.
- **Modes:** single-turn or multi-turn.

```bash
# Terminal 1: a server that answers from the ShareGPT responses
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
export VLLM_SIMULATOR_TRACE=$PWD/data/processed/sharegpt.responses.jsonl
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching \
  --enable-prompt-tokens-details --max-model-len 8192 --port 8000
```

Wait until `curl -s localhost:8000/health` answers. The first start can
take up to 2 minutes, while it loads the tokenizer and builds the KV cache.

```bash
# Terminal 2: replay 10 conversations, every turn (about 30 s)
python -m client_simulator.client_sim \
  --trace data/processed/sharegpt.conversations.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --mode multi \
  --conversation-rate 2 --think-time 1 --seed 1 --limit 10 \
  --output results/multi.jsonl --capture-metrics
```

`results/multi.jsonl` has one line per request, with its cached prompt
tokens. A turn whose prompt outgrows `--max-model-len` is recorded as
`skipped_context`, and the rest of that conversation is dropped
(`abandoned_turns` in the run summary). `results/multi.jsonl.run.json` and the `/metrics` snapshots sit
next to it. `--mode single` replays first turns only. All options:
[client_simulator/README.md](client_simulator/README.md).

### Task 2: single- vs multi-turn prefix sharing

One command replays the same sample twice, single-turn and multi-turn, each
on a fresh server with the cache event log on. It then computes the
metrics of [spec 004](specs/004-m2-prefix-sharing-analysis/spec.md):
- the cached fraction of each request;
- hits per cache block;
- the reuse interval and reuse distance of each block;
- hit sources and the hit rate over time.

It also draws 7 figures, each as PNG and PDF.

```bash
# Quick run (about 30 s): 20 conversations, no pacing
python -m analysis.run_m2_task2 --limit 20 --conversation-rate inf \
  --think-time 0 --out results/m2-task2-smoke
# Full reproduction (about 35 minutes): 1,000 conversations, 1 per second, think time 5 s
python -m analysis.run_m2_task2 --limit 1000 --out results/m2-task2
# Recompute the metrics and figures from a finished run, without replaying
python -m analysis.run_m2_task2 --out results/m2-task2 --analyze-only
```

**Outputs.** Each setting gets `<out>/single/` and `<out>/multi/`, holding:
- the client results and the cache event log (`events.jsonl`);
- `run.json`;
- the metric CSVs;
- `summary.json`, with its self-checks: the log's hit rate equals vLLM's
  `gpu_prefix_cache_hit_rate` (`sc003_pass`).

The figures go to `<out>/figures/`.

**What you should see.** In the full run, `block_hit_rate_log` in
`summary.json`, the share of block lookups that hit, is about 0.95 for
`multi/` and 0.13 for `single/`; our fresh rerun gave 0.953 and 0.127. The
quick run's 20 conversations only check the pipeline and are too few for
these numbers.

### Task 3: cache size, block size and eviction policy

One command replays Task 2's multi-turn sample once per configuration,
each on a fresh server. A configuration is an eviction policy (`lru`,
`fifo`, `lfu`), a block size and a KV-cache capacity in tokens. The
FIFO and LFU evictors are in `vllm/core/eviction_policies.py`; you choose
one with `VLLM_EVICTION_POLICY`. Definitions:
[spec 005](specs/005-m2-cache-tuning/spec.md).

```bash
# Quick run (about 1.5 minutes): 3 policies x block size 16 x 2 small capacities
python -m analysis.run_m2_task3 --grid smoke --out results/m2-task3-smoke
# Full reproduction, the reported 54 configurations (about 18 minutes each, about 16 h in all):
#   3 policies x block sizes 8, 16, 32 x capacities 2^14, 2^16, 2^18, 2^20, 2^21 tokens (45) ...
python -m analysis.run_m2_task3 --grid full --out results/m2-task3
#   ... plus 131,072 tokens for every policy and block size (9)
python -m analysis.run_m2_task3 --out results/m2-task3 --capacities 131072
# Recompute every run's metrics, the summary and the figures, without replaying
python -m analysis.run_m2_task3 --out results/m2-task3 --analyze-only
```

**Resuming.** An interrupted grid resumes when you rerun the same command;
finished runs are kept. `--policies`, `--block-sizes` and `--capacities`
pick any subset. A configuration whose cache cannot hold one
`--max-model-len` request is listed as `infeasible` and skipped, because
vLLM would refuse to start.

**Outputs:**
- one folder per configuration, `<out>/<policy>-bs<B>-cap<T>/`;
- `summary.csv`, one row each;
- `best_by_capacity.md`: the answer, the best configuration at each cache
  size, ranked by usable reuse (also as `.csv`);
- the figures in `<out>/figures/`.

The 54 run folders take about 22 GB. The quick run only checks the pipeline:
20 conversations in a tiny cache, so its ranking of the policies is not the
study's result. That is in
[report/milestone2/task3-cache-tuning.md](report/milestone2/task3-cache-tuning.md).

## 6. Milestone 3: eviction policies across workloads

### Task 1: the best policy for each workload

One runner replays a workload on three fresh servers, one per policy
(LRU, LFU, FIFO), at a fixed 65,536-token cache with 16-token blocks. It
then computes usable token reuse, hit rate, hits per block, reuse
intervals and evictions. `examples/<name>.json` describes each experiment.
Needs data block B. Definitions:
[spec 006](specs/006-m3-workload-eviction/spec.md).

How the workloads are replayed:
- **Bailian** is a token trace. It is sent to `/v1/completions` at its
  recorded timestamps.
- **AgentBank and CC-Bench** are chats, paced like ShareGPT: Poisson
  arrivals at 1 conversation/s, think time 5 s, in the full runs. The quick
  runs use 2 conversations/s and 1 s, to finish in minutes.

```bash
# Quick runs (1-3 minutes each): 20 Bailian requests, 20 AgentBank and 5 CC-Bench conversations
python -m analysis.run_experiment --config examples/bailian_experiment.json --out results/m3t1-bailian-smoke
python -m analysis.run_experiment --config examples/agentbank_smoke.json --out results/m3t1-agentbank-smoke
python -m analysis.run_experiment --config examples/ccbench_smoke.json --out results/m3t1-ccbench-smoke
# Figures from a finished run (no server)
python -m analysis.plot_experiment --out results/m3t1-bailian-smoke
```

```bash
# Full reproduction: the three reported comparisons
python -m analysis.run_experiment --config examples/bailian_full.json --out results/bailian-full-cap65536
python -m analysis.run_experiment --config examples/agentbank_1000.json --out results/agentbank-1000-cap65536
python -m analysis.run_experiment --config examples/ccbench_full.json --out results/ccbench-full-cap65536
# The report's figures
python -m analysis.plot_experiment --out results/bailian-full-cap65536 \
  --figures report/milestone3/figures/task1/bailian-full
python -m analysis.plot_experiment --out results/agentbank-1000-cap65536 \
  --figures report/milestone3/figures/task1/agentbank-1000
python -m analysis.plot_experiment --out results/ccbench-full-cap65536 \
  --figures report/milestone3/figures/task1/ccbench-full
```

**Runtime and disk:**

| Run | Time | Disk |
|---|---|---|
| Bailian | over 6 hours: it replays the whole 2-hour trace in real time, once per policy | — |
| AgentBank | about 1 hour: 19 min per policy | about 0.2 GB per policy |
| CC-Bench | about 1.5 hours: 20 min replay plus 10 min analysis per policy | about 7 GB per policy, 20 GB in all |

Check free space first (`df -h .`). In our rerun, AgentBank's LRU gave
exactly the reported 89.47%; CC-Bench's LRU gave 1.53%, against 1.48%
reported.

**Outputs:**
- `<out>/summary.csv`, one row per policy. The column to compare is
  `usable_reused_fraction`, the share of prompt tokens served from the
  cache;
- one folder per policy, `<out>/<policy>-bs16-cap65536/`, with `run.json`,
  the event log, the client results and `summary.json`;
- the figures, in `<out>/figures/` or the `--figures` folder.

`--analyze-only` recomputes the metrics without replaying.
`examples/bailian_3000.json` runs the 3,000-request sample.

**What you should see.** The quick runs only check the pipeline. Their
samples fit in the 65,536-token cache, so nothing is evicted and the three
policies give the same numbers. CC-Bench's 5 conversations even reuse
about 80% of their tokens, while the full run's 495 conversations thrash
the cache (1.5–2.1%). The policies differ only under the eviction pressure
of the full runs; their numbers are in the
[Task 1 note](report/milestone3/task1-workload-eviction.md).

### Task 2: a policy for a mixed workload

**The setup.** A *balanced mix* of 4 workloads, 7 traces (Bailian's four
traces count as one workload), each trace an equal share by requests:
- ShareGPT;
- AgentBank;
- CC-Bench;
- the Bailian chat, API, thinking and coder traces.

Each run lasts 30 minutes on one shared 131,072-token cache. Measurement
covers requests sent in [600 s, 1800 s). Seed 1 is used only for training
and tuning; seeds 2–4 are reported.

**The policies.**
- Baselines: LRU, LFU and FIFO.
- Three published-style scorers: TTDF, RACE-RP and Sampled LRFU
  (`vllm/core/reuse_evictors.py`, `vllm/core/sampled_lrfu.py`).
- Our workload-aware policy, `wa` (`vllm/core/wa_evictor.py`).
- Two reference points: Belady's offline optimum and a cache that never
  evicts.

How they work: [spec 007](specs/007-m3-mixed-workload-policy/spec.md).
Results:
- Phase 1 (baselines, TTDF, RACE-RP, Sampled LRFU):
  [report/milestone3/task2-mixed-workload-policy.md](report/milestone3/task2-mixed-workload-policy.md).
- Phase 2 (`wa`): the *Phase 2 first experiments* section of
  [specs/007-m3-mixed-workload-policy/plan.md](specs/007-m3-mixed-workload-policy/plan.md).

Needs data blocks A and C. Two kinds of run:
- an **engine run** (`grid`) is a real 30-minute `vllm serve` replay; a
  grid runs 6 at a time;
- a **cache replay** (`sim`) drives vLLM's block allocator directly with
  modelled timing. It takes 15 s to 4 min per run and is used for sweeps.

Before replays, limit NumPy to one thread per process:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

**Quick run (about 8 minutes):**

```bash
python -m analysis.run_m3_task2 plans --grid smoke           # a 3-minute balanced mix (seed 101)
python -m analysis.run_m3_task2 grid --grid smoke --allow-dirty --out results/m3-task2-smoke   # 1 LRU engine run
python -m analysis.run_m3_task2 plans --grid phase1          # the reported plans: balanced, seeds 1-4 (about 3.5 min)
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s2 \
  --policies lru 'wa:labels=oracle,buckets=1,half_life=600' \
  --capacities 131072 --window 600 1800 --out results/m3-quick.jsonl   # 2 cache replays
python -m analysis.sim_table results/m3-quick.jsonl --against lru
```

**What you should see:**
- The smoke engine run finishes in about 3 minutes. Its `grid.json`
  shows `"status": "finished"`, and all 542 requests complete. vLLM's hit
  rate is in the saved metrics:
  `grep '^vllm:gpu_prefix_cache_hit_rate' results/m3-task2-smoke/*/client.jsonl.metrics.after.txt`
  (0.44 in our run).
- The smoke plan is too short for the study's measurement window, so
  `sim_table` can't summarize it.
- `sim` prints each replay's usable reuse over the whole run. `sim_table`
  reports it over the measurement window [600 s, 1800 s), which is the
  study's metric. For seed 2, `wa` scores 57.44 against LRU's 54.22, a
  gain of 3.22 points.

**Reproduce the headline comparison (about 2.5 hours).** This is LRU,
TTDF, RACE-RP and `wa` at 64K/128K/256K in the cache replay, and LRU,
RACE-RP and `wa` in 12 engine runs at 128K. The Phase 1 grid's seed-1
ceiling run provides the timing that RACE-RP trains on.

```bash
python -m analysis.run_m3_task2 plans --grid phase1
python -m analysis.run_m3_task2 replay --plan data/processed/m3/balanced-human-s1 \
  --out results/m3-task2/balanced-human-s1-ceiling --capacity 11851680      # no-eviction run, 30 min
python -m analysis.race_train --sketch 16x6 --out results/m3-race           # RACE-RP's sketch, trained on seed 1
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s{2,3,4} \
  --policies lru lru_leaf ttdf race_rp:sketch=results/m3-race/sketch-L16-P6.json \
    'wa:labels=oracle,buckets=1,half_life=600' 'wa:labels=none,buckets=1,half_life=1800' belady \
  --capacities 65536 131072 262144 --window 600 1800 --out results/m3-explore/heldout.jsonl
python -m analysis.sim_table results/m3-explore/heldout.jsonl --against lru   # the replay table, per workload, gate
python -m analysis.run_m3_task2 grid --grid wa --out results/m3-task2        # 12 engine runs, about 1 h
python -m analysis.tables results/m3-task2/balanced-human-s{2,3,4}-*-cap131072
python -m analysis.sim_table --against lru \
  --engine results/m3-task2/balanced-human-s{2,3,4}-{lru,race_rp,wa_oracle,wa_none}-cap131072
```

`wa`'s settings were chosen on seed 1 from 12 candidates: labels oracle or
none, hit buckets on or off, half-life 120, 600 or 1,800 s. To rerun that
tuning:

```bash
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s1 \
  --policies $(for l in oracle none; do for b in 1 0; do for h in 120 600 1800; do \
    echo "wa:labels=$l,buckets=$b,half_life=$h"; done; done; done) \
  --capacities 131072 --window 600 1800 --out results/m3-explore/wa-tune-s1.jsonl
python -m analysis.sim_table results/m3-explore/wa-tune-s1.jsonl --seeds 1
```

<details><summary><b>Full Phase 1 study</b>: every number and figure of the Task 2 note (78 engine runs, about 8 h at 6 at a time, plus about 2 h of replays)</summary>

Run these in order. Each grid skips finished runs and resumes after an
interruption. A grid refuses a working tree with uncommitted changes; its
runs record their commit. `--shard 1/2` and `--shard 2/2` split a grid,
and its plans, across two machines.

```bash
# 1. Plans: the balanced mix (seeds 1-4, and fast pacing for seed 2), each workload alone,
#    and the extra mixes the capacity sweep compares
python -m analysis.run_m3_task2 plans --grid phase1
python -m analysis.run_m3_task2 plans --grid isolated
for s in 1 3 4; do python -m analysis.mix --mix balanced --pacing fast --seed $s \
  --data data --out data/processed/m3/balanced-fast-s$s; done
for m in chat-heavy agent-heavy; do for s in 1 2; do python -m analysis.mix --mix $m \
  --pacing human --seed $s --data data --out data/processed/m3/$m-human-s$s; done; done

# 2. Engine runs: baselines and ceilings (21), each workload alone (36), and seed 2 at 64K and 256K
python -m analysis.run_m3_task2 grid --grid phase1
python -m analysis.run_m3_task2 grid --grid isolated --jobs 12
for p in lru lfu fifo; do for c in 65536 262144; do python -m analysis.run_m3_task2 replay \
  --plan data/processed/m3/balanced-human-s2 --policy $p --capacity $c \
  --out results/m3-task2/balanced-human-s2-$p-cap$c; done; done

# 3. Cache replays: the capacity sweep with Belady (12 plans), and the measurement-window curve
python -m analysis.run_m3_task2 sim --plans data/processed/m3/{balanced-human,balanced-fast}-s{1,2,3,4} \
  data/processed/m3/{chat-heavy,agent-heavy}-human-s{1,2} --policies lru lfu fifo belady \
  --capacities 32768 49152 65536 98304 131072 196608 262144 393216 --out results/m3-sim/sweep.jsonl
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s{1,2,3,4} \
  --policies lru lfu fifo belady --capacities 524288 786432 1048576 2097152 4194304 \
  --out results/m3-sim/sweep.jsonl
python -m analysis.run_m3_task2 sim --plans data/processed/m3/{balanced-human,balanced-fast}-s{1,2,3,4} \
  data/processed/m3/{chat-heavy,agent-heavy}-human-s{1,2} --policies lru --capacities ceiling \
  --out results/m3-sim/sweep.jsonl
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s{2,3,4} \
  --policies lru lfu fifo belady \
  --capacities 32768 49152 65536 98304 131072 196608 262144 393216 ceiling \
  --window 600 1800 --out results/m3-sim/capacity-window.jsonl

# 4. TTDF, RACE-RP and Sampled LRFU: train, tune on seed 1, engine-run, replay held-out seeds
python -m analysis.race_train --sketch 16x6 --out results/m3-race
python -m analysis.race_train --sketch 16x1 16x2 16x3 16x4 16x6 16x8 64x1 64x2 64x3 64x4 64x6 64x8 \
  --out results/m3-race/tune
#    Seed-1 tuning: 49 settings = LRU + each policy's specified setting + 17 TTDF variants
#    + 12 RACE-RP sketches + 17 Sampled LRFU variants
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s1 --policies lru ttdf sampled_lrfu \
  $(for ts in 1 5 30; do for tl in 60 120 600; do for w in w_short=0.3,w_long=0.7 w_short=0.7,w_long=0.3; do \
    [ "$ts,$tl,$w" = "5,120,w_short=0.7,w_long=0.3" ] || echo "ttdf:tau_short=$ts,tau_long=$tl,$w"; done; done; done) \
  $(for L in 16 64; do for P in 1 2 3 4 6 8; do echo "race_rp:sketch=results/m3-race/tune/sketch-L$L-P$P.json"; done; done) \
  $(for h in 64 512 4096 32768 262144 2097152; do for n in 4 16 64; do \
    [ "$h,$n" = "512,4" ] || echo "sampled_lrfu:half_life=$h,sample_size=$n"; done; done) \
  --capacities 32768 65536 131072 262144 393216 --window 600 1800 --out results/m3-sim/tune-seed1.jsonl
python -m analysis.m3t2_policies select           # each policy's name and its specified or tuned setting
python -m analysis.run_m3_task2 grid --grid policies
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s{2,3,4} \
  --policies $(python -m analysis.m3t2_policies select | cut -f2) \
  --capacities 32768 49152 65536 98304 131072 196608 262144 393216 \
  --window 600 1800 --out results/m3-sim/capacity-window.jsonl
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s{2,3,4} \
  --policies lru lfu fifo ttdf sampled_lrfu race_rp:sketch=results/m3-race/sketch-L16-P6.json \
  --capacities 131072 --window 600 1800 --jobs 1 --out results/m3-sim/overhead.jsonl

# 5. Per-run tables, the study's side analyses, and the report
python -m analysis.tables results/m3-task2/*/
python -m analysis.identify --train data/processed/m3/balanced-human-s1 \
  --test data/processed/m3/balanced-human-s{2,3,4} --out results/m3-study/identify.json
python -m analysis.signals --train results/m3-task2/balanced-human-s1-ceiling \
  --test results/m3-task2/balanced-human-s{2,3,4}-ceiling --out results/m3-study/signals.json
python -m analysis.paper_checks \
  $(for s in 2 3 4; do echo --ceiling results/m3-task2/balanced-human-s$s-ceiling; done) \
  $(for s in 1 2 3 4; do echo --plan data/processed/m3/balanced-human-s$s; done) \
  --out results/m3-study/paper.json
python -m analysis.m3t2_study                     # results/m3-study/summary.json
python -m analysis.m3t2_report                    # results/m3-study/mixed-workload-reuse.html
python -m analysis.m3t2_figures                   # report/milestone3/figures/task2/
python -m analysis.m3t2_policies csv --out report/milestone3/data/task2
```

The cache replay is a design tool. The reported numbers come from engine
runs. The replay lands within 2.1 points of them, except Sampled LRFU at
32K on seed 2, which is 4.1 points lower (note, point 4). The
tracked CSVs in `report/milestone3/data/task2/` are the numbers behind the
note's tables and figures.
</details>

## 7. Reference

### Environment variables

| Variable | Meaning |
|---|---|
| `VLLM_USE_SIMULATOR=1` | Replace model execution by trace replay, and pin the CPU platform. |
| `VLLM_SIMULATOR_TRACE=<file>` | The response trace to replay. Without it, every request gets filler tokens. |
| `VLLM_CPU_KVCACHE_SPACE=<GiB>` | Simulated KV-cache size (default 4). `--num-gpu-blocks-override N` sets an exact block count instead. |
| `VLLM_CACHE_EVENT_LOG=<file>` | Log every prefix-cache event (hit, miss, insert, free, revive, evict) as JSON lines, for the analyses. |
| `VLLM_EVICTION_POLICY` | The prefix cache's eviction policy: `lru` (default, vLLM's own), `fifo`, `lfu`, `ttdf`, `race_rp`, `sampled_lrfu`, `lru_leaf`, `wa`. Read only with `--enable-prefix-caching`; unknown values are refused. |
| `VLLM_EVICTION_PARAMS` | The policy's settings, `key=value,...`. TTDF: `tau_short`, `tau_long`, `w_short`, `w_long`. RACE-RP: `sketch=<file>` (required), `alpha`. Sampled LRFU: `half_life`, `sample_size`, `seed`. `wa`: `labels` (`oracle` or `none`), `half_life`, `kappa`, `buckets`. All of them take `stats=<file>` for overhead counters. In a cache replay, the same settings follow the policy name after a colon. |

### The simulator in more detail

<details><summary>Trace format, replay rules, and how to see that the prefix cache worked</summary>

**Trace format.** A trace has one JSON object per line. Each object has a
`request_id` and either:
- `response`: text, tokenized with the served model's tokenizer; or
- `response_token_ids`: a list of token ids.

The server matches a request by its id, from the `X-Request-Id` header or
the chat body's `request_id`. It never compares prompts.

**Replay rules:**
- The response comes back one token per decode step, then EOS, so the
  request finishes with `finish_reason: "stop"`.
- `max_tokens`, `stop` and `min_tokens` work as usual. `/v1/completions`
  defaults to 16 tokens, so send a larger `max_tokens` to get whole answers.
- An unknown id gets filler tokens until `max_tokens`.
- `n > 1`, batched prompts and streaming all work.

**Seeing the prefix cache work.** Prefix caching works on full 16-token
blocks of the prompt, as in real vLLM. You can see it in three places:
- vLLM's counter, `curl -s localhost:8000/metrics | grep '^vllm:gpu_prefix_cache_hit_rate'`;
- per request, `usage.prompt_tokens_details.cached_tokens`, with
  `--enable-prompt-tokens-details`;
- per block, the event log (`VLLM_CACHE_EVENT_LOG`).

The Milestone 2 experiments check on every run that the first and third
agree.

**Useful flags:**
- `--enable-prefix-caching` (off by default in this vLLM version);
- `--block-size {8,16,32,64,128}`;
- `--num-gpu-blocks-override N`;
- `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-chunked-prefill`;
- `--disable-frontend-multiprocessing`, which runs the engine inside the
  API process and is easier to debug.

The KV cache must hold at least `--max-model-len` tokens.
</details>

### Tests

```bash
make test        # every milestone
make test-m1     # or test-m2, test-m3
make test-core   # vLLM's own scheduler and block-manager tests (CPU-safe subset)
make ci          # everything CI runs, in CI's order
make lint        # ruff, isort, yapf, codespell on the files changed since origin/milestone-3
```

Each test module declares its milestone with `pytestmark = pytest.mark.m1`
(or `m2`, `m3`). Without `make`: export
`VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1` before pytest starts, then run
`tools/run_project_tests.sh [all|m1|m2|m3]`.

| Milestone | Tests | Where |
|---|---|---|
| 1 | 67 | `tests/simulator/test_{simulator,engine,online}.py` |
| 2 | 136 | the cache event log, metrics, the Task 2 and Task 3 commands, the FIFO and LFU evictors (`tests/simulator`, `tests/analysis`, `tests/eviction`) |
| 3 | 1,197 + 1 xfail | the adapters (`tests/client_simulator`), the runners, the cache replay, Belady, the study and report builders (`tests/analysis`), and every new evictor (`tests/eviction`, `tests/simulator`) |

Every pull request runs them on GitHub (`project-ci`), along with the
linters and a check against [CONTRIBUTING.md](CONTRIBUTING.md)
(`make check-pr BODY=pr.md`).

### What changed from upstream vLLM

| Path | Purpose |
|---|---|
| `vllm/simulator.py` (new) | `Simulator` (trace replay) and `SimulatorExecutor`. |
| `vllm/engine/llm_engine.py`, `async_llm_engine.py`, `vllm/platforms/__init__.py` | Select the simulator executor; pin the CPU platform. |
| `vllm/entrypoints/…`, `vllm/utils.py` | macOS port-binding fixes; `cached_tokens` on `/v1/completions`. |
| `vllm/envs.py` | The variables above. |
| `vllm/core/block/cache_events.py` (new), `prefix_caching_block.py`, `cpu_gpu_block_allocator.py`, `block_manager.py` | The cache event log, the request context, and the evictor hooks (`record_hit`, `record_access`). |
| `vllm/core/evictor.py`, `eviction_policies.py` (new) | Policy selection, FIFO, LFU. vLLM's `LRUEvictor` is unchanged. |
| `vllm/core/reuse_evictors.py`, `sampled_lrfu.py`, `wa_evictor.py` (new) | TTDF, RACE-RP, `lru_leaf`, Sampled LRFU, `wa`. |
| `client_simulator/` (new) | Dataset adapters and the replay client. |
| `analysis/` (new) | Metrics, the experiment runners, the cache replay and Belady, the Milestone 3 study, the figures. |
| `tests/simulator`, `tests/analysis`, `tests/eviction`, `tests/client_simulator` (new) | The project's tests. |
| `examples/` | The simulator demo and the Task 1 experiment configurations. |
| `specs/`, `report/`, `CONTRIBUTING.md`, `Makefile`, `tools/`, `.github/workflows/` | Specs, the report notes and figures, team rules, commands and CI. |

### Troubleshooting

- **Requests reach another program (macOS).** `vllm serve` listens on
  0.0.0.0, and macOS lets it start even when another program holds
  127.0.0.1 at that port. Pass `--host 127.0.0.1`, or pick a free port
  (`lsof -nP -iTCP:8000 -sTCP:LISTEN`).
- **`Port 8000 is already in use`** from an experiment command: stop the
  other server, or pass `--port` (`run_m3_task2 grid` uses ports from
  `--base-port`, default 8100).
- **A grid refuses to start on a dirty tree.** Commit your changes, or
  pass `--allow-dirty`; the run then records that it was dirty.
- **`FileNotFoundError` from `os.getcwd()` while importing vLLM.** The
  terminal's current folder no longer exists, for example after an external
  drive was remounted. `cd` back into the repository by its full path.
- **The cache event log looks incomplete.** It is written in batches. Stop
  the server and check for the final `{"ev": "end"}` line.

### Known limitations

- **Rejected at startup:** LoRA, prompt adapters, tensor and pipeline
  parallelism, multi-step scheduling, speculative decoding, pooling models,
  and `--tokenizer-mode mistral`.
- **Prompt logprobs are not supported.** Beam search yields filler tokens.
- **`gpt2` has no chat template.** Use a chat model for
  `/v1/chat/completions`.
- **Timing comes from the event loop, not a GPU model.** The study is about
  cache behaviour, not latency.

### Project documents

- [CONTRIBUTING.md](CONTRIBUTING.md): branches, PRs, tests, CI.
- [specs/](specs/README.md): one spec, plan and task list per feature.
- [report/](report/README.md): the findings of each task, their figures,
  the changelogs and AI usage.
