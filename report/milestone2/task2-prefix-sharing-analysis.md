# Milestone 2, Task 2: prefix sharing in single- and multi-turn ShareGPT

> **In progress.** The setup is final; the results come from the
> 1,000-conversation run started on 2026-10-21.

**Question.** When ShareGPT is replayed as single-turn (first exchange only)
and as multi-turn (whole conversations), how much of each prompt comes from
the prefix cache, how often is each cached block reused, and how long does
a block wait between reuses? Metric definitions:
[spec 004](../../specs/004-m2-prefix-sharing-analysis/spec.md), metric
table.

## Setup

- **Sample:** the first 1,000 conversations of the preprocessed ShareGPT
  file: 6,833 turns; 306 conversations have one turn and the longest has
  300.
- **Client:** seed 0, 1 conversation/s (Poisson), think time 5 s, at most 32
  requests in flight.
- **Server:** `Qwen/Qwen2.5-0.5B-Instruct` (config and tokenizer only) on
  the simulator, with `--enable-prefix-caching
  --enable-prompt-tokens-details --max-model-len 8192`. The KV cache is at
  the defaults: 21,845 blocks of 16 tokens (4 GiB) with LRU eviction. Each
  setting starts on a fresh server, so its cache starts empty.
- **Measurement:** the cache event log (`VLLM_CACHE_EVENT_LOG`) together
  with the client's per-request results; the metrics are computed by
  `analysis/metrics.py`.
- **Command:** `python -m analysis.run_m2_task2 --limit 1000 --out
  results/m2-task2`, run at commit `e6849da8`, with `--port 8010` and
  absolute data paths. Each setting's `run.json` records the full
  configuration.
- **Self-checks** (in each `summary.json`): the log's hit rate equals vLLM's
  own (SC-003), and how many requests' `cached_tokens` differ from the hits
  logged for them (SC-004).

Phase 0 and the recorder's overhead are in
[`plan.md`](../../specs/004-m2-prefix-sharing-analysis/plan.md#phase-0-results-2026-10-20).

## Results

_To be written after the run: figures 1 to 6 from `figures/task2/` and a
table with the key numbers per setting._

## Interpretation

_To be written after the run: what the numbers show, including the
intra- and cross-conversation hit shares and the hits on blocks written
during decoding._

## Limitations

- **Simulated time.** The simulator computes nothing, so intervals in
  seconds reflect the client's pacing (1 conversation/s, 5 s think time) and
  queueing, not model speed. Figure 3b gives the same intervals in requests,
  which do not depend on speed.
- **One sample, one seed, one cache size.** Task 3 varies the cache.
- **Long conversations are trimmed** by the 8,192-token context (see
  [Task 1](task1-client-simulator.md#what-we-verified)).
- **Block hashes are only comparable within one run.**
