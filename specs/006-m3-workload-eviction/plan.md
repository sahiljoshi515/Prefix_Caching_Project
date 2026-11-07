# Implementation Plan: Eviction policies across workloads

**Spec**: [spec.md](spec.md) | **Branch**: named per `CONTRIBUTING.md` §4 |
**Date**: 2026-10-18

## Summary

Add three adapters to the client simulator's preprocessing (two text, one
synthesized from block hash ids), teach the client recorded timestamps and
token-id prompts, then reuse `005`'s sweep runner with a fixed capacity and
block size to compare LRU, FIFO and LFU per workload.

## Technical Context

**Language/Version**: Python 3.12 (3.9+ syntax)
**Primary Dependencies**: `huggingface_hub` (pinned) for dataset downloads;
the Bailian trace from its GitHub repository
**Testing**: `tests/client_simulator/test_adapters_*.py`,
`tests/client_simulator/test_timestamps_and_token_prompts.py`

## Constitution Check

- [x] I. Online replay through `vllm serve`
- [x] II. No vLLM files changed (client and analysis only)
- [x] III. Fixture-based adapter tests
- [x] IV. Metrics from `004`/`005` only
- [x] V. Datasets downloaded, never committed; manifests per run
- [x] VI, VII. Per PR

## Files

### New

| Path                                                                  | Purpose                                           |
| --------------------------------------------------------------------- | ------------------------------------------------- |
| `client_simulator/adapters/agentbank.py`                              | AgentBank adapter                                 |
| `client_simulator/adapters/ccbench.py`                                | CC-Bench-trajectories adapter                     |
| `client_simulator/adapters/qwenbailian.py`                            | Bailian adapter with hash-id → token synthesis    |
| `analysis/run_experiment.py`                                          | Shared chat/token/mixed policy runs and summaries |
| `tests/client_simulator/fixtures/{agentbank,ccbench,bailian}_small.*` | Handcrafted raw-format fixtures (< 50 KB each)    |

### Modified project files

| Path                                     | Change                                                                  |
| ---------------------------------------- | ----------------------------------------------------------------------- |
| `client_simulator/client_sim.py`         | Timestamp scheduling; `/v1/completions` path for token-id prompts       |
| `client_simulator/process_raw_traces.py` | Register adapters; write `response_token_ids` for synthesized responses |

## Design

### Synthesizing Bailian prompts

- The raw JSONL hash list has `ceil(input_length / 16)` entries, including
  the partial final block. For example, 791 tokens means 49 full blocks
  and a seven-token tail, although there are 50 hashes. The prior padding
  formula was incorrect for this format.
- Use the model tokenizer's ordinary, non-whitespace token IDs. A seeded
  RNG keyed by trace namespace and full-block hash produces 16 tokens.
  Equal full hashes reproduce the same tokens; different traces have
  separate namespaces. Hash collisions in the synthesized token sequences
  are theoretically possible but extremely unlikely.
- A request-specific seed produces exactly `input_length % 16` tail tokens.
  Another seed produces `output_length` response tokens. Original decoded
  text and output-to-future-input sharing cannot be recovered this way.
- Plain request dictionaries carry input/output IDs, EOS ID, timestamp, root
  conversation ID and original request type. Export separates the client
  input and server response files and keeps matching namespaced request IDs.
- Token replay submits the full input through `/v1/completions`, without a
  second chat template. It uses recorded seconds relative to the first
  request and checks exact returned token IDs through logprobs. No fixed
  think time or parent-dependent completion delay is added.

### Shared runner and metric integration

The simplified implementation uses dictionaries for token requests and one
`--workloads` config interface for Bailian and mixtures. The existing text
request classes and ShareGPT command remain as before. Shared analysis is
performed once in Task 2's analyzer rather than repeated in the runner.

- `analysis.run_experiment` reads prepared workloads from one JSON config.
  A chat workload uses the existing conversation client; a token workload
  uses recorded-time replay. Additional adapters can emit either format.
  Multiple entries overlay their streams against the same server, using
  one start clock, output file and global concurrency semaphore. Per-stream
  rates, seeds and offsets are explicit; this does not invent a mixed-
  workload ratio or implement a new eviction policy.
- Reuse `SimulatorServer`, `sweep.Config`, `run_sweep`, Task 2's analyzer
  and the existing metric formulas. Old Milestone 2 commands remain
  available to reproduce their reports; future workload experiments use
  the shared runner rather than new task-specific runners.
- Freeze selected inputs and combined response IDs, save checksums,
  configuration, commands and commit/dirty state. Source backup machinery
  is omitted. Refuse changed settings and modified samples on resume;
  `--analyze-only` permits analysis updates without replaying traffic.
- Normalize completion IDs (`cmpl-...-0`) alongside chat IDs. Use parent
  roots to identify intra-conversation hits. Raw completion responses have
  unknown API cached-token counts; retain those nulls. Reuse the existing
  `prefix_hits` and `add_usable_fractions` event-log measurements for usable
  reuse. Never present event-derived counts as independent API validation.
- Record statuses and timing per request and per-workload usable reuse in
  mixed runs. Check actual block capacity, complete logs, log/Prometheus
  agreement and identical request coverage across policies.

### Bailian experiment and report plan (2026-11-05)

Configuration: online vLLM v0 CPU simulator, no weights,
`Qwen/Qwen2.5-0.5B-Instruct`, 65,536-token capacity (4,096 blocks of 16),
32,768-token context, maximum 32 concurrent requests/sequences, synthesis
seed 0, prefix caching enabled, chunked prefill disabled. Each policy gets
an empty cache, identical prepared inputs/outputs and original arrivals.
Capacity follows Jasmine's 65,536-token pressure point, where ShareGPT's
usable reuse differs substantially across policies. Confirm the final
Bailian sample also causes evictions; do not choose a setting after seeing
which policy it favors.

1. Verify the adapter with a handcrafted fixture and replay the first 20
   requests under all three policies. This is a functional smoke, not a
   performance conclusion. Also verify a tiny mixed chat/token run.
2. The primary analysis uses the first 3,000 requests (829.828 seconds,
   about 42 minutes across policies plus startup), configured in
   `examples/bailian_3000.json`. At context 32,768 none require a context
   skip; the full trace has 19 overlong requests. Report actual exclusions
   rather than silently trim. The saved primary runs all completed with no
   failures, skips or preemptions. Additional segments/repeats are needed
   to assess generalization.
3. Collect block hit rate; overall usable prompt-token reuse (primary);
   mean per-request usable fraction; hits per distinct block (zero included)
   and per residency; reuse intervals in seconds and requests; eviction
   count/rate; cross-request versus intra-session sources; context skips,
   failures, preemptions and timing drift. Describe input/output lengths,
   turn counts and working set relative to capacity to explain differences.
4. Match the Overleaf report's claim/evidence style: a short setup and one
   compact table with rows LRU/FIFO/LFU and columns hit rate, overall usable
   reuse, mean request reuse and eviction rate; report exclusions in the
   caption/text. No conclusion is filled in before the experiment.
5. Main figures: (a) one policy comparison figure with separate bar panels
   for hit rate, usable token reuse and eviction rate, labeled with units;
   (b) a three-panel CDF figure for per-request usable fraction, hits per
   block and reuse intervals, like Section 3.2. Include zero-hit blocks,
   use a tail zoom if concentration hides differences, and state sample
   populations. A ten-second hit-rate-over-time plot is supporting evidence
   if it explains variation; keep additional plots outside the short report.
6. Explain the best observed policy using reuse frequency, spacing and
   cache pressure, or report a tie if differences are not persuasive.
   Limit the claim to this sample and configuration; synthetic output
   tokens, request-specific partial tails, context filtering, one trace
   segment and simulator-only latency all limit generalization.

### AgentBank and CC-Bench experiment plan (2026-11-07)

Same fixed configuration as Bailian: capacity 65,536 tokens, block size 16,
context 32,768 (Qwen2.5-0.5B's maximum), 32 concurrent requests, a fresh
server per policy. Both are chat workloads (`format: chat`) replayed like
Milestone 2's multi-turn ShareGPT: 1 conversation/s Poisson arrivals and a
5 s think time, since neither dataset has arrival times across sessions.
The policy comparison then reflects the workloads' content, not pacing.

- **AgentBank:** the first 1,000 conversations of a seeded random order
  over all 19 subsets (`examples/agentbank_1000.json`), 4,992 requests; all
  fit in 32,768 tokens.
- **CC-Bench:** all 370 sessions, split into 498 threads (session,
  sub-agent and post-`/compact` contexts), 18,184 requests
  (`examples/ccbench_full.json`). About 36% of the requests are over
  32,768 tokens; the client skips such a turn and the rest of its thread,
  and the run reports the share. Raising the context would require a cache
  larger than 65,536 tokens (vLLM refuses a context larger than the cache),
  so CC-Bench would no longer share the other workloads' capacity.
- **Not reproduced:** Claude Code's system prompt and tool definitions
  (about 12K tokens shared by every real session) are not in the logs and
  no stand-in is added; nor is the timing of sub-agents inside a session.

### Role mapping for agent trajectories

The system prompt or task description becomes `system`; the agent's actions
and reasoning become `assistant`; tool or environment observations become
`user`. Consecutive same-role messages are merged with a blank line, so
roles alternate as the client requires. Each assistant message is one replay
request with the full history, the same as multi-turn ShareGPT.

## Phases

1. **Phase 0 — Research**: download each dataset, record its exact format,
   size and licence in `client_simulator/README.md`; confirm the Bailian
   block size; choose per-workload sample sizes and arrival parameters.
2. **Phase 1**: adapters with fixtures and tests (one PR per adapter).
3. **Phase 2**: client timestamps and token prompts with tests.
4. **Phase 3**: `run_m3_task1.py`, runs, figures, write-up in
   `report/milestone3/task1-workload-eviction.md`.

## Verification

```bash
tools/run_project_tests.sh m3 -v      # adapter tests carry pytestmark = pytest.mark.m3
python -m analysis.run_experiment --config examples/bailian_experiment.json \
  --out results/smoke-m3t1
```

## Risks

| Risk                                                | Mitigation                                                                          |
| --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| A dataset's format differs from the assumption      | Phase 0 confirms before any code; the spec's assumptions are updated in the same PR |
| CC-Bench trajectories are few but very long         | Sample by request count, not conversation count; raise `--max-model-len`            |
| Synthetic tokens tokenize differently on round trip | Token-id prompts bypass text entirely (`/v1/completions` with a token list)         |
