# Feature Specification: Eviction policies across workloads

**Spec**: `006-m3-workload-eviction` | **Milestone/Task**: Milestone 3, Task 1
**Owner**: TBD (assign at Milestone 3 kickoff) | **Reviewers**: TBD
**Status**: Draft
**Created**: 2026-10-18 | **Depends on**: `003`, `004`, `005`

**Input**: Charter: "Identify and preprocess multiple LLM workloads beyond
ShareGPT, such as AgentBank, CC-Bench trajectories, and Qwen Bailian usage
traces. Extend the client simulator from Milestone 2 to replay these
workloads. Keep the KV-cache capacity and block size fixed while varying the
eviction policy. Compare eviction policies such as LRU, LFU, and FIFO for
each workload. Measure [...] prefix-cache hit rates and fraction of reused
tokens; cache-block reuse frequency and reuse intervals; cache eviction
frequency. Generate comparison plots and identify the best-performing
eviction policy for each workload. Analyze why different workloads favor
different eviction strategies."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Replay three more workloads (Priority: P1)

A team member preprocesses AgentBank, CC-Bench-trajectories and
qwen-bailian-usagetraces-anon with the same command used for ShareGPT
(`process_raw_traces --dataset <name>`) and replays each with the client
simulator.

**Why this priority**: every other part of Milestone 3 needs these traces.

**Independent Test**: per adapter, a handcrafted fixture in that dataset's
raw format; the produced conversations and response trace are checked line
by line, as for ShareGPT.

**Acceptance Scenarios**:

1. **Given** a text dataset (AgentBank, CC-Bench), **Then** the adapter
   yields `Conversation`s in the existing format, with tool and environment
   turns mapped onto `user`/`assistant` or `system` roles by a documented
   rule.
2. **Given** the Bailian trace, which records block hash ids and lengths
   rather than text, **Then** the adapter synthesizes token ids: each hash id
   maps to a fixed, seeded sequence of ordinary tokens of the trace's block
   size. Equal hash-id prefixes become equal token prefixes, so vLLM's
   prefix cache sees the same sharing structure. Responses are synthesized
   with the recorded output length.
3. **Given** a trace with timestamps (Bailian), **Then** the client sends
   requests at the recorded times, relative to the first; otherwise it uses
   Poisson arrivals (course spec, Task 1).
4. **Given** token-id prompts, **Then** the client uses `/v1/completions`
   with `prompt` as a token list (no chat template, since the tokens are
   synthetic), with the same request-id and result format.

---

### User Story 2 - Policy comparison per workload (Priority: P1)

One command replays every workload under LRU, FIFO and LFU at one fixed
capacity and block size, using `005`'s evictors and sweep runner.

**Acceptance Scenarios**:

1. **Given** the fixed configuration, **Then** every (workload, policy) pair
   runs on the same sample and seed, from a fresh server.
2. **Given** the runs, **Then** a summary table holds, per pair: hit rate,
   token-level reused fraction, mean hits per block, median and p90 reuse
   interval, eviction count and rate (`004`, `005` definitions).

---

### User Story 3 - Explain which policy wins where (Priority: P2)

Plots and a write-up name the best policy per workload and explain it with
each workload's reuse profile: reuse-interval distribution, share of
cross-request versus intra-session reuse, and working-set size relative to
capacity.

### Edge Cases

- **Context length.** Agent and coding trajectories can exceed 8,192 tokens.
  The run uses `--max-model-len 32768` (Qwen2.5-0.5B's maximum) for these
  workloads and reports the share of skipped turns per workload.
- **Very large datasets.** A seeded sample with a fixed request count per
  workload keeps run times comparable; the sample sizes are recorded.
- **Hash-id block size.** If the Bailian trace's block size differs from the
  fixed replay block size, the synthesized sharing would be wrong. The
  adapter refuses a mismatch.
- **Licensing and size.** Raw datasets stay out of git (constitution V);
  adapters download them into `data/raw/`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Adapters `agentbank`, `ccbench`, `bailian` MUST implement
  `check_download` and `load_conversations` (or `load_token_requests` for
  Bailian) and register in `process_raw_traces.ADAPTERS`.
- **FR-002**: Request ids MUST be prefixed with the workload name
  (`agentbank:`, `ccbench:`, `bailian:`), which `007` uses as the workload
  tag.
- **FR-003**: The client MUST support recorded timestamps and token-id
  prompts (User Story 1, scenarios 3 and 4) without changing ShareGPT
  behaviour.
- **FR-004**: The fixed capacity MUST be one where `005` showed the policies
  differ on ShareGPT (cache under pressure). The choice and its reason are
  recorded.
- **FR-005**: Each role-mapping or synthesis rule MUST be documented in the
  adapter's docstring and in `client_simulator/README.md`.

## Success Criteria *(mandatory)*

- **SC-001**: All four workloads (with ShareGPT) × 3 policies run from one
  command on the agreed samples.
- **SC-002**: Adapter tests pass on handcrafted fixtures for every dataset.
- **SC-003**: The write-up names a best policy per workload and supports it
  with at least one reuse-profile figure per workload.

## Assumptions

- Dataset formats are confirmed in Phase 0. Expected: AgentBank and CC-Bench
  provide multi-turn text trajectories; the Bailian trace provides per
  request a timestamp, input/output lengths, chat or session ids and a list
  of block hash ids.
- Model and tokenizer stay `Qwen/Qwen2.5-0.5B-Instruct`.

## Out of Scope

- Mixed workloads and the new policy (`007`).
