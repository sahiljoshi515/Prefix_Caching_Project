# Feature Specification: Tune cache parameters and eviction policies

**Spec**: `005-m2-cache-tuning` | **Milestone/Task**: Milestone 2, Task 3
**Owner**: Jasmine Lu | **Reviewers**: Xin Liang Chia, Marcus
**Status**: Done: Phase 1 in PR #10, Phase 2 in PR #11, Phase 3 (results)
in PR #12; findings in
[`report/milestone2/task3-cache-tuning.md`](../../report/milestone2/task3-cache-tuning.md)
**Created**: 2026-10-18 | **Depends on**: `003`, `004` (recorder, metrics,
server launcher)

**Input**: Charter: "Experiment with different KV-cache capacities and block
sizes. Implement and compare cache eviction policies such as LRU, LFU, and
FIFO. Evaluate how each configuration affects prefix-cache hit rates,
reused-token fractions, and block reuse patterns. Generate comparison plots
and identify configurations that maximize prefix sharing."

## Background

vLLM v0 has exactly one eviction policy. `EvictionPolicy` has the single
member `LRU` (`vllm/core/evictor.py:7-11`), `make_evictor` raises for
anything else, and nothing in the CLI, `CacheConfig` or the environment
selects a policy. The allocator always passes the default
(`prefix_caching_block.py:69`). Capacity and block size are already
configurable: `--num-gpu-blocks-override N` sets an exact block count, and
`--block-size` accepts 8, 16, 32, 64 or 128.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Choose the eviction policy (Priority: P1)

A team member starts `vllm serve` with `VLLM_EVICTION_POLICY=lfu` (or `lru`,
`fifo`) and the prefix cache evicts accordingly. Unset means LRU, exactly as
today.

**Why this priority**: required by the charter, and reused by Milestone 3.

**Independent Test**: unit tests drive each evictor directly with a
hand-built sequence of `add` / `remove` / `record_hit` / `evict` calls and
assert the evicted block ids.

**Acceptance Scenarios**:

1. **Given** blocks A, B, C freed at t=1, 2, 3 and A reused at t=4 then freed
   at t=5, **When** one block is evicted, **Then** LRU evicts B and FIFO
   evicts A.
2. **Given** blocks with 3, 1 and 1 hits whose last accesses are t=1, 2 and 3,
   **When** one block is evicted, **Then** LFU evicts the 1-hit block with
   the older access (t=2).
3. **Given** an unknown value such as `VLLM_EVICTION_POLICY=mru`, **Then**
   the server refuses to start and names the valid values.
4. **Given** the variable unset, **Then** `tests/core` passes unchanged.

---

### User Story 2 - Sweep capacity, block size and policy (Priority: P1)

One command runs a grid of configurations on the same ShareGPT multi-turn
sample and seed. Each configuration gets its own run folder, produced with
`004`'s launcher, recorder and metrics.

**Independent Test**: a 2×1×3 grid on the 20-conversation fixture finishes
in under 5 minutes and writes one summary row per configuration.

**Acceptance Scenarios**:

1. **Given** a grid, **When** a configuration's folder already holds a
   finished manifest, **Then** it is skipped, so an interrupted sweep
   resumes.
2. **Given** block sizes 8 to 128, **Then** capacity is set in **tokens**
   (`blocks × block_size` constant), so block-size comparisons are not
   confounded by cache size. Block-count sweeps are reported separately.

---

### User Story 3 - Compare and recommend (Priority: P2)

Plots and a short write-up identify the configurations that maximize prefix
sharing and explain why:

1. Hit rate and token-level reused fraction vs capacity, one line per policy.
2. Same vs block size at fixed token capacity.
3. Eviction count vs capacity per policy.
4. CDFs of hits per residency and of reuse interval for the three policies
   at one capacity under pressure.
5. A table of the best configuration per capacity budget.

### Edge Cases

- **Capacity below one request's needs.** vLLM rejects a cache smaller than
  `max_model_len` tokens at startup (`SimulatorExecutor.initialize_cache`).
  The sweep's minimum capacity is therefore `max_model_len / block_size`
  blocks. Configurations below it are reported as infeasible, not as
  failures.
- **Preemption.** Small caches preempt running requests. The preemption
  count (Prometheus `vllm:num_preemptions_total`) is recorded, because
  preemption also destroys reuse.
- **Ties.** Each policy's tie-break order is part of its definition (FR-003)
  and tested, so results do not depend on heap internals.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `VLLM_EVICTION_POLICY` (`lru`, `lfu` or `fifo`; default `lru`)
  MUST select the evictor of the GPU prefix-caching allocator. The chosen
  policy MUST be logged at startup.
- **FR-002**: New evictors MUST implement the existing `Evictor` interface
  (`evictor.py:14-53`), plus a `record_hit(block_id)` hook that the
  allocator calls on every lookup hit. Its default is a no-op, so LRU is
  unchanged.
- **FR-003**: Policy definitions, applied to blocks in the free pool (refcount
  0):
  - **LRU**: unchanged vLLM behaviour. Oldest `last_accessed` first; ties go
    to more `num_hashed_tokens` (deeper block), then lower block id.
  - **FIFO**: oldest **insertion** first, where insertion is the first time
    the content hash entered the free pool since it was last computed.
    Reuse does not refresh it. Ties as LRU.
  - **LFU**: fewest hits since the content hash was last computed (counted by
    `record_hit`, reset on eviction) first. Ties are broken by LRU order.
- **FR-004**: New evictors MUST NOT copy `LRUEvictor.update`'s stale-heap
  pattern (`evictor.py:120-121`): every priority change must remain
  evictable.
- **FR-005**: `evict()` MUST run in O(log n) amortized, so sweeps on 20k-block
  caches stay fast.
- **FR-006**: The sweep command MUST record, per configuration, everything
  constitution V lists, plus eviction and preemption counts.
- **FR-007**: Eviction count and eviction rate (evictions per 1,000 lookups)
  are defined here and reused by `006` and `007`. The source is `evict`
  events in `004`'s log.

### Metric Definitions

| Metric | Unit | Definition | Edge cases |
|---|---|---|---|
| Eviction count | count | Number of `evict` events in a run | 0 when the cache never fills |
| Eviction rate | per 1,000 lookups | `1000 × evictions / (hits + misses)` | — |
| All others | — | As defined in `004` | — |

## Success Criteria *(mandatory)*

- **SC-001**: With the variable unset, `tests/simulator` and `tests/core`
  pass and vLLM's behaviour is identical: the same replay gives the same
  per-request `cached_tokens`.
- **SC-002**: The evictor unit tests cover the scenarios above and every
  tie-break rule.
- **SC-003**: The sweep covers at least 5 capacities × 3 block sizes × 3
  policies on the agreed sample, and every figure in User Story 3 is
  produced from committed scripts.
- **SC-004**: The write-up states, with numbers, the best policy per
  capacity and the sensitivity to block size.

## Assumptions

- Workload: `004`'s multi-turn ShareGPT sample, rate and seed.
- Capacities are chosen relative to the sample's working set (the number of
  distinct full-block hashes in an infinite-cache run), from far below it to
  above it, so the curve shows the transition into no eviction.

## Out of Scope

- Workloads other than ShareGPT and the workload-aware policy (`006`, `007`).
- Changing vLLM's prefix-caching granularity or hashing.
