# Feature Specification: Replay ShareGPT and analyze prefix sharing

**Spec**: `004-m2-prefix-sharing-analysis` | **Milestone/Task**: Milestone 2,
Task 2
**Owner**: Xin Liang Chia | **Reviewers**: Jasmine Lu, Triet Chau
**Status**: Draft
**Created**: 2026-10-18 | **Depends on**: `003` (client simulator)

**Input**: Charter: "Replay ShareGPT in two settings: single-turn (first turn
only) and multi-turn (complete conversations). Instrument vLLM to collect:
Fraction of cached prefix tokens reused per request. Number of cache hits per
KV-cache block. Time intervals between consecutive reuses of each block.
Generate CDFs and other relevant plots comparing single-turn and multi-turn
workloads. Analyze the observed prefix-sharing patterns."
Course spec, Task 2, also asks for "any additional metrics you find
interesting".

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Per-block cache event log (Priority: P1)

A team member starts `vllm serve` with `VLLM_CACHE_EVENT_LOG=<path>` and gets
one JSONL line for every prefix-cache event: lookup hit, lookup miss, insert,
free, revive and evict. Each line has a timestamp, the block's content hash
and the request that caused it. Without the variable, vLLM behaves and
performs exactly as before.

**Why this priority**: hits per block and reuse intervals cannot be computed
from vLLM's existing metrics, which keep only one cumulative hit rate.
Tasks 3 (`005`), `006` and `007` reuse this log.

**Independent Test**: in-process engine test (like
`tests/simulator/test_engine.py`) sends the same 40-token prompt twice with
block size 16; the log shows, for the 2 full blocks, a miss and an insert on
the first request and a hit on the second, with the right request ids.

**Acceptance Scenarios**:

1. **Given** the variable is unset, **When** the simulator suite and
   `tests/core` run, **Then** they pass unchanged and no file is written.
2. **Given** a cache of 6 blocks and prompts that overflow it, **Then**
   `evict` events appear, each for a block that had a `free` event earlier.
3. **Given** a shared prefix used by two concurrent requests, **Then** the
   second request's lookup is a `hit` even though the block was never freed.

---

### User Story 2 - The three charter metrics from one run (Priority: P1)

A script turns one run's event log and the client's result JSONL into the
three charter metrics (definitions below) as tidy CSV files.

**Independent Test**: unit tests on handcrafted event logs whose answers are
worked out by hand (for example, hits at t=1, 4 and 10 give intervals 3 and
6).

---

### User Story 3 - Single-turn vs multi-turn comparison (Priority: P1)

One command runs both settings on the same conversation sample with the same
seed and default cache settings, then draws the comparison plots:

1. CDF of per-request cached fraction, single vs multi.
2. CDF of hits per block, single vs multi (zero-hit blocks included).
3. CDF of reuse interval in seconds, log x-axis, single vs multi.
4. Mean cached fraction by turn index (multi only).
5. Hit rate over time (windowed), both settings.

**Independent Test**: on a 20-conversation fixture, the command finishes in
under 2 minutes and writes all five figures plus the CSVs behind them.

---

### User Story 4 - Explain the patterns (Priority: P2)

A written analysis (for the report) explains what the plots show, with
numbers. It separates the two sources of sharing:

- **intra-conversation:** turn k reuses the history of turns 0 to k-1;
- **cross-conversation:** different conversations share a prefix, chiefly
  the chat template's default system prompt.

The log's request ids let each hit be attributed to one source.

### Edge Cases

- **Partial blocks.** Only full blocks are hashed, so a prompt shorter than
  one block can never hit, and the trailing partial block never hits.
  Cached fraction is therefore below 1 even for an exact repeat.
- **Concurrent sharing.** A block hit while still in use (refcount > 0) is a
  hit. It is never freed, so it has no revive event.
- **Late dedupe.** When a block fills during decode and its hash already
  exists (`promote_to_immutable_block`), vLLM remaps it. This is logged as
  `dedupe`, not `hit`: the tokens were computed anyway.
- **CPU (swap) allocator.** vLLM keeps a second prefix-caching allocator for
  swapped blocks. Only the GPU allocator's events are logged. Swap-in lookups
  count toward vLLM's hit rate, so they are logged as `hit`/`miss` with
  `"src": "swap_in"`.
- **Simulated time.** The simulator computes nothing, so wall-clock gaps are
  driven by client timing (arrival rate, think time) plus queueing. Reuse
  intervals are always reported with the run's rate and think time, and also
  in "requests in between", which does not depend on wall-clock speed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `VLLM_CACHE_EVENT_LOG` (path, default unset) MUST enable an
  event recorder in the GPU `PrefixCachingBlockAllocator`; unset MUST mean no
  recorder object and no measurable overhead.
- **FR-002**: Each event MUST be one JSONL line:
  `{"t": <monotonic seconds>, "ev": <kind>, "hash": <content hash>,
  "block": <physical id>, "req": <request id or null>}`, with `ev` one of
  `hit`, `miss`, `insert`, `dedupe`, `free`, `revive`, `evict`. Lookups
  made while swapping a preempted request back in carry `"src": "swap_in"`;
  all other lookups carry `"src": "prefill"`.
- **FR-003**: The recorder MUST buffer and flush in batches, and on engine
  shutdown, so logging does not dominate step time.
- **FR-004**: Request attribution MUST come from the block manager setting
  the current request id around allocation, not from new allocator
  arguments, to keep upstream edits small.
- **FR-005**: Metric code MUST live in `analysis/` as pure functions of the
  event log and result JSONL, with no vLLM import, so it is unit-testable and
  reusable by `005` to `007`.
- **FR-006**: One command (`python -m analysis.run_m2_task2`) MUST run both
  settings, write a run folder per setting
  (`results/m2-task2/<setting>/`) with the client output, event log, manifest
  and metrics CSVs, and render the figures to `figures/m2-task2/`.
- **FR-007**: The manifest MUST record everything constitution V lists,
  including the KV-cache block count that vLLM logs at startup.

### Metric Definitions

These definitions are shared project-wide (constitution IV).

| Metric | Unit | Definition | Edge cases |
|---|---|---|---|
| Per-request cached fraction | ratio 0–1 | `cached_tokens / prompt_tokens` for a completed request, where `cached_tokens` is vLLM's `num_cached_tokens` (`len(computed_block_nums) × block_size`) and `prompt_tokens` counts the templated prompt | No hit → `0.0`, never missing. Skipped requests are excluded and counted separately |
| Token-level reused fraction | ratio 0–1 | `Σ cached_tokens / Σ prompt_tokens` over completed requests | Reported next to the per-request mean; the two differ when long prompts hit more |
| Hits per block | count | Number of `hit` events for a content hash over the run, summed across its residencies (insert to evict) | Blocks with zero hits are included in the CDF; the share with ≥ 1 hit is reported |
| Hits per residency | count | `hit` events between one `insert` (or `revive` after evict-and-recompute) and the next `evict` of that hash | Supports `005`'s eviction analysis |
| Reuse interval | seconds | For each content hash, the gap between consecutive accesses, where an access is its `insert` or a `hit` | A hash accessed once contributes no interval. Also reported as the number of requests that arrived in between |
| Block hit rate | ratio 0–1 | `hits / (hits + misses)` over lookups. vLLM's `gpu_prefix_cache_hit_rate` computes this cumulatively | Windowed version (default 10 s) for the time plot |

### Key Entities

- **Cache event**: one line of the event log (FR-002).
- **Run folder**: client JSONL, `run.json` manifest, event log, `/metrics`
  snapshots, metrics CSVs.

## Success Criteria *(mandatory)*

- **SC-001**: With `VLLM_CACHE_EVENT_LOG` unset, `tests/simulator` and
  `tests/core` pass and a 100-conversation replay's duration changes by less
  than 2%.
- **SC-002**: Hand-derived expected values match in every `tests/analysis`
  case.
- **SC-003**: The log agrees with vLLM's own count:
  `gpu_prefix_cache_hit_rate` from `/metrics` equals `hit/(hit+miss)` over
  all logged lookups (both `src` values), to 4 decimal places.
- **SC-004**: For every request, `cached_tokens ≤ block_size × (hits the log
  attributes to it)`. Equality is expected except where a hit was on a block
  another request was still computing, which vLLM does not count as computed;
  the number of such requests is reported.
- **SC-005**: Both settings run on the agreed sample (default 1,000
  conversations, seed 0) and all five figures are produced from committed
  scripts.

## Assumptions

- Default cache parameters = vLLM's defaults on the CPU platform: block size
  16, `VLLM_CPU_KVCACHE_SPACE` 4 GiB, LRU. Task 3 varies them.
- Sample: the first 1,000 conversations of the preprocessed file after a
  seeded shuffle; conversation rate 1/s, think time 5 s. Finalized in Phase 0
  from a throughput measurement.

## Out of Scope

- Changing the eviction policy or cache size (`005`).
- Workloads other than ShareGPT (`006`).
