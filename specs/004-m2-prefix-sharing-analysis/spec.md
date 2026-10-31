# Feature Specification: Replay ShareGPT and analyze prefix sharing

**Spec**: `004-m2-prefix-sharing-analysis` | **Milestone/Task**: Milestone 2,
Task 2
**Owner**: Xin Liang Chia | **Reviewers**: Jasmine Lu, Triet Chau
**Status**: In progress (branch `milestone2-marcus-prefix-sharing-analysis`)
**Created**: 2026-10-18 | **Updated**: 2026-10-21 |
**Depends on**: `003` (client simulator)

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
dedupe, free, revive and evict. Each line has a timestamp, the block's
content hash, the request that caused it and the block-manager operation it
happened in. Without the variable, vLLM behaves and performs exactly as
before.

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
seed and default cache settings, then draws the comparison plots (file
names as `run_m2_task2` writes them):

- `fig1`: CDF of per-request cached fraction, single vs multi.
- `fig2`: CDF of hits per block, single vs multi (zero-hit blocks included).
- `fig3`: CDF of reuse interval in seconds, log x-axis, single vs multi.
- `fig3b`: the same intervals counted in requests allocated in between,
  which does not depend on simulator speed (see Edge Cases).
- `fig4`: mean cached fraction by turn index (multi only).
- `fig5`: hit rate over time (windowed), both settings.
- `fig6`: hit sources, intra vs cross conversation and prompt vs
  decode-written blocks (for User Story 4).

**Independent Test**: on a 20-conversation fixture, the command finishes in
under 2 minutes and writes all seven figures plus the CSVs behind them.

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
  `"op": "swap_in"`. (v0 swaps only groups with more than one sequence, so
  with `n=1` requests this should not occur; preemption recomputes.)
- **Blocks written during decode.** A block that fills with generated tokens
  is inserted from `append_slots`. In multi-turn replay, turn k+1's prompt
  contains turn k's answer, so it can hit blocks turn k wrote while decoding.
  These inserts carry `"op": "append_slots"` and the request id, so such hits
  can be counted separately.
- **vLLM fixes `cached_tokens` early.** vLLM computes a waiting request's
  cached-token count the first time the scheduler considers it, and keeps
  that value while the request waits (`ComputedBlocksTracker`,
  `prefix_caching_block.py:979-1010`). Blocks that become cached, or are
  evicted, before the request is actually allocated are not reflected. The
  event log shows what allocation really found; SC-004 measures the
  difference.
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
  "block": <physical id>, "req": <request id or null>, "op": <operation or
  null>}`, with `ev` one of `hit`, `miss`, `insert`, `dedupe`, `free`,
  `revive`, `evict`, and `op` the block-manager call it happened in:
  `allocate` (prompt), `append_slots` (decode), `free`, `swap_in` or
  `swap_out`. The first line is a header with `time.time()` and
  `time.monotonic()` so the log can be aligned with the client's UTC
  start time.
- **FR-003**: The recorder MUST buffer lines and flush every 4,096 events or
  1 second, whichever comes first, and at engine shutdown. Closing writes a
  final `{"ev": "end"}` line; the analysis MUST warn when it is missing (the
  engine was killed and the tail may be lost).
- **FR-004**: Request attribution MUST come from the block manager setting
  the current request id and operation around `allocate`, `append_slots`,
  `free`, `swap_in` and `swap_out`, not from new allocator arguments, to
  keep upstream edits small. `append_slots` and `free` receive only a
  sequence, so the recorder keeps a sequence-id-to-request-id map, filled
  in `allocate`.
- **FR-005**: Metric code MUST live in `analysis/` as pure functions of the
  event log and result JSONL, with no vLLM import, so it is unit-testable and
  reusable by `005` to `007`.
- **FR-006**: One command (`python -m analysis.run_m2_task2`) MUST run both
  settings, write a run folder per setting
  (`results/m2-task2/<setting>/`) with the client output, event log, manifest
  and metrics CSVs, and render the figures to `results/m2-task2/figures/`.
  The figures the report uses are committed under
  `report/milestone2/figures/task2/`.
- **FR-007**: The manifest MUST record everything constitution V lists,
  including the KV-cache block count that vLLM logs at startup.

### Metric Definitions

These definitions are shared project-wide (constitution IV).

| Metric | Unit | Definition | Edge cases |
|---|---|---|---|
| Per-request cached fraction | ratio 0–1 | `cached_tokens / prompt_tokens` for a completed request, where `cached_tokens` is vLLM's `num_cached_tokens` (`len(computed_block_nums) × block_size`) and `prompt_tokens` counts the templated prompt | No hit → `0.0`, never missing. Skipped requests are excluded and counted separately |
| Token-level reused fraction | ratio 0–1 | `Σ cached_tokens / Σ prompt_tokens` over completed requests | Reported next to the per-request mean; the two differ when long prompts hit more |
| Usable reused fraction | ratio 0–1 | As the two rows above, with `cached_tokens` replaced by `block_size × ` the request's hits before its first miss in its first prefill (`allocate` lookups in the event log), capped at `prompt_tokens`: the contiguous cached prefix, which is all the scheduler can skip | Added for `005`. vLLM's `cached_tokens` comes from `find_cached_blocks_prefix`, a binary search that assumes a cached block's predecessors are cached too. A policy that evicts a conversation's oldest blocks first (FIFO) leaves misses before cached blocks, and vLLM overcounts. Under LRU the two agree when no two prefills in one scheduler step share a newly inserted block and no request is deferred (`AllocStatus.LATER`) or preempted, as in `005`'s paced runs. Otherwise they can differ in both directions: a hit on a block inserted in the same step is recomputed (this metric overstates), and a preempted or deferred request's count is not its first prefill's (vLLM's can be larger). Use this metric whenever the policy is not LRU (`005`, `006`, `007`), and treat it as an upper bound under burst load |
| Hits per block | count | Number of `hit` events for a content hash over the run, summed across its residencies (insert to evict) | Blocks with zero hits are included in the CDF; the share with ≥ 1 hit is reported |
| Hits per residency | count | `hit` events between one `insert` (or `revive` after evict-and-recompute) and the next `evict` of that hash | Supports `005`'s eviction analysis |
| Reuse interval | seconds | For each content hash, the gap between consecutive accesses, where an access is its `insert` or a `hit` | A hash accessed once contributes no interval. Also reported as the number of requests allocated in between (distinct `req` values whose first `allocate` event falls inside the gap) |
| Block hit rate | ratio 0–1 | `hits / (hits + misses)` over lookups. vLLM's `gpu_prefix_cache_hit_rate` computes this cumulatively | Windowed version (default 10 s) for the time plot |
| Hit source | shares summing to 1 | Each `hit` is labelled by the request that inserted the block in the current residency: same conversation (intra), other conversation (cross); and by the insert's `op` (`allocate` = prompt block, `append_slots` = block of generated tokens) | Additional metric for US4. Hits whose insert precedes the log are labelled unknown |

### Key Entities

- **Cache event**: one line of the event log (FR-002).
- **Run folder**: client JSONL, `run.json` manifest, event log, `/metrics`
  snapshots, metrics CSVs.

## Success Criteria *(mandatory)*

- **SC-001**: With `VLLM_CACHE_EVENT_LOG` unset, `tests/simulator` and
  `tests/core` pass and a throughput-bound 100-conversation replay
  (`--conversation-rate inf --think-time 0`, so client pacing does not hide
  engine time) takes less than 2% longer than on `milestone-2`. The
  recorder-on time is reported too.
- **SC-002**: Hand-derived expected values match in every `tests/analysis`
  case.
- **SC-003**: The log agrees with vLLM's own count:
  `gpu_prefix_cache_hit_rate` from `/metrics` equals `hit/(hit+miss)` over
  all logged lookups (including `swap_in` lookups), to 4 decimal places.
- **SC-004**: For every completed request, the run reports
  `d = cached_tokens − block_size × (allocate-op hits the log attributes to
  it)`. `d = 0` is expected for most requests. `d < 0` happens when a hit was
  on a block another request was still computing (vLLM does not count it as
  computed) or one cached while the request waited; `d > 0` when a block
  vLLM counted was evicted before allocation (see Edge Cases). The manifest
  records how many requests fall in each case, and the analysis explains any
  share above 1%.
- **SC-005**: Both settings run on the agreed sample (default 1,000
  conversations, seed 0) and all seven figures are produced from committed
  scripts.

## Assumptions

- Default cache parameters = vLLM's defaults on the CPU platform: block size
  16, `VLLM_CPU_KVCACHE_SPACE` 4 GiB, LRU. Task 3 varies them.
- Sample: the first 1,000 conversations of the preprocessed file, in file
  order (the client's `--limit`; it has no shuffle), Poisson arrivals with
  seed 0, conversation rate 1/s, think time 5 s. Kept after the Phase 0
  measurement (`plan.md`, Phase 0 results).
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (tokenizer and config only), served
  with `--enable-prefix-caching --enable-prompt-tokens-details
  --max-model-len 8192`, as in `client_simulator/README.md`. Its chat
  template inserts a default system prompt, so the first block of every
  prompt is shared across conversations.

## Out of Scope

- Changing the eviction policy or cache size (`005`).
- Workloads other than ShareGPT (`006`).
