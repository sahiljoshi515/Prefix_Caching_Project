# Feature Specification: A workload-aware eviction policy for mixed workloads

**Spec**: `007-m3-mixed-workload-policy` | **Milestone/Task**: Milestone 3,
Task 2
**Owners**: Sahil Joshi, Xin Liang Chia | **Reviewers**: Triet Chau, Jasmine
Lu
**Status**: Specified (revised 2026-10-31); not started
**Created**: 2026-10-18 | **Depends on**: `004`, `005`, `006`

**Input**: Charter (Sahil Joshi and Xin Liang Chia): "Create mixed-workload
traces by combining requests from different workload types. Analyze
differences in prefix reuse patterns, reuse intervals, and cache pressure
across workloads. Design a workload-aware eviction policy that determines
which KV-cache blocks to retain or evict. Investigate how to identify
workload types and whether workload-specific information can improve
eviction decisions. Implement the proposed policy in the modified vLLM
simulator. Compare the proposed policy against standard baselines (LRU, LFU,
FIFO) under identical cache configurations and request traces. Evaluate
whether the proposed policy improves prefix-cache reuse across heterogeneous
workloads. Present results using appropriate plots and discuss the benefits,
limitations, and trade-offs of the proposed approach."
Course spec, Milestone 3 Task 2: "design an eviction policy that performs
best under a mixed workload. Things to consider: What is the difference in
their KV cache reusing time interval? What is the difference in their KV
cache reusing portion? How do I know a request belongs to which workload
type? Please compare the effectiveness of your designed eviction policy
against the baselines (e.g., LRU and LFE for all requests.)"

## Why this task

Prefix caching saves work only while a later request still finds its blocks
cached. The cache is small, so the eviction policy decides which saved work
survives. Milestone 2 studied one workload, ShareGPT chat. A real server sees
chat, coding agents and tool-using agents at once, and they compete for one
cache. LRU, LFU and FIFO apply one rule to every block, whatever workload it
belongs to. This task asks whether knowing whose block it is lets a policy
keep the right ones.

**Example** (to illustrate; User Story 2 measures whether it happens). A
chat user gets an answer, and the conversation's history goes to the free
pool; the user's next turn arrives about 5 s later and needs that history
again. Meanwhile agent requests run and finish. Suppose most of their blocks
are never reused. Their freed blocks are now more recent than the chat
history, so when the next agent requests need room, LRU evicts the chat
history first. The chat turn then recomputes its whole history, while the
agent blocks LRU kept are never used again. LRU cannot tell "recent but
never reused" from "older but about to return". A policy that has learned
"this agent workload's blocks rarely return, chat blocks return within
about 5 s" would evict the agent blocks first. Milestone 2 has already shown
the cost of ignoring how a workload reuses its blocks: LFU evicted freshly
freed chat blocks, which have no hits yet, just before their conversations
returned (Background 3).

Each step of the charter has a reason, and they build one argument: observe
(1 and 2), propose (3 and 4), build (5), test fairly (6 and 7), explain (8).

| Charter step | Why it is needed | Where in this spec |
|---|---|---|
| 1. Create mixed-workload traces | Workloads compete for the cache only when they run together; running each alone (`006`) cannot show it | User Story 1, FR-001 |
| 2. Analyze reuse patterns, reuse intervals and cache pressure | A rule can only use differences that were measured: how soon each workload returns, how much of each prompt it reuses, which one fills the cache (the course's first two questions) | User Story 2 |
| 3. Design a workload-aware policy | Turns the measured differences into a rule, for example protecting a workload's freed blocks for about its typical reuse interval and evicting first the blocks of workloads that rarely return | User Story 3, FR-004, FR-005, FR-010 |
| 4. Identify workload types, and whether that helps | A real server is not told a request's workload (the course's third question). The no-label mode shows whether the label itself helps; if it does not, that is a finding too | User Story 3 (scenarios 1 to 3), FR-002 |
| 5. Implement the policy in the simulator | Results must be measured, not argued; the simulator runs vLLM's own scheduler and allocator on a laptop | FR-003, `plan.md` |
| 6. Compare against LRU, LFU and FIFO with identical cache and trace | Only the policy may change; otherwise a win could come from the setup or from noise (identical runs differ by about 0.01, Background 5) | User Story 3 (scenario 4), FR-006, FR-007 |
| 7. Evaluate across heterogeneous workloads | A policy can raise the total by starving one workload; the question is who gains and who pays | User Story 4, SC-003 |
| 8. Plots, benefits, limitations and trade-offs | The report must show and explain the result, including its costs: more complexity, labels a real server may not have, and parameters that may fit only our simulated timing | FR-011, User Story 4, SC-004 |

## Background: what Milestone 2 established

Each finding below shapes a requirement of this spec.

1. **Eviction only matters under pressure.** At vLLM's default cache
   (349,520 tokens), eviction cost ShareGPT multi-turn 0.1% of its lookups
   (Task 2). In Task 3, LRU came within 0.003 of a cache that never evicts
   from 262,144 tokens up, and all three policies tied at 2,097,152 tokens,
   where nothing is evicted. A better policy can only show where LRU falls
   clearly short of that ceiling (FR-007).
2. **vLLM v0 assumes cached blocks form a prefix.** `find_cached_blocks_prefix`
   (`vllm/core/block/prefix_caching_block.py:653`) binary-searches for a
   prompt's first uncached block. A policy that evicts a block while later
   blocks of the same prompt stay cached leaves a hole, and vLLM then
   overcounts `cached_tokens`. The scheduler skips tokens whose KV was never
   written. The simulator does not notice, but on a GPU the output would be
   wrong. FIFO does this: at 65,536 tokens its hit rate equals LRU's, but
   only half its hits are usable (0.378 against 0.761 of prompt tokens;
   Task 3, PR #12). Hence FR-005 and the usable reused fraction (FR-008).
3. **Counting hits evicts the blocks multi-turn reuse needs.** A freshly
   freed block has no hits yet, so LFU evicts it a median 1.5 s after it is
   freed, before its conversation returns 5 s later. At 262,144 tokens LFU
   serves 0.176 of prompt tokens from the cache against LRU's 0.949
   (Task 3). Any frequency component must protect recently freed blocks of
   active sessions.
4. **The client sets the reuse timing.** ShareGPT's multi-turn reuse
   interval is the client's 5 s think time plus a few milliseconds (Task 2):
   a property of our pacing, not of the dataset. A policy tuned to it could
   learn that constant instead of the workload (FR-010, User Story 4).
5. **Runs differ by about 0.01 near the pressure cliff.** Two independent
   runs of LRU at 65,536 tokens gave 0.764 and 0.752 (Task 3). Margins
   below the seed spread are ties (SC-003).
6. **Runs are slow and large.** A 1,000-conversation ShareGPT multi-turn run
   replays for about 18 minutes. At the default cache its event log is
   286 MB (Task 2); Task 3's 45 run folders took about 14 hours of replay
   and 22 GB.
7. **The request context exists only while logging.** The block manager
   knows which request it is serving only through the event log's
   attribution, which is off when `VLLM_CACHE_EVENT_LOG` is unset
   (`vllm/core/block/cache_events.py`, `attributed`). A policy cannot depend
   on a measurement tool being on (FR-002).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build mixed traces (Priority: P1)

A team member builds a mixed trace from `006`'s per-workload traces. The
inputs are a mix (each workload's share of requests), a total request
budget, a time window and a seed. One server replays the result with the
client from `006`.

**Why this priority**: every other story runs on these traces.

**Independent Test**: mixing two fixtures 50/50 gives the expected counts,
keeps each session's turns in order and intact, and is identical across runs
with the same seed.

**Acceptance Scenarios**:

1. **Given** a mix and a seed, **Then** whole sessions (conversations,
   trajectories) are sampled per workload, never split, until each
   workload's share of requests is within 5 points of its target. The
   builder reports each workload's share of requests and of prompt tokens,
   since agent trajectories are long and their token share can far exceed
   their request share.
2. **Given** the mix, **Then** each workload keeps its own arrival process
   on one shared timeline. Bailian requests keep their recorded times,
   relative to the window's start. The other workloads start sessions as
   Poisson processes, at rates that give the target shares over the window.
3. **Given** the mixed trace, **Then** every request id keeps its workload
   prefix (`006` FR-002). The trace's manifest records the mix, the seed,
   and each workload's pacing (session rate, think time).

---

### User Story 2 - Characterize the mix (Priority: P1)

From an LRU run of each mix, with `004`'s event log on, per-workload
statistics answer the course's three questions with numbers. They also
measure how much a better policy could gain.

**Acceptance Scenarios**:

1. **Reuse interval:** per-workload CDFs of the reuse interval, in seconds
   and in requests started in between (`004`). The second does not depend
   on the client's pacing or the simulator's speed.
2. **Reuse portion:** per workload, the usable reused fraction (FR-008),
   hits per block, and the shares of hits within the same session, across
   sessions, and across workloads.
3. **Cache pressure:** per workload, its share of the cached blocks over
   time, how its working set grows, its evictions, and its misses on blocks
   evicted earlier. An eviction counts against the workload whose block was
   evicted (the last request that held it). The event log attributes it to
   the request whose allocation caused it, so the analysis must re-attribute
   it.
4. **Ceiling:** the same mix and seed, replayed at a capacity that never
   evicts, gives each workload's usable reuse without eviction. The gap
   between that and LRU is the most any policy can gain.

---

### User Story 3 - The workload-aware policy (Priority: P1)

A new evictor, `VLLM_EVICTION_POLICY=wa`, decides what to evict from each
block's workload label and from per-workload statistics it learns while
serving. Its design comes from User Story 2's evidence (FR-004), and it is
implemented like `005`'s policies.

**Acceptance Scenarios**:

1. **Oracle labels** (`VLLM_WA_LABELS=oracle`): a block's label is the
   workload of the last request that held it, the request whose `free` put
   it in the free pool. The label is the request id's workload prefix.
2. **Inferred labels** (`inferred`): labels come from a classifier over
   features a server sees without the request id. Examples are the prompt
   length, the hash of the prompt's first blocks, whether the prompt extends
   a cached prefix, and the time since that prefix was last used. The
   classifier is trained on runs whose seeds the reported comparison does
   not use. Its accuracy and confusion matrix against the oracle labels are
   reported.
3. **No labels** (`none`): every block gets the same label, and the policy
   otherwise runs unchanged. Comparing `oracle` with `none` answers whether
   workload information itself improves eviction (charter).
4. **Given** the same capacity, block size, trace and seed, **Then** `wa` in
   each label mode and LRU, FIFO and LFU are compared on every metric of
   `004` and `005`.
5. **Given** the same replay with the event log on and off, **Then** `wa`
   produces the same per-request `cached_tokens` both times.

---

### User Story 4 - Evaluate honestly (Priority: P2)

Results cover at least three mixes, with at least three seeds each, at the
agreed capacity. They report means with their spread, and the write-up
discusses benefits, limitations, trade-offs and overhead.

**Acceptance Scenarios**:

1. **Per workload:** each mix's results are also reported per workload, so
   a policy that gains overall by starving one workload shows up. The
   write-up states the largest per-workload loss against LRU.
2. **Homogeneous traces:** `wa` also runs on each single-workload trace, to
   show whether it hurts a workload on its own. The baseline runs come from
   `006` wherever the configuration matches.
3. **Pacing sensitivity:** one mix is replayed with a think time and
   session rates different from those the parameters were tuned on, with
   the parameters unchanged.
4. **Overhead:** replay time and `evict()` cost under `wa` against LRU, on
   the same trace.

### Edge Cases

- **Blocks shared across workloads** (for example a common system prompt):
  such a block takes the label of the last request that held it. The label
  rule is tested, and the share of such blocks and of cross-workload hits is
  reported (User Story 2).
- **Unknown workload** in inferred mode: falls back to a default class
  that behaves like LRU.
- **Context length:** one server serves the whole mix, so every workload
  runs at `006`'s `--max-model-len` (32,768 tokens if agent and coding
  traces need it). ShareGPT's numbers in a mix are therefore compared with
  `006`'s ShareGPT run, not with Task 2's 8,192-token runs. The capacity
  must hold one maximum-length request (`005`).
- **Preemption:** small caches preempt running requests (`005`), so every
  run records its preemption count.
- **Bookkeeping:** `evict()` stays O(log n) amortized (`005` FR-005), with
  no stale heap entries (`005` FR-004).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `analysis/mix.py` MUST build mixed traces from `006`'s
  per-workload outputs, given request shares, a request budget, a time
  window and a seed. It MUST write a manifest (User Story 1).
- **FR-002**: The workload label of the request that frees a block MUST
  reach the evictor whether or not `VLLM_CACHE_EVENT_LOG` is set, without
  new allocator arguments (Background 7).
- **FR-003**: The `wa` evictor MUST implement the `Evictor` interface plus
  `record_hit` (`005` FR-002), live in `vllm/core/eviction_policies.py`, and
  be selected with `VLLM_EVICTION_POLICY=wa`. `VLLM_WA_LABELS=oracle |
  inferred | none` selects the label mode (default `oracle`), and the
  server refuses unknown values (`005` FR-001).
- **FR-004**: The design MUST be written in `plan.md` and reviewed before
  implementation: the scoring rule, its parameters, how they were chosen,
  and the evidence from User Story 2.
- **FR-005**: `wa` MUST keep the cached blocks prefix-closed. It never
  evicts a block while a later block of the same prompt stays cached, so
  vLLM's `cached_tokens` stays exact and the policy would also be correct on
  a GPU (Background 2). Unit tests check this, and every `wa` run must
  report 0 requests with a hit after a miss. If a design must break this
  rule, `plan.md` justifies why, and the report states that the policy is
  not safe on a GPU with vLLM v0.
- **FR-006**: Parameters (and the inferred-label classifier) MUST be tuned
  on seeds that the reported comparison does not use.
- **FR-007**: Within a comparison, every policy MUST use the same capacity,
  block size, trace and seed. The capacity MUST put the mix under pressure:
  LRU's usable reuse must fall short of the no-eviction ceiling by more than
  the seed spread. `006`'s fixed capacity is used if it does; otherwise a
  smaller capacity is chosen, with its reason recorded, before any `wa` run
  is compared.
- **FR-008**: The primary metric is the usable reused fraction from `004`'s
  metric table (added by PR #12). Reported next to it: block hit rate,
  vLLM's `cached_tokens` share, requests with a hit after a miss, eviction
  rate, and misses on blocks evicted earlier. Per-workload values group
  requests by their oracle label.
- **FR-009**: One committed command MUST run the whole grid. It extends
  `005`'s sweep runner (`analysis/sweep.py`) with axes for the trace and the
  label mode, resumes after an interruption, and records everything
  constitution V lists plus the mix manifest.
- **FR-010**: `wa` MUST NOT read the client's pacing settings (think time,
  rates) or the trace files. What it knows about reuse timing it learns
  online from the requests it serves.
- **FR-011**: The write-up MUST include at least these figures, each
  produced by committed scripts:
  1. Per-workload reuse-interval CDFs under LRU, in seconds and in
     requests.
  2. Per-workload usable reuse and hits per block under LRU.
  3. Each workload's share of the cache over time, LRU against `wa`.
  4. Usable reuse per policy and mix, with the seed spread and the
     no-eviction ceiling.
  5. Each workload's change against LRU within each mix.
  6. The inferred labels' confusion matrix.

## Success Criteria *(mandatory)*

- **SC-001**: The mix builder's and `wa`'s tests pass. For `wa` these cover
  hand-derived victims on a 3-workload toy trace, the label rule, prefix
  closure, and refusing unknown modes. With `VLLM_EVICTION_POLICY` unset,
  `tests/core` and `005`'s tests pass unchanged.
- **SC-002**: One committed command produces, for at least 3 mixes × 3
  seeds at the agreed capacity, runs of LRU, FIFO, LFU and `wa` in each
  label mode, plus the no-eviction ceiling. It also produces `wa` on each
  single-workload trace.
- **SC-003**: The report states, per mix and with the seed spread, whether
  `wa` beats the best baseline on usable reuse, by how much, for which
  workloads, and where it does not. Margins within the seed spread are
  reported as ties. A negative result with a sound explanation meets this
  criterion.
- **SC-004**: The report answers the course's three questions with numbers:
  how the workloads' reuse intervals differ, how their reused portions
  differ, and how accurately a request's workload can be identified. It
  also says whether identifying the workload pays off (`oracle` against
  `inferred` against `none`).

## Assumptions

- The model is `Qwen/Qwen2.5-0.5B-Instruct`, and block size and capacity
  are those fixed in `006`, subject to FR-007.
- Oracle labels come from the request-id prefixes of `006` FR-002, which
  follow vLLM's `chatcmpl-` or `cmpl-` prefix.
- Until `006`'s adapters land, ShareGPT single-turn and multi-turn, under
  distinct workload prefixes, stand in for two workloads. Task 2 showed
  their reuse differs: 13% against 95% of lookups hit. The mix builder, the
  characterization and a first `wa` can be built and tested on them.
- Sample sizes are chosen so the whole grid replays in about a day across
  the owners' laptops (Background 6). `plan.md` records the estimate after
  Phase 0.
- The project is due 2026-11-30 (course spec).

## Out of Scope

- Changing vLLM's scheduler or admission policy; only eviction changes.
- Making FIFO or any other baseline safe on a GPU. Baselines run as `005`
  defines them and are measured with the usable reused fraction.
