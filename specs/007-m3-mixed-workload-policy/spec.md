# Feature Specification: A workload-aware eviction policy for mixed workloads

**Spec**: `007-m3-mixed-workload-policy` | **Milestone/Task**: Milestone 3,
Task 2
**Owners**: Sahil Joshi, Xin Liang Chia | **Reviewers**: Triet Chau, Jasmine
Lu
**Status**: Phase 1 (study and baselines) done 2026-11-05; Phase 2 (`wa`)
revised 2026-11-07, not started
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
| 6. Compare against LRU, LFU and FIFO with identical cache and trace | Only the policy may change; otherwise a win could come from the setup or from noise (identical runs differ by about 0.01, Background 5). Phase 1's label-free policies and a leaf-only LRU control join the comparison, so a gain can be traced to its cause (Background 11) | User Story 3 (scenario 4), FR-006, FR-007, FR-012, FR-013 |
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
   Phase 1 therefore paces workloads without timestamps with real pauses
   borrowed from the closest recorded loop ("human" pacing, User Story 1):
   ShareGPT from bailian-chat's turn gaps, AgentBank from CC-Bench's
   recorded tool latencies. Their reuse intervals stay inputs, not
   findings; only Bailian's and CC-Bench's are measured.
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

## Background, continued: what Phase 1 found

Phase 1 (2026-11-05, `report/milestone3/task2-mixed-workload-policy.md`) ran
the balanced mix at 131,072 tokens: engine runs on seeds 2–4, seed 1 kept
for training, and the cache replay (`analysis/cachesim.py`) for other
capacities. Each finding below shapes a Phase 2 requirement.

8. **The mix is under pressure, and LRU is the bar.** LRU serves 55.8% of
   prompt tokens from the cache, LFU 53.9% and FIFO 50.3%, against 78.1%
   without eviction (engine). Belady reaches 68.3% in the replay at the same
   capacity. FR-007 holds, and the room a policy has is real but bounded.
9. **Who loses, and why.** Of LRU's 22.3-point shortfall against no
   eviction, Bailian accounts for 13.1 points, CC-Bench 6.4, ShareGPT 2.4
   and AgentBank 0.3. LRU serves 99.5% of reuses that wait under 10 s and
   almost none after 30 s. Half of ShareGPT's reuses wait more than 1.1 min,
   so in the mix it keeps 10.0% of its prompt tokens against 83.3% alone.
   Its pauses are borrowed (Background 4), so this reflects our pacing as
   much as ShareGPT.
10. **Workloads can be told apart without labels.** A decision tree on
    prompt length, prefix hashes, reuse history and inter-arrival times,
    trained on seed 1, labels 89.7% of the held-out requests (84.2% when
    each workload counts equally). The largest confusion is AgentBank
    labelled ShareGPT (16%): both start with Qwen's default system prompt.
    CC-Bench's tasks repeat across seeds, which flatters the prefix hashes.
11. **Label-free policies already beat LRU, for a reason not yet known.**
    RACE-RP serves 58.9% and TTDF 56.7%, both above LRU on every seed.
    Sampled LRFU serves 43.9% because it leaves holes. TTDF and RACE-RP
    evict only idle blocks with no cached block after them in a prompt; LRU
    has no such rule, so part of their lead may come from the rule rather
    than from their scores (FR-012). RACE-RP costs 44.1 µs per eviction in
    the replay and 58.4 µs on the engine, against about 1 µs for LRU.
12. **Frequency helps one workload, alone.** Bailian on its own keeps 24.7%
    under LFU against 19.8% under LRU. In the mix, LFU ties LRU on Bailian.
13. **Published policies rarely beat LRU on prefix caches.** Liu et al.
    found that most of the 14 policies they tested, including the
    workload-aware policy of the ATC'25 Bailian study, do about as well as
    LRU or slightly worse on production prefix-cache traces (Related work).
    A consistent gain of a few points is a real result, and a tie is a
    plausible outcome (SC-003).

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
   relative to the window's start; seed k replays each trace's k-th window,
   so seeds never share a session. The other workloads start sessions as
   Poisson processes over the window, and each later turn is sent its
   pause after the previous one completes. "Human" pacing (the default)
   borrows real pauses: ShareGPT samples bailian-chat's text turn gaps by
   turn number, less the production answer time; CC-Bench uses its own
   recorded tool latencies; AgentBank samples those pooled. "Fast" pacing
   keeps Phase 0's 5/1/2 s constants as the sensitivity case.
3. **Given** the window, **Then** the plan stops at its end: a turn planned
   at or after the stop (by zero-latency send time) is dropped with the
   rest of its session, and shares count only kept turns. Every policy
   then replays the same requests, and each request records its planned
   send time so the analysis can skip a warm-up.
4. **Given** the mixed trace, **Then** every request id keeps its workload
   prefix (`006` FR-002). The trace's manifest records the mix, the seed,
   each workload's pacing source and pause quantiles, the turns dropped at
   the stop, each Bailian trace's thinning (the share of the window's
   sessions kept) and every input file's sha256.

---

### User Story 2 - Characterize the mix (Priority: P1)

Per-workload statistics answer the course's three questions with numbers,
and measure how much a better policy could gain. A workload's own
properties (how soon and how much it reuses) come from the ceiling run
(scenario 4), never from an LRU run, whose statistics only see the blocks
LRU kept. Each policy's run gives that policy's outcome. Requests planned
in the first 600 s (the warm-up) are not measured, and block statistics
are right-censored at the end of the run.

**Acceptance Scenarios**:

1. **Reuse interval:** per-workload CDFs of the reuse interval, in seconds
   and in requests started in between (`004`), split by whether the
   previous access was the same session, another session of the workload,
   or another workload. Requests in between scale with the pacing too, so
   neither is a property of a dataset whose pacing is borrowed. Next to it,
   the **idle age**: the time from a block's free to the next lookup of its
   hash, which is what an evictor sees; hits on blocks still held by a
   running request are excluded from it.
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
block's workload label and from per-label statistics it learns while
serving. Its design comes from User Story 2's evidence and Phase 1's
findings (FR-004). The starting point is `plan.md`'s class hit-density
design: it ranks idle leaf blocks by the hits they are expected to earn per
unit of cache time, estimated online for each class (LHD, Related work).
The ATC'25 Bailian study evicts by the same kind of per-category reuse
estimate. `wa` is implemented like `005`'s and Phase 1's policies.

**Acceptance Scenarios**:

1. **Oracle labels** (`labels=oracle`): a block's label is the
   workload of the last request that held it, the request whose `free` put
   it in the free pool. The label is the request id's workload prefix.
2. **Inferred labels** (`inferred`; dropped, scope of 2026-11-10, `plan.md`: the classifier's
   accuracy is reported, and its labels would lie between oracle and none):
   labels come from a classifier over
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
   each label mode is compared with every policy of FR-013 on every metric
   of `004` and `005`. The comparison with TTDF and RACE-RP says whether a
   label adds anything a label-free score already captures.
5. **Given** the same replay with the event log on and off, **Then** `wa`
   produces the same per-request `cached_tokens` both times.
6. **Reduces to the control:** with one label and every refinement off,
   `wa` chooses the same victims as `lru_leaf` (FR-012) on the same trace.
   This checks the implementation and makes the label's own effect
   measurable.

---

### User Story 4 - Evaluate honestly (Priority: P2)

Results cover the balanced mix with three seeds at the agreed capacity
(the chat-heavy and agent-heavy mixes were dropped, scope of 2026-11-10, `plan.md`). Engine runs
cover it; the capacity sweep may use the cache replay, which is labelled
wherever it is used and checked against the engine (SC-002). Results report means with their spread, and the write-up
discusses benefits, limitations, trade-offs and overhead.

**Acceptance Scenarios**:

1. **Per workload:** each mix's results are also reported per workload, so
   a policy that gains overall by starving one workload shows up. The
   write-up states the largest per-workload loss against LRU.
2. **Homogeneous traces** (dropped, scope of 2026-11-10, `plan.md`): `wa` also runs on each
   single-workload trace, to
   show whether it hurts a workload on its own. The baseline runs come from
   `006` wherever the configuration matches.
3. **Pacing sensitivity** (dropped, scope of 2026-11-10, `plan.md`): one mix is replayed with
   a think time and
   session rates different from those the parameters were tuned on, with
   the parameters unchanged.
4. **Overhead:** replay time and `evict()` cost under `wa` against LRU, on
   the same trace.
5. **Attribution:** `lru_leaf` minus LRU measures the leaf-only rule, and
   TTDF, RACE-RP and `wa`, each minus `lru_leaf`, measure their scores
   (FR-012, SC-005).
6. **Label errors** (dropped, scope of 2026-11-10, `plan.md`): `wa` also runs with oracle
   labels corrupted at known
   rates: the classifier's confusion matrix at 1×, 2× and 4× its error,
   and uniform errors of 10%, 20% and 40%. The report plots usable reuse
   against the error rate, next to `lru_leaf` and `wa` in `none` mode
   (FR-014).
7. **Outside check** (optional; dropped, scope of 2026-11-10, `plan.md`, with S3-FIFO run in
   vLLM instead): the mixes are also replayed in PrefixSim,
   an open simulator with 46 policies, among them a reimplementation of
   the ATC'25 workload-aware policy, ARC, S3-FIFO, SIEVE and LHD (FR-015).
   PrefixSim counts hits after a miss, which vLLM v0 cannot use, so usable
   reuse is recomputed from its gap dump before any number is compared.

### Edge Cases

- **Blocks shared across workloads** (for example a common system prompt):
  such a block takes the label of the last request that held it. The label
  rule is tested, and the share of such blocks and of cross-workload hits is
  reported (User Story 2).
- **Unknown or uncertain workload** in inferred mode: a request the
  classifier cannot place with enough confidence (a threshold tuned on
  seed 1) gets the pooled statistics, as in `none` mode.
- **Cold start:** a class with few observations uses statistics shrunk
  toward the pooled ones, so a new or rare workload is not ranked on noise.
- **Memory:** per-class statistics, and any record `wa` keeps of blocks it
  evicted, are bounded; their size is reported with the overhead (User
  Story 4).
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
  `record_hit` (`005` FR-002), be registered in
  `vllm/core/eviction_policies.py`, and be selected with
  `VLLM_EVICTION_POLICY=wa`. It may build on Phase 1's `LeafScoreEvictor`
  (`vllm/core/reuse_evictors.py`). `labels=oracle | none` in
  `VLLM_EVICTION_PARAMS` selects the label mode (default `oracle`), and
  the server refuses unknown values (`005` FR-001).
- **FR-004**: The design MUST be written in `plan.md` and reviewed before
  implementation: the scoring rule, its parameters, how they were chosen,
  and the evidence from User Story 2. The starting point is `plan.md`'s
  class hit-density design, with the refinements `plan.md` lists for
  review (T005).
- **FR-005**: `wa` MUST keep the cached blocks prefix-closed. It never
  evicts a block while a later block of the same prompt stays cached, so
  vLLM's `cached_tokens` stays exact and the policy would also be correct on
  a GPU (Background 2). Unit tests check this, and every `wa` run must
  report 0 requests with a hit after a miss. If a design must break this
  rule, `plan.md` justifies why, and the report states that the policy is
  not safe on a GPU with vLLM v0.
- **FR-006**: Parameters (and the inferred-label classifier) MUST be tuned
  on seeds that the reported comparison does not use. Numbers on a tuning
  seed are labelled as training in every table.
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
  requests by their oracle label. Also reported: the mean over workloads
  with each workload weighted equally, and the 99th percentile of uncached
  prompt tokens per request.
- **FR-009**: One committed command MUST run the whole grid. It extends
  `005`'s sweep runner (`analysis/sweep.py`) with axes for the trace, the
  label mode and the label-error rate, resumes after an interruption, and
  records everything
  constitution V lists plus the mix manifest.
- **FR-010**: `wa` MUST NOT read the client's pacing settings (think time,
  rates) or the trace files. What it knows about reuse timing it learns
  online from the requests it serves.
- **FR-011**: The write-up MUST include at least these figures, each
  produced by committed scripts:
  1. Per-workload reuse-interval CDFs under LRU, in seconds and in
     requests.
  2. Per-workload usable reuse and hits per block under LRU.
  3. Each workload's share of the cache over time, LRU against `wa`
     (dropped, scope of 2026-11-10, `plan.md`).
  4. Usable reuse per policy, with the seed spread and the no-eviction
     ceiling.
  5. Each workload's change against LRU.
  6. The inferred labels' confusion matrix.
  7. Usable reuse against the injected label-error rate (dropped with
     User Story 4, scenario 6).
- **FR-012**: A control evictor, `VLLM_EVICTION_POLICY=lru_leaf`, MUST
  order idle leaf blocks (blocks with no cached block after them in a
  prompt) by last access, with ties going to the deeper block. It uses the
  same candidate set as TTDF and RACE-RP and runs wherever LRU runs
  (Background 11).
- **FR-013**: Every comparison MUST include LRU, LFU and FIFO (the course's
  baselines), `lru_leaf` (the control), TTDF and RACE-RP (Phase 1's
  label-free policies), `wa` in each label mode, S3-FIFO over the same
  idle leaves (`s3fifo_leaf`, a strong label-free policy from outside the
  project), and the no-eviction ceiling. Belady is reported in the cache replay as a reference; for
  usable reuse it is a strong reference, not a proven bound.
- **FR-014**: `plan.md` MUST state how `wa` limits the damage of a wrong
  label. (The label-error runs of User Story 4, scenario 6, were dropped,
  scope of 2026-11-10, `plan.md`.)
- **FR-015** (dropped, scope of 2026-11-10, `plan.md`; S3-FIFO runs in vLLM instead): The
  mixes SHOULD be exported to PrefixSim's JSONL input
  (`hash_ids`, `timestamp`, `session_id`, and the workload as `type`). Its
  LRU, ARC, S3-FIFO, SIEVE, LHD, `workload_aware` and Belady then run at the
  same capacity in blocks, and usable reuse is recomputed from
  `--dump-holes`. PrefixSim's LRU must agree with the cache replay's
  before its other numbers are reported.

## Success Criteria *(mandatory)*

- **SC-001**: The mix builder's and `wa`'s tests pass. For `wa` these cover
  hand-derived victims on a 3-workload toy trace, the label rule, prefix
  closure, and refusing unknown modes. With `VLLM_EVICTION_POLICY` unset,
  `tests/core` and `005`'s tests pass unchanged.
- **SC-002**: One committed command produces, at the agreed capacity:
  engine runs of the balanced mix on 3 seeds for every policy of FR-013,
  with `wa` in each label mode, and the replay's agreement with the engine
  for every policy the engine ran. (The other mixes and the
  single-workload traces were dropped, scope of 2026-11-10, `plan.md`.)
- **SC-003**: The report states, per mix and with the seed spread, whether
  `wa` beats the best of LRU, LFU and FIFO on usable reuse, how it compares
  with `lru_leaf` and RACE-RP, by how much, for which workloads, and where
  it does not. Margins within the seed spread are reported as ties. `wa`
  goes to engine runs only if it passes `plan.md`'s decision gate in the
  replay; otherwise the replay result is the reported negative result. A
  negative result with a sound explanation meets this criterion.
- **SC-004**: The report answers the course's three questions with numbers:
  how the workloads' reuse intervals differ, how their reused portions
  differ, and how accurately a request's workload can be identified. It
  also says whether identifying the workload pays off (`oracle` against
  `none`, which bracket a classifier's labels), and whether a label adds
  anything beyond the label-free RACE-RP and S3-FIFO.
- **SC-005**: The report separates the leaf-only rule from the scores,
  with the differences of User Story 4 (scenario 5) and their seed spread.

## Assumptions

- The model is `Qwen/Qwen2.5-0.5B-Instruct`, and block size and capacity
  are those fixed in `006`, subject to FR-007.
- Oracle labels come from the request-id prefixes of `006` FR-002, which
  follow vLLM's `chatcmpl-` or `cmpl-` prefix.
- Phase 0 started on ShareGPT single- and multi-turn stand-ins; `006`'s
  adapters have since replaced them (T012).
- Sample sizes are chosen so the whole grid replays in about a day across
  the owners' laptops (Background 6). `plan.md` records the estimate after
  Phase 0.
- Each owner's laptop has `006`'s processed data (AgentBank, CC-Bench and
  the four Bailian traces) before Phase 2's runs.
- Phase 2 follows `plan.md`'s dates: `wa` in oracle and none modes, with its
  tests, by 2026-11-19; the evaluation by 2026-11-27.
- The project is due 2026-11-30 (course spec).

## Out of Scope

- Changing vLLM's scheduler or admission policy; only eviction changes.
- Making FIFO or any other baseline safe on a GPU. Baselines run as `005`
  defines them and are measured with the usable reused fraction.
- Tiered storage (CPU or SSD), compressing KV instead of evicting it, and
  compute-weighted metrics such as PrefixSim's compute saving ratio. In the
  usable reused fraction every token weighs the same.

## Related work

The papers this revision draws on.

- Wang et al., "KVCache Cache in the Wild: Characterizing and Optimizing
  KVCache Cache at a Large Cloud Provider", USENIX ATC 2025
  (<https://www.usenix.org/conference/atc25/presentation/wang-jiahao>). The
  source of the Bailian traces; evicts by per-category reuse probability.
- Beckmann et al., "LHD: Improving Cache Hit Rate by Maximizing Hit
  Density", NSDI 2018
  (<https://www.usenix.org/conference/nsdi18/presentation/beckmann>). Hit
  density learned online per class; the basis of `plan.md`'s design.
- Liu, Yu and Yang, "When Fancy Eviction Fails: Rethinking Cache Replacement
  For LLM Prefix Reuse", arXiv 2609.28870, 2026
  (<https://arxiv.org/abs/2609.28870>), and its simulator PrefixSim
  (<https://github.com/cacheMon/PrefixSim>). Background 13 and FR-015.
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model
  Programs", NeurIPS 2024 (<https://arxiv.org/abs/2312.07104>).
  RadixAttention evicts the least recently used leaf, the rule `lru_leaf`
  isolates (FR-012).
- Chłędowski et al., "Robust Learning-Augmented Caching: An Experimental
  Study", ICML 2021 (<https://icml.cc/virtual/2021/poster/8581>). Falling
  back to a classic policy when a learned one does worse (FR-014).
