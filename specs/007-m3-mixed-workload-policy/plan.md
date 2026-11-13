# Implementation Plan: A workload-aware eviction policy for mixed workloads

**Spec**: [spec.md](spec.md) | **Branch**: named per `CONTRIBUTING.md` §4 |
**Date**: 2026-10-18, revised 2026-10-31

## Summary

Build mixed traces from `006`'s per-workload traces, characterize per-workload
reuse from an LRU run, then design, implement and evaluate a workload-aware
evictor (`wa`) against LRU, FIFO and LFU, with oracle, inferred and no
workload labels. The scoring rule below is the starting hypothesis; FR-004
requires it to be confirmed or revised from the characterization before
implementation. The spec's Background lists the Milestone 2 findings that
constrain it.

## Technical Context

**Language/Version**: Python 3.12 (3.9+ syntax)
**Primary Dependencies**: `005`'s evictor framework and sweep runner,
`004`'s recorder and metrics (with PR #12's usable reused fraction);
`scikit-learn` only if the inferred-label classifier needs it (pin it first)
**Testing**: `tests/eviction/test_wa_evictor.py`,
`tests/simulator/test_wa_policy_env.py`, `tests/analysis/test_mix.py`

## Constitution Check

- [x] I. Evictor runs inside the real v0 allocator; online replay
- [x] II. Upstream edits limited to the label hand-off below
- [x] III. Toy-trace tests with hand-derived victims; seeds fixed
- [x] IV. Metrics from `004`/`005`
- [x] V. Mixed traces rebuilt from scripts; manifests per run
- [x] VI, VII. Per PR

## Files

### New

| Path | Purpose |
|---|---|
| `analysis/mix.py` | Mixed-trace builder |
| `analysis/characterize.py` | Per-workload reuse profiles from an event log |
| `analysis/run_m3_task2.py` | Mixes × seeds × policies × label modes |
| `tests/analysis/test_mix.py`, `tests/eviction/test_wa_evictor.py` | Tests (marked `m3`) |
| `tests/simulator/test_wa_policy_env.py` | `wa` in an in-process engine: label hand-off, the same `cached_tokens` with the event log on and off, no hit after a miss (marked `m3`) |

### Modified

| Path | Change | Reason |
|---|---|---|
| `vllm/core/eviction_policies.py` (from `005`) | Add `WorkloadAwareEvictor` | Policy home |
| `vllm/core/evictor.py` | Add `WA` to `EvictionPolicy` and dispatch | Registry |
| `vllm/core/block/cache_events.py` (from `004`) | Track the current request even when the event log is off, and expose it so the evictor reads the freeing request's label | Today the context exists only while logging (spec, Background 7); FR-002 forbids new allocator arguments |
| `vllm/envs.py` | `VLLM_WA_LABELS` (`oracle`, `inferred`, `none`) | Mode switch |
| `analysis/sweep.py` (from `005`) | Trace and label-mode axes | One resumable grid command (FR-009) |

## Design (starting hypothesis)

### Constraints from Milestone 2

- **Prefix closure (FR-005).** Within one prompt, a later block must never
  outlive an earlier one. LRU gets this for free: a hit touches a block and
  all its predecessors, and ties evict the deepest block first. A score that
  only falls with idle time, with LRU's deepest-first tie-break, keeps it
  too. Frequency terms or per-label weights can break it (a shared early
  block labelled with a low-priority workload). Every run's
  `requests_with_hits_after_miss` is the check.
- **Protect recently freed blocks.** Plain LFU loses because a freed block
  has no hits yet (spec, Background 3). Each workload gets a protection
  window learned online, for example a high percentile of its observed
  reuse intervals, during which its freed blocks are not evicted ahead of
  older ones.
- **No pacing constants (FR-010).** Windows and probabilities come from the
  intervals the evictor observes, never from the client's think time or
  rates, so the pacing-sensitivity run (User Story 4) can show whether the
  policy adapts.

### Scoring

Approximate Belady (evict the block whose next use is furthest away) with
per-workload statistics learned online:

- For each workload `w`, keep a decayed histogram of observed reuse
  intervals and the probability `p_w` that a block of `w` is reused at all
  (from hits recorded since insertion).
- For a free block `b` of workload `w`, idle for time `a`, estimate
  `P(reuse within horizon H | idle a)` from `w`'s histogram, conditioned on
  `a` (blocks idle longer than most of `w`'s intervals are unlikely to
  return), and multiply by `num_hashed_tokens`-based depth weighting
  (shallow blocks protect more downstream reuse).
- Evict the block with the lowest score. Recompute lazily: a heap keyed by
  the score at insertion time, rescored on pop (bounded rescoring per
  eviction keeps O(log n) amortized).

Alternatives to evaluate in User Story 2 before committing:

- **Static partitioning** with per-workload quotas adapted by marginal hit
  gain (ARC-like). Simpler, but wastes space when a workload idles.
- **Per-workload policy choice**: LRU within chat and LFU within agent
  workloads, with a global arbitration by `p_w`. Uses `006`'s per-workload
  winners directly.

### Workload identification

- **Oracle**: request-id prefix (`sharegpt:`, `agentbank:`, `ccbench:`,
  `bailian:`) of the request that frees the block.
- **None**: one label for every block, the ablation that shows what the
  labels themselves are worth.
- **Inferred**: features observable by a server: prompt length, number of
  messages (turns), whether the first full block matches a known
  system-prompt hash, inter-arrival time within a session. Start with a small
  decision tree trained on one seed and evaluated on another (FR-005).

## Phase 0 evidence and the proposed design (FR-004, pending review)

Branch `milestone3-sahil-mixed-workload`, 2026-10-31. Every number below
comes from a committed command; the engine runs that check the cache
simulator are in `results/m3-validate/`.

### What the characterization shows

The balanced mix (seed 1) has 3,593 requests over 20 minutes, one seventh
each from ShareGPT, AgentBank, CC-Bench and the four Bailian traces
(`analysis/mix.py`). Measured with `analysis/characterize.py` on the engine's
LRU run at 131,072 tokens (`results/m3-validate/balanced-s1-lru-cap131072`):

- **Reuse interval.** Closed-loop sessions (ShareGPT, AgentBank, CC-Bench)
  read a freed block again after the client's think time (5, 1 and 2 s). In
  production, Bailian users return after a median 111 s (chat), 29 s
  (thinking) and 118 s (coder), with a p90 of 4 to 16 minutes (raw traces).
  Under LRU at this capacity a freed block survives about 15 s, so
  closed-loop histories are kept and Bailian histories are lost.
- **Reuse portion.** Closed-loop workloads are at their no-eviction ceiling
  under LRU (0.85 to 0.96 usable). LRU's 12.9-point shortfall against the
  no-eviction ceiling (0.608 against 0.737, both engine runs) is 96% in the
  Bailian traces (coder 48%, chat 23%, thinking 21%, API 4% of the lost
  tokens) and 4% in ShareGPT. Bailian's hits come mostly from other sessions
  (shared system prompts and templates: 77% to 100% of its hits), and only
  1% to 9% of its blocks are ever hit.
- **Identification.** A request's prefix family (the chained hash of its
  k-th block) names its workload with purity 0.71 at k = 1 and 0.9997 at
  k = 2 within a run. A family table learned on seed 1 and scored on seed 2
  is 66% accurate at k = 1. The Bailian traces' templates recur (81% to 99%
  accuracy), but ShareGPT, AgentBank and CC-Bench all start with Qwen's
  default system prompt, so their first block cannot tell them apart.
- **Simulator.** `analysis/cachesim.py` drives vLLM's own allocator and
  evictors. Against seven engine runs (LRU, FIFO and LFU at 131,072 and
  393,216 tokens, and no eviction), with each run's timing, it looks up the
  same blocks, evicts the same set (overlap at least 0.9999), and matches
  usable reuse within 0.0002 per workload. It picks LRU's victims in the
  same order (1,015,404 evictions). Tuning uses one no-eviction engine
  run's timing (LRU exact, FIFO within 0.15 and LFU within 1.1 points).
- **Baselines.** In the simulator's capacity sweep, LRU leads from 49,152
  tokens up. LFU beats it at 32,768 (0.268 against 0.218), where the
  frequency of shared templates is the best signal. FIFO's usable reuse is
  far below its hit rate at every capacity (holes).

### Proposed policy: class hit-density eviction

Rank free blocks by expected hits per unit of cache time, estimated per
class from what the evictor sees (LHD, Beckmann et al., NSDI 2018), with
LRU order inside a class:

1. **Leaf-only free pool (FR-005).** The evictor tracks each cached block's
   cached children, through a new no-op `Evictor.record_insert(block_id,
   hash, parent_block_id)` hook at the allocator's insert. Only childless
   free blocks are candidates. A free block's cached descendants are always
   free, so every free block stays evictable, and the cached set stays
   prefix-closed whatever the scores.
2. **Classes.** A free block's class is `(label, hits bucket)`, where the
   hits bucket is 0, 1 or 2+ hits since the block was computed. The label
   depends on the mode:
   - `oracle`: the freeing request's workload (the FR-002 context);
   - `inferred`: its prefix family hashed into B buckets (depth-k chained
     hash mod B), with no request id;
   - `none`: one label for every block.
3. **Online statistics per class** (decayed, FR-010). For each free that
   ends: its age at the end, and whether it ended in a revive (a hit) or an
   eviction, in log-spaced age bins.
4. **Eviction.** Each class keeps its free leaves in LRU order. The victim
   is the head (oldest block) of the class with the lowest hit density at
   that head's age, `HD_c(a) = Σ_{b ≥ a} revives_c(b) / Σ_{b ≥ a} ends_c(b) ·
   (age_b - a)`, with ties going to the older block. Eviction costs O(C)
   for C classes.
5. **It reduces to LRU.** With one label and the hits bucket off there is
   one queue, and its head is the LRU block. So `none` without buckets is
   exactly LRU, which a regression test checks.

Why this design: LRU's single clock keeps a closed-loop block for 15 s,
although such a block returns within 5 s or not at all. The same clock
drops Bailian blocks that are less likely to return but would use the cache
time better. A per-class hit density that depends on age captures both,
and the hits bucket keeps the template signal that made LFU win at 32,768
tokens.

**Alternative** (simpler, evaluated in the simulator first):
family-offset LRU, with key `last access + Δ_c` and `Δ_c = p_c × W_c`
(reuse probability times p90 return gap), on the same leaf-only pool.

**Ablations**:
- labels: oracle, inferred, none;
- hits buckets on and off;
- family depth k ∈ {1, 2, 4};
- buckets B ∈ {8, 64, 512};
- statistics half-life.

**Decision gate.** Parameters are tuned on seed 1 in the simulator
(FR-006). The design goes to engine runs only if, on seeds 2 to 4 of at
least two mixes at the FR-007 capacity, it beats LRU by more than the seed
spread without lowering any workload by more than 2 points. Otherwise the
alternative is tried, and a negative result is reported (SC-003). With the
scope of 2026-11-10 (one mix, below) the gate applies to the balanced mix:
in the replay at 131,072 tokens `wa` with oracle labels is above LRU on
every seed and lowers no workload by more than 2 points, so it went to
the engine.

### Refinements for review (2026-11-07, spec revision for Phase 2)

Proposed after Phase 1 and a literature study; part of T005's review,
none of them built yet.

1. **Candidate set.** Use Phase 1's `LeafScoreEvictor`
   (`vllm/core/reuse_evictors.py`) for the leaf-only pool instead of a new
   `record_insert` hook. TTDF and RACE-RP already rely on it.
2. **Control.** Add `lru_leaf` (spec FR-012) on the same base. Step 5's
   claim "`none` without buckets is exactly LRU" becomes "is exactly
   `lru_leaf`", and that equality is the regression test (spec User Story
   3, scenario 6).
3. **Seeing past its own evictions.** Hit-density statistics only see
   blocks the policy kept. A sampled, bounded record of evicted hashes (for
   example 1 in 8, held up to 20 min) lets a later lookup record the age at
   which the block would have been reused. LHD's explorers and UCP's
   sampled monitors solve the same problem.
4. **Shrinkage.** A class's estimate is pulled toward the pooled estimate
   in proportion to how few observations it has (spec Edge Cases, cold
   start).
5. **Label safeguard** (spec FR-014). In inferred mode, a request below a
   confidence threshold tuned on seed 1 gets the pooled class. Optionally,
   a shadow `lru_leaf` over hashes compares misses over a sliding window,
   and `wa` follows its choice while it is doing worse (Chłędowski et al.,
   ICML 2021).
6. **Label-error runs.** Oracle labels corrupted by the classifier's
   confusion matrix at 1×, 2× and 4× its error, and uniformly at 10%, 20%
   and 40% (spec User Story 4, scenario 6).

## Phase 1 study (2026-11-04, branch `milestone3_task2`)

Phase 1 measured the course's three questions on engine runs before any
policy code, with a design reviewed for confounds first. What changed from
Phase 0, and why:

- **Pacing.** Phase 0's constant pauses (5/1/2 s) set ShareGPT's and the
  agents' reuse intervals by fiat, and they decide who LRU serves. Plans
  now default to `--pacing human`: ShareGPT borrows bailian-chat's text
  turn gaps (by turn, less 0.03 s per parent output token, at least 1 s);
  CC-Bench uses its own recorded tool latencies (p50 0.13 s, p90 22 s);
  AgentBank samples those. The old constants are `--pacing fast`, run as
  the sensitivity case. ShareGPT's and AgentBank's intervals stay inputs.
- **Window.** Plans are 1,800 s and stop at the window (`--tail 0`), so
  every policy replays the same requests (identical lookup counts per seed
  are checked). Metrics cover requests planned in [600, 1800) s
  (`analysis/m3t2_config.json`).
- **Seeds.** Seed k replays each Bailian trace's k-th half hour, so no
  session repeats across seeds; seed 1 trains (the identification ladder,
  the signal models), seeds 2 to 4 are reported.
- **Properties from the ceiling.** How soon and how much a workload reuses
  comes from the no-eviction run, never from an LRU run, whose statistics
  see only the blocks LRU kept. `characterize.reuse_profile` had such a bias
  (its access gaps forgot the last access at an eviction); it now uses
  `metrics.reuse_intervals`.
- **Capacity.** 131,072 tokens (T013): in the simulator pre-flight LRU's
  usable reuse (0.527) is 0.22 below the ceiling (0.746), four times the
  seed spread (0.05).
- **Runs.** 21 engine runs at about 31 minutes each, 6 at a time. Batch 2
  ran while analysis jobs loaded the machine (request latency doubled), so
  its six runs were replayed on a quiet machine; the contended copies are
  kept in `results/m3-task2-contended/` as a timing-sensitivity check.

The results, figures and design signals are in
[`task2-mixed-workload-policy.md`](../../report/milestone3/task2-mixed-workload-policy.md)
and its results page. They revise the proposed design above in two ways:
past hits predict reuse strongly (so frequency belongs in the score, with
protection for blocks too young to have been reused), and per-category
reuse times are not exponential (shared prompts return within seconds,
history within minutes), so the score should be learned per prompt family
or class from observed returns rather than fitted to one curve.

### Report revision (2026-11-04)

The results page was cut to six points at the owners' request (reuse
interval, reuse portion, identification, capacity, each workload alone,
alone against sharing). What this added after the Phase 1 freeze:

- `analysis/m3t2_config.json` gained `served_interval_edges_s` (0, 10, 30
  and 300 s), the bins the page quotes for the share of reuses served by
  wait. It changes no earlier number.
- The `isolated` grid: each workload's requests of the reported seeds'
  balanced plans, unchanged, so its traffic intensity is the same alone as
  in the mix. About 1% of a workload's no-eviction reuse comes from another
  workload (at most 0.9% averaged over seeds, 1.5% on one seed), so the
  alone runs use the mix's ceiling.
- Identification over the four workloads (Bailian's traces as one): one
  CART tree per signal set, depth chosen on seed 1, scored on seeds 2–4.
  Inheriting a parent's label is dropped there: in a trial it carried one
  wrong label across a whole session and lowered accuracy, so each request
  is labelled on its own.
- The capacity curve counts the measurement window ([600, 1800) s) like
  the engine numbers, so the page's points agree at 131,072 tokens.

### TTDF and RACE-RP (2026-11-05)

Two label-free policies, specified by the owners, implemented in
`vllm/core/reuse_evictors.py` and compared with LRU, LFU and FIFO on the
mixed workload only (points 1, 2, 4 and a new point 7). The definitions
are implemented as given; where the request left a choice open, this is
what the code does:

- **Access.** A request using a cached block: a lookup hit, or the block's
  insertion or deduplication when it fills (the allocator's new
  `Evictor.record_access` hook, a no-op for LRU, LFU and FIFO). So each
  request accesses each block of its prompt once, and a block computed
  again after an eviction counts as accessed.
- **Time.** The block manager's event time: `time.time()` when the
  `allocate`, `append_slots` or `free` call began (`RequestContext.time`,
  `cache_events.event_time`); in the cache replay, the simulated time. All
  accesses and evictions of one call share it.
- **Candidates.** Idle blocks (refcount 0) with no cached child. Because
  the cache only ever drops such leaves, it stays prefix-closed, and an
  ancestor never goes before its descendants. Ties: greater depth, then
  older last access, then lower block id.
- **Windows.** "In the previous 10 / 120 s" counts accesses in
  (t − 10, t] / (t − 120, t]. The family count is the access count of the
  block's second block over (t − 120, t] (the block itself at depth 1 or
  2): each request uses each block once, so those are the requests sharing
  the first two blocks.
- **Ghosts.** An evicted block's access times are kept until 180 s after
  its last access (checked lazily); a cached block's are always kept, pruned
  to the 120 s before its latest access, which changes no count.
- **Training trace.** Seed 1's plan replayed through vLLM's allocator with
  room for every block (nothing is evicted, so the trace does not depend on
  a policy), timed by seed 1's no-eviction engine run
  (`analysis/race_train.py`). Idle starts are refcount-0 frees; an
  observation at age a needs no access in (start, start + a] and
  start + a ≤ trace end − 180 s. Standardization uses the population
  standard deviation; hyperplanes are `numpy.random.default_rng(0)`.
- **Scan reuse.** The candidates are scored once per event time and kept as
  a heap for the rest of that call; a block that becomes a leaf is scored
  and pushed, and a RACE-RP access to a second block (a family count
  change) forces a fresh scan. Tests check every victim against a
  brute-force scan.
- **Tuning (seed 1 only).** Cache replay at 32K, 64K, 128K, 256K and 384K
  tokens over the window: TTDF's τ_short ∈ {1, 5, 30} s, τ_long ∈
  {60, 120, 600} s, w_short ∈ {0.3, 0.7} (w_long = 1 − w_short); RACE-RP's
  L ∈ {16, 64} and P ∈ {1, 2, 3, 4, 6, 8} (bandwidth and repetitions, as
  in Coleman and Shrivastava's RACE paper), α = 5. The best mean picks the
  setting; a tie keeps the specified one. Seeds 2–4 are then run once with
  the specified and the picked settings.
- **Runs.** The `policies` grid: both policies at 131,072 tokens on seeds
  2–4, and seed 2 at 32K and 384K to check the replay. The capacity sweep
  adds both policies to the windowed replay. `m3t2_config.json` gains
  `new_policies`; nothing earlier changes, and the alone runs (points 5
  and 6) stay with the baselines.
- **Instrumentation.** Besides the hook and the event time: the
  `VLLM_EVICTION_PARAMS` variable; in the cache replay, settings after a
  colon in the policy name, a timer on every policy's `evict` and the
  workload of each evicted block's last request. LRU, LFU and FIFO behave
  as before.

### Sampled LRFU (2026-11-05)

A third label-free policy, also specified by the owners
(`vllm/core/sampled_lrfu.py`, policy `sampled_lrfu`), on the mixed workload
only. Per cached block: a score s and the logical time τ of its last
access; an access at t sets s ← 2^(−(t−τ)/H)·s + 1 and τ ← t (a new block
starts at s = 1); to evict, up to k blocks are drawn without replacement
from the idle ones and the lowest 2^(−(T−τ)/H)·s goes, ties to the lower
block id. Choices the code makes:

- **Clock.** One step per `record_access` (hit, insertion,
  deduplication), in the order the block manager calls it; evictions do
  not advance it. A block is counted at most once per request (the request
  context names the request; a block the same request uses again, as after
  a preemption, is not counted twice).
- **Pool.** Every idle (refcount 0) block, as the definition says and as
  LRU, LFU and FIFO choose from; unlike TTDF and RACE-RP there is no
  leaf-only rule, so it can evict a block in the middle of a cached prompt.
  An array with a position map and swap-removal; `random.Random(seed)`.
- **Defaults** H = 512 block uses (under a second of this mix's traffic)
  and k = 4. Seed-1 tuning: H ∈ {64, 512, 4096, 32768, 262144, 2097152},
  k ∈ {4, 16, 64}, chosen as for TTDF and RACE-RP. The `policies` grid adds
  its five engine runs (15 in all).

## `wa` removed (2026-11-13)

Marcus removed `wa` from the code: `vllm/core/wa_evictor.py`, its
registration (`EvictionPolicy.WA`, `VLLM_EVICTION_POLICY=wa`), its tests,
and the runner's `wa` grid with its `run_policy` record. The study's
policies are now LRU, LFU and FIFO, TTDF, RACE-RP, Sampled LRFU and the
`lru_leaf` control. Phase 2's design and the results below stay as the
record; their `wa` commands no longer run. The prototype is at
`589fa730`. The engine runs of LRU and RACE-RP at 128K made by the `wa`
grid stand on their own.

Task 2's designed policy is now **RACE-RP**, which the owners specified in
Phase 1 (*TTDF and RACE-RP* above):
- It is the best policy at every size from 32K to 256K in the replay (Phase
  1 note), and on the engine at 128K it scores 58.74% against LRU's
  55.80%, +2.94 points on seeds 2–4. `wa` with oracle labels tied it
  (59.07%).
- It reads no workload labels. The course's question of how to tell a
  request's workload is answered by Phase 1's classifier (89.7% accurate).
  `wa`'s two label modes measured what the label adds: 0.86 points on the
  engine at 128K (59.07% against 58.21%).
- Its limitation: at 128K it lowers ShareGPT by 2.1 points against LRU,
  just past the 2-point gate. No policy passed the gate at every capacity.

## Scope (2026-11-10)

Marcus chose to finish the basic study rather than the full grid of
User Story 4, with the report still to write before 2026-11-30.

- **Kept:** the balanced mix, seeds 2–4. Engine runs at 131,072 tokens of
  LRU, RACE-RP and `wa` with oracle labels and with none
  (`grid --grid wa`), beside Phase 1's LFU and FIFO;
  replays at 64K, 128K and 256K, with `lru_leaf` as the control.
  Per-workload results, the gate and the overhead.
- **Dropped:**
  - the inferred label mode and the label-error runs: oracle and none
    bracket any classifier's labels, and Phase 1 reports the classifier's
    accuracy (89.7%) and confusion matrix;
  - the chat-heavy and agent-heavy mixes and the single-workload runs:
    the balanced mix already mixes seven workloads, and the per-workload
    results show which gain and which pay;
  - `wa`'s pacing-sensitivity run, and the cache-share-over-time figure;
  - PrefixSim and its outside policies;
  - the exploratory probe and the second leaf-LRU control (removed;
    their code is in `090be78d`);
  - S3-FIFO over the idle leaves (`s3fifo_leaf`), tried and removed the
    same day to keep the comparison to the course's baselines, RACE-RP
    and `wa`. It was not above LRU (replay: a tie at 64K and 128K, −3.05
    points at 256K; engine at 128K: 54.14% against 55.80%); its code and
    runs are in `4848ead7` to `3b429e99`.

`spec.md` and `tasks.md` mark each dropped requirement and task.

## Phase 2 first experiments (2026-11-08, revised 2026-11-09 and 2026-11-10; branch `milestone3-marcus-wa-spec`)

Run before T005's review, to give it evidence: the cache replay (model
timing) at three capacities, then engine runs at 131,072 tokens, on the
balanced mix with human pacing, measured over 600–1800 s.
Settings were chosen on seed 1 only; seeds 2–4 were run once with the
chosen settings. The first version of this section reported `wa` from code
with two scoring faults that PR #16's review found (see **Review fixes**
below); the `wa` numbers here are from the fixed code, with its settings
chosen again on seed 1. Those runs used the working tree that the
revision commits, on top of `631340a2` (the rows record it as dirty), and
replayed LRU, TTDF and RACE-RP in the same batch, which reproduced their
earlier numbers exactly.

```bash
python -m analysis.run_m3_task2 plans --grid phase1
# seed 1 without eviction, on the engine: RACE-RP's training
python -m analysis.run_m3_task2 replay --plan data/processed/m3/balanced-human-s1 \
    --out results/m3-task2/balanced-human-s1-ceiling --capacity 11851680
python -m analysis.race_train --sketch 16x6 --out results/m3-race
# wa settings on seed 1 at 131,072 tokens: labels oracle|none x buckets 1|0
# x half_life 120|600|1800
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s1 \
    --policies 'wa:labels=oracle,buckets=1,half_life=600' ... \
    --capacities 131072 --window 600 1800 --out results/m3-explore/wa-fixed-tune-s1.jsonl
python -m analysis.sim_table results/m3-explore/wa-fixed-tune-s1.jsonl --seeds 1
# held-out seeds
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s{2,3,4} \
    --policies lru ttdf race_rp:sketch=results/m3-race/sketch-L16-P6.json \
    'wa:labels=oracle,buckets=1,half_life=600' 'wa:labels=none,buckets=1,half_life=1800' \
    --capacities 65536 131072 262144 --window 600 1800 \
    --out results/m3-explore/wa-fixed-heldout.jsonl
python -m analysis.sim_table results/m3-explore/wa-fixed-heldout.jsonl --against lru
```

`lru_leaf` ran the same way in the first batch
(`results/m3-explore/heldout-baselines.jsonl`).

**Reproduction.** With the data, plans and RACE-RP sketch rebuilt on
another laptop, the replay at 131,072 tokens gives LRU 55.79%, TTDF 56.75%,
RACE-RP 58.73% and no eviction 78.06%, as in the Phase 1 note.

Usable reuse, seeds 2–4 (mean; per-seed verdicts by the note's rule):

| Policy | 64K | 128K | 256K |
|---|---|---|---|
| `wa`, oracle labels (buckets on, half-life 600 s) | 53.92% | 59.19% | 63.51% |
| `wa`, no labels (buckets on, half-life 1,800 s) | 53.18% | 58.14% | 62.55% |
| RACE-RP | 54.28% | 58.73% | 63.15% |
| TTDF | 51.65% | 56.75% | 61.98% |
| LRU, `lru_leaf` | 50.74% | 55.79% | 61.13% |

- **The leaf-only rule changes nothing.** `lru_leaf` evicts the same
  blocks as LRU at every capacity (same evictions per workload, same usable
  reuse), so TTDF's and RACE-RP's gains come from their scores (spec
  SC-005, in the replay). A second control, ordered by the scheduler's
  stamp instead of the access clock, did the same; it was removed as
  redundant (2026-11-10).
- **`wa` with oracle labels** is above LRU on every seed at every capacity
  (+3.19, +3.40, +2.38 points) and ties RACE-RP at every capacity (−0.36,
  +0.46, +0.36; the seeds disagree on the sign at 64K and 256K, and at 128K
  the mean is below half a point). It is above `wa` without labels at every
  capacity (+0.75, +1.05, +0.96), so the label pays off.
- **`wa` without labels** is above LRU on every seed (+2.44, +2.36,
  +1.42), ties RACE-RP at 64K and 128K and is below it at 256K (−0.60).
- **Per workload, against LRU** (`sim_table` lists every workload down by
  more than 2 points, the decision gate above):

  | Policy | 64K | 128K | 256K |
  |---|---|---|---|
  | `wa`, oracle labels | none | none | bailian-thinking −4.11 |
  | `wa`, no labels | none | ShareGPT −3.88 | ShareGPT −2.37 |
  | RACE-RP | none | ShareGPT −2.14 | none |
  | TTDF | none | none | ShareGPT −5.97 |

  `wa` with oracle labels passes the gate at 64K and 128K and fails it at
  256K; `wa` without labels, the version a server could run without being
  told each request's workload, fails it at 128K and 256K. The other
  policies fail it too: no policy so far passes at every capacity.
- **Cost per eviction in the replay**, same batch (8 runs at a time):

  | Policy | 64K | 128K | 256K |
  |---|---|---|---|
  | LRU | 0.9 µs | 1.3 µs | 1.5 µs |
  | TTDF | 2.7 µs | 4.2 µs | 7.0 µs |
  | `wa`, no labels | 25.8 µs | 43.1 µs | 84.6 µs |
  | `wa`, oracle labels | 24.6 µs | 47.6 µs | 99.0 µs |
  | RACE-RP | 35.1 µs | 49.3 µs | 116.9 µs |

  Rescoring the candidates after every end (fix 2) does one scan per
  eviction, of about 33 candidates at 128K; `wa` now costs about as much
  as RACE-RP.
- **An exploratory label probe** (survival curves per workload from the
  seed-1 engine run) was above LRU on every seed with oracle labels (+1.9,
  +2.1, +1.7 points) and tied it without labels. `wa`'s oracle-against-none
  comparison measures the same thing, so the probe and its survival-table
  script were removed (2026-11-10; they are in `090be78d`).

**Engine runs (2026-11-10).** 12 runs at 131,072 tokens on seeds 2–4:
LRU, RACE-RP and `wa` with oracle labels and with none, each on its own
`vllm serve` (simulator executor), 6 at a time.
They ran from clean commits (`4d484319` for the first six, `c32b413e` for
the rest; `wa`, LRU and RACE-RP are the same code in both).

```bash
python -m analysis.run_m3_task2 grid --grid wa --out results/m3-task2
python -m analysis.tables results/m3-task2/balanced-human-s{2,3,4}-*-cap131072
python -m analysis.sim_table --against lru \
    --engine results/m3-task2/balanced-human-s{2,3,4}-{lru,race_rp,wa_oracle,wa_none}-cap131072
```

| Policy | Usable reuse | Against LRU, per seed | Against RACE-RP | Per eviction |
|---|---|---|---|---|
| `wa`, oracle labels | 59.07% | +2.91 +3.45 +3.46, higher | +0.33, tie | 50 µs |
| RACE-RP | 58.74% | +2.67 +2.47 +3.69, higher | | 60 µs |
| `wa`, no labels | 58.21% | +2.64 +2.20 +2.40, higher | −0.53, lower | 45 µs |
| LRU | 55.80% | | | |
| *Phase 1 note:* LFU / FIFO / no eviction | 53.94% / 50.29% / 78.06% | | | |

- **The replay's verdicts hold on the engine.** `wa` with oracle labels is
  above LRU on every seed and ties RACE-RP, oracle labels are above none on
  every seed (+0.27, +1.25, +1.05; mean +0.86). Each policy is within 0.3
  points of its replay mean. The one change:
  `wa` without labels falls just below RACE-RP (every seed below, mean
  −0.53) where the replay had a tie.
- **The gate at 131,072 tokens:** `wa` with oracle labels lowers no
  workload by more than 2 points; `wa` without labels lowers ShareGPT by
  4.05, and RACE-RP ShareGPT by 2.11.
- **Every run is sound.** Each completed all its requests (5,386, 5,403
  and 5,396 per seed), none had a hit after a miss, so the cache stayed
  prefix-closed (FR-005), and vLLM's `cached_tokens` equals the usable
  prefix for every request. With
  oracle labels `wa` learned 21 classes (7 workloads × 3 reuse counts), so
  the labels reach it on the engine; with none, 3.
- **Reproducible.** This laptop's LRU is within 0.01 points of the Phase 1
  note's engine runs on every seed, and RACE-RP within 0.15. A first set
  of the same 12 runs, on the same code through a scratch
  script, gives the same means within 0.13 points. Run by run, LRU and
  RACE-RP repeat within 0.15 points but `wa` within 0.6 (seed 3 with
  oracle labels: 61.36% and 61.93%), so a repeated `wa` run moves by
  about as much as its margin over RACE-RP.
- **Discarded runs.** Six runs fell behind while other programs overloaded
  the laptop (load above 600), and two of their reruns spanned an
  11-minute sleep, which shifts the wall-clock ages `wa` scores by. Each
  was stopped, deleted and run again; the server logs of the 12 kept runs
  have no gap over 8 s.

**Review fixes (2026-11-09).** PR #16's review found two faults in the
first `wa`, both now covered by regression tests that fail on the old code:

1. *Ends younger than the block.* The density at age `a` summed the ends
   from `a`'s bin on, including that bin's ends younger than `a`. When most
   of them were, the remaining lifetime went negative and was capped, so a
   block just past its class's reuse time got the highest score instead of
   the lowest (3.6% of score calls in the seeds 2–4 replays, by the
   review's count). The bin's ends now count only if their mean age is at
   least `a`.
2. *Stale scores.* The candidates were scored once per scheduler step,
   while every hit and eviction in the step changes the statistics. They
   are now scored again after every recorded end.

Also from the review: a dedupe (a request computed the block itself) no
longer counts as a hit, for the idle period or the hit buckets. None
happens in the replay, so this matters on the engine only. With the fixes,
seed 1 picks the same settings (54.13% with oracle labels, 53.81% without;
first version 54.74% and 54.08%), and the held-out table moves down by
0.2–0.9 points.

**How much the bin rule matters.** The bins are 27% wide, so the rule for
a block's own bin is a design choice. On seed 1 at 128K, four rules are
within 0.6 points and none leads in both label modes (oracle / none):
the committed rule 54.13 / 53.81; ages rounded down to the bin's lower
edge (the review's alternative) 54.37 / 53.75; the bin's ends spread
evenly over it 54.75 / 53.59; the committed rule with bins 4 times finer
54.16 / 53.65. The review measured the rounded-down rule with fresh scores
on seeds 2–4: 59.86% at 128K with oracle labels, 0.67 points above the
committed rule and on every seed above RACE-RP. Whether `wa` beats RACE-RP
or ties it therefore depends on this choice; the committed rule is the one
that scores a block past its bin's ends below a younger block, as the
review asked.

**For T005's review.**
- **Scoring rule.** The prototype scores each leaf at its own age, as LHD
  does, not the head of each class's LRU queue as step 4 of the design
  says. It therefore does not reduce to `lru_leaf` with one label (spec
  User Story 3, scenario 6). Either build the head-of-class rule, or change
  the scenario. The bin rule above is part of this choice.
- **The gate.** Bailian-thinking at 256K with oracle labels, and ShareGPT
  without labels (128K and 256K, replay and engine), lose more than 2
  points.
- **Tuning.** TTDF and RACE-RP keep their specified settings,
  while `wa` had 12 settings tried on seed 1 (twice: before and after the
  fixes) and four bin rules.

## Phases

Target dates are proposals for the owners to confirm; the project is due
2026-11-30.

1. **Phase 0** (by 2026-11-09): mixed-trace builder and LRU
   characterization with the no-eviction ceiling (US1, US2). Starts on the
   ShareGPT single- and multi-turn stand-ins, then switches to `006`'s
   traces as they land. Ends with the run-time estimate for the full grid.
2. **Phase 1** (by 2026-11-12): finalize scoring and the capacity (FR-007)
   from the characterization; write both here and get them reviewed.
3. **Phase 2** (by 2026-11-19): implement `wa` (oracle and none), tests,
   first comparison.
4. **Phase 3** (by 2026-11-27): inferred labels; full evaluation grid,
   split across both owners' laptops; write-up in
   `report/milestone3/task2-mixed-workload-policy.md`.

## Verification

```bash
tools/run_project_tests.sh m3 -v      # test modules carry pytestmark = pytest.mark.m3
python -m pytest tests/core --ignore=tests/core/block/e2e --ignore=tests/core/test_num_computed_tokens_update.py
python -m analysis.run_m3_task2 plans --grid smoke && python -m analysis.run_m3_task2 grid --grid smoke --allow-dirty
```

## Risks

| Risk | Mitigation |
|---|---|
| Little time between `006` and the deadline (2026-11-30) | Phase 0 starts now on the ShareGPT single- and multi-turn stand-ins and switches to `006`'s traces when its adapters land; the simplest alternative (per-workload policy choice) is the fallback design |
| The grid takes too long (Task 3: about 18 minutes and up to hundreds of MB per run) | Size the samples from Phase 0's estimate; the sweep resumes, so the two owners can run halves of the grid on their own laptops |
| The best-scoring design breaks prefix closure | Prefer designs that keep it (Constraints); otherwise report it as not safe on a GPU (FR-005) |
| Overfitting parameters to one trace | FR-005 seed split; report single-workload results too |
| Inferred labels too inaccurate to help | That is a reportable finding; oracle results bound the benefit |
