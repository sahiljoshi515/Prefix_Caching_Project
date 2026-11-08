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
alternative is tried, and a negative result is reported (SC-003).

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

## Phase 2 first experiments (2026-11-08, branch `milestone3-marcus-wa-spec`)

Run before T005's review, to give it evidence: cache replay only (model
timing), the balanced mix with human pacing, measured over 600–1800 s.
Settings were chosen on seed 1 only; seeds 2–4 were run once with the
chosen settings. The replay ran this branch's code before it was committed,
on top of `ef5fe385` (the rows record that commit as dirty); the commits
since change formatting only.

```bash
python -m analysis.run_m3_task2 plans --grid phase1
# seed 1 without eviction, on the engine: RACE-RP's training and the probe's table
python -m analysis.run_m3_task2 replay --plan data/processed/m3/balanced-human-s1 \
    --out results/m3-task2/balanced-human-s1-ceiling --capacity 11851680
python -m analysis.race_train --sketch 16x6 --out results/m3-race
python -m analysis.hazard_table --run results/m3-task2/balanced-human-s1-ceiling \
    --out results/m3-explore/hazard-s1.json
# wa settings on seed 1 at 131,072 tokens: labels oracle|none x buckets 1|0
# x half_life 120|600|1800
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s1 \
    --policies lru 'wa:labels=oracle,buckets=1,half_life=600' ... \
    --capacities 131072 --window 600 1800 --out results/m3-explore/wa-tune-s1.jsonl
# held-out seeds
python -m analysis.run_m3_task2 sim --plans data/processed/m3/balanced-human-s{2,3,4} \
    --policies lru lru_leaf lru_leaf_free ttdf \
    race_rp:sketch=results/m3-race/sketch-L16-P6.json \
    'wa:labels=oracle,buckets=1,half_life=600' 'wa:labels=none,buckets=1,half_life=1800' \
    --capacities 65536 131072 262144 --window 600 1800 --out results/m3-explore/<rows>.jsonl
python -m analysis.sim_table results/m3-explore/<rows>.jsonl ... --against lru
```

**Reproduction.** With the data, plans and RACE-RP sketch rebuilt on
another laptop, the replay at 131,072 tokens gives LRU 55.79%, TTDF 56.75%,
RACE-RP 58.73% and no eviction 78.06%, as in the Phase 1 note.

Usable reuse, seeds 2–4 (mean; per-seed verdict against LRU by the note's
rule):

| Policy | 64K | 128K | 256K |
|---|---|---|---|
| `wa`, oracle labels (buckets on, half-life 600 s) | 54.58% | 60.13% | 64.42% |
| `wa`, no labels (buckets on, half-life 1,800 s) | 53.40% | 58.88% | 63.34% |
| RACE-RP | 54.28% | 58.73% | 63.15% |
| TTDF | 51.65% | 56.75% | 61.98% |
| LRU, `lru_leaf`, `lru_leaf_free` | 50.74% | 55.79% | 61.13% |

- **The leaf-only rule changes nothing.** `lru_leaf` and `lru_leaf_free`
  evict the same blocks as LRU at every capacity (same evictions per
  workload, same usable reuse), so TTDF's and RACE-RP's gains come from
  their scores (spec SC-005, in the replay).
- **`wa` with oracle labels** is above LRU on every seed at every capacity
  (+3.85, +4.34, +3.29 points), above RACE-RP at 128K and 256K (+1.39,
  +1.27) and tied with it at 64K (+0.30). It is above `wa` without labels
  by 1.1–1.2 points at every capacity, so the label pays off.
- **`wa` without labels** ties RACE-RP at 128K and 256K and is below it at
  64K (−0.88).
- **Per workload, oracle against LRU:** at 128K the largest loss is
  bailian-api's (−0.55) and ShareGPT gains 8.3; at 256K bailian-thinking
  loses 4.9 points while ShareGPT gains 31.4. The decision gate above (no
  workload down by more than 2 points) therefore holds at 128K and fails at
  256K.
- **Cost per eviction in the replay** at 128K: LRU 1.6 µs, `wa` without
  labels 23 µs, `wa` with oracle labels 51 µs, RACE-RP 63 µs.
- **The label probe** (`wa_probe`: survival curves per workload from the
  seed-1 engine run, horizon chosen on seed 1) is above LRU on every seed
  with oracle labels (+1.9, +2.1, +1.7 points) and ties it without labels.
  It helps ShareGPT and CC-Bench at Bailian's expense. `wa` supersedes it.

**For T005's review.**
- **Scoring rule.** The prototype scores each leaf at its own age, as LHD
  does, not the head of each class's LRU queue as step 4 of the design
  says. It therefore does not reduce to `lru_leaf` with one label (spec
  User Story 3, scenario 6). Either build the head-of-class rule, or change
  the scenario.
- **Bailian at 256K.** The loss to bailian-thinking needs a look before
  the engine runs.
- **Not yet run:** engine runs of `wa`, inferred labels, label errors and
  the chat-heavy and agent-heavy mixes. TTDF and RACE-RP keep their
  specified settings, while `wa` had 12 settings tried on seed 1.

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
