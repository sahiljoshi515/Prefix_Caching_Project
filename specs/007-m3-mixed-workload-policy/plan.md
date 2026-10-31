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
python -m analysis.run_m3_task2 --grid smoke --out results/smoke-m3t2
```

## Risks

| Risk | Mitigation |
|---|---|
| Little time between `006` and the deadline (2026-11-30) | Phase 0 starts now on the ShareGPT single- and multi-turn stand-ins and switches to `006`'s traces when its adapters land; the simplest alternative (per-workload policy choice) is the fallback design |
| The grid takes too long (Task 3: about 18 minutes and up to hundreds of MB per run) | Size the samples from Phase 0's estimate; the sweep resumes, so the two owners can run halves of the grid on their own laptops |
| The best-scoring design breaks prefix closure | Prefer designs that keep it (Constraints); otherwise report it as not safe on a GPU (FR-005) |
| Overfitting parameters to one trace | FR-005 seed split; report single-workload results too |
| Inferred labels too inaccurate to help | That is a reportable finding; oracle results bound the benefit |
