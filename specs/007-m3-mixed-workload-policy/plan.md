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
