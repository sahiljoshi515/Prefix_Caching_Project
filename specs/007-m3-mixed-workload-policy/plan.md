# Implementation Plan: A workload-aware eviction policy for mixed workloads

**Spec**: [spec.md](spec.md) | **Branch**: named per `CONTRIBUTING.md` §4 |
**Date**: 2026-10-18

## Summary

Build mixed traces from `006`'s per-workload traces, characterize per-workload
reuse from an LRU run, then design, implement and evaluate a workload-aware
evictor (`wa`) against LRU, FIFO and LFU, with oracle and inferred workload
labels. The scoring rule below is the starting hypothesis; FR-004 requires it
to be confirmed or revised from the characterization before implementation.

## Technical Context

**Language/Version**: Python 3.12 (3.9+ syntax)
**Primary Dependencies**: `005`'s evictor framework, `004`'s recorder and
metrics; `scikit-learn` only if the inferred-label classifier needs it (pin
it first)
**Testing**: `tests/eviction/test_wa_evictor.py`,
`tests/analysis/test_mix.py`

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

### Modified

| Path | Change | Reason |
|---|---|---|
| `vllm/core/eviction_policies.py` (from `005`) | Add `WorkloadAwareEvictor` | Policy home |
| `vllm/core/evictor.py` | Add `WA` to `EvictionPolicy` and dispatch | Registry |
| `vllm/core/block/cache_events.py` (from `004`) | Expose `current_request()` so the evictor can read the label at insertion | Reuses the context instead of new arguments (FR-002) |
| `vllm/envs.py` | `VLLM_WA_LABELS` | Mode switch |

## Design (starting hypothesis)

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
  `bailian:`).
- **Inferred**: features observable by a server: prompt length, number of
  messages (turns), whether the first full block matches a known
  system-prompt hash, inter-arrival time within a session. Start with a small
  decision tree trained on one seed and evaluated on another (FR-005).

## Phases

1. **Phase 0**: mixed-trace builder and LRU characterization (US1, US2).
2. **Phase 1**: finalize scoring from the characterization; write it here.
3. **Phase 2**: implement `wa` (oracle), tests, first comparison.
4. **Phase 3**: inferred labels; full evaluation grid; write-up in
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
| Little time between `006` and the deadline (2026-11-30) | Phase 0 starts as soon as `006` Phase 1 lands; the simplest alternative (per-workload policy choice) is the fallback design |
| Overfitting parameters to one trace | FR-005 seed split; report single-workload results too |
| Inferred labels too inaccurate to help | That is a reportable finding; oracle results bound the benefit |
