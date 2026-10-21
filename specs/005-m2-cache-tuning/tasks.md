# Tasks: Tune cache parameters and eviction policies

**Input**: [spec.md](spec.md), [plan.md](plan.md)

## Phase 0: Research

- [ ] T001 From the `004` multi-turn run, compute the working set (distinct
  full-block hashes × block size) and choose the capacity points

## Phase 1: User Story 1 — selectable eviction policy (P1)

- [ ] T002 [US1] Test first: scenarios 1 and 2 of the spec, tie-breaks,
  reuse after free, evict-then-reinsert resets FIFO order and LFU count
  (`tests/eviction/test_eviction_policies.py`, marked `m2`)
- [ ] T003 [US1] `FIFOEvictor`, `LFUEvictor` (`vllm/core/eviction_policies.py`)
- [ ] T004 [US1] Enum members, dispatch, `Evictor.record_hit` default
  (`vllm/core/evictor.py`)
- [ ] T005 [US1] Variable, allocator wiring, hit hook, startup log
  (`vllm/envs.py`, `vllm/core/block/prefix_caching_block.py`)
- [ ] T006 [US1] Engine test: the variable selects the evictor; bad value
  fails; unset is LRU (`tests/simulator/test_eviction_policy_env.py`)
- [ ] T007 [US1] SC-001: same replay with the variable unset gives identical
  `cached_tokens` as `main`

**Checkpoint**: PR 1 merged; `tests/core` green.

## Phase 2: User Story 2 — sweep runner (P1)

- [ ] T008 [P] [US2] Test first: grid expansion, token-capacity to block
  count, resume skips finished runs (`tests/analysis/test_sweep.py`)
- [ ] T009 [US2] Sweep runner (`analysis/sweep.py`)
- [ ] T010 [US2] Task 3 grid and summary (`analysis/run_m2_task3.py`)

## Phase 3: User Story 3 — results (P2)

- [ ] T011 [US3] Run the full grid; commit only report figures
- [ ] T012 [US3] Figures 1 to 5 of the spec (`analysis/plots.py`)
- [ ] T013 [US3] Write-up with the recommendation table
  (`report/milestone2/task3.md`)

## Dependencies

- Needs `004` Phases 1 and 2 merged (recorder, metrics, launcher).
- `006` and `007` reuse the evictors and the sweep runner.
