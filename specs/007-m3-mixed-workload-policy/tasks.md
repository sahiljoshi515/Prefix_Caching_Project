# Tasks: A workload-aware eviction policy for mixed workloads

**Input**: [spec.md](spec.md), [plan.md](plan.md)

## Phase 0: User Stories 1 and 2 — mixed traces and characterization (P1)

- [ ] T001 [US1] Test first: two-fixture 50/50 mix, ordering, determinism
  (`tests/analysis/test_mix.py`)
- [ ] T002 [US1] Mixed-trace builder (`analysis/mix.py`)
- [ ] T003 [US2] Per-workload reuse profiles (`analysis/characterize.py`)
- [ ] T004 [US2] LRU run on three mixes; characterization figures

## Phase 1: Design

- [ ] T005 Finalize the scoring rule from T004's evidence; update `plan.md`
  (FR-004) and get review before coding

## Phase 2: User Story 3 — the policy (P1)

- [ ] T006 [US3] Test first: 3-workload toy trace with hand-derived victims;
  shared-block label rule (`tests/eviction/test_wa_evictor.py`)
- [ ] T007 [US3] `WorkloadAwareEvictor` and registration
  (`vllm/core/eviction_policies.py`, `vllm/core/evictor.py`)
- [ ] T008 [US3] Label hand-off via the request context, oracle mode
  (`vllm/core/block/cache_events.py`, `vllm/envs.py`)
- [ ] T009 [US3] Inferred-label classifier; accuracy against oracle

## Phase 3: User Story 4 — evaluation (P2)

- [ ] T010 [US4] `analysis/run_m3_task2.py` grid: mixes × seeds × policies ×
  label modes, plus single-workload traces
- [ ] T011 [US4] Figures and write-up (`report/notes/m3-task2.md`)

## Dependencies

- Needs `006` Phase 1 (adapters) and its baselines; `005` evictor framework;
  `004` recorder.
