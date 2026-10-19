# Tasks: Eviction policies across workloads

**Input**: [spec.md](spec.md), [plan.md](plan.md)

## Phase 0: Research

- [ ] T001 [P] AgentBank: format, size, licence, role mapping; update
  `client_simulator/README.md`
- [ ] T002 [P] CC-Bench-trajectories: same
- [ ] T003 [P] qwen-bailian-usagetraces-anon: same, plus its block size and
  timestamp unit
- [ ] T004 Choose the fixed capacity and block size from `005` results;
  record why

## Phase 1: User Story 1 — adapters (P1)

- [ ] T005 [P] [US1] AgentBank adapter + fixture + tests
- [ ] T006 [P] [US1] CC-Bench adapter + fixture + tests
- [ ] T007 [P] [US1] Bailian adapter with token synthesis + fixture + tests
- [ ] T008 [US1] Client: recorded timestamps and token-id prompts + tests
  (`client_simulator/client_sim.py`, `schema.py`)

## Phase 2: User Story 2 — comparison runs (P1)

- [ ] T009 [US2] `analysis/run_m3_task1.py` (workload × policy, summary)
- [ ] T010 [US2] Run all pairs on the agreed samples

## Phase 3: User Story 3 — analysis (P2)

- [ ] T011 [US3] Per-workload reuse-profile figures and the comparison plot
- [ ] T012 [US3] Write-up (`report/notes/m3-task1.md`)

## Dependencies

- Needs `005` merged (evictors, sweep runner) and `003` merged (client).
- `007` needs Phase 1 (workload-tagged traces) and the Phase 2 baselines.
