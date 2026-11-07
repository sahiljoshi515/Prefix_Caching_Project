# Tasks: Eviction policies across workloads

**Input**: [spec.md](spec.md), [plan.md](plan.md)

## Phase 0: Research

- [x] T001 [P] AgentBank: format, size, licence, role mapping; update
  `client_simulator/README.md`
- [x] T002 [P] CC-Bench-trajectories: same
- [x] T003 [P] qwen-bailian-usagetraces-anon: same, plus its block size and
  timestamp unit
- [x] T004 Choose the fixed capacity and block size from `005` results;
  record why (`plan.md`: 65,536 tokens, block size 16)

## Phase 1: User Story 1 — adapters (P1)

- [x] T005 [P] [US1] AgentBank adapter + fixture + tests
- [x] T006 [P] [US1] CC-Bench adapter + fixture + tests
- [x] T007 [P] [US1] Bailian adapter with token synthesis + fixture + tests
- [x] T008 [US1] Client: recorded timestamps and token-id prompts + tests
  (`client_simulator/client_sim.py`, `schema.py`)

## Phase 2: User Story 2 — comparison runs (P1)

- [x] T009 [US2] `analysis/run_experiment.py` (shared prepared chat/token/mixed runner)
- [ ] T010 [US2] Run all pairs on the agreed samples

## Phase 3: User Story 3 — analysis (P2)

- [ ] T011 [US3] Per-workload reuse-profile figures and the comparison plot
- [ ] T012 [US3] Write-up (`report/milestone3/task1-workload-eviction.md`)

## Dependencies

- Needs `005` merged (evictors, sweep runner) and `003` merged (client).
- `007` needs Phase 1 (workload-tagged traces) and the Phase 2 baselines.

Bailian implementation scope (2026-11-05): adapter, paired token export,
recorded-time client, completion-aware metrics and shared runner are added.
Other text adapters, full workload comparisons, figures and final findings
remain pending. The planned fixed configuration is in `plan.md`.

Bailian analysis subset complete (2026-11-05): first 500 requests at capacity
65,536/block size 16 under LRU/FIFO/LFU, validity audit, comparison bars,
zero-inclusive CDFs, windowed hit rate and the Task 1 note. T010/T011/T012
remain open for the other workloads. Additional segments/repeats are not
part of the current evidence.

The primary Bailian sample is now 3,000 requests (829.828 s); 500 requests
remain a pilot. The three saved runs passed audit with no exclusions,
failures or preemptions, and their figures/plain-text findings replace the
pilot as the main analysis. Other workloads and repeat/segment stability
checks remain open.
