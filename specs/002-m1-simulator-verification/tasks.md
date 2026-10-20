# Tasks: Independent verification of the Milestone 1 simulator

**Input**: [spec.md](spec.md), [plan.md](plan.md)

## Phase 1: Setup

- [ ] T001 Open the tracking issue "Milestone 1 verification" linking this
  spec

## Phase 2: User Stories 1 and 2 (P1), in parallel per verifier

- [ ] T002 [P] [US1] Triet Chau: run the plan's procedure, post the report
- [ ] T003 [P] [US1] Xin Liang Chia: run the plan's procedure, post the report
- [ ] T004 [P] [US2] Each verifier: confirm the four acceptance scenarios of
  User Story 2 by name in the test log

## Phase 3: Follow-up

- [ ] T005 File one issue per deviation; fix each through a `fix/` or
  `docs/` PR. Known already: the README's "What changed" table says
  `tests/simulator` has 59 tests; it has 67 (measured 2026-10-18)
- [ ] T006 Close the tracking issue when SC-001 and SC-002 hold

## Dependencies

- T002 and T003 are independent. T005 depends on their findings.
