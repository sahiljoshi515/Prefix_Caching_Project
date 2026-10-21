# Tasks: Client simulator and ShareGPT preprocessing

**Input**: [spec.md](spec.md), [plan.md](plan.md)

## Phase 1–3: Implementation (PR #6)

- [x] T001 [US1] ShareGPT download and fragment rebuilding
  (`client_simulator/adapters/sharegpt.py`)
- [x] T002 [US1] Preprocessing CLI writing client and server files
  (`client_simulator/process_raw_traces.py`)
- [x] T003 [US2] Poisson scheduling, single/multi replay, context check,
  request-id mapping (`client_simulator/client_sim.py`)
- [x] T004 [US3] Result JSONL, run manifest, metrics snapshots
  (`client_simulator/client_sim.py`)

## Phase 4: Review fixes in PR #6 (Sahil Joshi, `e026ae82`)

- [x] T005 [US3] Record missing `cached_tokens` as 0 (FR-007, plan #1)
- [x] T008 [P] Fix ruff, isort and yapf findings (FR-011, plan #4, #5)
- [x] T009 [P] Remove committed `results/` files (FR-012, plan #6)
- [x] T013 Merge PR #6 (2026-10-19); unblocks `001` Phase 5 and `004`

## Phase 5: Follow-up PR (open)

- [ ] T006 [US2] Keep replaying after an answer mismatch (FR-008, plan #2)
- [ ] T007 [US2] Non-zero exit only on failures (FR-009, plan #3)
- [x] T014 [P] Add `/results/` to `.gitignore` (FR-012, plan #6); done in
  the Task 2 PR (spec 004, T013)
- [ ] T010 [P] Add `__init__.py` files and the module docstring (plan #7, #8)
- [ ] T011 Tests for FR-010 (`tests/client_simulator/test_sharegpt.py`,
  `test_client_sim.py`, fixture under `tests/client_simulator/fixtures/`)
- [ ] T012 Record PR #6's AI usage for the report (the PR is merged, so add
  it to `report/ai-usage.md` instead of the description)

## Dependencies

- T006, T007, T010 and T014 are independent; T011 tests T006 and T007.
