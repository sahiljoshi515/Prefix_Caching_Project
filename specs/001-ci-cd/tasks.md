# Tasks: Project CI/CD

**Input**: [spec.md](spec.md), [plan.md](plan.md)

## Phase 1: Setup

- [x] T001 Measure the suites locally: `tests/simulator` 67 passed in about
  21 s, `tests/core` subset 944 passed in about 13 s (2026-10-18)
- [x] T002 Reproduce the `mypy (3.12)` failure and confirm the fix:
  `uvx --python 3.12 mypy==1.11.1 ... vllm/executor` fails without
  setuptools and passes with `setuptools<76`

## Phase 2: User Story 1 + 3 — project tests on every PR (P1, P2)

- [x] T003 [US1] Add `project-ci` with the install, simulator, client/analysis
  and core steps (`.github/workflows/project-ci.yml`)
- [x] T004 [US1] Push the branch, open the PR, confirm `project-ci / tests` is
  green and under 15 minutes (SC-003): PR #7, 2 min 41 s; torch
  `2.5.1+cpu`; m1 67 passed, m2/m3 "no tests yet", tests/core 944 passed
- [ ] T005 [US1] On a throwaway commit, break one simulator assertion and
  confirm only the simulator step fails (SC-002); drop the commit

**Checkpoint**: the PR shows `project-ci / tests` passing.

- [x] T024 [US1] Milestone markers `m1`/`m2`/`m3` in `pyproject.toml`;
  tag the 3 Milestone 1 test modules
- [x] T025 [US1] `tools/run_project_tests.sh` (all, one milestone,
  check-markers); m1 gives 67 passed, m2 "no tests yet", an unmarked test
  makes check-markers exit 1; shellcheck clean
- [x] T026 [US1] One CI step per milestone plus the marker check
  (`.github/workflows/project-ci.yml`)

## Phase 3: User Story 2 — only meaningful checks (P1)

- [x] T006 [P] [US2] Install `setuptools<76` in the mypy job
  (`.github/workflows/mypy.yaml`)
- [x] T007 [P] [US2] Scope `lint-and-deploy` to `examples/chart-helm/**`
  (`.github/workflows/lint-and-deploy.yaml`)
- [x] T008 [P] [US2] Delete `.github/workflows/reminder_comment.yml`
- [x] T009 [P] [US2] Delete `.github/workflows/publish.yml`
- [ ] T010 [US2] On the PR, confirm no failing or irrelevant checks (SC-001).
  PR #7: all green except `pr_reminder`, which runs from `main`'s copy
  (`pull_request_target`) until this PR merges; recheck on the next PR

## Phase 3b: User Story 7 — PR requirements check (P1)

- [x] T019 [US7] Test first: one or more cases per rule, plus a real git
  repository (`tools/pr_check/test_check_pr.py`, 39 passed)
- [x] T020 [US7] Checker (`tools/pr_check/check_pr.py`); ruff, isort, yapf
  and codespell clean
- [x] T021 [US7] Workflow (`.github/workflows/pr-requirements.yml`)
- [x] T022 [US7] Replay on history: PR #6 as opened gives 10 errors and 2
  warnings, all genuine; the Milestone 1 range flags its 4 unlisted
  upstream files
- [x] T023 [US7] On this branch's PR, confirm the check passes with a
  complete description (PR #7: `pr-requirements / check` passed); the
  failing cases are covered by `tools/pr_check/test_check_pr.py`

## Phase 4: User Story 6 — repository metadata (P3)

- [x] T011 [P] [US6] Replace `.github/PULL_REQUEST_TEMPLATE.md` with the
  `CONTRIBUTING.md` §4 items
- [x] T012 [P] [US6] Remove the `pip` ecosystem and upstream reviewers from
  `.github/dependabot.yml`
- [ ] T013 [US6] Replace `.github/CODEOWNERS` (lists vLLM maintainers) with
  the team's GitHub handles. Blocked: needs Xin Liang Chia's and Jasmine
  Lu's handles (known: `sahiljoshi515`, `trietmchau`)
- [ ] T014 [US6] Ask the repository owner to mark `project-ci / tests`,
  `pr-requirements / check`, `ruff`, `yapf`, `mypy` and `codespell` as
  required checks on `main`

## Phase 5: User Story 4 — e2e replay smoke test (P2)

Unblocked: `003` merged on 2026-10-19 (PR #6).

- [ ] T015 [US4] Commit a 20-conversation ShareGPT-format fixture, under
  100 KB (`tests/fixtures/sharegpt_small.json`)
- [ ] T016 [US4] Add an e2e step or job: preprocess the fixture, start
  `vllm serve` with the simulator, replay single and multi, assert all
  completed and multi-turn mean cached fraction > 0
  (`.github/workflows/project-ci.yml`)

## Phase 6: User Story 5 — report and submission delivery (P3)

Blocked until `report/` (MLSys 2025 LaTeX) exists.

- [ ] T017 [US5] Build the report PDF on changes to `report/`, check the page
  limit, upload the artifact (`.github/workflows/report.yml`)
- [ ] T018 [US5] On `submission-*` tags, publish a release with the PDF and a
  `git bundle --all` (`.github/workflows/submission.yml`)

## Dependencies

- T004, T005 and T010 need the branch pushed (needs the author's go-ahead).
- Phase 5 can start now (`003` is on `main`); Phase 6 needs `report/`.
- T013 needs two teammates' GitHub handles; T014 needs the owner.
