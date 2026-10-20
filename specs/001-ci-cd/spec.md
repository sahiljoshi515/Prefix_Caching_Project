# Feature Specification: Project CI/CD

**Spec**: `001-ci-cd` | **Milestone/Task**: cross-cutting (supports all)
**Owner**: Marcus | **Reviewers**: any one other member
**Status**: In progress
**Created**: 2026-10-18 | **Depends on**: none

**Input**: Charter: "create a clean pull request (PR) once the feature has
been implemented and thoroughly tested. Other team members can then review
the changes [...] before we merge the PR into the main branch."
`CONTRIBUTING.md` §6.5: "Once continuous integration is configured, all
required checks shall pass before a PR is merged."

## Background

The repository inherits vLLM's GitHub workflows. On PR #6 (the first
Milestone 2 PR), 6 of 11 checks failed, and 3 of those failures had nothing
to do with the PR:

| Check | Why it failed | Relevant to us? |
|---|---|---|
| `lint-and-deploy` | Helm chart test; `docker run minio/minio` is denied ("pull access denied") | No |
| `pr_reminder` | Posts vLLM's Buildkite reminder; no permission on a fork PR | No |
| `mypy (3.12)` | `pkg_resources` not found; also fails on `main` | Yes, but broken |
| `ruff`, `yapf` | Real style errors in the PR | Yes |

Meanwhile nothing in CI runs our own tests: `tests/simulator` (Milestone 1)
only runs on members' laptops. Reviewers therefore cannot tell from a PR's
checks whether it works.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Every PR runs the project's tests (Priority: P1)

A PR author pushes a branch and opens a PR against `main`. Within 15 minutes
a check named `project-ci / tests` reports whether the simulator
test suite passes on a clean Linux machine installed exactly as the README
says.

**Why this priority**: it is the check the charter's review process relies
on, and every later milestone adds tests to it.

**Independent Test**: open a PR that changes only a comment; the check runs
and passes. Push a commit that breaks an assertion in `tests/simulator`; the
check fails and names the test.

**Acceptance Scenarios**:

1. **Given** a PR against `main`, **When** it is opened or updated,
   **Then** the `tests` job installs the project with
   `constraints-simulator.txt` and CPU-only PyTorch, and runs the project's
   tests one milestone per step (`tools/run_project_tests.sh m1`, `m2`,
   `m3`), so the run page shows which milestone broke.
2. **Given** a new test module without a milestone marker, **When** the job
   runs, **Then** the step "Every project test is marked with its milestone"
   fails and names the test, so no test is silently left out of every
   milestone step.
3. **Given** a member on a laptop, **When** they run
   `tools/run_project_tests.sh m2`, **Then** exactly the Milestone 2 tests
   run, with the same environment variables as CI.
4. **Given** a push to `main`, **When** it lands, **Then** the same job
   runs, so `main`'s status is always visible.

---

### User Story 2 - Only meaningful checks, all green on a clean PR (Priority: P1)

A reviewer sees a PR's check list and every failure is about that PR.

**Why this priority**: permanent red checks teach the team to ignore CI.

**Independent Test**: a documentation-only PR shows no failing check.

**Acceptance Scenarios**:

1. **Given** a PR that does not touch `examples/chart-helm/`, **When**
   checks run, **Then** `lint-and-deploy` does not run.
2. **Given** any PR, **When** it is opened, **Then** no vLLM Buildkite
   reminder is posted.
3. **Given** `main` at the current commit, **When** `mypy` runs on Python
   3.9 to 3.12, **Then** all four jobs pass.
4. **Given** a `v*` tag is pushed, **When** workflows trigger, **Then**
   nothing tries to build and publish vLLM release wheels.

---

### User Story 3 - vLLM's block-manager tests run too (Priority: P2)

Milestone 2 Task 3 and Milestone 3 modify `vllm/core/` (evictors, block
manager). The `tests/core` subset from `CONTRIBUTING.md` §6.2 guards it. It
takes about 13 seconds (944 tests, measured 2026-10-18 on a laptop), so it
runs in the same job on every PR instead of behind a path filter. A
path-filtered workflow cannot be a required check, because it is reported as
pending when skipped.

**Independent Test**: break `LRUEvictor.evict`; the `tests/core` step fails.

---

### User Story 4 - End-to-end replay smoke test (Priority: P2)

Now that the client simulator (`003`) is merged, every PR should also start
`vllm serve` with the simulator, replays a small committed ShareGPT-format
fixture in single-turn and multi-turn mode, and checks that every request
completes with its recorded answer and that multi-turn has a nonzero cached
fraction.

**Why this priority**: it is the only check that exercises the client, the
server and the cache together, which is exactly what the report measures.

**Independent Test**: break the request-id mapping in the client; the smoke
test fails because answers no longer match.

---

### User Story 5 - Deliverables are built, not hand-assembled (Priority: P3)

Continuous delivery for a course project means the graded artifacts are
produced by CI:

1. On every push to `main` that changes `report/`, the LaTeX report is built
   and the PDF is uploaded as a workflow artifact, and the build fails if it
   exceeds 6 pages before the references.
2. When a tag `submission-*` is pushed, a workflow attaches to a GitHub
   Release the report PDF and a `git bundle` of the full history (the `.git`
   folder the course requires, in a portable form).

**Independent Test**: push tag `submission-test` on a fork; the release
contains `report.pdf` and `prefix-caching.bundle`, and
`git clone prefix-caching.bundle` reproduces the full history.

---

### User Story 7 - Every PR is checked against the team's rules (Priority: P1)

Tests show that the code works; they do not show that a PR follows
`CONTRIBUTING.md`. A check named `pr-requirements / check` reads the PR
itself and fails it when a rule is broken, with an annotation per problem and
a summary table on the run page. The checker also runs locally
(`python tools/pr_check/check_pr.py --base origin/main --head HEAD
--body-file <description.md>`).

**Why this priority**: PR #6 was opened without the description items,
with experiment output committed and without tests. A reviewer had to find
all of that by hand.

**Independent Test**: replayed on PR #6 as first opened, the checker reports
the 6 missing description sections and the 4 committed `results/` files as
errors, and the missing tests and the commit `implement client sim` as
warnings.

**Rules**:

| Rule | Level | Source |
|---|---|---|
| The six description sections exist and are filled (not the template text) | error | CONTRIBUTING.md §4 |
| Every modified upstream vLLM file (one that exists at `94d545a1`) is named under "Upstream changes" | error | constitution II |
| Nothing under `vllm/v1/` changes | error | constitution I |
| Nothing under `data/`, `results/`, `.venv/`; no file over 1 MB | error | CONTRIBUTING.md §7 |
| Python code under `vllm/`, `client_simulator/` or `analysis/` changes with a test change | warning | CONTRIBUTING.md §6 |
| Commit subjects ≤ 72 characters, capitalized, no period; a body is present | warning | CONTRIBUTING.md §5 |
| "Milestone and task" links a spec | warning | constitution, workflow |
| Branch name is `<type>-<name>-<description>` (e.g. `milestone2-triet-client-simulator`) | warning | CONTRIBUTING.md §4 |

Missing tests and commit messages are warnings, not errors. A refactor may
need no new test, and fixing an old commit message means rewriting branch
history, which `CONTRIBUTING.md` §4 discourages.

**Acceptance Scenarios**:

1. **Given** a PR whose description still contains the template text,
   **Then** the check fails and names each unfilled section.
2. **Given** the author then edits the description, **Then** the check runs
   again (the `edited` event) and passes.
3. **Given** a PR from Dependabot, **Then** the check is skipped.

---

### User Story 6 - Repository metadata fits the team (Priority: P3)

- The PR template asks for the six items of `CONTRIBUTING.md` §4 instead of
  vLLM's text.
- Dependabot does not propose bumps of the deliberately pinned Python
  dependencies, and does not request vLLM maintainers as reviewers.

### Edge Cases

- **A fork PR lacks secrets.** The workflows need none: everything they
  download (PyPI, the PyTorch CPU index, Hugging Face config/tokenizer files
  for `gpt2` and `Qwen/Qwen2.5-0.5B-Instruct`) is public.
- **Hugging Face is down or rate limits.** The job fails at the download
  step with a clear message rather than inside a test. Caching the HF
  directory between runs is a later improvement.
- **A dependency resolver picks CUDA PyTorch.** The install step must use the
  PyTorch CPU index; otherwise it downloads several GB and may run out of disk.
- **Tests that need real model execution.** These are already excluded by
  `CONTRIBUTING.md` §6.2 and stay excluded.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A workflow `project-ci` MUST run on `pull_request` to `main`
  or a `milestone-*` branch and on `push` to them, and cancel superseded
  runs of the same branch. The other checks use the same branch filter.
- **FR-002**: It MUST install the project as the README's Linux instructions
  do (Python 3.12, `VLLM_TARGET_DEVICE=empty`, `-c constraints-simulator.txt`)
  but from the PyTorch CPU index.
- **FR-003**: It MUST run the project's tests through
  `tools/run_project_tests.sh`, one step per milestone, after a step that
  fails when any project test lacks an `m1`/`m2`/`m3` marker. The script
  sets `VLLM_USE_SIMULATOR=1` and `VLLM_NO_USAGE_STATS=1`, runs only the
  project test folders (`tests/simulator`, `tests/client_simulator`,
  `tests/analysis`, `tests/eviction`), and treats "no tests for this
  milestone yet" as success.
- **FR-004**: `project-ci` MUST also run the §6.2 `tests/core` subset, as a
  separate step so its failure is reported on its own.
- **FR-005**: `lint-and-deploy` MUST run only when `examples/chart-helm/**` or
  its own file changes.
- **FR-006**: `reminder_comment.yml` and `publish.yml` MUST be removed.
- **FR-007**: The `mypy` workflow MUST pass on `main` for Python 3.9 to 3.12.
- **FR-008**: Existing ruff, isort, yapf and codespell checks MUST stay as
  they are; they pass on `main` and catch real problems (PR #6).
- **FR-009**: New workflows MUST pin third-party actions by commit SHA, as the
  existing workflows do, and request only `contents: read` unless they publish.
- **FR-010**: The PR template MUST list the six PR-description items of
  `CONTRIBUTING.md` §4.
- **FR-011**: Dependabot MUST NOT open `pip` update PRs and MUST NOT request
  upstream reviewers.
- **FR-012** *(US4)*: A workflow MUST run the end-to-end replay smoke test
  once `client_simulator/` is on `main`.
- **FR-013** *(US5)*: Workflows MUST build the report PDF and publish a
  submission release as described in US5 once `report/` exists.
- **FR-014** *(US7)*: A workflow `pr-requirements` MUST run
  `tools/pr_check/check_pr.py` on every PR event (opened, edited,
  reopened, synchronize, ready for review), applying the rules of US7, and
  MUST read the PR description from the event file, never by interpolating
  it into a shell command.
- **FR-015** *(US7)*: The checker MUST use only the standard library, so the
  check needs no project install and finishes in under a minute, and it MUST
  have its own unit tests, which run in the same workflow.

### Key Entities

- **Required checks**: `project-ci / tests`, `pr-requirements / check`,
  `ruff`, `yapf`, `mypy`, `codespell`. A repository admin marks these as required in branch
  protection for `main` (a settings change, not a file).

## Success Criteria *(mandatory)*

- **SC-001**: A documentation-only PR shows zero failing checks.
- **SC-002**: A PR that breaks one simulator test shows exactly one failing
  check, `project-ci / tests`, naming that test.
- **SC-003**: `project-ci / tests` finishes in under 15 minutes.
- **SC-004**: After this spec's PR merges, every workflow run on `main` is
  green.

## Assumptions

- GitHub-hosted `ubuntu-latest` runners, free minutes for a public repo.
- Branch protection is configured by the repository owner (Sahil Joshi)
  after this lands; the workflows do not depend on it.

## Out of Scope

- Running vLLM's full upstream test suite (it needs GPUs).
- Buildkite, Docker image builds, Helm charts, PyPI publishing.
- Running the full-size experiments in CI. They take too long and need the
  full datasets; CI runs fixtures only.
