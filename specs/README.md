# Specs

Spec-driven development for the COMP 536 prefix-sharing project, in the
layout of [GitHub Spec Kit](https://github.com/github/spec-kit). The rules
every spec follows are in
[`.specify/memory/constitution.md`](../.specify/memory/constitution.md); new
specs start from [`.specify/templates/`](../.specify/templates/).

Each folder holds:

- `spec.md`: **what** and **why**: user stories, requirements, metric
  definitions, success criteria
- `plan.md`: **how**: files, upstream edits with reasons, design decisions,
  verification commands
- `tasks.md`: ordered checklist; boxes are ticked in the PR that does the
  task

## Index

| Spec | Charter item | Owner | Status | Depends on |
|---|---|---|---|---|
| [001-ci-cd](001-ci-cd/spec.md) | Clean, tested PRs (cross-cutting) | Marcus | In progress | — |
| [002-m1-simulator-verification](002-m1-simulator-verification/spec.md) | Milestone 1 verification | Triet Chau, Xin Liang Chia | Awaiting verification | — |
| [003-m2-client-simulator](003-m2-client-simulator/spec.md) | M2 Task 1: client simulator | Triet Chau | Merged (PR #6); follow-ups open | M1 |
| [004-m2-prefix-sharing-analysis](004-m2-prefix-sharing-analysis/spec.md) | M2 Task 2: replay and analyze | Xin Liang Chia | In progress | 003 |
| [005-m2-cache-tuning](005-m2-cache-tuning/spec.md) | M2 Task 3: cache parameters, LRU/LFU/FIFO | Jasmine Lu | Draft | 003, 004 |
| [006-m3-workload-eviction](006-m3-workload-eviction/spec.md) | M3 Task 1: policies per workload | TBD | Draft | 003, 004, 005 |
| [007-m3-mixed-workload-policy](007-m3-mixed-workload-policy/spec.md) | M3 Task 2: workload-aware policy | TBD | Draft | 004, 005, 006 |

## Dependency order

```
001 CI/CD ─────────────── runs every spec's tests
002 M1 verification

003 client simulator ──► 004 recorder + metrics ──► 005 evictors + sweep
                                                   │
                         006 adapters + per-workload comparison ◄┘
                                                   │
                         007 mixed traces + workload-aware policy ◄┘
```

Shared building blocks, so later specs do not re-implement them:

| Built in | Reused by | What |
|---|---|---|
| 003 | 004–007 | Client simulator, request-id format, result JSONL |
| 004 | 005–007 | `VLLM_CACHE_EVENT_LOG` recorder, `analysis/metrics.py`, `analysis/server.py` |
| 005 | 006, 007 | `VLLM_EVICTION_POLICY`, evictor framework with `record_hit`, sweep runner |
| 006 | 007 | Workload adapters and workload-prefixed request ids |

## Working with a spec

1. Read the spec and plan before starting; raise disagreements in review of
   the spec, not the code.
2. Work through `tasks.md` in order. Tests come before or with the code they
   test.
3. Open the PR with `.github/PULL_REQUEST_TEMPLATE.md`; link the spec and
   tick the finished tasks in the same PR.
4. If the implementation changes the design, update `plan.md` in the same PR
   and say so in the description.
5. When the results are final, write them up in the task's note under
   `report/notes/` (see [`report/README.md`](../report/README.md)). The spec
   holds the plan; findings go into the note.

## Running the tests

Only the project's own tests, never vLLM's full upstream suite. No need to
activate the virtualenv; `make` uses `.venv` directly:

```bash
make setup                     # once: create .venv and install (macOS/Linux)
make test                      # every milestone
make test-m2                   # one milestone: test-m1, test-m2, test-m3
make test-m1 ARGS="-k prefix"  # extra pytest arguments
make ci                        # everything CI runs, in the same order
make lint                      # the repo's linters on files you changed
make check-pr BODY=pr.md       # the PR-requirements check, before opening a PR
make help                      # all commands
```

CI runs the same `make` targets, so a green run locally means a green run
on GitHub (apart from Linux-only differences).

Every new test module declares its milestone on one line, and CI fails a
test that has none:

```python
pytestmark = pytest.mark.m2
```

Project tests live in `tests/simulator`, `tests/client_simulator`,
`tests/analysis` and `tests/eviction`; a new folder is added to
`PROJECT_TEST_DIRS` in the script.

## Deadline

Everything, including the report, is due **2026-11-30 23:59** (the Canvas
deadline wins if it differs). Milestone 3 depends on most of Milestone 2, so
`004` and `005` Phases 1–2 are on the critical path.
