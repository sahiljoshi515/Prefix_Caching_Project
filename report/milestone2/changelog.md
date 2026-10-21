# Milestone 2: changelog

What the group changed in Milestone 2, task by task and newest first, so the
milestone can be reviewed in one place. Each entry says what changed, who
made the change, and where (commit or PR). It comes from the git history
and the pull requests. Results are in the per-task notes, and each task's
owner and status are in the [spec index](../../specs/README.md).

Commit dates are the author's local date; merge dates are GitHub's (UTC).
Last updated 2026-10-21. Each PR that adds to the milestone adds its entry
here.

## Task 1: client simulator and ShareGPT preprocessing

[Spec 003](../../specs/003-m2-client-simulator/spec.md). Merged in PR #6;
follow-ups open (spec 003, Phase 5).

### Unreleased (Task 2 branch)

- **Fixed:** `/results/` is now in `.gitignore`, closing follow-up T014
  (marcus, `0bdb37be`).

### 2026-10-19: PR #6 merged into `main` by sahiljoshi515

- **Added** (trietmchau, `ee944717`, 2026-10-18):
  - ShareGPT adapter: downloads the pinned
    `ShareGPT_V3_unfiltered_cleaned_split.json` once and rebuilds whole
    conversations from its `<id>_<offset>` fragments, skipping broken
    groups with counted reasons (`client_simulator/adapters/sharegpt.py`).
  - Preprocessing command: writes the client's conversations and the
    server's response trace, one line per turn with the id
    `<conversation_id>:<turn>` (`process_raw_traces.py`). The full dataset
    gives 49,704 conversations and 328,525 turns.
  - Replay client: Poisson or immediate conversation arrivals from a seed,
    single- or multi-turn, a fixed think time and a concurrency bound. Chat
    requests go through the server so the model's chat template applies; a
    `/tokenize` check skips turns longer than the context. Writes one JSON
    line per request, a `run.json` manifest and `/metrics` snapshots
    (`client_sim.py`, `schema.py`, `client_simulator/README.md`).
- **Fixed** (sahiljoshi515, `e026ae82`, 2026-10-18):
  - A request with no cache hit records `cached_tokens` 0 and
    `cached_fraction` 0.0 instead of null, because vLLM omits the field.
  - The ruff, isort and yapf findings.
  - Preprocessing no longer imports the replay client (`request_id()` moved
    to `schema.py`).
- **Removed** (sahiljoshi515, `e026ae82`): the placeholder AgentBank,
  CC-Bench and qwen-bailian adapters, and the committed `results/` files.

## Task 2: replay ShareGPT and analyze prefix sharing

[Spec 004](../../specs/004-m2-prefix-sharing-analysis/spec.md). In progress
on branch `milestone2-marcus-prefix-sharing-analysis` (marcus); the
1,000-conversation run is under way, and the PR opens after PR #8.

### Unreleased: 2026-10-21

- **Changed:**
  - Spec 004 brought up to date: 16 tasks ticked, the Phase 0 measurements
    and the overhead check recorded, one PR for the whole task, seven
    figures (`c887fc81`).
  - README: `VLLM_CACHE_EVENT_LOG`, the Milestone 2 tests, a section on how
    to tell the prefix cache worked, and troubleshooting (`91a9c13b`).

### Unreleased: 2026-10-20

- **Added:**
  - Cache event log: with `VLLM_CACHE_EVENT_LOG` set, vLLM writes one JSON
    line per prefix-cache event (hit, miss, insert, dedupe, free, revive,
    evict) with the request and block-manager operation behind it; nothing
    changes when it is unset. 11 engine tests (`72f2dc2c`).
  - Metrics library `analysis/`: per-request cached fraction, hits per
    block, reuse intervals in seconds and in requests, windowed hit rate,
    and hit sources (same or other conversation, prompt or generated
    block). 19 tests with hand-derived answers (`0bdb37be`).
  - Experiment command `python -m analysis.run_m2_task2`, the server
    launcher and seven figures. It runs both settings and checks that the
    log matches vLLM's own hit rate (SC-003). 9 tests; matplotlib pinned
    (`81b58699`).
- **Changed:** spec 004 revised after reading vLLM's v0 allocator; eight
  findings changed the design (`1a889aaf`).
- **Fixed:** type annotations in two tests, for mypy (`e6849da8`).

## Task 3: cache size, block size and eviction policy

[Spec 005](../../specs/005-m2-cache-tuning/spec.md), drafted in PR #7. Not
started.

## Across tasks: team process, CI and documentation

### Unreleased: 2026-10-21 (Task 2 branch)

- **Added:**
  - `report/` for the technical report: a README with the shape of a note
    and a draft paper outline, and the AI-usage record with PRs #7 and #8
    (`532371fb`, `5e1a2f9d`).
  - `CONTRIBUTING.md` Section 8, Documentation, and a PR-template checkbox
    for updated documents (`532371fb`).
  - This changelog (`fc120862`).
- **Changed:** `report.md` moved to `report/milestone1/simulator.md`, with
  its test count corrected (`295c1c8e`).

### PR #8, open (ChiaXinLiang, commits by marcus)

- **Changed:** CI checks also run on PRs into `milestone-*` branches, and
  the Makefile and `CONTRIBUTING.md` follow the default branch
  (`35f3b49f`, 2026-10-20).

### 2026-10-20: `milestone-2` becomes the default branch (sahiljoshi515)

### 2026-10-20: PR #7 merged into `main` by sahiljoshi515

- **Added** (ChiaXinLiang, commits by marcus, 2026-10-18):
  - `CONTRIBUTING.md`: branching, commits, tests, experiments and AI use
    (`278d2718`).
  - The spec-kit constitution, templates and specs 001 to 007, one per
    milestone task (`51e37917`).
  - Project CI: the tests per milestone on every PR, and a check of each
    PR against `CONTRIBUTING.md` (`947d7dfd`, `9c503d1a`).
  - A Makefile with shared setup, test and lint commands (`0bfcb2c8`).
  - The README for Milestone 2 (`483faf70`).

## Open items

- PR #8 needs review; until it merges, PRs into `milestone-2` run no checks.
- Task 1 follow-ups (spec 003, Phase 5): keep replaying after an answer
  mismatch, exit non-zero only on failures, unit tests, and PR #6's entry in
  [`ai-usage.md`](../ai-usage.md).
- Task 2: the full run, its note and figures, then the PR.
- Task 3 has not started; it builds on Task 2's event log and `analysis/`.
- The branch protection ruleset targets no branch, so nothing stops a direct
  push to `main` (spec 001, T014; repository settings, owner only).
