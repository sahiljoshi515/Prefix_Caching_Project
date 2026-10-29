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

- **Changed:** the Task 1 note records the full run: 5,097 of 5,098
  multi-turn requests that reached the server replayed exactly, the one
  mismatch comes from an `<|endoftext|>` string in a recording, the context
  limit removed 25% of multi-turn turns, and the ShareGPT rebuild kept
  99.1% of conversations (`33393edb`).
- **Added:** the Task 1 note,
  [`task1-client-simulator.md`](task1-client-simulator.md): what was built
  and what Task 2's runs verified about it, including a likely connection
  race between vLLM's 5 s keep-alive and the 5 s think time, now follow-up
  T015 in spec 003 (marcus, `b79d7be3`).
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

[Spec 004](../../specs/004-m2-prefix-sharing-analysis/spec.md). Done on
branch `milestone2-marcus-prefix-sharing-analysis` (marcus), with the
1,000-conversation results recorded; in review as PR #9.

### Unreleased: 2026-10-21

- **Added:** the results of the 1,000-conversation run, in the Task 2 note
  with seven figures (`figures/task2/`). Multi-turn hits on 95% of block
  lookups against 13% for single-turn; 98% of multi-turn hits are within
  the conversation, three quarters on blocks of generated tokens; both
  self-checks pass. Spec 004's T017 and T018 are ticked (`33393edb`).
- **Added:** the Task 2 note,
  [`task2-prefix-sharing-analysis.md`](task2-prefix-sharing-analysis.md),
  with its setup; the results follow the 1,000-conversation run
  (`b79d7be3`).
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

[Spec 005](../../specs/005-m2-cache-tuning/spec.md), drafted in PR #7
(Jasmine Lu). Phases 1 and 2 are merged into `milestone-2`; Phase 3 is in
review as PR #12.

### Unreleased: 2026-10-24 (Phase 3, PR #12)

- **Added:** the results of the 45-run grid (3 policies × block sizes 8,
  16, 32 × 5 capacities from 16,384 to 2,097,152 tokens, on Task 2's
  1,000-conversation sample), in the Task 3 note with five figures
  (`figures/task3/`) and the best configuration per capacity. Reuse climbs
  from 2% to 95% between 16,384 and 262,144 tokens and stays there; LRU is
  best; FIFO matches LRU's hit rate but, under pressure, half its hits sit
  behind a miss and cannot be used (0.38 against 0.76 of prompt tokens at
  65,536); plain LFU collapses whenever the cache evicts, because it
  evicts each conversation's newest blocks first; block size 8 is
  slightly best.
- **Added:** the usable reused fraction (spec 004's metric table): reuse
  counted from the contiguous cached prefix in the event log
  (`metrics.prefix_hits`, `metrics.add_usable_fractions`), reported by
  `analyze_run` and ranked on by `run_m2_task3`. vLLM's `cached_tokens`
  overcounts under FIFO because `find_cached_blocks_prefix` assumes
  cached blocks form a prefix; found in review. 3 tests.
- **Added:** `run_m2_task3` draws spec 005's figures and writes
  `best_by_capacity.csv`/`.md`; `analysis/plots.py` gains the policies'
  colours and a small-multiples plot. 4 tests.

### 2026-10-26: Phase 2, PR #11 re-landed as PR #14, merged into `milestone-2` by ChiaXinLiang

- **Added:** the sweep runner, `python -m analysis.run_m2_task3`: one fresh
  server per configuration of policy, block size and capacity (in tokens),
  resumable, with a `smoke` grid (6 runs, 2 minutes) and the `full` grid
  (45 runs). Each run is analyzed with Task 2's metrics and summarized in
  one row of `summary.csv`. Eviction count and rate (spec 005, FR-007) are
  in `analysis/metrics.py`, and Task 2's `analyze_run` reports them;
  `run_setting` takes extra server arguments, an environment and manifest
  fields. 22 tests.

### 2026-10-26: Phase 1, PR #10 merged into `milestone-2` by ChiaXinLiang

- **Added:** FIFO and LFU evictors next to vLLM's LRU
  (`vllm/core/eviction_policies.py`), chosen for the GPU prefix cache with
  `VLLM_EVICTION_POLICY=lru|fifo|lfu`; unset means LRU. The allocator
  reports each lookup hit to the evictor through `Evictor.record_hit`, which
  LFU counts. 41 evictor tests and 9 engine tests (`6ade43db`).
- **Changed:** spec 005's plan records four changes from the draft: the
  variable is read where the GPU allocator is built; it yields a policy,
  which vLLM's own `make_evictor` turns into an evictor; FIFO orders by time
  so a request's blocks keep LRU's deepest-first tie-break; and counters are
  keyed by block id.

### Unreleased (Task 2 branch)

- **Added:** a placeholder note,
  [`task3-cache-tuning.md`](task3-cache-tuning.md), with the sections to
  fill (marcus, `b79d7be3`).

## Across tasks: team process, CI and documentation

### Unreleased: 2026-10-21 (Task 2 branch)

- **Added:**
  - `report/` for the technical report: a README with the shape of a note
    and a draft paper outline, and the AI-usage record with PRs #7 and #8
    (`532371fb`, `5e1a2f9d`).
  - `CONTRIBUTING.md` Section 8, Documentation, and a PR-template checkbox
    for updated documents (`532371fb`).
  - This changelog (`fc120862`, `bef0b579`).
- **Changed:** `report/` split into one folder per milestone, with each
  milestone's changelog, notes and figures inside it and `ai-usage.md` at
  the top (`22598a4c`). Milestone 1's own files are in its changelog.
- **Changed:** task notes are named after their specs
  (`task<K>-<name>.md`), every task has a note or a placeholder, and
  Milestone 3 has its folder and changelog (`b79d7be3`).

### PR #13, open (sahiljoshi515)

- **Fixed:** the `codespell` check, failing on `milestone-2` since PR #9's
  PDF figures were committed: `*.pdf` and `*.png` are skipped
  (`521ed9e6`, 2026-10-24). Unblocks PR #10.

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
  [`ai-usage.md`](../ai-usage.md), and the likely keep-alive race (T015).
- Task 2: review and merge PR #9; its checks run once PR #8 merges.
- Task 3: Phase 1 (evictors) in review as PR #10, Phase 2 (sweep runner)
  as PR #11 after #10, Phase 3 (the grid's results and the note) as PR #12
  after #11.
- The branch protection ruleset targets no branch, so nothing stops a direct
  push to `main` (spec 001, T014; repository settings, owner only).
