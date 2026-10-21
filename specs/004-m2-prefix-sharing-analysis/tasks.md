# Tasks: Replay ShareGPT and analyze prefix sharing

**Input**: [spec.md](spec.md), [plan.md](plan.md)
**Branch**: `milestone2-marcus-prefix-sharing-analysis`; one PR for all of
Task 2 into `milestone-2`, merged with a merge commit.

## Phase 0: Prerequisites and research

- [ ] T001 CI runs on PRs into `milestone-2`: add `milestone-*` to the
  workflow branch filters; `Makefile` `BASE` and the CONTRIBUTING.md
  `check_pr.py` command follow the default branch (separate PR,
  `ci-marcus-milestone-branches`: PR #8, awaiting review)
- [x] T002 Preprocess ShareGPT locally (`client_simulator/README.md`)
- [x] T003 Replay 100 conversations in each mode with the defaults; record
  in `plan.md` the duration, the "Simulated KV cache" block count, the
  skipped/failed counts and the chosen sample size, rate and think time
- [x] T004 Check whether the chosen sample evicts at the default cache size
  (needs T008; otherwise compare `num_gpu_blocks` with the sample's total
  distinct prompt blocks)

## Phase 1: User Story 1 — cache event log (P1)

- [x] T005 [US1] Test first (`tests/simulator/test_cache_events.py`, marked
  `m2`), on gpt2 with block size 16:
  - a 40-token prompt sent twice in sequence: request 1 logs `miss`,
    `insert` for 2 blocks; request 2 logs `hit` and `revive` for both, with
    the right `req` and `op: allocate`;
  - decode filling the third block logs `insert` with `op: append_slots`;
  - `--num-gpu-blocks-override 6` with prompts that overflow it logs
    `evict`, each preceded by a `free` of the same hash;
  - two requests sharing a prefix in one step: the second logs `hit` without
    `revive`;
  - variable unset: no file, and `_events is None` on the allocator;
  - header first, and `end` last after the recorder is closed.
- [x] T006 [US1] Declare `VLLM_CACHE_EVENT_LOG` (`vllm/envs.py`)
- [x] T007 [US1] Recorder, sequence map and context
  (`vllm/core/block/cache_events.py`)
- [x] T008 [US1] Hooks in the allocator, GPU-only attachment, block-manager
  context (`prefix_caching_block.py`, `cpu_gpu_block_allocator.py`,
  `block_manager.py`)
- [x] T009 [US1] Overhead check (SC-001): 100 conversations,
  `--conversation-rate inf --think-time 0`, three runs each on
  `milestone-2`, this branch with the recorder off, and on; record in
  `plan.md` and the PR

**Checkpoint**: `make test-m1 test-m2 test-core` green.

## Phase 2: User Story 2 — metrics library (P1)

- [x] T010 [P] [US2] Test first (`tests/analysis/test_metrics.py`, marked
  `m2`): handcrafted logs for every row of the spec's metric table,
  including zero-hit blocks, single-access hashes, evict then re-insert,
  a missing `end` line, and hits on decode-written blocks
- [x] T011 [P] [US2] Log loader (`analysis/events.py`)
- [x] T012 [US2] Metric functions and hit-source attribution
  (`analysis/metrics.py`)
- [x] T013 Pin and install `matplotlib==3.9.2`
  (`constraints-simulator.txt`, `Makefile`); add `/results/` to
  `.gitignore`

**Checkpoint**: `make test-m2` runs `tests/analysis` (`project-ci` does the
same once the PR is open).

## Phase 3: User Stories 3 and 4 — experiments and analysis

- [x] T014 [US3] Server launcher shared with later specs
  (`analysis/server.py`)
- [x] T015 [US3] Plots (`analysis/plots.py`): the seven figures of US3
- [x] T016 [US3] Experiment command with the SC-003/SC-004 checks in the
  manifest (`analysis/run_m2_task2.py`); 20-conversation smoke run under
  2 minutes
- [ ] T017 [US3] Run the agreed sample in both settings; commit only the
  figures used in the report (`report/milestone2/figures/task2/`)
- [ ] T018 [US4] Write the analysis with numbers, including the
  intra/cross and prompt/decode-written hit shares
  (`report/milestone2/task2.md`)
- [x] T019 Document the commands in `README.md`; tick these tasks

## Dependencies

- `003` is merged (client results with the zero-hit fix).
- T001 (PR #8) merged before the Task 2 PR is opened, so the PR is checked.
- `005` builds on this branch's recorder, `analysis/metrics.py` and
  `analysis/server.py`, so it starts once the Task 2 PR is merged.
