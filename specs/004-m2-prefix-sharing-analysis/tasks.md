# Tasks: Replay ShareGPT and analyze prefix sharing

**Input**: [spec.md](spec.md), [plan.md](plan.md)

## Phase 0: Research

- [ ] T001 Measure single and multi replay throughput for 100 conversations;
  record sample size, rate and think time decisions in `plan.md`
- [ ] T002 Record the default KV-cache block count from the server log and
  whether the chosen sample triggers any eviction

## Phase 1: User Story 1 — cache event log (P1)

- [ ] T003 Declare `VLLM_CACHE_EVENT_LOG` (`vllm/envs.py`)
- [ ] T004 [US1] Test first: engine test with block size 16, a repeated
  40-token prompt (2 misses + 2 inserts, then 2 hits), a 6-block cache that
  forces evictions, and two concurrent requests sharing a prefix
  (`tests/simulator/test_cache_events.py`)
- [ ] T005 [US1] Recorder and `current_request`
  (`vllm/core/block/cache_events.py`)
- [ ] T006 [US1] Hooks in the allocator, GPU-only attachment, request
  context (`prefix_caching_block.py`, `cpu_gpu_block_allocator.py`,
  `block_manager.py`)
- [ ] T007 [US1] Overhead check, recorder off versus no hooks, 100
  conversations (SC-001)

**Checkpoint**: PR 1 merged; `tests/simulator` and `tests/core` green.

## Phase 2: User Story 2 — metrics library (P1)

- [ ] T008 [P] [US2] Test first: handcrafted logs for each metric in the
  spec's table, including zero-hit blocks, single-access hashes and evict
  then re-insert (`tests/analysis/test_metrics.py`)
- [ ] T009 [P] [US2] Log loader (`analysis/events.py`)
- [ ] T010 [US2] Metric functions (`analysis/metrics.py`)
- [ ] T011 [US2] Intra/cross-conversation attribution (`analysis/metrics.py`)

**Checkpoint**: PR 2 merged; `project-ci` runs `tests/analysis`.

## Phase 3: User Stories 3 and 4 — experiments and analysis

- [ ] T012 [US3] Server launcher shared with later specs
  (`analysis/server.py`)
- [ ] T013 [US3] Plots (`analysis/plots.py`)
- [ ] T014 [US3] Experiment command with SC-003/SC-004 self-checks
  (`analysis/run_m2_task2.py`)
- [ ] T015 [US3] Run the agreed sample in both settings; commit only the
  report figures
- [ ] T016 [US4] Write the analysis with numbers
  (`report/notes/m2-task2.md`)
- [ ] T017 Pin `matplotlib` in `constraints-simulator.txt` and document the
  commands in `README.md`

## Dependencies

- Needs `003` merged (client results with zero-hit fix).
- `005` starts once Phase 1 and Phase 2 are merged; it reuses the recorder,
  `analysis/metrics.py` and `analysis/server.py`.
