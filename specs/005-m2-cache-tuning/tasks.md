# Tasks: Tune cache parameters and eviction policies

**Input**: [spec.md](spec.md), [plan.md](plan.md)

## Phase 0: Research

- [x] T001 From the `004` multi-turn run, compute the working set (distinct
  full-block hashes × block size) and choose the capacity points: 114,964
  blocks × 16 = 1,839,424 tokens (Task 2 note); points in the plan, Sweep

## Phase 1: User Story 1 — selectable eviction policy (P1)

- [x] T002 [US1] Test first: scenarios 1 and 2 of the spec, tie-breaks,
  reuse after free, evict-then-reinsert resets FIFO order and LFU count
  (`tests/eviction/test_eviction_policies.py`, marked `m2`)
- [x] T003 [US1] `FIFOEvictor`, `LFUEvictor` (`vllm/core/eviction_policies.py`)
- [x] T004 [US1] Enum members, dispatch, `Evictor.record_hit` default
  (`vllm/core/evictor.py`)
- [x] T005 [US1] Variable, allocator wiring, hit hook, startup log
  (`vllm/envs.py`, `vllm/core/block/prefix_caching_block.py`,
  `vllm/core/block/cpu_gpu_block_allocator.py`; see plan, Changes from the
  draft)
- [x] T006 [US1] Engine test: the variable selects the evictor; bad value
  fails; unset is LRU (`tests/simulator/test_eviction_policy_env.py`)
- [x] T007 [US1] SC-001: same replay with the variable unset gives identical
  `cached_tokens` as `main`. Checked 2026-10-23 against `milestone-2`
  (`728c0d8a`): the first 30 ShareGPT conversations, multi-turn,
  `--conversation-rate inf --think-time 0 --max-concurrency 1` (so the
  order is deterministic), `--num-gpu-blocks-override 1024` (so it evicts).
  Both gave 147 requests and 5,209 evictions, with identical per-request
  `cached_tokens` and identical per-request hit/miss/evict sequences in the
  event log

**Checkpoint**: PR 1 merged; `tests/core` green.

## Phase 2: User Story 2 — sweep runner (P1)

- [x] T008 [P] [US2] Test first: grid expansion, token-capacity to block
  count, resume skips finished runs (`tests/analysis/test_sweep.py`)
- [x] T009 [US2] Sweep runner (`analysis/sweep.py`)
- [x] T010 [US2] Task 3 grid and summary (`analysis/run_m2_task3.py`)

## Phase 3: User Story 3 — results (P2)

- [x] T011 [US3] Run the full grid; commit only report figures (45 runs at
  `9f18c4c8`, 2026-10-23/24, and 9 more at 131,072 tokens at `4270ae94`,
  2026-10-31; every run passed SC-003 of spec 004)
- [x] T012 [US3] Figures 1 to 5 of the spec (`analysis/plots.py`,
  `analysis/run_m2_task3.py`; item 5 is `best_by_capacity.md`)
- [x] T013 [US3] Write-up with the recommendation table
  (`report/milestone2/task3-cache-tuning.md`)

## Dependencies

- Needs `004` Phases 1 and 2 merged (recorder, metrics, launcher).
- `006` and `007` reuse the evictors and the sweep runner.
