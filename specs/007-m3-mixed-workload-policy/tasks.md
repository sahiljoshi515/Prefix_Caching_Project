# Tasks: A workload-aware eviction policy for mixed workloads

**Input**: [spec.md](spec.md), [plan.md](plan.md)

`wa` was removed from the code on 2026-11-13 (`plan.md`, *`wa` removed*):
T006–T008 are dropped, and the `wa` parts of T016 with them.

## Phase 0: User Stories 1 and 2 — mixed traces and characterization (P1)

- [x] T001 [US1] Test first: two-fixture 50/50 mix, ordering, determinism,
  request and token shares (`tests/analysis/test_mix.py`); Phase 1 adds
  pacing, the plan's stop and disjoint Bailian windows
- [x] T002 [US1] Mixed-trace builder with its manifest (`analysis/mix.py`,
  FR-001); Phase 1: human pacing from recorded loops, 1,800 s stationary
  plans, seed k on each Bailian trace's k-th window
- [x] T012 [US1] Stand-in workloads: superseded by `006`'s adapters, which
  the earlier work added (AgentBank, CC-Bench, four Bailian traces)
- [x] T003 [US2] Per-workload reuse profiles, with each eviction attributed
  to the workload whose block was evicted (`analysis/characterize.py`,
  `analysis/tables.py`, `analysis/m3t2_study.py`)
- [x] T004 [US2] Engine runs of the balanced mix (seeds 2–4 reported, seed
  1 for training) under LRU, LFU and FIFO with their no-eviction ceilings;
  the reuse study, its figures and results page; chat-heavy and
  agent-heavy mixes in the simulator only (`analysis/run_m3_task2.py
  grid --grid phase1`, about 31 minutes per run, 6 at a time)

## Phase 1: Design

- [ ] T005 Finalize the scoring rule from T004's evidence; update `plan.md`
  (FR-004) and get review before coding, including the refinements of
  2026-11-07 and the label safeguard (FR-014)
- [x] T013 Choose the capacity under pressure and record why (FR-007):
  131,072 tokens; in the simulator pre-flight, LRU's usable reuse (0.527)
  is 0.22 below the ceiling (0.746), against a seed spread of 0.05

## Phase 2: User Story 3 — the policy (P1)

- [x] T006 [US3] Test first: 3-workload toy trace with hand-derived victims,
  the shared-block label rule, prefix closure, unknown modes refused
  (`tests/eviction/test_wa_evictor.py`). Dropped (`wa` removed 2026-11-13)
- [x] T007 [US3] `WorkloadAwareEvictor` and registration
  (`vllm/core/eviction_policies.py`, `vllm/core/evictor.py`); a prototype
  with oracle and none modes is in `vllm/core/wa_evictor.py`, with
  replay and engine results in `plan.md` (2026-11-08, revised after PR
  #16's review and on 2026-11-10), pending T005's review. Dropped (`wa`
  removed 2026-11-13)
- [x] T008 [US3] Label hand-off that works with the event log off, oracle
  and none modes (`vllm/core/block/cache_events.py`, `vllm/envs.py`,
  FR-002); engine test: same `cached_tokens` with the log on and off
  (`tests/simulator/test_wa_policy_env.py`). The design holds: `wa`
  switches the request context on itself (`LeafScoreEvictor`), so labels
  reach it with the log off; the engine test is not written. Dropped
  (`wa` removed 2026-11-13)
- [x] T009 [US3] Inferred-label classifier, trained on seeds the comparison
  does not use (FR-006); accuracy and confusion matrix against oracle
  (`analysis/identify.py`; trained on seed 1, tested on seeds 2–4)

## Phase 3: User Story 4 — evaluation (P2)

- [ ] T016 [US4] `lru_leaf` control on `LeafScoreEvictor`, with a test that
  `wa` in `none` mode with refinements off picks the same victims (FR-012;
  `tests/eviction/test_lru_leaf.py`); replay and engine runs at 131,072
  tokens on seeds 2–4 (SC-005). Done: the control and its tests, and
  the replay (it evicts the same blocks as LRU); not done: its engine runs
  (LRU's stand in, the replay showing them equal) and the `wa` equality
  test, which waits for T005
- [x] T017 [US4] Label-error runs: corrupt oracle labels at the rates of
  User Story 4, scenario 6, in the replay (FR-014, figure 7 of FR-011).
  Dropped (scope of 2026-11-10, `plan.md`)
- [x] T019 [US4] S3-FIFO over the idle leaves (`s3fifo_leaf`), with its
  tests, in the replay and on the engine. Removed the same day (scope of
  2026-11-10, `plan.md`): not above LRU; its code is in `4848ead7`
- [x] T018 [US4] Optional: export the mixes to PrefixSim's JSONL, run its
  LRU, ARC, S3-FIFO, SIEVE, LHD, `workload_aware` and Belady with
  `--dump-holes`, and recompute usable reuse; check its LRU against the
  replay first (FR-015). Dropped (scope of 2026-11-10)
- [ ] T010 [US4] Grid command extending `analysis/sweep.py` (FR-009): mixes
  × seeds × policies (FR-013) × label modes × label-error rates, ceilings,
  single-workload traces
  (`analysis/run_m3_task2.py`; Phase 1 has the `plans`, `grid` and `sim`
  commands). Done for the scope of 2026-11-10: `grid --grid wa` runs both
  label modes beside LRU and RACE-RP on the balanced mix; the
  mixes, label-error and single-workload axes were dropped
- [x] T014 [US4] Pacing-sensitivity run: one mix at another think time and
  rate, parameters unchanged (Phase 1 ran the baselines with fast pacing on
  seed 2). Dropped for `wa` (scope of 2026-11-10)
- [x] T015 [US4] Overhead: replay time and `evict()` cost, `wa` against LRU
  (`plan.md`, Phase 2 experiments: replay and engine)
- [ ] T011 [US4] Figures 4 and 5 of FR-011 (3 dropped; 1, 2 and 6 are
  Phase 1's) and the write-up
  (`report/milestone3/task2-mixed-workload-policy.md`)

## Dependencies

- Needs `006` Phase 1 (adapters) and its baselines for the real mixes; T012
  lets Phase 0 start before then. Also needs `005`'s evictor framework and
  sweep runner, and `004`'s recorder with the usable reused fraction
  (PR #12).
- T005 needs T004; T006 to T009 need T005; T010 needs T007 and T008.
- T016 needs nothing new and can start before T005's review; T006's
  regression test needs it.
