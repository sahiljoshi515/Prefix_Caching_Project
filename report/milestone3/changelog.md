# Milestone 3: changelog

What the group changed in Milestone 3, task by task and newest first, so the
milestone can be reviewed in one place. Each entry says what changed, who
made the change, and where (commit or PR). Results are in the per-task
notes, and each task's owner and status are in the
[spec index](../../specs/README.md).

Commit dates are the author's local date; merge dates are GitHub's (UTC).
Last updated 2026-11-08. Each PR that adds to the milestone adds its entry
here.

## Repository

### 2026-11-04: `milestone-2` renamed to `milestone-3`, the default branch (sahiljoshi515)

- The branch keeps its history (head `7d191529`); GitHub moved the open PRs
  to `milestone-3`. CI already runs on `milestone-*` branches.
- `Makefile` (`BASE`), `CONTRIBUTING.md` and `tools/pr_check/check_pr.py`
  now compare against `origin/milestone-3`.
- In an existing clone: `git branch -m milestone-2 milestone-3`, then
  `git fetch origin --prune`, `git branch -u origin/milestone-3 milestone-3`
  and `git remote set-head origin -a`.

## Task 1: eviction policies across workloads

[Spec 006](../../specs/006-m3-workload-eviction/spec.md). Not started.

### 2026-10-23: PR #9 merged into `milestone-2` by ChiaXinLiang

- **Added:** this changelog and a placeholder note,
  [`task1-workload-eviction.md`](task1-workload-eviction.md) (marcus,
  `b79d7be3`).

### 2026-10-20: PR #7 merged into `main` by sahiljoshi515

- **Added:** spec 006, a draft (ChiaXinLiang, commits by marcus,
  `51e37917`, 2026-10-18).

## Task 2: a workload-aware policy for mixed workloads

[Spec 007](../../specs/007-m3-mixed-workload-policy/spec.md). Owners: Sahil
Joshi and Xin Liang Chia. Phase 1 (the reuse study and the baselines) done
on branch `milestone3_task2`; the policy itself is Phase 2.

### Unreleased (branch `milestone3-marcus-wa-spec`, PR #16): Phase 2 spec and first experiments

- **Changed:** spec 007 revised for Phase 2 from Phase 1's results: the
  findings as Background 8–13, the class hit-density design as `wa`'s
  starting point, and FR-012 to FR-015 (a leaf-only LRU control, the full
  comparison set, label errors and a safeguard, an optional PrefixSim
  check); `plan.md` lists refinements for T005's review and `tasks.md` adds
  T016–T018 (marcus, `0cc2e85a`).
- **Added:** the `lru_leaf` and `lru_leaf_free` controls, an exploratory
  label probe (`wa_probe`, with `analysis/hazard_table.py`) and a first
  `wa` evictor with oracle and none modes, learned online (marcus,
  `090be78d`); `analysis/sim_table.py` for the replay tables. The
  runner's `wa` grid: 12 engine runs at 128K on seeds 2–4, LRU, RACE-RP
  and `wa` with oracle labels and with none, whose run.json names the
  grid policy (`run_policy`) so a later grid finds them; `replay
  --params` for one run's policy settings; `sim_table --engine` for
  engine runs, with the study's usable reuse.
- **Removed:** the second leaf-LRU control `lru_leaf_free` (it evicted the
  same blocks as `lru_leaf`), and the label probe `wa_probe` with
  `analysis/hazard_table.py` (`wa` with and without labels measures what
  the probe did). Their results stay in `plan.md`.
- **Ran:** cache replay of the balanced mix, settings on seed 1, seeds 2–4
  at 64K, 128K and 256K, recorded in `plan.md`. The leaf-only rule evicts
  the same blocks as LRU. After the review fixes, `wa` with oracle labels
  is above LRU on every seed (+3.4 points at 128K) and ties RACE-RP at
  every capacity; without labels it ties RACE-RP at 64K and 128K and is
  below it at 256K. The per-workload gate (no workload down more than 2
  points against LRU) fails for `wa` with oracle labels at 256K
  (bailian-thinking −4.1) and without labels at 128K and 256K (ShareGPT
  −3.9, −2.4); RACE-RP and TTDF fail it at one capacity each. Not yet run
  on the engine.
- **Fixed (review):** `wa`'s density no longer counts ends younger than
  the block in its own age bin, which gave blocks just past their class's
  reuse time the highest score; its candidates are scored again after
  every hit and eviction, not once per scheduler step; a dedupe is not a
  hit. The first version's table (60.13% at 128K with oracle labels) is
  replaced. Also: `wa` and the probe use `cache_events.workload_label`;
  the eviction tests share a fixture that restores the request context's
  time; `sim_table` refuses rows of different setups, leaves out the
  tuning seed and lists every gate failure.

### Unreleased (branch `milestone3_task2`, PR pending): Phase 1

- **Added:** the M3 workload adapters (AgentBank, CC-Bench, four Bailian
  traces), the mixed-workload plan builder, token-level plan replay, the
  cache simulator and its validation, and the FR-002 request context,
  merged from the earlier local branch `milestone3-sahil-mixed-workload`
  (`2310990f`).
- **Changed:** plans pace text workloads with real pauses borrowed from
  recorded loops (ShareGPT from Bailian chat's turn gaps, CC-Bench from its
  own recorded tool latencies, AgentBank from CC-Bench's), stop at the
  1,800 s window so every policy replays the same requests, and give seed k
  each Bailian trace's k-th window (`01a81876`). Spec 007 amended to match
  (`4264d3bd`).
- **Added:** the grid runner (`plans`, `grid`, `sim`; `5e7e4558`,
  `f94f8e58`), a Belady reference in the simulator (`993bab02`), per-run
  lookup, residency and request tables with one 004 reuse-interval
  implementation (`cf0395df`), label-free workload identification and a
  ranking of reuse signals (`975efbe9`), the ATC'25 paper checks
  (`1f1fe61c`), and the Phase 1 study, results page and figures
  (`de3b08d4`).
- **Ran:** 21 engine runs (balanced mix; seeds 2–4 reported, seed 1 for
  training, seed 2 with Phase 0's fast pacing, one LRU replicate), at
  131,072 tokens and 16-token blocks, plus a simulator sweep of cache size
  for LRU, LFU, FIFO and Belady. Results:
  [`task2-mixed-workload-policy.md`](task2-mixed-workload-policy.md).
- **Added:** each workload run alone with the requests it sends in the mix
  (`analysis/isolate.py`, the `isolated` grid; `56980db6`): 36 engine runs
  (4 workloads x LRU, LFU, FIFO x seeds 2–4). The cache replay can count
  only the measurement window (`sim --window`).
- **Changed:** the results page, its figures and the results note now cover
  six points only: reuse interval and reuse portion on the mix, workload
  identification over the four workloads (each observable signal alone and
  together, accuracy and confusion matrix), usable reuse against cache size
  from 32K to 384K tokens with Belady, and each workload alone against
  sharing the cache.
- **Added:** three label-free policies for the mixed workload, specified
  by the owners. **TTDF** (two decayed use counts, 5 s and 120 s) and
  **RACE-RP** (RACE sketches of reuse, trained on seed 1;
  `analysis/race_train.py`) evict only idle blocks with no cached child
  (`vllm/core/reuse_evictors.py`, `58cab544`, `dbde4253`). **Sampled LRFU**
  (a score halving every 512 block uses; the lowest of 4 random idle
  blocks goes) is O(1) per access and eviction
  (`vllm/core/sampled_lrfu.py`, `1fd0ddc5`). vLLM gains only an
  `Evictor.record_access` hook and a per-call event time; LRU, LFU and FIFO
  evict as before.
- **Ran:** seed-1 tuning in the cache replay; 15 engine runs (the three
  policies at 131,072 tokens on seeds 2–4, and seed 2 at 32K and 384K);
  the held-out capacity sweep for each policy as specified and as tuned.
  The results page adds them to points 1, 2 and 4 and a new point 7
  (`edad452a`, `a0314233`, `2f171e09`).

### Unreleased (branch `milestone3-marcus-mixed-workload-spec`)

- **Changed:** spec 007 revised from Milestone 2's findings (marcus). The
  spec now names its owners and gains two sections: why each step of the
  task is needed, with an example of LRU evicting the wrong workload's
  blocks, and a Background of what Milestone 2 found. New
  requirements: the policy keeps cached blocks prefix-closed, because vLLM
  v0 overcounts `cached_tokens` otherwise (FR-005). The usable reused
  fraction is the primary metric (FR-008). The comparison capacity must
  put the mix under pressure (FR-007). The policy learns reuse timing
  online instead of reading the client's pacing (FR-010). The workload
  label reaches the evictor with the event log off (FR-002).
- **Added:** spec 007's `none` label mode, to measure what the labels
  themselves are worth; a no-eviction ceiling per mix; per-workload results
  and a pacing-sensitivity run; ShareGPT single- and multi-turn as stand-in
  workloads until spec 006's adapters land; and a list of required figures
  (FR-011). The plan gains its design constraints and proposed target
  dates, and the tasks gain T012 to T015.

### 2026-10-23: PR #9 merged into `milestone-2` by ChiaXinLiang

- **Added:** a placeholder note,
  [`task2-mixed-workload-policy.md`](task2-mixed-workload-policy.md)
  (marcus, `b79d7be3`).

### 2026-10-20: PR #7 merged into `main` by sahiljoshi515

- **Added:** spec 007, a draft (ChiaXinLiang, commits by marcus,
  `51e37917`, 2026-10-18).

## Open items

- Assign the owners of spec 006 (spec 007's are Sahil Joshi and Xin
  Liang Chia).
- Both tasks build on Milestone 2: the event log and `analysis/` (Task 2)
  and the eviction policies (Task 3).
