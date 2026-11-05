# Milestone 3: changelog

What the group changed in Milestone 3, task by task and newest first, so the
milestone can be reviewed in one place. Each entry says what changed, who
made the change, and where (commit or PR). Results are in the per-task
notes, and each task's owner and status are in the
[spec index](../../specs/README.md).

Commit dates are the author's local date; merge dates are GitHub's (UTC).
Last updated 2026-11-04. Each PR that adds to the milestone adds its entry
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
