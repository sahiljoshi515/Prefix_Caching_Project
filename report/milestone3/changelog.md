# Milestone 3: changelog

What the group changed in Milestone 3, task by task and newest first, so the
milestone can be reviewed in one place. Each entry says what changed, who
made the change, and where (commit or PR). Results are in the per-task
notes, and each task's owner and status are in the
[spec index](../../specs/README.md).

Commit dates are the author's local date; merge dates are GitHub's (UTC).
Last updated 2026-10-31. Each PR that adds to the milestone adds its entry
here.

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
Joshi and Xin Liang Chia. Spec revised; implementation not started.

### Unreleased (branch `milestone3-marcus-mixed-workload-spec`)

- **Changed:** spec 007 revised from Milestone 2's findings (marcus). The
  spec now names its owners and gains a Background section. New
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
