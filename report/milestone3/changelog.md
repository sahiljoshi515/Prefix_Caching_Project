# Milestone 3: changelog

What the group changed in Milestone 3, task by task and newest first, so the
milestone can be reviewed in one place. Each entry says what changed, who
made the change, and where (commit or PR). Results are in the per-task
notes, and each task's owner and status are in the
[spec index](../../specs/README.md).

Commit dates are the author's local date; merge dates are GitHub's (UTC).
Last updated 2026-11-09. Each PR that adds to the milestone adds its entry
here.

## Task 1: eviction policies across workloads

[Spec 006](../../specs/006-m3-workload-eviction/spec.md).

### Unreleased (`milestone3-jasmine-agentbank-ccbench`)

- **Added:** AgentBank and CC-Bench adapters with fixtures and tests
  (`client_simulator/adapters/agentbank.py`, `ccbench.py`), their configs
  (`examples/agentbank_1000.json`, `examples/ccbench_full.json`) and
  `pyarrow==18.0.0` for the parquet files (imleznupar, `c9aa4e13`).
- **Fixed:** the client retries `/tokenize` once on a dropped connection,
  the keep-alive race that failed one request in three of the runs
  (spec 003, T015) (imleznupar, `1d785f91`).
- **Added:** the AgentBank (1,000 trajectories) and CC-Bench (all 370
  sessions) comparisons, their figures, and the best policy per workload in
  [`task1-workload-eviction.md`](task1-workload-eviction.md): LRU for
  AgentBank (89.47% usable reuse), LFU for Bailian, and no policy for
  CC-Bench, where LFU is only slightly ahead because the cache thrashes
  under every policy at 1 conversation/s (imleznupar).
- **Fixed (review):** the CC-Bench adapter follows each session's
  `uuid`/`parentUuid` tree, drops list-form local commands and synthetic
  API errors, and does not replay a resumed session's copied history
  (copies with the same uuids are skipped; copies with new uuids are
  matched by text and extend their conversation); AgentBank and CC-Bench
  tests cover these cases, and the client tests `first_turn`. CC-Bench was
  rerun: no policy helps at this load. The note scopes both workloads'
  rankings to this load against Task 2's runs and explains LFU's AgentBank
  result by blocks of finished conversations filling the cache
  (imleznupar, `cbd9a3ba`, `c7830bd8`, `cce52746`).

### Unreleased (Milestone 2, Task 2 branch)

- **Added:** this changelog and a placeholder note,
  [`task1-workload-eviction.md`](task1-workload-eviction.md) (marcus,
  `b79d7be3`).

### 2026-10-20: PR #7 merged into `main` by sahiljoshi515

- **Added:** spec 006, a draft (ChiaXinLiang, commits by marcus,
  `51e37917`, 2026-10-18).

## Task 2: a workload-aware policy for mixed workloads

[Spec 007](../../specs/007-m3-mixed-workload-policy/spec.md). Not started.

### Unreleased (Milestone 2, Task 2 branch)

- **Added:** a placeholder note,
  [`task2-mixed-workload-policy.md`](task2-mixed-workload-policy.md)
  (marcus, `b79d7be3`).

### 2026-10-20: PR #7 merged into `main` by sahiljoshi515

- **Added:** spec 007, a draft (ChiaXinLiang, commits by marcus,
  `51e37917`, 2026-10-18).

## Open items

- Assign the owners of specs 006 and 007 at the Milestone 3 kickoff.
- Both tasks build on Milestone 2: the event log and `analysis/` (Task 2)
  and the eviction policies (Task 3).
