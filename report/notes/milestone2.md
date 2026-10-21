# Milestone 2: record of work

A record of what the group has done in Milestone 2, so anyone can review
the milestone's progress in one place. It comes from the git history and the
pull requests. Results and their interpretation are in the per-task notes;
each task's owner and status are in the [spec index](../../specs/README.md).

Last updated 2026-10-21. Add a row in the PR that adds to the milestone.

## Tasks

| Task | Spec | Work so far |
|---|---|---|
| 1. Client simulator and ShareGPT preprocessing | [`003`](../../specs/003-m2-client-simulator/spec.md) | PR #6, merged 2026-10-19; follow-ups open (spec 003, Phase 5) |
| 2. Replay ShareGPT and analyze prefix sharing | [`004`](../../specs/004-m2-prefix-sharing-analysis/spec.md) | Branch `milestone2-marcus-prefix-sharing-analysis`: code, tests and docs done, the 1,000-conversation run under way; PR opens after PR #8 |
| 3. Cache size, block size, eviction policy | [`005`](../../specs/005-m2-cache-tuning/spec.md) | Spec drafted; no branch or PR yet |

## Contributions

By date. GitHub account first, and the git commit author where it differs.
Before this milestone, sahiljoshi515 built the Milestone 1 simulator (four
commits on 2026-10-10, see [`m1-simulator.md`](m1-simulator.md)).

| Date | Who | What | Where |
|---|---|---|---|
| 2026-10-18 | trietmchau | Task 1: ShareGPT download and conversation rebuilding, the preprocessing command, and the replay client (Poisson arrivals, single and multi-turn, chat template through the server, per-request results) | `ee944717`, PR #6 |
| 2026-10-18 | sahiljoshi515 | Review fixes in PR #6: lint, a missing `cached_tokens` recorded as 0, committed results removed | `e026ae82`, PR #6 |
| 2026-10-18 | ChiaXinLiang (commits by marcus) | Team rules (`CONTRIBUTING.md`), the spec-kit constitution and specs 001 to 007, project CI (tests per milestone, the PR-requirements check), the Makefile, the README for Milestone 2 | PR #7 |
| 2026-10-19 | sahiljoshi515 | Merged PR #6 into `main` | |
| 2026-10-20 | sahiljoshi515 | Merged PR #7; made `milestone-2` the default branch | |
| 2026-10-20 | ChiaXinLiang (commits by marcus) | CI checks for PRs into the milestone branches | PR #8, open |
| 2026-10-20 to 21 | marcus | Task 2: the cache event log in vLLM, the `analysis/` metrics, server launcher, experiment command and figures, 39 tests, the Phase 0 measurements; the `report/` folder and the documentation rules | Branch `milestone2-marcus-prefix-sharing-analysis` |

## Open items

- PR #8 needs review; until it merges, PRs into `milestone-2` run no checks.
- Task 1 follow-ups (spec 003, Phase 5): keep replaying after an answer
  mismatch, exit non-zero only on failures, unit tests, and PR #6's entry in
  [`ai-usage.md`](ai-usage.md).
- Task 2: the full run, its note and figures, then the PR.
- Task 3 has not started; it builds on Task 2's event log and `analysis/`.
- The branch protection ruleset targets no branch, so nothing stops a direct
  push to `main` (spec 001, T014; repository settings, owner only).
