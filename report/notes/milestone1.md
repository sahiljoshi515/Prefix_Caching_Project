# Milestone 1: changelog

What the group changed in Milestone 1, newest first, so the milestone can be
reviewed in one place. Each entry says what changed, who made the change,
and where (commit or PR). It comes from the git history and the pull
requests. How the simulator works and how it was tested are in
[`m1-simulator.md`](m1-simulator.md).

Commit dates are the author's local date; merge dates are GitHub's (UTC).
Last updated 2026-10-21. Each PR that changes Milestone 1's code or
documents adds its entry here.

## Simulator: replace model execution with trace replay

Built by sahiljoshi515. Independent verification:
[spec 002](../../specs/002-m1-simulator-verification/spec.md), not done yet.

### Unreleased (Task 2 branch)

- **Changed:** `report.md` moved to `report/notes/m1-simulator.md`; its
  "What was modified" table now gives 67 tests in 3 test files, matching its
  own section 4, instead of 59 (marcus, `295c1c8e`).

### 2026-10-20: PR #7 merged into `main` by sahiljoshi515

- **Changed:** the three test modules declare `pytestmark =
  pytest.mark.m1`, so CI runs them as the Milestone 1 suite
  (`make test-m1`) (marcus, `947d7dfd`, 2026-10-18).

### 2026-10-10: committed to `main` by sahiljoshi515

Committed directly, without a PR; the team's PR rules came later
(`CONTRIBUTING.md`, PR #7).

- **Added** (`072288d4`): the simulator.
  - `vllm/simulator.py`: `Simulator` replays each request's response from
    a JSONL trace (`VLLM_SIMULATOR_TRACE`), matched by request id, then
    ends it with EOS; a request with no trace entry gets filler tokens.
    `SimulatorExecutor` plugs it into the engine in place of the model,
    sizes the simulated KV cache the way vLLM's CPU backend does, records
    each request's cached prompt tokens, and rejects configurations it
    cannot emulate at startup.
  - `VLLM_USE_SIMULATOR` and `VLLM_SIMULATOR_TRACE` (`vllm/envs.py`); the
    executor is selected in `llm_engine.py` and `async_llm_engine.py` (4
    lines each); the CPU platform is pinned while simulating
    (`vllm/platforms/__init__.py`).
- **Added** (`4393eb5d`): 67 tests in three layers: the `Simulator` alone
  (28), an in-process engine (19), and a real `vllm serve` driven through
  the OpenAI client (20).
- **Added** (`c045b639`): dependency pins (`constraints-simulator.txt`),
  the project README (upstream's kept as `README_vllm.md`), an example
  trace and client (`examples/simulator/`), and the Milestone 1 report
  (`report.md`).
- **Fixed** (`acfeafee`): `vllm serve` on macOS. The API server bound its
  port twice, which macOS rejects, so the server never started; and
  shutdown called a port lookup that needs root on macOS, so every Ctrl-C
  ended in a traceback.

## Verification (spec 002)

### 2026-10-20: PR #7 merged into `main` by sahiljoshi515

- **Added:** spec 002, independent verification of the simulator by Triet
  Chau and Xin Liang Chia on their own machines (marcus, `51e37917`,
  2026-10-18). Its six tasks are open. The 67 tests run in CI
  (`project-ci`) on PRs into `main`, which covers part of it but not the
  independent reports.

## Open items

- Spec 002: both verifiers run its procedure and report (T001 to T006).
- Milestone 1's entry in [`ai-usage.md`](ai-usage.md). Section 7 of the
  report says an AI assistant (Claude) was used, and there is no PR
  description to copy it from, so its author writes it.
