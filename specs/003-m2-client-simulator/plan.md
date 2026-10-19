# Implementation Plan: Client simulator and ShareGPT preprocessing

**Spec**: [spec.md](spec.md) | **Branch**: `client-simulator` (PR #6,
merged) |
**Date**: 2026-10-18

## Summary

PR #6 implemented User Stories 1 to 3 and was merged on 2026-10-19 after
Sahil Joshi's fix commit `e026ae82`. This plan records the design and the
state of each review item; the open ones become a follow-up PR.

## Technical Context

**Language/Version**: Python 3.12 (3.9+ syntax required by CI's yapf)
**Primary Dependencies**: `httpx` 0.27.2 (already pinned), standard library
**Storage**: `data/raw/` and `data/processed/` (gitignored), `results/`
(gitignored)
**Testing**: `tests/client_simulator/` (new), run by `project-ci`

## Constitution Check

- [x] I. Online mode via `vllm serve`; CPU only
- [x] II. No vLLM files changed
- [ ] III. Tests missing (FR-010)
- [x] IV. `cached_fraction` counts misses as 0 (FR-007, `e026ae82`)
- [ ] V. Bulk results removed, but `results/` is not gitignored (FR-012)
- [x] VI. Reviewed PR
- [ ] VII. AI usage not stated in PR #6's description

## Files (as merged)

| Path | Purpose |
|---|---|
| `client_simulator/schema.py` | `Message`, `Conversation`, `ReplayRequest`, `ScheduledConversation`, `request_id()` |
| `client_simulator/adapters/sharegpt.py` | Download and rebuild ShareGPT conversations |
| `client_simulator/process_raw_traces.py` | CLI: adapter → client JSONL + server response trace |
| `client_simulator/client_sim.py` | Scheduling, replay, result logging, CLI |
| `client_simulator/README.md` | Usage |

## Review items and their state

Line numbers refer to `main` at `cceecdf5`.

| # | Where | Change | State |
|---|---|---|---|
| 1 | `client_sim.py` (usage parsing) | Record a missing `cached_tokens` as `0`/`0.0`. A server started without `--enable-prompt-tokens-details` looks the same as one with zero hits, so the flag must stay on (the README's server command includes it) | Done (`e026ae82`) |
| 2 | `client_sim.py:175-180, 215-218` | On mismatch set `answer_match: false` and keep replaying the conversation, instead of marking it `failed` and breaking out | Open |
| 3 | `client_sim.py:342` | Exit 1 only on `failed`, not on `skipped_context` | Open |
| 4 | `process_raw_traces.py` | Replace the parenthesized `with (...)` that yapf cannot parse | Done (`e026ae82`) |
| 5 | `adapters/sharegpt.py` | Remove unused imports; wrap long lines | Done (`e026ae82`) |
| 6 | `results/`, `.gitignore` | Remove committed results; gitignore `/results/` | Files removed (`e026ae82`); gitignore open |
| 7 | `client_simulator/__init__.py`, `adapters/__init__.py` | Add, so tools treat them as packages | Open |
| 8 | `client_sim.py:1` | Add a module docstring (`--help` prints `__doc__`) | Open |
| 9 | `tests/client_simulator/` | Tests listed in FR-010, using a handcrafted fixture and `httpx.MockTransport` so no server is needed | Open |
| 10 | `adapters/agentbank.py` etc. | Placeholder adapters | Removed (`e026ae82`); `006` creates them |

## Verification

```bash
ruff check client_simulator tests/client_simulator
isort --check-only client_simulator tests/client_simulator
yapf --diff --recursive client_simulator tests/client_simulator
tools/run_project_tests.sh m2 -v      # tests carry pytestmark = pytest.mark.m2
# end to end, as in client_simulator/README.md, with --limit 20 in both modes
```

## Risks

| Risk | Mitigation |
|---|---|
| Full ShareGPT `json.load` needs a few GB of RAM | Acceptable on laptops; stream with `ijson` only if a member's machine cannot |
| The `/tokenize` calls double request count and add server load | Measure the effect on latency fields; cache counts per request id if needed |
