# Implementation Plan: [FEATURE NAME]

**Spec**: [link to spec.md] | **Branch**: `[branch]` | **Date**: [YYYY-MM-DD]

## Summary

[The requirement in one paragraph, and the chosen approach]

## Technical Context

**Language/Version**: Python 3.12 (3.9+ compatible syntax)
**Primary Dependencies**: [pinned in constraints-simulator.txt]
**Storage**: [files written: JSONL, CSV, PNG, ...]
**Testing**: pytest, `VLLM_USE_SIMULATOR=1`, [suites]
**Target Platform**: CPU-only macOS / Linux laptop, GitHub Actions ubuntu
**Performance Goals**: [e.g. replay N conversations in M minutes]
**Constraints**: [memory, determinism, ...]

## Constitution Check

*Gate: must pass before Phase 1 and be re-checked after it.*

- [ ] I. v0 engine, online serving, CPU only, no weights
- [ ] II. Upstream edits listed below with reasons; defaults unchanged
- [ ] III. Tests named for every requirement; deterministic
- [ ] IV. Shared metric definitions and implementation reused
- [ ] V. Experiment configuration recorded; no data committed
- [ ] VI. Reviewed PR, merge commit
- [ ] VII. AI usage recorded

## Files

### New

| Path | Purpose |
|---|---|
| [path] | [purpose] |

### Modified upstream vLLM files

| Path | Change | Reason |
|---|---|---|
| [path] | [what] | [why it cannot live in a new module] |

## Design

[Data flow, interfaces, key decisions. For each decision: the options
considered and why this one was chosen.]

## Phases

1. **Phase 0 — Research**: [open questions to answer first]
2. **Phase 1 — Core**: [...]
3. **Phase 2 — Experiments / polish**: [...]

## Verification

[Exact commands that demonstrate each success criterion]

## Risks

| Risk | Mitigation |
|---|---|
| [risk] | [mitigation] |
