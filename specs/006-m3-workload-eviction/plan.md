# Implementation Plan: Eviction policies across workloads

**Spec**: [spec.md](spec.md) | **Branch**: named per `CONTRIBUTING.md` §4 |
**Date**: 2026-10-18

## Summary

Add three adapters to the client simulator's preprocessing (two text, one
synthesized from block hash ids), teach the client recorded timestamps and
token-id prompts, then reuse `005`'s sweep runner with a fixed capacity and
block size to compare LRU, FIFO and LFU per workload.

## Technical Context

**Language/Version**: Python 3.12 (3.9+ syntax)
**Primary Dependencies**: `huggingface_hub` (pinned) for dataset downloads;
the Bailian trace from its GitHub repository
**Testing**: `tests/client_simulator/test_adapters_*.py`,
`tests/client_simulator/test_timestamps_and_token_prompts.py`

## Constitution Check

- [x] I. Online replay through `vllm serve`
- [x] II. No vLLM files changed (client and analysis only)
- [x] III. Fixture-based adapter tests
- [x] IV. Metrics from `004`/`005` only
- [x] V. Datasets downloaded, never committed; manifests per run
- [x] VI, VII. Per PR

## Files

### New

| Path | Purpose |
|---|---|
| `client_simulator/adapters/agentbank.py` | AgentBank adapter |
| `client_simulator/adapters/ccbench.py` | CC-Bench-trajectories adapter |
| `client_simulator/adapters/qwenbailian.py` | Bailian adapter with hash-id → token synthesis |
| `analysis/run_m3_task1.py` | Workload × policy runs, summary, figures |
| `tests/client_simulator/fixtures/{agentbank,ccbench,bailian}_small.*` | Handcrafted raw-format fixtures (< 50 KB each) |

### Modified project files

| Path | Change |
|---|---|
| `client_simulator/schema.py` | Optional `timestamp_s` per conversation turn; a `TokenRequest` entity for token-id prompts |
| `client_simulator/client_sim.py` | Timestamp scheduling; `/v1/completions` path for token-id prompts |
| `client_simulator/process_raw_traces.py` | Register adapters; write `response_token_ids` for synthesized responses |

## Design

### Synthesizing Bailian prompts

- Vocabulary for synthesis: the tokenizer's ordinary tokens (ids below the
  first special token), excluding ids that decode to whitespace only.
- `tokens(hash_id) = Random(seed ^ hash_id).choices(vocab, k=block_size)`:
  deterministic and collision-free in practice for 16-token blocks.
- A request's prompt is the concatenation over its `hash_ids`; the final
  partial block is padded by `input_length - len(hash_ids) × block_size`
  tokens drawn from a request-specific seed, so it never matches anything,
  just as a unique suffix would not.
- Response: `output_length` ordinary tokens, written as
  `response_token_ids` in the server trace.

### Role mapping for agent trajectories

The system prompt or task description becomes `system`; the agent's actions
and reasoning become `assistant`; tool or environment observations become
`user`. Consecutive same-role messages are merged with a blank line, so
roles alternate as the client requires. Each assistant message is one replay
request with the full history, the same as multi-turn ShareGPT.

## Phases

1. **Phase 0 — Research**: download each dataset, record its exact format,
   size and licence in `client_simulator/README.md`; confirm the Bailian
   block size; choose per-workload sample sizes and arrival parameters.
2. **Phase 1**: adapters with fixtures and tests (one PR per adapter).
3. **Phase 2**: client timestamps and token prompts with tests.
4. **Phase 3**: `run_m3_task1.py`, runs, figures, write-up in
   `report/milestone3/task1.md`.

## Verification

```bash
tools/run_project_tests.sh m3 -v      # adapter tests carry pytestmark = pytest.mark.m3
python -m analysis.run_m3_task1 --limit 20 --out results/smoke-m3t1
```

## Risks

| Risk | Mitigation |
|---|---|
| A dataset's format differs from the assumption | Phase 0 confirms before any code; the spec's assumptions are updated in the same PR |
| CC-Bench trajectories are few but very long | Sample by request count, not conversation count; raise `--max-model-len` |
| Synthetic tokens tokenize differently on round trip | Token-id prompts bypass text entirely (`/v1/completions` with a token list) |
