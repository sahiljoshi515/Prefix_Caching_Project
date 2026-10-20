# Implementation Plan: Replay ShareGPT and analyze prefix sharing

**Spec**: [spec.md](spec.md) | **Branch**: named per `CONTRIBUTING.md` §4 |
**Date**: 2026-10-18

## Summary

Add an opt-in cache event recorder to vLLM's prefix-caching allocator, then a
vLLM-free `analysis/` package that turns event logs and client results into
the charter's metrics and figures, and one command that runs the single-turn
and multi-turn experiments end to end. The recorder and `analysis/` are the
shared measurement layer for `005`, `006` and `007`.

## Technical Context

**Language/Version**: Python 3.12 (3.9+ syntax)
**Primary Dependencies**: standard library for the recorder; `numpy` (pinned)
and `matplotlib` (new; to be pinned in `constraints-simulator.txt`) for
analysis
**Storage**: `results/<experiment>/<setting>/` (gitignored); figures in
`figures/` (only figures used in the report are committed)
**Testing**: `tests/simulator/test_cache_events.py` (engine-level),
`tests/analysis/` (pure functions)
**Performance Goals**: recorder overhead < 2% of replay duration (SC-001)

## Constitution Check

- [x] I. Recorder runs inside the real v0 allocator; experiments use
      `vllm serve`
- [x] II. Upstream edits listed below; all are no-ops when
      `VLLM_CACHE_EVENT_LOG` is unset
- [x] III. Hand-derived test cases (tasks T004, T008)
- [x] IV. This spec owns the shared metric definitions; `analysis/metrics.py`
      is the single implementation
- [x] V. Manifest per run; data and bulk results gitignored
- [x] VI. One PR per phase (recorder, analysis, experiments)
- [x] VII. AI usage noted per PR

## Files

### New

| Path | Purpose |
|---|---|
| `vllm/core/block/cache_events.py` | `CacheEventRecorder` (buffered JSONL writer, flush on exit), `current_request(request_id)` context manager, `recorder_from_env()` |
| `analysis/__init__.py` | Package marker |
| `analysis/events.py` | Load an event log into typed rows |
| `analysis/metrics.py` | The metric definitions of the spec, as pure functions |
| `analysis/plots.py` | CDF, by-turn and over-time plots (matplotlib, consistent style) |
| `analysis/server.py` | Launch and stop `vllm serve` with given args and env; wait for `/health`; capture the "Simulated KV cache: N blocks" line. Reused by `005` to `007` |
| `analysis/run_m2_task2.py` | Runs both settings and writes run folders, CSVs and figures |
| `tests/simulator/test_cache_events.py` | Engine-level recorder tests |
| `tests/analysis/test_metrics.py` | Hand-derived metric tests |

### Modified upstream vLLM files

| Path | Change | Reason |
|---|---|---|
| `vllm/envs.py` | Declare `VLLM_CACHE_EVENT_LOG` | Same pattern as `VLLM_SIMULATOR_TRACE` |
| `vllm/core/block/prefix_caching_block.py` | `self._events = None` in `__init__`; one `if self._events:` call at: lookup hit/miss in `allocate_immutable_block` (:170-176), insert and dedupe in `promote_to_immutable_block` (:456-474), free in `_decr_refcount_cached_block` (:259-262), revive in `_incr_refcount_cached_block` (:235-238), evict in `_maybe_allocate_evicted_block_id` (:319-327) | The events exist only inside the allocator |
| `vllm/core/block/cpu_gpu_block_allocator.py` | In `create` (:78-88), attach the recorder to the GPU allocator only | Keeps swap-space blocks out of the log |
| `vllm/core/block_manager.py` | Wrap the bodies of `allocate` (:164) and `swap_in` with `current_request(seq_group.request_id)` | Request attribution without new allocator arguments (FR-004) |

## Design

### Event recorder

- One module-level recorder per engine process, created from the variable
  when the GPU allocator is built. `None` when unset, so each hook costs one
  attribute check.
- Lines are buffered in a list and written every 4,096 events and at
  `atexit`. The first line is a header with `time.time()` and
  `time.monotonic()`, so a log can be aligned with the client's UTC start
  time.
- Content hashes are Python `hash()` values. They are stable within one
  process, which is all the analysis needs. Across runs, blocks are compared
  by distribution, never by hash.
- `current_request` uses a module-level variable, not `contextvars`: the v0
  engine core runs in one thread, and allocation is synchronous.

Options considered:

- **Scrape `/metrics` only.** It gives just a cumulative hit rate, with no
  per-block data. Rejected.
- **Pass request ids through allocator calls.** It would change four
  signatures used by `tests/core`. Rejected in favour of the context.
- **A step counter instead of time.** The spec asks for time gaps. The
  "requests in between" variant covers rate-independent analysis.

### Attribution of sharing

The engine request id is `chatcmpl-<conversation_id>:<turn>`. For each `hit`,
the analysis finds the `insert` of that hash (same residency). It labels the
hit **intra-conversation** if the inserting request belongs to the same
conversation, and **cross-conversation** otherwise. Hits on the first block of
every prompt, the chat template's system prompt, show up as
cross-conversation, which is the expected signature.

### Experiment command

```
python -m analysis.run_m2_task2 \
  --conversations data/processed/sharegpt.conversations.jsonl \
  --responses data/processed/sharegpt.responses.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --limit 1000 --seed 0 \
  --conversation-rate 1 --think-time 5 --out results/m2-task2
```

For each setting in (single, multi): start the server with a fresh process
(empty cache) and `VLLM_CACHE_EVENT_LOG`, run the client, snapshot
`/metrics`, stop the server, compute metrics, then plot both settings
together.

## Phases

1. **Phase 0 — Research**: measure replay throughput on a laptop for 100
   conversations in each mode; fix the sample size and rate so that one
   setting takes 30 minutes or less. Check how often the default cache (about
   20k blocks of 16 tokens at 4 GiB for Qwen2.5-0.5B; read the exact count
   from the server log) evicts at all for the chosen sample.
2. **Phase 1 — Recorder** (PR 1): recorder, hooks, engine tests, overhead
   check.
3. **Phase 2 — Analysis library** (PR 2): `analysis/events.py`,
   `metrics.py`, `plots.py`, unit tests.
4. **Phase 3 — Experiments** (PR 3): `server.py`, `run_m2_task2.py`, figures
   for the report, written analysis in `report/notes/m2-task2.md`.

## Verification

```bash
tools/run_project_tests.sh m2 -v      # new test modules carry pytestmark = pytest.mark.m2
python -m pytest tests/core --ignore=tests/core/block/e2e --ignore=tests/core/test_num_computed_tokens_update.py
python -m analysis.run_m2_task2 --limit 20 --out results/smoke   # < 2 min
```

SC-003 and SC-004 are checked automatically at the end of
`run_m2_task2` and printed in the manifest.

## Risks

| Risk | Mitigation |
|---|---|
| Default cache never evicts for the sample, so reuse intervals reflect only workload | Report it explicitly; Task 3 studies pressure |
| Event logs grow large (one line per block lookup) | Gzip on close; about 100 bytes per event; measure in Phase 0 |
| `hash()` randomization (PYTHONHASHSEED) differs between runs | Never compare hashes across runs (see Design) |
| Upstream hook placement drifts from line numbers | Hooks are named by method; tests fail if an event kind goes missing |
