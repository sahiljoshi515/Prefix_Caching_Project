# Implementation Plan: Replay ShareGPT and analyze prefix sharing

**Spec**: [spec.md](spec.md) | **Branch**:
`milestone2-marcus-prefix-sharing-analysis` (from `milestone-2`) |
**Date**: 2026-10-18, revised 2026-10-20 after reading the code at `72cfea62`,
and 2026-10-21 with the Phase 0 results and the single-PR decision

## Summary

Add an opt-in cache event recorder to vLLM's prefix-caching allocator, then a
vLLM-free `analysis/` package that turns event logs and client results into
the charter's metrics and figures, and one command that runs the single-turn
and multi-turn experiments end to end. The recorder and `analysis/` are the
shared measurement layer for `005`, `006` and `007`.

## Technical Context

**Language/Version**: Python 3.12 (3.9+ syntax)
**Primary Dependencies**: standard library for the recorder; `numpy`
(1.26.4, pinned) and `matplotlib` (new, `3.9.2`, pinned in
`constraints-simulator.txt` and installed by `make setup`) for analysis
**Storage**: `results/<experiment>/<setting>/` (gitignored: `/results/` is
in `.gitignore`); figures in `results/<experiment>/figures/`, and the ones the
report uses are committed under `report/figures/m2-task2/`
**Testing**: `tests/simulator/test_cache_events.py` (engine-level, gpt2
config only, like `tests/simulator/test_engine.py`),
`tests/analysis/test_metrics.py` (pure functions), and
`tests/analysis/test_run_m2_task2.py` and `test_server.py` (the experiment
command and launcher without a server). All carry
`pytestmark = pytest.mark.m2`.
**Performance Goals**: recorder-off overhead < 2% (SC-001); measured in
[Overhead](#overhead-sc-001-t009)

## Findings from the code (2026-10-20)

These change details of the 2026-10-18 draft:

1. **Decode inserts.** Blocks filled by generated tokens are promoted inside
   `BlockSpaceManager.append_slots`. Wrapping only `allocate` would log them
   with `req: null`, which breaks intra/cross attribution in multi-turn runs,
   where turn k+1 reuses turn k's answer. So `append_slots` and `free` are
   wrapped too.
2. **No request id on `Sequence`.** `append_slots(seq)` and `free(seq)` get a
   sequence only. The recorder keeps `seq_id -> request_id`, filled in
   `allocate` and dropped in `free`.
3. **Miss path inserts at allocation.** On a miss, `allocate_immutable_block`
   allocates a mutable block and appends the tokens, which calls
   `promote_to_immutable_block` at once. Expected order: `miss`, optional
   `evict` (block reuse), `insert`. The block is only marked computed after
   the step (`mark_blocks_as_computed`), which is why a second request in the
   same step can `hit` it while vLLM reports no cached tokens for it.
4. **`cached_tokens` source.** The simulator sets it from
   `len(computed_block_nums) × block_size` (`vllm/simulator.py:197-201`),
   which comes from `ComputedBlocksTracker.get_num_cached_tokens`. That value
   is fixed the first time a waiting request is considered
   (`prefix_caching_block.py:979-1010`). SC-004 is now a reported
   difference, not an assertion.
5. **Hit-rate metric is exact.** `CacheMetricData` averages equal-sized
   chunks of queries, so `gpu_prefix_cache_hit_rate` equals
   `hits / lookups` up to float rounding. SC-003 can be checked to 4 decimal
   places.
6. **Shutdown.** `vllm serve` runs the engine in a `spawn` child.
   On shutdown the API server sends SIGTERM, which the engine turns into
   `KeyboardInterrupt`, so `atexit` runs; after 4 s it is killed. The
   recorder therefore also flushes every second and writes an `end` line on
   close.
7. **Shared first block.** Qwen2.5's template adds a 24-token default system
   prompt, so with block size 16 exactly one block is shared by every
   conversation (three with block size 8). Expect about one cross-conversation
   hit per request from this alone.
8. **Prefix caching is off by default in v0** (`arg_utils.py:208-209`), so
   every run passes `--enable-prefix-caching`. Chunked prefill stays off
   (default for max-model-len ≤ 32K).

## Constitution Check

- [x] I. Recorder runs inside the real v0 allocator; experiments use
      `vllm serve`
- [x] II. Upstream edits listed below; all are no-ops when
      `VLLM_CACHE_EVENT_LOG` is unset
- [x] III. Hand-derived test cases (T005, T010); new modules marked `m2`
- [x] IV. This spec owns the shared metric definitions; `analysis/metrics.py`
      is the single implementation
- [x] V. Manifest per run; data and bulk results gitignored
- [x] VI. One PR for all of Task 2 from this branch (team decision,
      2026-10-20), merged into `milestone-2` with a merge commit
- [x] VII. AI usage noted per PR

## Files

### New

| Path | Purpose |
|---|---|
| `vllm/core/block/cache_events.py` | `CacheEventRecorder` (buffered JSONL writer, count/time flush, `end` line on close, sequence-to-request map), `recorder_from_env()` (one recorder per path per process), `close_all()`, and the `attributed(op)` decorator for block-manager methods |
| `analysis/__init__.py` | Package marker |
| `analysis/events.py` | Load an event log (plain or `.gz`) into typed rows; check header and `end` line |
| `analysis/metrics.py` | The metric definitions of the spec, as pure functions, plus hit-source attribution |
| `analysis/plots.py` | CDF, by-turn and over-time plots (matplotlib, one style) |
| `analysis/server.py` | Launch and stop `vllm serve` with given args and env; wait for `/health`; read the "Simulated KV cache: N blocks" line. Reused by `005` to `007` |
| `analysis/run_m2_task2.py` | Runs both settings, writes run folders, CSVs, figures and the SC-003/SC-004 checks |
| `tests/simulator/test_cache_events.py` | Engine-level recorder tests |
| `tests/analysis/__init__.py`, `tests/analysis/test_metrics.py` | Hand-derived metric tests |
| `tests/analysis/test_run_m2_task2.py` | Sample writing, CSVs and self-checks, figures, refusal to overwrite, all without a server |
| `tests/analysis/test_server.py` | Launcher: busy port refused, binds 127.0.0.1, reads the KV-cache size |

### Modified upstream vLLM files

| Path | Change | Reason |
|---|---|---|
| `vllm/envs.py` | Declare `VLLM_CACHE_EVENT_LOG` next to `VLLM_SIMULATOR_TRACE` (:76, :480) | Same pattern as the simulator |
| `vllm/core/block/prefix_caching_block.py` | Optional `event_recorder` argument stored as `self._events` in `__init__` (:64); one `if self._events is not None:` call at: `hit`/`miss` in `allocate_immutable_block` (:170-176); `revive` in `_incr_refcount_cached_block` (:236-238); `free` in `_decr_refcount_cached_block` (:261); `evict` in `_maybe_allocate_evicted_block_id` (:319); `insert` and `dedupe` in `promote_to_immutable_block` (:458-469) | The events exist only inside the allocator |
| `vllm/core/block/cpu_gpu_block_allocator.py` | In `create` (:79), attach the recorder to the GPU `PrefixCachingBlockAllocator` only | Keeps swap-space blocks out of the log |
| `vllm/core/block_manager.py` | `self._events` from `recorder_from_env()` in `__init__`; `@cache_events.attributed(op)` on `allocate` (:164), `append_slots` (:234), `free` (:252), `swap_in` (:359) and `swap_out` (:412). The decorator calls the method directly when `self._events` is None | Request attribution without new allocator arguments or re-indented method bodies (FR-004) |

### Other repository files

| Path | Change |
|---|---|
| `constraints-simulator.txt`, `Makefile` (`setup`) | Pin and install `matplotlib==3.9.2` |
| `.gitignore` | Add `/results/` (also spec 003 follow-up 6) |
| `README.md` | Task 2 commands |

## Design

### Event recorder

- `recorder_from_env()` returns `None` when the variable is unset, so each
  hook costs one attribute check. Otherwise it returns the process's
  recorder for that path (tests use a fresh `tmp_path` per engine).
- Lines are buffered and written every 4,096 events or 1 second, at
  `atexit`, and on `close()`. The first line is a header (`time.time()`,
  `time.monotonic()`, pid, block size); the last is `{"ev": "end"}`.
- Content hashes are Python `hash()` values: stable within one process,
  which is all the analysis needs. Runs are compared by distribution, never
  by hash.
- The current request and op are module-level state, not `contextvars`: the
  v0 engine core runs in one thread and allocation is synchronous.
- `free` events come from `_decr_refcount_cached_block` only when the
  refcount reaches 0 (the block enters the evictor). A decrement that leaves
  other users is not an event.

Options considered:

- **Scrape `/metrics` only.** Just a cumulative hit rate, no per-block
  data. Rejected.
- **Pass request ids through allocator calls.** Changes signatures used by
  `tests/core`. Rejected in favour of the context.
- **Set the context in the scheduler instead of the block manager.** It
  has the `SequenceGroup` everywhere, but needs edits at four call sites in
  a much larger file. Rejected.

### Attribution of sharing

The engine request id is `chatcmpl-<conversation_id>:<turn>` (the client
sends `<conversation_id>:<turn>` as `X-Request-Id`). For each `hit`, the
analysis finds the `insert` of that hash in the current residency. The hit is
**intra-conversation** if the inserting request has the same conversation,
**cross-conversation** otherwise, and **decode-written** if that insert's op
was `append_slots`.

### Experiment command

```
python -m analysis.run_m2_task2 \
  --conversations data/processed/sharegpt.conversations.jsonl \
  --responses data/processed/sharegpt.responses.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --limit 1000 --seed 0 \
  --conversation-rate 1 --think-time 5 --out results/m2-task2
```

For each setting in (single, multi): start a fresh server (empty cache) with
`VLLM_CACHE_EVENT_LOG=<run>/events.jsonl`, run the client
(`python -m client_simulator.client_sim ... --capture-metrics`; exit code 1
is expected when some prompts exceed the context and is read from its
`run.json` instead), stop the server, compute metrics, then plot both
settings together. Refuses to overwrite an existing run folder.

## Phases

0. **Prerequisites**
   - CI: every workflow filters on `branches: [main]`, so PRs into
     `milestone-2` currently get **no checks**. Fix in a separate small PR
     (`ci-marcus-milestone-branches`): add `milestone-*` to the filters, and
     make `BASE` in the Makefile and the `check_pr.py` command in
     CONTRIBUTING.md follow the default branch.
   - Data: preprocess ShareGPT locally (about 670 MB download, see
     `client_simulator/README.md`).
   - Research: replay 100 conversations per mode; fix the sample size, rate
     and think time so one setting takes 30 minutes or less; record the KV
     block count and whether the sample evicts at all.
1. **Phase 1 — Recorder**: recorder, hooks, engine tests, overhead check.
2. **Phase 2 — Analysis library**: `analysis/events.py`, `metrics.py`, unit
   tests, `matplotlib` pin, `/results/` ignore.
3. **Phase 3 — Experiments**: `server.py`, `plots.py`, `run_m2_task2.py`,
   report figures, written analysis in `report/notes/m2-task2.md`.

All three phases go into one PR (Constitution Check, VI); the phases are the
order of work and of commits.

### Phase 0 results (2026-10-20)

100 conversations per mode at the defaults (seed 0, 1 conversation/s, think
time 5 s), recorder on:

| | Single-turn | Multi-turn |
|---|---|---|
| Replay time | 121 s | 260 s |
| Requests completed | 100 | 507 |
| Skipped (prompt over 8,192 tokens) | 0 | 6 |
| Events logged | 5,280 | 194,259 |
| Event log | 0.7 MB | 26.8 MB (138 bytes per event) |
| Distinct blocks inserted | 2,191 (22 per conversation) | 11,318 (113 per conversation) |
| Evictions | 0 | 0 |

- **KV cache** at the defaults: 21,845 blocks of 16 tokens
  (`VLLM_CPU_KVCACHE_SPACE` 4 GiB, 12,288 bytes per token for Qwen2.5-0.5B).
- **Sample kept**: the first 1,000 conversations, 1/s, think time 5 s. They
  hold 6,833 turns; 306 conversations have one turn and the longest has 300,
  which bounds the multi-turn run at about 300 × 5 s = 25 minutes of think
  time.
- **Eviction (T004)**: cached blocks stay until evicted, so a run evicts once
  it has inserted more distinct blocks than the cache holds. Single-turn
  sends one request per conversation, so 1,000 conversations insert about
  10 × 2,191 ≈ 21,900 blocks and reach the 21,845 limit only near the end.
  Multi-turn inserted 11,318 blocks for the first 100 conversations (592
  turns, 513 of them sent before over-long turns ended their
  conversations), and the sample has 11.5 times as many turns, so it evicts
  well before the end. The full run's `evict` counts give the exact numbers.

### Overhead (SC-001, T009)

100 conversations multi-turn (507 requests), `--conversation-rate inf
--think-time 0`, three runs each, replay time in seconds:

| Code | Runs | Mean | vs `milestone-2` |
|---|---|---|---|
| `milestone-2` (`72cfea62`) | 4.93, 5.06, 5.76 | 5.25 | |
| This branch, recorder off | 5.08, 5.00, 5.39 | 5.16 | −1.8% |
| This branch, recorder on | 5.46, 5.48, 5.64 | 5.53 | +5.3% |

With the recorder off, the difference is inside the run-to-run spread
(up to 0.8 s between runs of the same code), so SC-001 holds. With it on,
replay is about 5% slower; experiments always run with it on, so their
timings are comparable with each other.

## Verification

```bash
make test-m2 ARGS=-v      # recorder and metric tests
make test-m1 test-core    # unchanged behaviour with the recorder off
python -m analysis.run_m2_task2 --limit 20 --conversation-rate inf \
  --think-time 0 --out results/smoke                         # < 2 min
```

SC-003 and SC-004 are computed at the end of `run_m2_task2` and written to
the manifest.

## Risks

| Risk | Mitigation |
|---|---|
| Default cache (21,845 blocks at 4 GiB for Qwen2.5-0.5B) never evicts for the sample, so reuse intervals reflect only the workload | Phase 0: multi-turn evicts at 1,000 conversations, single-turn barely; report the `evict` counts; Task 3 studies cache pressure |
| Event logs grow large (one line per block lookup) | Measured 138 bytes per event (Phase 0); the full run's size goes into the note; gzip after the run if needed |
| `hash()` randomization differs between runs | Never compare hashes across runs |
| Hook placement drifts from line numbers | Hooks are named by method; tests fail if an event kind goes missing |
| Engine killed before the final flush | 1-second flush; missing `end` line is reported |
| PRs into `milestone-2` run no CI | Phase 0 CI fix first |
