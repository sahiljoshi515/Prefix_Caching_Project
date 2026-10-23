# Implementation Plan: Tune cache parameters and eviction policies

**Spec**: [spec.md](spec.md) | **Branch**:
`milestone2-jasmine-eviction-policies` (Phase 1, from `milestone-2`) |
**Date**: 2026-10-18, revised 2026-10-23 with Phase 1 as built

## Summary

Make the prefix-cache eviction policy selectable through an environment
variable and add FIFO and LFU evictors next to vLLM's LRU, with a one-line
hit hook in the allocator. Then build a resumable sweep over capacity, block
size and policy on top of `004`'s launcher, recorder and metrics, and plot
the comparisons.

## Technical Context

**Language/Version**: Python 3.12 (3.9+ syntax)
**Primary Dependencies**: `heapq`, `004`'s `analysis/` package
**Testing**: `tests/eviction/test_eviction_policies.py` (pure evictor unit
tests), `tests/simulator/test_eviction_policy_env.py` (engine wiring),
`tests/analysis/test_sweep.py` (grid, resume)
**Performance Goals**: `evict()` O(log n) amortized (FR-005)

## Constitution Check

- [x] I. Evictors run inside the real v0 allocator; runs use `vllm serve`
- [x] II. Upstream edits below; LRU path untouched when the variable is unset
- [x] III. Deterministic evictor tests with hand-derived victims
- [x] IV. Reuses `004` metrics; defines only eviction count and rate
- [x] V. Sweep writes a manifest per configuration
- [x] VI. PRs per phase
- [x] VII. AI usage per PR

## Files

### New

| Path | Purpose |
|---|---|
| `vllm/core/eviction_policies.py` | `FIFOEvictor`, `LFUEvictor`, and `eviction_policy_from_env()` |
| `analysis/sweep.py` | Grid definition, resumable runner (one server per configuration) |
| `analysis/run_m2_task3.py` | The Task 3 grid, summary CSV, figures |
| `tests/eviction/test_eviction_policies.py` | Evictor unit tests (a project folder, so `tools/run_project_tests.sh` finds it; marked `m2`) |
| `tests/eviction/__init__.py` | Package marker, as in `tests/analysis/` |
| `tests/simulator/test_eviction_policy_env.py` | The variable selects the GPU evictor only; bad values fail; the hit hook reaches LFU; FIFO evicts a different block than LRU and LFU in a 6-block cache |
| `tests/analysis/test_sweep.py` | Grid expansion, token-capacity conversion, resume |

### Modified upstream vLLM files

| Path | Change | Reason |
|---|---|---|
| `vllm/envs.py` | Declare `VLLM_EVICTION_POLICY` | Same pattern as the simulator variables |
| `vllm/core/evictor.py` | Add `FIFO`, `LFU` to `EvictionPolicy`; `make_evictor` dispatches to `eviction_policies`; add `Evictor.record_hit` with an empty default | Natural registry; the default keeps LRU unchanged |
| `vllm/core/block/prefix_caching_block.py` | `allocate_immutable_block` hit branch: `self.evictor.record_hit(cached_block_id)` | LFU needs hits that happen while the block is in use, which the evictor otherwise never sees |
| `vllm/core/block/cpu_gpu_block_allocator.py` | `create`: pass `eviction_policy=eviction_policies.eviction_policy_from_env()` to the GPU `PrefixCachingBlockAllocator` | FR-001 names the GPU allocator; the swap-space allocator stays on LRU. Same place `004` attaches its recorder |

## Design

### Why an environment variable, not a CLI flag

A `--eviction-policy` flag would need edits in `arg_utils.py`, `CacheConfig`,
`Scheduler`, `SelfAttnBlockSpaceManager` and `CpuGpuBlockAllocator.create`
(five upstream files) to reach the allocator. The variable needs one read at
allocator construction. This follows the precedent of `VLLM_USE_SIMULATOR`,
and the sweep records it in the manifest. The cost is that it does not show
up in vLLM's `cache_config_info` metric, so the startup log line (FR-001) and
the manifest carry it instead.

### Changes from the 2026-10-18 draft (Phase 1)

1. **The variable is read in `CpuGpuBlockAllocator.create`, not in
   `PrefixCachingBlockAllocator.__init__`.** The allocator class is also
   used for swap-space blocks and cannot tell which one it is, so reading
   the variable in `__init__` would change the CPU allocator too. `create`
   builds both, so it passes the policy to the GPU one only, next to `004`'s
   recorder. `prefix_caching_block.py` then needs only the hit hook.
2. **`eviction_policy_from_env()` returns an `EvictionPolicy`, not an
   evictor,** and the allocator builds the evictor with vLLM's own
   `make_evictor`. It logs the policy (FR-001) and raises on unknown values
   (US1, scenario 3). Unset or empty means LRU.
3. **FIFO's insertion order is a time, not a counter:** the block's
   `last_accessed` when it first entered the free pool. A request frees all
   its blocks with the same `last_accessed`, so a counter would order them by
   the order `free` visits them and evict the root of a prefix chain first,
   which makes every block after it unreachable. With a time, those blocks
   tie and FR-003's tie-break (deeper block first) applies, as for LRU.
4. **Counters are keyed by physical block id, not content hash.**
   `record_hit` receives a block id (FR-002), and a block id keeps one
   content from the time it is cached until it is evicted, when the counters
   are cleared. That is the per-hash count FR-003 defines.

### Evictor data structures

- **FIFO**: a dict `block id → insertion time` that survives
  `remove`/`add` (reuse), cleared on `evict`. A heap keyed by
  `(insertion_time, -num_hashed_tokens, block_id)` with lazy deletion: an
  entry is live only if the block is in the free table with that same key.
- **LFU**: `hits[hash]` incremented by `record_hit`; heap keyed by
  `(hits, last_accessed, -num_hashed_tokens, block_id)`. A changed key
  pushes a new entry (no in-place mutation), and stale entries are skipped on
  pop. The heap is rebuilt when it holds more than 2× as many entries as
  there are free blocks (and at least 64), the same
  idea as `LRUEvictor._cleanup` but without its orphaning bug (FR-004).
- `record_hit` on a block that is not in the free pool (refcount > 0) only
  updates the counter. The count takes effect when the block is freed.

### Sweep

- Grid axes: policy ∈ {lru, fifo, lfu}; block size ∈ {8, 16, 32, 64, 128}
  (at fixed token capacity); capacity ∈ 5 to 7 points spanning the working
  set (from the `004` infinite-cache run).
- Each configuration starts a fresh server (`analysis.server.launch`) with
  `--block-size`, `--num-gpu-blocks-override`, `VLLM_EVICTION_POLICY` and
  `VLLM_CACHE_EVENT_LOG`, replays the sample, and computes metrics into
  `results/m2-task3/<policy>-bs<B>-cap<T>/`.
- `summary.csv` holds one row per configuration; figures are drawn from it.

## Phases

1. **Phase 0**: from the `004` infinite-cache run, compute the working set in
   tokens and choose the capacity points.
2. **Phase 1** (PR 1): evictors, hit hook, variable, tests.
3. **Phase 2** (PR 2): sweep runner and tests.
4. **Phase 3** (PR 3): run the grid, figures, write-up in
   `report/milestone2/task3-cache-tuning.md`.

## Verification

```bash
tools/run_project_tests.sh m2 -v      # test modules carry pytestmark = pytest.mark.m2
python -m pytest tests/core --ignore=tests/core/block/e2e --ignore=tests/core/test_num_computed_tokens_update.py
python -m analysis.run_m2_task3 --limit 20 --grid smoke --out results/smoke-task3
```

## Risks

| Risk | Mitigation |
|---|---|
| The grid takes too long on a laptop | Resume support; split the grid across members' machines; the smallest useful grid is 5 × 3 × 3 = 45 runs |
| LFU keeps once-hot blocks forever (cache pollution) | Expected; it is a finding. Optionally add LFU with decay as a fourth policy if time allows |
| `max_model_len` sets a minimum capacity | Lower `--max-model-len` for the sweep (for example 4096) and record it; ShareGPT turns over the limit are skipped by the client |
