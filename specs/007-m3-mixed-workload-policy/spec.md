# Feature Specification: A workload-aware eviction policy for mixed workloads

**Spec**: `007-m3-mixed-workload-policy` | **Milestone/Task**: Milestone 3,
Task 2
**Owner**: TBD (assign at Milestone 3 kickoff) | **Reviewers**: TBD
**Status**: Draft
**Created**: 2026-10-18 | **Depends on**: `004`, `005`, `006`

**Input**: Charter: "Create mixed-workload traces by combining requests from
different workload types. Analyze differences in prefix reuse patterns,
reuse intervals, and cache pressure across workloads. Design a
workload-aware eviction policy that determines which KV-cache blocks to
retain or evict. Investigate how to identify workload types and whether
workload-specific information can improve eviction decisions. Implement the
proposed policy in the modified vLLM simulator. Compare the proposed policy
against standard baselines (LRU, LFU, FIFO) under identical cache
configurations and request traces. Evaluate whether the proposed policy
improves prefix-cache reuse across heterogeneous workloads. Present results
using appropriate plots and discuss the benefits, limitations, and
trade-offs."
Course spec: "What is the difference in their KV cache reusing time interval?
[...] reusing portion? How do I know a request belongs to which workload
type?"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build mixed traces (Priority: P1)

A team member builds a mixed trace from the per-workload traces of `006` with
a given mix (for example 40% ShareGPT, 20% each of the others), a total
request budget and a seed. The result is replayable by the unchanged client.

**Independent Test**: mixing two fixtures 50/50 gives the expected counts,
keeps each conversation's turns in order and intact, and is identical across
runs with the same seed.

**Acceptance Scenarios**:

1. **Given** a mix and seed, **Then** whole conversations or sessions are
   sampled per workload (never split), and each keeps its own arrival
   process: recorded timestamps for Bailian, Poisson for the others,
   superimposed on one timeline.
2. **Given** the mixed trace, **Then** every request id keeps its workload
   prefix.

---

### User Story 2 - Characterize the mix (Priority: P1)

From an LRU run of the mixed trace (with `004`'s event log), per-workload
reuse-interval CDFs, reused-token fractions, hits per block and
working-set growth over time answer the course spec's three questions
quantitatively.

---

### User Story 3 - The workload-aware policy (Priority: P1)

A new evictor `VLLM_EVICTION_POLICY=wa` uses each block's workload label to
decide what to evict. It is designed from User Story 2's findings and
implemented in the simulator like `005`'s policies.

**Acceptance Scenarios**:

1. **Given** labels from request ids (oracle mode), **Then** each inserted
   block carries the label of the request that computed it.
2. **Given** inferred mode, **Then** labels come from a classifier over
   features a real server could observe (no request-id prefix). Its accuracy
   against the oracle is reported.
3. **Given** the same capacity, block size, mixed trace and seed, **Then**
   `wa` and the three baselines are compared on every metric of `004`/`005`.

---

### User Story 4 - Evaluate honestly (Priority: P2)

Results cover at least three mixes and three seeds per mix, report means with
spread, include each single-workload trace (to show the policy does not hurt
homogeneous workloads), and discuss benefits, limitations and overheads.

### Edge Cases

- **Shared blocks across workloads** (a block hit by requests of two
  workloads): the label policy is defined, for example "label of first
  inserter, updated to the most recent hitter", and tested.
- **Unknown workload** in inferred mode: fall back to the policy's default
  class (behaves like LRU).
- **Bookkeeping overhead**: `evict()` stays O(log n) amortized (`005`
  FR-005).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `analysis/mix.py` MUST build mixed traces from `006` outputs
  given proportions, a request budget and a seed.
- **FR-002**: The allocator MUST make the current request's workload label
  available to the evictor at insertion, reusing `004`'s request context (no
  new allocator arguments).
- **FR-003**: The `wa` evictor MUST implement the `Evictor` interface plus
  `record_hit` (`005`), live in `vllm/core/eviction_policies.py`, and be
  selected by `VLLM_EVICTION_POLICY=wa`. Labelling mode comes from
  `VLLM_WA_LABELS=oracle|inferred`.
- **FR-004**: Its design (scoring rule, parameters, how they were chosen)
  MUST be written in `plan.md` before implementation, with the evidence from
  User Story 2.
- **FR-005**: Parameters MUST be tuned on a different seed than the one used
  for the reported comparison.
- **FR-006**: Comparisons MUST use identical capacity, block size, trace and
  seed across policies.

## Success Criteria *(mandatory)*

- **SC-001**: Mixed-trace builder and `wa` evictor tests pass, with
  hand-derived eviction victims for a 3-workload toy trace.
- **SC-002**: Results for ≥ 3 mixes × ≥ 3 seeds × 4 policies × 2 label modes
  are produced by one committed command.
- **SC-003**: The report states, with spread, whether `wa` beats the best
  baseline on hit rate and reused-token fraction, by how much, and where it
  does not. A negative result with a sound explanation meets this criterion.

## Assumptions

- Capacity and block size as fixed in `006`.
- Workload labels in oracle mode come from the request-id prefixes defined in
  `006` FR-002.

## Out of Scope

- Changing vLLM's scheduler or admission policy; only eviction changes.
