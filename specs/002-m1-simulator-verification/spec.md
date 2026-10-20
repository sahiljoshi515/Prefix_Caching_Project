# Feature Specification: Independent verification of the Milestone 1 simulator

**Spec**: `002-m1-simulator-verification` | **Milestone/Task**: Milestone 1
**Owner**: Triet Chau and Xin Liang Chia (each verifies independently)
**Implemented by**: Sahil Joshi (commits `072288d4`, `4393eb5d`, `c045b639`)
**Status**: Awaiting verification
**Created**: 2026-10-18 | **Depends on**: none

**Input**: Charter: "Milestone 1 (Done by Sahil Joshi). Triet Chau and Xin
Liang Chia must test this on their machines independently and verify."
Course spec, Milestone 1: "bypass actual GPU inference [...] keeping all
other components functional, including continuous batching, scheduling, and
request management [...] Please write a set of tests to verify that your
modified vLLM implementation works correctly before proceeding to the next
milestone."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A teammate reproduces the setup from the README alone (Priority: P1)

A team member who did not write Milestone 1 clones the repository on their
own laptop, follows only `README.md`, and ends with a working simulator.

**Why this priority**: the course grades the README's setup instructions,
and Milestones 2 and 3 run on every member's machine.

**Independent Test**: a fresh clone into a new directory with no existing
`.venv`.

**Acceptance Scenarios**:

1. **Given** a fresh clone, **When** the verifier runs the README *Setup*
   commands for their OS, **Then** both install checks print `CpuPlatform`
   and `SimulatorExecutor`.
2. **Given** the install, **When** they run
   `python -m pytest tests/simulator -v -o asyncio_default_fixture_loop_scope=function`
   with `VLLM_USE_SIMULATOR=1`, **Then** all 67 tests pass.
3. **Given** a running `vllm serve gpt2` with the example trace, **When** they
   run `python examples/simulator/send_requests.py`, **Then** the three
   responses match the trace and the prefix cache hit rate is above 0.

---

### User Story 2 - The simulator keeps vLLM's machinery (Priority: P1)

The verifier checks the course's Milestone 1 claim, not just that tests
pass: scheduling, batching and prefix caching still run, and only the forward
pass is replaced.

**Independent Test**: read `vllm/simulator.py` and the four-line hooks in
`vllm/engine/llm_engine.py` and `vllm/engine/async_llm_engine.py`; confirm
with the tests named below.

**Acceptance Scenarios**:

1. **Given** two simultaneous requests, **Then** they are batched in one step
   (`test_simultaneous_requests_are_batched_together`).
2. **Given** a repeated 40-token prompt, **Then** 32 tokens are reported
   cached (`test_prefix_cache_hit_on_repeated_prompt`).
3. **Given** a 6-block cache, **Then** preemption happens and requests still
   finish (`tests/simulator/test_engine.py`, preemption test).
4. **Given** online mode, **Then** `/metrics` reports
   `vllm:gpu_prefix_cache_hit_rate > 0`
   (`test_metrics_report_prefix_cache_hits`).

### Edge Cases

- Windows: the README covers macOS and Linux only. A Windows verifier uses
  WSL2 (Linux instructions) and records that.
- Apple Silicon versus Intel Macs, Python 3.9 to 3.12: record the exact
  versions. A failure on a supported version is a defect.
- Network-restricted networks: the first run downloads tokenizer files from
  Hugging Face; record if a mirror or `HF_HUB_OFFLINE=1` with a warm cache
  was needed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Each verifier MUST use a fresh clone and a fresh virtual
  environment, following only the README.
- **FR-002**: Each verifier MUST record OS and version, CPU architecture,
  Python version, `uv --version`, the commit hash, and the full output of the
  test run.
- **FR-003**: Each verifier MUST file the record as a comment on the
  tracking issue "Milestone 1 verification" (one comment per verifier).
- **FR-004**: Every deviation from the README (an extra step, a changed
  command, an error) MUST be filed as a separate issue and fixed through a
  PR that updates the README or the code.

## Success Criteria *(mandatory)*

- **SC-001**: Two verification records exist from two different machines,
  each with 67 of 67 tests passing.
- **SC-002**: Every deviation found has a merged fix, or a README note if it
  is environment-specific.
- **SC-003**: Once `001-ci-cd` lands, the same suite also passes on Linux CI,
  which serves as a third, automated verification.

## Assumptions

- "Test this on their machines" means the README path plus the test suite
  plus the example replay; it does not include re-deriving the design.

## Out of Scope

- Changing the simulator's behaviour. A verifier who finds a defect files an
  issue; the fix gets its own PR.
