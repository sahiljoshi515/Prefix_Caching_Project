# Feature Specification: Client simulator and ShareGPT preprocessing

**Spec**: `003-m2-client-simulator` | **Milestone/Task**: Milestone 2, Task 1
**Owner**: Triet Chau | **Reviewers**: Marcus, Xin Liang Chia
**Status**: Merged (PR #6, 2026-10-19, with fix commit `e026ae82`);
follow-ups open (see `tasks.md` Phase 5)
**Created**: 2026-10-18 (written after the implementation, from the charter
and the PR #6 review) | **Depends on**: Milestone 1

**Input**: Charter: "Download and preprocess the ShareGPT dataset. Implement
a client that sends requests to our modified vLLM server using the
appropriate model chat template. Support both single-turn and multi-turn
conversations. Simulate request arrival times using a Poisson process when
timestamps are unavailable. Ensure that requests are correctly mapped to
their prerecorded responses."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prepare ShareGPT once (Priority: P1)

A team member runs one command that downloads ShareGPT (if not already
present), rebuilds whole conversations from the fragments the dataset splits
them into, and writes two files: the client's conversations and the server's
response trace.

**Independent Test**: run it on a handcrafted 4-record fixture with a split
conversation, a gap, and a trailing unanswered user turn; check both outputs
line by line.

**Acceptance Scenarios**:

1. **Given** fragments `A_0` and `A_2` of one conversation, **When**
   preprocessed, **Then** one conversation `sharegpt:A` with 4 messages is
   written.
2. **Given** a conversation with a missing fragment or out-of-order roles,
   **Then** it is skipped and counted in the printed summary.
3. **Given** a conversation ending with an unanswered user message, **Then**
   that message is dropped.
4. **Given** the outputs, **Then** every response-trace line is
   `{"request_id": "<conversation_id>:<turn>", "response": <text>}` and every
   turn of every conversation has exactly one line.

---

### User Story 2 - Replay conversations against the server (Priority: P1)

A team member starts `vllm serve` with the response trace and runs the client
in `single` (first exchange only) or `multi` (all turns, history included)
mode, at a chosen conversation rate, think time, seed and limit.

**Independent Test**: replay 10 conversations against a local simulator
server; every request completes with its recorded answer.

**Acceptance Scenarios**:

1. **Given** `--conversation-rate R` (finite), **Then** conversation starts
   follow a Poisson process with rate R from a seeded RNG, and two runs with
   the same seed have identical planned start times.
2. **Given** `--conversation-rate inf`, **Then** all conversations start at
   once, bounded by `--max-concurrency`.
3. **Given** multi mode, **Then** turn k of a conversation is sent only after
   turn k-1's response arrives plus `--think-time` seconds, and its messages
   are the recorded history up to turn k.
4. **Given** any request, **Then** it carries `X-Request-Id:
   <conversation_id>:<turn>`, and the server replays exactly the recorded
   answer (`answer_match: true`).
5. **Given** a prompt plus answer longer than `max_model_len`, **Then** the
   turn is skipped with `status: skipped_context` and the conversation's
   remaining turns are counted as abandoned.

---

### User Story 3 - Per-request results for analysis (Priority: P1)

Task 2 (`004`) consumes one JSONL line per request plus a run manifest and
before/after `/metrics` snapshots.

**Acceptance Scenarios**:

1. **Given** a completed request, **Then** its line contains `request_id`,
   `conversation_id`, `turn_index`, timing fields (`ready_s`, `sent_s`,
   `completed_s`, `latency_s`), `prompt_tokens`, `completion_tokens`,
   `cached_tokens`, `cached_fraction`, `finish_reason`, `answer_match` and
   `status`.
2. **Given** a request with no prefix-cache hit, **Then** `cached_tokens` is
   `0` and `cached_fraction` is `0.0`, never `null`. vLLM omits
   `prompt_tokens_details` when nothing hit
   (`vllm/entrypoints/openai/serving_chat.py:752`).
3. **Given** a run, **Then** `<output>.run.json` records the full
   configuration, the start time, the final status and the counts.

### Edge Cases

- Detokenized output can differ from the recorded text (special-token
  strings inside ShareGPT text, normalization). The turn is flagged
  `answer_match: false`, but the conversation continues, because the next
  turn's history comes from the recording anyway. Ending the conversation
  would shorten long conversations and bias the multi-turn statistics.
- Skipped turns (context limit) are normal for ShareGPT at
  `--max-model-len 8192` and do not make the run exit non-zero. Transport or
  server failures do.
- Qwen2.5's chat template inserts a default system prompt when none is given,
  so every conversation shares that prefix. This is the template's real
  behaviour and is kept, but `004` reports it separately.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Preprocessing MUST download the pinned ShareGPT file only if it
  is absent, and MUST be deterministic (same input, byte-identical output).
- **FR-002**: Preprocessing MUST rebuild conversations from `<id>_<offset>`
  fragments and skip, with counted reasons, conflicting, gapped,
  wrong-role-order or empty groups.
- **FR-003**: The client MUST send chat requests through
  `/v1/chat/completions`, so the server applies the served model's chat
  template; prompt length checks MUST use the server's `/tokenize` with the
  same template.
- **FR-004**: The client MUST support `single` and `multi` modes, Poisson or
  immediate conversation arrivals, a fixed think time, a seed, a limit and a
  concurrency bound.
- **FR-005**: Request ids MUST be `<conversation_id>:<turn_index>` on both
  sides, generated by one shared function.
- **FR-006**: The client MUST set `max_tokens` to the recorded answer's token
  count plus 1 (for the EOS the simulator appends), and `temperature` 0.
- **FR-007**: A missing `cached_tokens` MUST be recorded as `0` when the
  server runs with `--enable-prompt-tokens-details`. *(Done in `e026ae82`)*
- **FR-008**: An answer mismatch MUST be recorded and MUST NOT end the
  conversation. *(Open: follow-up)*
- **FR-009**: The exit status MUST be non-zero only for failed requests, not
  for `skipped_context`. *(Open: follow-up)*
- **FR-010**: Unit tests MUST cover fragment rebuilding, `iter_requests`,
  `schedule_conversations` determinism, JSONL validation and the
  `cached_tokens` default. *(Open: follow-up)*
- **FR-011**: The code MUST pass ruff, isort and yapf. *(Done in `e026ae82`)*
- **FR-012**: Experiment output (`results/`) MUST NOT be committed
  (constitution V); `results/` is gitignored. *(Partly done: the files were
  removed in `e026ae82`; the `.gitignore` entry is still missing)*
- **FR-013**: Adapters for later workloads (`006`) MUST implement
  `check_download(raw_path)` and `load_conversations(raw_path)` and plug into
  `process_raw_traces.ADAPTERS`.

### Key Entities

- **Conversation**: `conversation_id`, alternating `user`/`assistant`
  `messages`, starting with `user` and ending with `assistant`.
- **ReplayRequest**: one turn: history before the answer, and the answer.
- **Result row**: one line of the client's output JSONL (User Story 3).

## Success Criteria *(mandatory)*

- **SC-001**: 1,000 ShareGPT conversations replay in multi mode with 100% of
  sent requests `answer_match: true` or individually explained.
- **SC-002**: Two runs with the same seed and limit produce identical
  `planned_conversation_start_s` columns.
- **SC-003**: Every request in the client output has a non-null
  `cached_fraction` (completed requests) or a `status` explaining why not.

## Assumptions

- Model: `Qwen/Qwen2.5-0.5B-Instruct` (has a chat template; about 11 MB of
  config and tokenizer files).
- ShareGPT file: `ShareGPT_V3_unfiltered_cleaned_split.json` from
  `anon8231489123/ShareGPT_Vicuna_unfiltered`.
- ShareGPT has no timestamps, so arrivals are Poisson (course spec, Task 1).

## Out of Scope

- Metric computation and plots (`004`).
- Workload adapters beyond ShareGPT (`006`).
