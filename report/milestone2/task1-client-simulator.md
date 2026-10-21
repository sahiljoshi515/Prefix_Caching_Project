# Milestone 2, Task 1: client simulator

**Question.** Replay ShareGPT against the simulator the way users would send
it: through the model's chat template, as single- or multi-turn
conversations, with Poisson arrivals, and with every request mapped to its
recorded answer. Full statement:
[spec 003](../../specs/003-m2-client-simulator/spec.md).

Built by trietmchau (PR #6) with fixes by sahiljoshi515; what changed when is
in the [changelog](changelog.md). How to run it is in
[`client_simulator/README.md`](../../client_simulator/README.md).

## What was built

### Data preparation

- **Source:** ShareGPT V3 (`ShareGPT_V3_unfiltered_cleaned_split.json` on
  Hugging Face, about 670 MB), downloaded once.
- **Rebuilding conversations:** ShareGPT stores long conversations as
  fragments named `<id>_<offset>`. The preprocessor puts the messages back
  in order and skips any conversation whose fragments conflict, leave gaps,
  start with the assistant or break the user/assistant alternation,
  counting each reason. A final user message with no answer is dropped.
- **Output:** the client's conversations, each message stored once, and the
  server's response trace, one line per turn with the id
  `<conversation_id>:<turn>`. The full dataset gives 49,704 conversations
  and 328,525 turns, 6.6 turns per conversation on average.
- **What rebuilding did:** 94,145 fragments formed 50,142 conversations, of
  which 49,704 (99.1%) were kept; 13,392 of those had been split across
  several fragments. The 438 skipped ones used other roles than user and
  assistant (224), started with the assistant (109), broke the alternation
  (96) or were empty (9). Eight unanswered final user messages were
  dropped.

### Replay client

- **Arrivals:** conversations start as a Poisson process at
  `--conversation-rate` (seeded exponential gaps), or all at once with
  `inf`. At most `--max-concurrency` requests are in flight (default 32).
- **Turns:** within a conversation, turn k is sent after turn k−1's answer
  plus `--think-time`. Multi-turn sends the recorded history; single-turn
  sends only the first exchange.
- **Chat template:** requests go to `/v1/chat/completions`, so the server
  applies the model's template; Qwen2.5's adds a default system prompt to
  every conversation. Before sending, the client asks the server's
  `/tokenize` for the prompt and answer lengths, and a turn that would not
  fit in `--max-model-len` is skipped (`skipped_context`), which also ends
  its conversation.
- **Mapping to recorded answers:** the `X-Request-Id` header carries
  `<conversation_id>:<turn>`, which the simulator looks up. `max_tokens` is
  the recorded answer's length plus one for the end-of-sequence token, and
  the client checks that the reply equals the recording (`answer_match`).
- **Output:** one JSON line per request (timing, token counts,
  `cached_tokens`, status), a `run.json` manifest, and the server's
  `/metrics` before and after the run.

## What we verified

From Task 2's runs: seed 0, 1 conversation/s, think time 5 s,
`Qwen/Qwen2.5-0.5B-Instruct`, `--max-model-len 8192`.

| Run | Completed | Skipped (too long) | Failed |
|---|---|---|---|
| Phase 0: 100 conversations, single-turn | 100 | 0 | 0 |
| Phase 0: 100 conversations, multi-turn | 507 | 6, which ended 79 later turns | 0 |
| Full run: 1,000 conversations, single-turn | 997 | 3 | 0 |
| Full run: 1,000 conversations, multi-turn | 5,097 | 60, which ended 1,671 later turns | 2, which ended 3 later turns |

- **Answers match the recording.** A request only counts as completed when
  the reply equals the recorded answer and finished normally. In the full
  multi-turn run, 5,098 requests reached the server and 5,097 replayed
  exactly; the one mismatch is explained below, so spec 003's SC-001 holds
  at 1,000 conversations.
- **One mismatch, explained.** The recorded answer of
  `sharegpt:Fakxkr4:6` contains the text `<|endoftext|>`. Qwen's tokenizer
  turns that text into its real end-of-text token, so the simulated reply
  stops there ("Skipped 1 messages" instead of "Skipped 1
  messages<|endoftext|>"). Spec 003 lists this case: special-token strings
  inside ShareGPT text.
- **The context limit trims long conversations.** In the full multi-turn
  run, 938 of 1,000 conversations replayed whole. 60 reached a turn too
  long for 8,192 tokens, which removed 1,731 of 6,833 turns (25%), always
  the turns with the longest histories. Multi-turn statistics are
  therefore biased toward shorter histories.

## Open issues

- **A connection race, likely.** In the full multi-turn run, request
  `sharegpt:QWJhYvA:5` failed instantly, before it was sent, with an empty
  error message and no error on the server. It happened once in 5,159
  requests. vLLM's HTTP server closes idle
  connections after 5 s (`TIMEOUT_KEEP_ALIVE`,
  `vllm/entrypoints/openai/api_server.py:70`), exactly our think time, so
  the client can reuse a pooled connection just as the server closes it.
  Not confirmed yet. A client keep-alive shorter than 5 s, or one retry on
  connection errors, would avoid it (spec 003, T015). Since a failed turn
  ends its conversation, each such failure also drops the conversation's
  remaining turns; this one was the conversation's last turn.
- **Spec 003's follow-ups** (Phase 5 of its
  [tasks](../../specs/003-m2-client-simulator/tasks.md)): keep replaying
  after an answer mismatch, exit with an error only on failures, unit tests,
  and PR #6's AI-usage entry.

## Limitations

- ShareGPT has no timestamps, so arrival times are synthetic (Poisson) and
  every user "thinks" for the same fixed time.
- Every conversation shares the chat template's default system prompt. That
  is real template behaviour, but it adds one shared block per conversation;
  Task 2 reports this source of sharing separately.
