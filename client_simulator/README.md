# Preprocessing replay traces

Run from the repo root with the virtual environment:

```bash
python -m client_simulator.process_raw_traces \
  --dataset sharegpt \
  --raw data/raw/sharegpt.json \
  --output data/processed/sharegpt.responses.jsonl \
  --client-output data/processed/sharegpt.conversations.jsonl
```

Preprocessing downloads raw data only if missing and reconstructs conversations
from raw fragments. It outputs:

- **Client conversations:** one JSON object per reconstructed conversation,
  with `conversation_id` and a `messages` list. Each message is stored once.
- **Server responses:** one JSON object per turn, containing only `request_id`
  and `response`. This is the file the server loads:

```bash
export VLLM_SIMULATOR_TRACE="$PWD/data/processed/sharegpt.responses.jsonl"
```

## Reading client conversations

The replay CLI uses these helpers to construct histories on demand:

```python
from pathlib import Path
from client_simulator.client_sim import iter_requests, load_conversations

path = Path("data/processed/sharegpt.conversations.jsonl")
for conversation in load_conversations(path):
    for request in iter_requests(conversation, mode="multi"):
        # Send request.messages with X-Request-Id=request.request_id.
        print(request.request_id)
```

Use `mode="single"` to emit only the first exchange of each conversation.

Future conversation adapters implement `check_download(raw_path)` and
`load_conversations(raw_path)`.

## Replaying conversations

Start the simulator server in one terminal (activate the project virtualenv):

```bash
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
export VLLM_SIMULATOR_TRACE="$PWD/data/processed/sharegpt.responses.jsonl"
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --max-model-len 8192 --port 8000
```

After server is active, run the client in another terminal:

```bash
python -m client_simulator.client_sim \
  --trace data/processed/sharegpt.conversations.jsonl \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --mode multi --conversation-rate 2 --think-time 1 \
  --seed 1 --limit 10 --max-concurrency 32 \
  --output results/multi.jsonl --capture-metrics
```

Use `--mode single` and a different output path for the first exchange only.
Omit `--limit` to replay the whole file. The limit controls the client only.  
`--conversation-rate inf`launches conversations immediately, otherwise
uses Poisson conversation arrivals. `--think-time` is a fixed delay after
each successful response.

Conversations overlap, but turns within each conversation are sequential.

## Results

- The chosen `--output` is a per request JSONL log.
- `OUTPUT.run.json` records configuration, UTC start time, run status, and
  final counts. `finished` means replay ended, check counts for failed and
  skipped requests.
- With `--capture-metrics`, `OUTPUT.metrics.before.txt` and
  `OUTPUT.metrics.after.txt` save the server's Prometheus metrics.

## AgentBank and CC-Bench agent trajectories

Both are text trajectories, prepared into the same conversation format as
ShareGPT and replayed through `/v1/chat/completions` with the model's chat
template. Each assistant message is one request with the full history
before it. Reading them needs `pyarrow` (`make setup` installs it).

| | AgentBank | CC-Bench trajectories |
|---|---|---|
| Source | [Solaris99/AgentBank](https://huggingface.co/datasets/Solaris99/AgentBank), revision `6d718ff7` | [zai-org/CC-Bench-trajectories](https://huggingface.co/datasets/zai-org/CC-Bench-trajectories), revision `792c6d32` |
| Licence | Apache-2.0 | MIT |
| Raw format | 19 subsets (ALFWorld, WebShop, Mind2Web, InterCode, GSM8K, ...), one parquet file each; rows `id`, `conversations` (`from`: `human`/`gpt`, `value`) | One parquet file; each row's `trajectory` is a Claude Code session log (JSON events linked by `uuid`/`parentUuid`) |
| Size | 53,205 trajectories, 243,598 requests | 370 sessions (74 coding tasks × 5 models), 498 threads, 18,184 requests |
| Conversation id | `agentbank:<subset>:<id>` (ids repeat across subsets) | `ccbench:<row id>:<thread>` |

Role mapping (spec 006): the task and environment observations (AgentBank
`human`) and the developer's prompts and tool results (CC-Bench) become
`user`; the agent's reasoning, actions and tool calls become `assistant`.
Neither dataset has a system prompt. Consecutive messages with the same
role are merged with a blank line; a trajectory must start with a user
message, and a final unanswered user message is dropped.

CC-Bench logs need more rules:

- **Threads.** An event without a parent starts a thread and every other
  event joins its parent's thread. Roots are session starts, sub-agent
  (sidechain) starts and `/compact` boundaries, which replace the history
  with a summary; each thread is its own conversation with its own context.
  Within a thread, events keep file order.
- **Rendering.** A tool call is `<tool_use name="NAME">INPUT_JSON</tool_use>`,
  a result `<tool_result>TEXT</tool_result>` (an image `[image]`). Thinking
  blocks are dropped.
- **Dropped events:** `system` and `summary` events, `isMeta` messages,
  local slash commands (`<command-name>`, `<local-command-stdout>`) and API
  error messages.
- **Not in the logs, so not replayed:** Claude Code's system prompt and
  tool definitions (about 12K tokens that every real session shares), and
  when a sub-agent ran inside its parent session.

Both adapters write conversations in a seeded random order (`--seed`), so
the first N lines are a seeded sample: of all 19 AgentBank subsets mixed,
and of CC-Bench without the five models' runs of one task arriving
together. The raw files are downloaded into `--raw` if missing.

```bash
python -m client_simulator.process_raw_traces --dataset agentbank \
  --raw data/raw/agentbank --seed 0 \
  --output data/processed/agentbank.responses.jsonl \
  --client-output data/processed/agentbank.conversations.jsonl
python -m client_simulator.process_raw_traces --dataset ccbench \
  --raw data/raw/ccbench/train.parquet --seed 0 \
  --output data/processed/ccbench.responses.jsonl \
  --client-output data/processed/ccbench.conversations.jsonl
```

Context: with Qwen's tokenizer, AgentBank's first 1,000 conversations
(4,992 requests) all fit in 32,768 tokens (median prompt 460 tokens).
CC-Bench prompts are long (median 12,778 tokens among those that fit):
about 36% of its requests are over 32,768. The client skips such a turn
and the rest of its conversation, so CC-Bench replays each thread until
its history passes 32K, and the run reports the skipped share.

## Bailian token traces and shared experiments

Bailian Trace A is a two-hour anonymous production trace. The local file
has 43,058 requests; timestamps are in seconds. Source/schema/license:
[Qwen Bailian](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)
(Apache-2.0). Hashes describe already-templated token inputs, so replay
must not apply another chat template. The adapter supports the upstream
16-token Trace A/B, thinking and coder filenames; other releases require
format verification first.

Prepare a 500-request sample (or omit `--limit` for the entire file):

```bash
python -m client_simulator.process_raw_traces --dataset bailian \
  --raw data/raw/qwen_traceA_blksz_16.jsonl --seed 0 --limit 500 \
  --client-output data/processed/bailian/requests.jsonl \
  --output data/processed/bailian/responses.jsonl
```

Existing raw files are reused; otherwise the adapter downloads the named
upstream trace (Trace A by default). Existing token outputs are refused;
choose new paths for a different seed/sample. Full-block hashes become
reproducible 16-token sequences from ordinary model tokens. The final
partial block is request-specific and preserves the exact input length.
Synthetic outputs preserve output length but do not reconstruct original
answer text, answer-to-next-input sharing or true cross-dataset sharing.
Image/search/file labels are
retained as metadata; replay does not perform those operations.

Client JSONL contains `request_id`, parent-root `conversation_id`, zero-based
`turn_index`, `timestamp_s`, `prompt_token_ids`, `output_length`,
`eos_token_id`, `block_size` and `request_type`. Server JSONL contains the
same `request_id` and `response_token_ids`, in the same order. The adapter
namespaces IDs and synthesis by the source filename; keep that name stable
when rebuilding a sample.

The shared runner below starts the server and calls the client for you.
For a server already running, the client can read the same workload config.
Workload limits, rates and think times come from that JSON file:

```bash
python -m client_simulator.client_sim \
  --workloads examples/bailian_experiment.json \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output results/bailian-manual.jsonl --capture-metrics
```

The client submits complete token inputs to `/v1/completions` at recorded
times relative to the first request, validates exact token IDs, and includes
one EOS token in the context budget. It records lateness/concurrency waits,
latency, failures and context skips. A skipped parent does not suppress a
later timestamped request. The ShareGPT chat/Poisson path keeps its existing
behavior.

### One runner for individual and mixed workloads

Use the shared runner instead of adding a runner for each dataset:

```bash
# First 20 prepared requests, all three policies on fresh local servers.
python -m analysis.run_experiment --config examples/bailian_experiment.json \
  --out results/bailian-smoke
# Recompute the existing metrics without replaying.
python -m analysis.run_experiment --config examples/bailian_experiment.json \
  --out results/bailian-smoke --analyze-only
# A mixture uses the same command with another configuration.
python -m analysis.run_experiment --config examples/mixed_experiment.json \
  --out results/mixed-smoke
```

The example fixes capacity 65,536 tokens, block size 16, context 32,768,
concurrency 32 and seed 0. For the larger experiment, change the workload's
`limit` to 500 and choose a new `--out`. The 500 arrivals span 167.407 s;
three recorded-time runs take about nine minutes plus startup. The 20-request
smoke is a transport/measurement check and may not cause cache pressure.
The final sample/repeats and mixed proportions remain experimental choices.

Config paths are relative to the config file. Each `workloads` entry has a
unique `name`, `format` (`chat` or `tokens`), prepared `trace`, `responses`
and optional `limit`. Chat entries can set `mode`, `conversation_rate`,
`think_time` and `seed`. Token entries use recorded timestamps. `offset_s`
shifts either stream relative to the shared start. Multiple entries overlay
on one server and share one concurrency limit; sample counts/rates/offsets
explicitly define the mix, with no automatic balancing. Input request IDs
must be unique across workloads. Future text adapters emit the existing
chat format; future timestamped token adapters emit the token format.

`experiment.json` freezes the selected workload files, merged responses,
settings, request IDs, checksums and commit/dirty state. A different sample
or setting needs a new output directory. Use a new output directory when
changing replay code too; the runner does not back up source files.
`--analyze-only` recomputes measurements from saved traffic.
Each policy directory holds server/events/client logs, metric snapshots,
`run.json`, `summary.json` and the shared analysis CSVs. `summary.csv`
compares policies; each summary also has per-workload statuses and usable
reuse, timing quantiles and validity checks. Raw results stay under ignored
`results/`; do not mix these runs with old experiment folders.

**Measurement:** the pinned completion endpoint omits API cached-token
counts. Its raw `cached_tokens` and API cached fraction remain null, not
zero. Use event-derived `usable_cached_tokens`, `usable_cached_fraction`
and summary `usable_reused_fraction` to compare policies. These count only
consecutive cached blocks before the first miss. API `token_reused_fraction`
remains unknown for token/mixed runs; API/log gap checks are unknown there,
not an independent success check. Block hit rate is independently checked
against the server's metric. Metrics retain zero-hit blocks and use parent
roots for intra-conversation attribution.

The experiment and report figure plan is in
[spec 006](../specs/006-m3-workload-eviction/plan.md#bailian-experiment-and-report-plan-2026-11-05).
Generate report figures from a finished comparison, without replaying:

```bash
python -m analysis.plot_experiment --out results/bailian-500-cap65536 \
  --figures report/milestone3/figures/task1/bailian
```

The command makes combined report panels and individual PNG/PDF bars/CDFs,
plus windowed hit rate, using the existing plotting/metric helpers. Zero-hit
blocks and zero intervals remain included; an inset exposes the block-CDF
tail. It checks populations and aggregate values before drawing and records
input/plot-source checksums. The
[Milestone 3 note](../report/milestone3/task1-workload-eviction.md) contains
results, reading instructions, a short Overleaf paragraph and limitations.

The primary completed Bailian comparison uses 3,000 requests. Regenerate
its figures with:

```bash
python -m analysis.plot_experiment --out results/bailian-3000-cap65536 \
  --figures report/milestone3/figures/task1/bailian-3000
```

The 500-request files/figures remain an earlier pilot. The result note uses
the larger sample and includes the short analysis in plain text.
