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
