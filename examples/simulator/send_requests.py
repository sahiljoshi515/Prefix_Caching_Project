"""Replay the prompts of a trace file against a running simulator server.

Start the server first (see README.md), for example::

    export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
    export VLLM_SIMULATOR_TRACE=$PWD/examples/simulator/trace_example.jsonl
    vllm serve gpt2 --enable-prefix-caching --port 8000

then run ``python examples/simulator/send_requests.py``.  Each request is
tagged with its trace ``request_id`` through the ``X-Request-Id`` header,
which is how the server-side simulator finds the response to replay.
"""
import argparse
import json
import time
from pathlib import Path

import requests
from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--trace", default=Path(__file__).with_name("trace_example.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    with open(args.trace) as f:
        records = [json.loads(line) for line in f if line.strip()]

    for record in records:
        start = time.perf_counter()
        completion = client.completions.create(
            model=args.model,
            prompt=record["prompt"],
            max_tokens=args.max_tokens,
            temperature=0,
            extra_headers={"X-Request-Id": record["request_id"]})
        choice = completion.choices[0]
        elapsed_ms = 1000 * (time.perf_counter() - start)
        print(f"{record['request_id']}: {choice.text!r} "
              f"[{choice.finish_reason}, {completion.usage.completion_tokens} "
              f"tokens, {elapsed_ms:.0f} ms]")

    # Prefix-cache statistics come from vLLM's own metrics, untouched.
    metrics = requests.get(args.base_url.replace("/v1", "/metrics")).text
    for line in metrics.splitlines():
        if line.startswith("vllm:gpu_prefix_cache_hit_rate"):
            print("prefix cache hit rate (from /metrics):", line.split()[-1])


if __name__ == "__main__":
    main()
