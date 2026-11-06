"""Prepare compact client conversations and server responses before serving."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from client_simulator.adapters import qwenbailian, sharegpt
from client_simulator.schema import Conversation, request_id

ADAPTERS = {
    "sharegpt": sharegpt,
    "bailian": qwenbailian,
}


def write_traces(conversations: Iterable[Conversation], client_path: Path,
                 server_path: Path) -> None:
    """Store each conversation once and each server answer once"""
    if client_path.resolve() == server_path.resolve():
        raise ValueError("Client and server trace paths must be different")
    client_path.parent.mkdir(parents=True, exist_ok=True)
    server_path.parent.mkdir(parents=True, exist_ok=True)
    client_tmp = client_path.with_name(client_path.name + ".tmp")
    server_tmp = server_path.with_name(server_path.name + ".tmp")
    conversation_count = 0
    request_count = 0
    seen = set()
    try:
        with client_tmp.open("w", encoding="utf-8") as client_file, \
                server_tmp.open("w", encoding="utf-8") as server_file:
            for conversation in conversations:
                if conversation.conversation_id in seen:
                    raise ValueError("Duplicate conversation ID: "
                                     f"{conversation.conversation_id}")
                seen.add(conversation.conversation_id)
                record = {
                    "conversation_id": conversation.conversation_id,
                    "messages": [asdict(m) for m in conversation.messages],
                }
                client_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n")
                # History slices are only needed at replay time.
                for index in range(1, len(conversation.messages), 2):
                    rid = request_id(conversation.conversation_id, index // 2)
                    answer = {
                        "request_id": rid,
                        "response": conversation.messages[index].content,
                    }
                    server_file.write(
                        json.dumps(answer, ensure_ascii=False) + "\n")
                    request_count += 1
                conversation_count += 1
        client_tmp.replace(client_path)
        server_tmp.replace(server_path)
    finally:
        client_tmp.unlink(missing_ok=True)
        server_tmp.unlink(missing_ok=True)


def write_token_traces(requests, client_path, server_path):
    """Write paired input-token and response-token files."""
    if client_path.resolve() == server_path.resolve():
        raise ValueError("Client and server trace paths must be different")
    client_path.parent.mkdir(parents=True, exist_ok=True)
    server_path.parent.mkdir(parents=True, exist_ok=True)
    if client_path.exists() or server_path.exists():
        raise ValueError(
            "Token output already exists; choose new output paths")
    client_tmp = client_path.with_suffix(client_path.suffix + ".tmp")
    server_tmp = server_path.with_suffix(server_path.suffix + ".tmp")
    count, seen = 0, set()
    try:
        with client_tmp.open("w") as client, server_tmp.open("w") as server:
            for request in requests:
                if request["request_id"] in seen:
                    raise ValueError("Duplicate token request ID")
                seen.add(request["request_id"])
                row = dict(request)
                answer = row.pop("response_token_ids")
                row["output_length"] = len(answer)
                client.write(json.dumps(row) + "\n")
                server.write(
                    json.dumps({
                        "request_id": row["request_id"],
                        "response_token_ids": answer
                    }) + "\n")
                count += 1
        client_tmp.replace(client_path)
        server_tmp.replace(server_path)
    finally:
        client_tmp.unlink(missing_ok=True)
        server_tmp.unlink(missing_ok=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=ADAPTERS)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Server response JSONL; point VLLM_SIMULATOR_TRACE here.")
    parser.add_argument(
        "--client-output",
        type=Path,
        help="Client JSONL; defaults to OUTPUT_STEM.conversations.jsonl.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    client_path = args.client_output or args.output.with_name(
        args.output.stem + ".conversations.jsonl")
    paths = [args.raw.resolve(), args.output.resolve(), client_path.resolve()]
    if len(set(paths)) != len(paths):
        parser.error("Raw, client, and server paths must all be different")

    adapter = ADAPTERS[args.dataset]
    adapter.check_download(args.raw)
    if hasattr(adapter, "load_token_requests"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        count = write_token_traces(
            adapter.load_token_requests(args.raw, tokenizer, args.seed,
                                        args.limit, args.block_size),
            client_path, args.output)
        print(f"Prepared {count} token requests")
    else:
        from itertools import islice
        write_traces(islice(adapter.load_conversations(args.raw), args.limit),
                     client_path, args.output)


if __name__ == "__main__":
    main()
