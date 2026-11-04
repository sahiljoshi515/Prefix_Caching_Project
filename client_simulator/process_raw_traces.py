"""Prepare compact client conversations and server responses before serving."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from client_simulator.adapters import agentbank, bailian, ccbench, sharegpt
from client_simulator.schema import Conversation, request_id
from client_simulator.token_plan import write_plan

# Text datasets provide `load_conversations`; the Bailian traces, which have
# no text, provide `load_token_requests` and become token-level plans.
ADAPTERS = {
    "sharegpt": sharegpt,
    "agentbank": agentbank,
    "ccbench": ccbench,
    **bailian.ADAPTERS,
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
                if conversation.turn_gaps_s is not None:
                    record["turn_gaps_s"] = conversation.turn_gaps_s
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=ADAPTERS)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Server response JSONL; point VLLM_SIMULATOR_TRACE here.")
    parser.add_argument(
        "--client-output",
        type=Path,
        help="Client JSONL; defaults to OUTPUT_STEM.conversations.jsonl "
        "(OUTPUT_STEM.plan.jsonl for token-level datasets).")
    args = parser.parse_args()
    adapter = ADAPTERS[args.dataset]
    token_level = hasattr(adapter, "load_token_requests")
    suffix = ".plan.jsonl" if token_level else ".conversations.jsonl"
    client_path = args.client_output or args.output.with_name(
        args.output.stem + suffix)
    paths = [args.raw.resolve(), args.output.resolve(), client_path.resolve()]
    if len(set(paths)) != len(paths):
        parser.error("Raw, client, and server paths must all be different")

    adapter.check_download(args.raw)
    if token_level:
        count = write_plan(adapter.load_token_requests(args.raw), client_path,
                           args.output)
        print(f"Wrote {count} token-level requests to {client_path}")
    else:
        write_traces(adapter.load_conversations(args.raw), client_path,
                     args.output)


if __name__ == "__main__":
    main()
