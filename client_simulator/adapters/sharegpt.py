import json
import re
from collections import Counter
from pathlib import Path
import shutil
from typing import Iterator
import urllib.request

from client_simulator.schema import Conversation, Message, ReplayRequest
from client_simulator.client_sim import iter_requests

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def check_download(raw_path: Path) -> None:
    if raw_path.is_file():
        print(f"Using existing dataset: {raw_path}")
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(raw_path.name + ".download")

    print(f"Downloading ShareGPT to {raw_path}...")
    try:
        with urllib.request.urlopen(SHAREGPT_URL, timeout=60) as source:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)

        temporary.replace(raw_path)
    finally:
        temporary.unlink(missing_ok=True)

def load_sharegpt(raw_path: Path) -> Iterator[Conversation]:
    """Validate ShareGPT's fragments int IR"""
    with raw_path.open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list):
        raise ValueError("ShareGPT data must be a JSON array")

    groups = {}
    counts = Counter()
    for record_index, record in enumerate(records):
        cur_conversation = f"{raw_path}: record[{record_index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{cur_conversation}: expected an object")
        source_id = record.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"{cur_conversation}: missing or invalid conversation ID")
        match = re.fullmatch(r"(.+)_(\d+)", source_id)
        base_id = match.group(1) if match else source_id
        offset = int(match.group(2)) if match else 0
        raw_messages = record.get("conversations")
        if not isinstance(raw_messages, list):
            raise ValueError(f"{cur_conversation}: expected a conversation list")

        group = groups.setdefault(base_id, {
            "positions": {}, "conflict": False, "fragments": 0,
        })
        group["fragments"] += 1
        counts["source_fragments"] += 1
        if not raw_messages:
            counts["empty_fragments"] += 1
        for message_index, raw_message in enumerate(raw_messages):
            cur_message = f"{cur_conversation}, message[{message_index}]"
            if not isinstance(raw_message, dict):
                raise ValueError(f"{cur_message}: expected an object")
            if not isinstance(raw_message.get("from"), str):
                raise ValueError(f"{cur_message}: 'from' must be a string")
            if not isinstance(raw_message.get("value"), str):
                raise ValueError(f"{cur_message}: content must be a string")

            position = offset + message_index
            positions = group["positions"]
            if position in positions:
                if positions[position] != raw_message:
                    group["conflict"] = True
                    counts["conflicting_overlap_occurrences"] += 1
                else:
                    counts["matching_overlap_occurrences"] += 1
            else:
                positions[position] = raw_message

    # The position maps reference the messages, so discard the outer array
    del records
    roles = {"human": "user", "gpt": "assistant"}
    counts["conversation_groups"] = len(groups)
    for base_id, group in groups.items():
        if group["conflict"]:
            counts["skipped_conflicting_groups"] += 1
            continue
        positions = group["positions"]
        if not positions:
            counts["skipped_empty_groups"] += 1
            continue
        ordered = sorted(positions)
        if ordered[0] != 0:
            counts["skipped_missing_start_groups"] += 1
            continue
        if any(position != index for index, position in enumerate(ordered)):
            counts["skipped_gap_groups"] += 1
            continue
        raw_messages = [positions[position] for position in ordered]
        if any(message["from"] not in roles for message in raw_messages):
            counts["skipped_unsupported_role_groups"] += 1
            continue
        if raw_messages[0]["from"] != "human":
            counts["skipped_assistant_first_groups"] += 1
            continue
        if any(message["from"] != ("human" if index % 2 == 0 else "gpt")
               for index, message in enumerate(raw_messages)):
            counts["skipped_invalid_order_groups"] += 1
            continue
        if raw_messages[-1]["from"] == "human":
            raw_messages = raw_messages[:-1]
            counts["unanswered_final_users_removed"] += 1
        if not raw_messages:
            counts["skipped_no_exchange_groups"] += 1
            continue

        messages = [Message(role=roles[message["from"]],
                            content=message["value"])
                    for message in raw_messages]
        counts["retained_conversations"] += 1
        counts["replay_requests"] += len(messages) // 2
        if group["fragments"] > 1:
            counts["retained_multi_fragment_groups"] += 1
        yield Conversation(conversation_id=f"sharegpt:{base_id}",
                           messages=messages)

    print("ShareGPT reconstruction: " + json.dumps(dict(counts), sort_keys=True))


def load_conversations(raw_path: Path) -> Iterator[Conversation]:
    yield from load_sharegpt(raw_path)
