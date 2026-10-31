"""CC-Bench coding-agent trajectories (HF ``zai-org/CC-Bench-trajectories``).

Each parquet row is one task solved by one model (74 tasks x 5 models), with
the agent's Claude Code session log as a JSON list in ``trajectory``.

Role mapping (spec 006 FR-005):

- Only main-thread ``user`` and ``assistant`` events count. ``summary`` and
  ``system`` events (context compaction, API errors) and side-chain events
  (sub-agents, ``isSidechain``) are dropped.
- ``user`` content becomes the ``user`` message. A ``tool_result`` part is
  written as ``<tool_result>\\n...\\n</tool_result>``, an image as
  ``[image]``.
- ``assistant`` content becomes the ``assistant`` message. A ``text`` part
  is kept, a ``tool_use`` part is written as ``<tool_use name="...">`` +
  its JSON input + ``</tool_use>``, and a ``thinking`` part is dropped
  (hidden reasoning is not sent back as context).
- Consecutive messages with the same role are joined with a blank line, so
  the conversation alternates user/assistant. A trailing ``user`` message is
  dropped; a trajectory that starts with ``assistant`` is skipped.
- Context compaction is not replayed: the conversation keeps growing, and
  the turns beyond the server's context are skipped at replay time.

Conversation ids are ``ccbench:<task_id>-<model>``, for example
``ccbench:12-glm-4-5``. The five models' runs of one task start with the same
user prompt, so a mix should sample at most one run per task.
"""
import json
import re
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from client_simulator.schema import Conversation, Message

URL = ("https://huggingface.co/datasets/zai-org/CC-Bench-trajectories/"
       "resolve/main/train.parquet")


def check_download(raw_path: Path) -> None:
    if raw_path.is_file():
        print(f"Using existing dataset: {raw_path}")
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(raw_path.name + ".download")
    print(f"Downloading CC-Bench to {raw_path}...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as source, \
                temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        temporary.replace(raw_path)
    finally:
        temporary.unlink(missing_ok=True)


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def part_text(part: dict) -> Optional[str]:
    """Text of one content part, or None for parts that are dropped."""
    kind = part.get("type")
    if kind == "text":
        return part.get("text", "")
    if kind == "tool_use":
        arguments = json.dumps(part.get("input"), ensure_ascii=False)
        return f'<tool_use name="{part.get("name", "")}">{arguments}</tool_use>'
    if kind == "tool_result":
        content = part.get("content")
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") if p.get("type") == "text" else "[image]"
                for p in content)
        return f"<tool_result>\n{content or ''}\n</tool_result>"
    if kind == "image":
        return "[image]"
    return None  # thinking


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    texts = (part_text(part) for part in content or [])
    return "\n".join(text for text in texts if text is not None)


def to_conversation(row: dict, counts: Counter) -> Optional[Conversation]:
    messages: list[Message] = []
    for event in json.loads(row["trajectory"]):
        if event.get("type") not in ("user", "assistant"):
            counts["dropped_other_events"] += 1
            continue
        if event.get("isSidechain"):
            counts["dropped_sidechain_events"] += 1
            continue
        role = event["type"]
        text = message_text((event.get("message") or {}).get("content"))
        if messages and messages[-1].role == role:
            messages[-1] = Message(role, messages[-1].content + "\n\n" + text)
            counts["merged_messages"] += 1
        else:
            messages.append(Message(role, text))
    if messages and messages[-1].role == "user":
        messages.pop()
        counts["unanswered_final_users_removed"] += 1
    if not messages:
        counts["skipped_no_exchange"] += 1
        return None
    if messages[0].role != "user":
        counts["skipped_assistant_first"] += 1
        return None
    cid = f"ccbench:{row['task_id']}-{model_slug(row['model_name'])}"
    return Conversation(cid, messages)


def load_conversations(raw_path: Path) -> Iterator[Conversation]:
    import pyarrow.parquet as pq  # only this dataset needs it

    counts: Counter = Counter()
    for row in pq.read_table(raw_path).to_pylist():
        counts["source_trajectories"] += 1
        conversation = to_conversation(row, counts)
        if conversation is not None:
            counts["retained_conversations"] += 1
            yield conversation
    print("CC-Bench conversion: " + json.dumps(dict(counts), sort_keys=True))
