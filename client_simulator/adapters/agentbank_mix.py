"""AgentBank agent trajectories (HF ``Solaris99/AgentBank``) as conversations,
as Milestone 3 Task 2's mix uses them (dataset ``agentbank-mix``).

Milestone 3 Task 1 replays AgentBank with ``agentbank.py``, whose ids and
order differ. Task 2's plans were built from this module's output (spec 007;
their manifests record its SHA-256), so it is kept as written for Task 2.

The dataset has one parquet file per task (alfworld, webshop, mind2web,
intercode_sql, ...). Each row is ``{id, conversations: [{from, value}]}``.

Role mapping (spec 006 FR-005):

- ``human`` -> ``user``: the task instruction, environment observations and
  tool output, which an agent framework sends to the model.
- ``gpt`` -> ``assistant``: the model's thoughts and actions.
- Consecutive messages with the same role are joined with a blank line, so
  the conversation alternates user/assistant as the client requires.
- A trailing ``user`` message has no recorded answer and is dropped. A
  trajectory that starts with ``gpt``, has another role, or has no exchange
  left is skipped and counted.

Each assistant message becomes one request. Conversation ids are
``agentbank:<task>-<dataset id>``, for example
``agentbank:alfworld-alfworld_0`` (dataset ids repeat across tasks: ``mbpp``
and ``mbpp_before`` both have ``mbpp_0``).
"""
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from client_simulator.schema import Conversation, Message

TASKS = ("alfred", "alfworld", "apps", "gsm8k", "hotpotqa", "humaneval",
         "intercode_bash", "intercode_sql", "iqa", "math", "mathqa", "mbpp",
         "mbpp_before", "mind2web", "rearrange", "strategyqa", "triviaqa",
         "webarena", "webshop")
URL = ("https://huggingface.co/datasets/Solaris99/AgentBank/resolve/main/"
       "{task}/train-00000-of-00001.parquet")
ROLES = {"human": "user", "gpt": "assistant"}


def check_download(raw_dir: Path) -> None:
    """Download any missing task file into `raw_dir/<task>.parquet`."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        path = raw_dir / f"{task}.parquet"
        if path.is_file():
            continue
        temporary = path.with_name(path.name + ".download")
        print(f"Downloading AgentBank {task} to {path}...")
        try:
            with urllib.request.urlopen(URL.format(task=task),
                                        timeout=60) as source, \
                    temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def to_conversation(record: dict, counts: Counter,
                    task: str) -> Optional[Conversation]:
    """One dataset row as a conversation, or None (counted) if unusable."""
    messages: list[Message] = []
    for raw in record.get("conversations") or []:
        role = ROLES.get(raw.get("from"))
        content = raw.get("value")
        if role is None or not isinstance(content, str):
            counts["skipped_unsupported_role"] += 1
            return None
        if messages and messages[-1].role == role:
            messages[-1] = Message(role,
                                   messages[-1].content + "\n\n" + content)
            counts["merged_messages"] += 1
        else:
            messages.append(Message(role, content))
    if messages and messages[-1].role == "user":
        messages.pop()
        counts["unanswered_final_users_removed"] += 1
    if not messages:
        counts["skipped_no_exchange"] += 1
        return None
    if messages[0].role != "user":
        counts["skipped_assistant_first"] += 1
        return None
    return Conversation(f"agentbank:{task}-{record['id']}", messages)


def load_conversations(raw_dir: Path) -> Iterator[Conversation]:
    import pyarrow.parquet as pq  # only this dataset needs it

    counts: Counter = Counter()
    for task in TASKS:
        for record in pq.read_table(raw_dir / f"{task}.parquet").to_pylist():
            counts["source_trajectories"] += 1
            conversation = to_conversation(record, counts, task)
            if conversation is not None:
                counts["retained_conversations"] += 1
                yield conversation
    print("AgentBank conversion: " + json.dumps(dict(counts), sort_keys=True))
