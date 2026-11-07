"""AgentBank agent trajectories (spec 006, FR-001).

Source: https://huggingface.co/datasets/Solaris99/AgentBank (Apache-2.0),
19 task subsets, one parquet file each, with rows ``id`` and
``conversations`` (a list of ``{"from", "value"}``).

Role mapping (spec 006, plan "Role mapping for agent trajectories"):
``human`` is the task description or an environment observation and becomes
``user``; ``gpt`` is the agent's reasoning and action and becomes
``assistant``. The dataset has no system prompt. Consecutive messages with
the same role are merged with a blank line, a trajectory must start with a
user message, and a final unanswered user message is dropped. Each assistant
message is one replay request with the full history, as in multi-turn
ShareGPT.

Ids repeat across subsets (``mbpp`` and ``mbpp_before``), so a conversation
id is ``agentbank:<subset>:<id>``. Conversations come out in a seeded random
order over all subsets, so the first N of the prepared file are a seeded
sample of the whole dataset.
"""
import json
import random
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from client_simulator.schema import Conversation, Message

REVISION = "6d718ff70e64b6c82236b839075763a4d5187180"
URL = ("https://huggingface.co/datasets/Solaris99/AgentBank/resolve/"
       f"{REVISION}/{{subset}}/train-00000-of-00001.parquet")
SUBSETS = ("alfred", "alfworld", "apps", "gsm8k", "hotpotqa", "humaneval",
           "intercode_bash", "intercode_sql", "iqa", "math", "mathqa", "mbpp",
           "mbpp_before", "mind2web", "rearrange", "strategyqa", "triviaqa",
           "webarena", "webshop")
ROLES = {"human": "user", "gpt": "assistant"}


def check_download(raw_dir: Path) -> None:
    """Download each subset's parquet file into ``raw_dir`` unless present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        path = raw_dir / f"{subset}.parquet"
        if path.is_file():
            continue
        temporary = path.with_name(path.name + ".download")
        print(f"Downloading AgentBank {subset} to {path}...")
        try:
            with urllib.request.urlopen(URL.format(subset=subset),
                                        timeout=60) as source, \
                    temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Using AgentBank subsets in {raw_dir}")


def conversation(subset: str, record: Dict[str, Any],
                 counts: Counter) -> Optional[Conversation]:
    """One AgentBank row as a `Conversation`, or None if it has no exchange."""
    messages: List[Message] = []
    for index, raw in enumerate(record["conversations"]):
        role = ROLES.get(raw.get("from"))
        if role is None or not isinstance(raw.get("value"), str):
            raise ValueError(f"agentbank:{subset}:{record['id']}: "
                             f"message[{index}] has an unknown role or value")
        if messages and messages[-1].role == role:
            messages[-1].content += "\n\n" + raw["value"]
            counts["merged_messages"] += 1
        else:
            messages.append(Message(role=role, content=raw["value"]))
    if messages and messages[0].role != "user":
        counts["skipped_assistant_first"] += 1
        return None
    if messages and messages[-1].role == "user":
        messages.pop()
        counts["unanswered_final_users_removed"] += 1
    if not messages:
        counts["skipped_no_exchange"] += 1
        return None
    return Conversation(conversation_id=f"agentbank:{subset}:{record['id']}",
                        messages=messages)


def conversations(records: Iterable[Tuple[str, Dict[str, Any]]],
                  seed: int = 0) -> Iterator[Conversation]:
    """Convert ``(subset, row)`` pairs, in a seeded random order."""
    rows = list(records)
    random.Random(seed).shuffle(rows)
    counts: Counter = Counter()
    for subset, record in rows:
        counts["source_trajectories"] += 1
        result = conversation(subset, record, counts)
        if result is not None:
            counts["retained_conversations"] += 1
            counts["replay_requests"] += len(result.messages) // 2
            yield result
    print("AgentBank: " + json.dumps(dict(counts), sort_keys=True))


def read_subsets(raw_dir: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    import pyarrow.parquet as pq
    for subset in SUBSETS:
        table = pq.read_table(raw_dir / f"{subset}.parquet",
                              columns=["id", "conversations"])
        for record in table.to_pylist():
            yield subset, record


def load_conversations(raw_dir: Path, seed: int = 0) -> Iterator[Conversation]:
    yield from conversations(read_subsets(raw_dir), seed)
