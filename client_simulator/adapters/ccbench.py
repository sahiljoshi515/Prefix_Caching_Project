"""CC-Bench Claude Code trajectories (spec 006, FR-001).

Source: https://huggingface.co/datasets/zai-org/CC-Bench-trajectories (MIT),
one parquet file with 370 rows (74 coding tasks x 5 models). Each row's
``trajectory`` is a Claude Code session log: a JSON list of events linked by
``uuid``/``parentUuid``.

Rules, applied per row:

- **Threads.** An event with no parent (or a parent not seen yet) starts a
  thread; every other event joins its parent's thread. Roots are session
  starts, sub-agent ("sidechain") starts and ``/compact`` boundaries, which
  replace the history with a summary. Each thread is one conversation,
  ``ccbench:<row id>:<thread index>``. Within a thread, events keep file
  order; siblings in the log are parallel tool calls and their results.
- **Roles** (spec 006, plan "Role mapping for agent trajectories"): the
  human's prompts and tool results become ``user``; the model's text and
  tool calls become ``assistant``. A tool call is written as
  ``<tool_use name="NAME">INPUT_JSON</tool_use>`` and a result as
  ``<tool_result>TEXT</tool_result>`` (an image as ``[image]``). Thinking
  blocks are dropped.
- **Dropped events:** ``system`` and ``summary`` events, ``isMeta`` user
  messages, local slash-command messages (``<command-name>``,
  ``<local-command-stdout>``) and API error messages. None of them is model
  output, and Claude Code does not send most of them to the model.
- **Merging:** Claude Code logs each block of a model response as its own
  event; consecutive messages with the same role are merged with a blank
  line, so one assistant message is one model request with the full thread
  history before it. Response ids are not used: some backends reuse one id
  across separate responses. A thread must start with a user message; a final
  unanswered user message is dropped.

Not in the logs, so not replayed: Claude Code's system prompt and tool
definitions (about 12K tokens shared by every session), and the timing of
sub-agents inside their parent session. Rows come out in a seeded random
order (the five models' runs of a task are adjacent in the file).
"""
import json
import random
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from client_simulator.schema import Conversation, Message

REVISION = "792c6d3221db4d9ed475702ed61a176dd8948152"
URL = ("https://huggingface.co/datasets/zai-org/CC-Bench-trajectories/"
       f"resolve/{REVISION}/train.parquet")
LOCAL_COMMANDS = ("<command-name>", "<command-message>",
                  "<local-command-stdout>")

Event = Dict[str, Any]


def check_download(raw_path: Path) -> None:
    if raw_path.is_file():
        print(f"Using existing dataset: {raw_path}")
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(raw_path.name + ".download")
    print(f"Downloading CC-Bench trajectories to {raw_path}...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as source, \
                temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        temporary.replace(raw_path)
    finally:
        temporary.unlink(missing_ok=True)


def threads(events: Iterable[Event]) -> List[List[Event]]:
    """Split a session log into threads that each have their own context."""
    thread_of: Dict[str, int] = {}
    result: List[List[Event]] = []
    for event in events:
        uuid = event.get("uuid")
        if uuid is None:  # "summary" rows are not part of any context
            continue
        parent = event.get("parentUuid")
        if parent in thread_of:
            index = thread_of[parent]
        else:
            index = len(result)
            result.append([])
        thread_of[uuid] = index
        result[index].append(event)
    return result


def render_block(block: Dict[str, Any]) -> Optional[str]:
    kind = block.get("type")
    if kind == "text":
        return block.get("text") or None
    if kind == "tool_use":
        arguments = json.dumps(block.get("input"), ensure_ascii=False)
        return f'<tool_use name="{block.get("name")}">{arguments}</tool_use>'
    if kind == "tool_result":
        return f"<tool_result>{render_content(block.get('content'))}" \
               "</tool_result>"
    if kind == "image":
        return "[image]"
    return None  # thinking and unknown blocks


def render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = (render_block(b) for b in content if isinstance(b, dict))
        return "\n\n".join(p for p in parts if p)
    return ""


def message(event: Event, counts: Counter) -> Optional[Message]:
    """The event's message as replayed, or None if it is dropped."""
    kind = event.get("type")
    if kind not in ("user", "assistant"):
        counts[f"dropped_{kind}_events"] += 1
        return None
    if event.get("isMeta") or event.get("isApiErrorMessage"):
        counts["dropped_meta_or_api_error"] += 1
        return None
    content = (event.get("message") or {}).get("content")
    if isinstance(content,
                  str) and content.lstrip().startswith(LOCAL_COMMANDS):
        counts["dropped_local_commands"] += 1
        return None
    text = render_content(content)
    if not text:
        counts["dropped_empty"] += 1
        return None
    return Message(role=kind, content=text)


def thread_conversation(conversation_id: str, events: List[Event],
                        counts: Counter) -> Optional[Conversation]:
    messages: List[Message] = []
    for event in events:
        current = message(event, counts)
        if current is None:
            continue
        if messages and messages[-1].role == current.role:
            messages[-1].content += "\n\n" + current.content
        else:
            messages.append(current)
    while messages and messages[0].role != "user":
        messages.pop(0)
        counts["leading_assistant_messages_removed"] += 1
    if messages and messages[-1].role == "user":
        messages.pop()
        counts["unanswered_final_users_removed"] += 1
    if not messages:
        counts["skipped_threads_without_exchange"] += 1
        return None
    return Conversation(conversation_id=conversation_id, messages=messages)


def conversations(rows: Iterable[Dict[str, Any]],
                  seed: int = 0) -> Iterator[Conversation]:
    """Convert dataset rows (``id``, ``trajectory``), in a seeded order."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    counts: Counter = Counter()
    for row in rows:
        counts["source_trajectories"] += 1
        for index, events in enumerate(threads(json.loads(row["trajectory"]))):
            result = thread_conversation(f"ccbench:{row['id']}:{index}",
                                         events, counts)
            if result is not None:
                counts["retained_conversations"] += 1
                counts["replay_requests"] += len(result.messages) // 2
                yield result
    print("CC-Bench: " + json.dumps(dict(counts), sort_keys=True))


def read_rows(raw_path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(raw_path, columns=["id", "trajectory"]).to_pylist()


def load_conversations(raw_path: Path,
                       seed: int = 0) -> Iterator[Conversation]:
    yield from conversations(read_rows(raw_path), seed)
