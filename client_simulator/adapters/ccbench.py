"""CC-Bench Claude Code trajectories (spec 006, FR-001).

Source: https://huggingface.co/datasets/zai-org/CC-Bench-trajectories (MIT),
one parquet file with 370 rows (74 coding tasks x 5 models). Each row's
``trajectory`` is a Claude Code session log: a JSON list of events linked by
``uuid``/``parentUuid``.

Rules, applied per row:

- **Threads.** Events form a tree through ``parentUuid``. An event with no
  parent (or a parent not seen yet) is a root. Roots are session starts,
  sub-agent ("sidechain") starts, ``/compact`` boundaries and older
  "This session is being continued..." summaries, which replace the history
  with a summary, and ``/clear`` restarts. Each root's tree is one thread,
  ``ccbench:<row id>:<thread index>``. An event whose uuid was already seen
  is skipped: a resumed session copies the earlier history into the log.
- **Branches.** Parallel tool calls and their results are chained one after
  another, so a fork in the tree is a re-run: the same request was sent
  again (after a rewind or retry) and the log keeps the abandoned branch.
  Each request's history is the path from its root down to it. The first
  root-to-leaf path is the thread's conversation; each later path is a
  conversation ``<thread id>b<k>`` whose turns shared with an earlier path
  (the same events) are history only (``first_turn``), so no request is
  replayed twice.
- **Roles** (spec 006, plan "Role mapping for agent trajectories"): the
  human's prompts and tool results become ``user``; the model's text and
  tool calls become ``assistant``. A tool call is written as
  ``<tool_use name="NAME">INPUT_JSON</tool_use>`` and a result as
  ``<tool_result>TEXT</tool_result>`` (an image as ``[image]``). Thinking
  blocks are dropped.
- **Dropped events:** ``system`` and ``summary`` events, ``isMeta`` user
  messages, local slash-command messages (``<command-name>``,
  ``<local-command-stdout>``, as a string or a list of blocks), API error
  messages and other messages from the ``<synthetic>`` model. None of them
  is model output, and Claude Code does not send most of them to the
  model.
- **Merging:** Claude Code logs each block of a model response as its own
  event; consecutive messages with the same role are merged with a blank
  line, so one assistant message is one model request with the full history
  before it. Response ids are not used: some backends reuse one id across
  separate responses. A path must start with a user message; a final
  unanswered user message is dropped.

Not in the logs, so not replayed: Claude Code's system prompt and tool
definitions (about 12K tokens shared by every session; the sessions had
them, the logs do not), and the timing of
sub-agents inside their parent session. Rows come out in a seeded random
order (the five models' runs of a task are adjacent in the file).
"""
import json
import random
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from client_simulator.schema import Conversation, Message

REVISION = "792c6d3221db4d9ed475702ed61a176dd8948152"
URL = ("https://huggingface.co/datasets/zai-org/CC-Bench-trajectories/"
       f"resolve/{REVISION}/train.parquet")
LOCAL_COMMANDS = ("<command-name>", "<command-message>",
                  "<local-command-stdout>")

Event = Dict[str, Any]
Children = Dict[str, List[Event]]


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


def tree(events: Iterable[Event],
         counts: Counter) -> Tuple[List[Event], Children]:
    """The roots and each event's children, in file order."""
    roots: List[Event] = []
    children: Children = {}
    for event in events:
        uuid = event.get("uuid")
        if uuid is None:  # "summary" rows are not part of any context
            continue
        if uuid in children:  # a resumed session's copy of earlier history
            counts["duplicate_events_skipped"] += 1
            continue
        parent = event.get("parentUuid")
        if parent in children:
            children[parent].append(event)
        else:
            roots.append(event)
        children[uuid] = []
    return roots, children


def paths(root: Event, children: Children) -> Iterator[List[Event]]:
    """Every root-to-leaf path, branches in file order (no recursion: a
    session can be over a thousand events deep)."""
    stack = [(root, 0)]
    path: List[Event] = []
    while stack:
        event, depth = stack.pop()
        del path[depth:]
        path.append(event)
        below = children[event["uuid"]]
        if not below:
            yield list(path)
        stack.extend((child, depth + 1) for child in reversed(below))


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
    body = event.get("message") or {}
    if (event.get("isMeta") or event.get("isApiErrorMessage")
            or body.get("model") == "<synthetic>"):
        counts["dropped_meta_or_api_error"] += 1
        return None
    text = render_content(body.get("content"))
    if text.lstrip().startswith(LOCAL_COMMANDS):
        counts["dropped_local_commands"] += 1
        return None
    if not text:
        counts["dropped_empty"] += 1
        return None
    return Message(role=kind, content=text)


Turn = Tuple[Tuple[str, ...], Tuple[str, ...]]


def path_messages(path: List[Event], rendered: Dict[str, Optional[Message]],
                  counts: Counter) -> Tuple[List[Message], List[Turn]]:
    """A path's messages and, for each turn, the uuids of its events."""
    messages: List[Message] = []
    sources: List[List[str]] = []
    for event in path:
        if event["uuid"] not in rendered:  # each event is counted once
            rendered[event["uuid"]] = message(event, counts)
        current = rendered[event["uuid"]]
        if current is None:
            continue
        if messages and messages[-1].role == current.role:
            messages[-1] = Message(
                current.role, messages[-1].content + "\n\n" + current.content)
            sources[-1].append(event["uuid"])
        else:
            messages.append(current)
            sources.append([event["uuid"]])
    while messages and messages[0].role != "user":
        messages.pop(0)
        sources.pop(0)
        counts["leading_assistant_messages_removed"] += 1
    if messages and messages[-1].role == "user":
        messages.pop()
        sources.pop()
        counts["unanswered_final_users_removed"] += 1
    turns = [(tuple(sources[i]), tuple(sources[i + 1]))
             for i in range(0, len(sources), 2)]
    return messages, turns


def shared_turns(turns: List[Turn], earlier: List[List[Turn]]) -> int:
    """How many leading turns an earlier path already replayed."""
    best = 0
    for other in earlier:
        same = 0
        while (same < min(len(turns), len(other))
               and turns[same] == other[same]):
            same += 1
        best = max(best, same)
    return best


def row_conversations(row: Dict[str, Any],
                      counts: Counter) -> Iterator[Conversation]:
    roots, children = tree(json.loads(row["trajectory"]), counts)
    rendered: Dict[str, Optional[Message]] = {}
    for index, root in enumerate(roots):
        thread_id = f"ccbench:{row['id']}:{index}"
        earlier: List[List[Turn]] = []
        for path in paths(root, children):
            messages, turns = path_messages(path, rendered, counts)
            first = shared_turns(turns, earlier)
            if first == len(turns):
                counts["paths_without_new_turns"] += 1
                continue
            branch = len(earlier)
            earlier.append(turns)
            if branch:
                counts["rerun_branches"] += 1
            counts["replay_requests"] += len(turns) - first
            yield Conversation(conversation_id=thread_id +
                               (f"b{branch}" if branch else ""),
                               messages=messages,
                               first_turn=first)


def conversations(rows: Iterable[Dict[str, Any]],
                  seed: int = 0) -> Iterator[Conversation]:
    """Convert dataset rows (``id``, ``trajectory``), in a seeded order."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    counts: Counter = Counter()
    for row in rows:
        counts["source_trajectories"] += 1
        for conversation in row_conversations(row, counts):
            counts["retained_conversations"] += 1
            yield conversation
    print("CC-Bench: " + json.dumps(dict(counts), sort_keys=True))


def read_rows(raw_path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(raw_path, columns=["id", "trajectory"]).to_pylist()


def load_conversations(raw_path: Path,
                       seed: int = 0) -> Iterator[Conversation]:
    yield from conversations(read_rows(raw_path), seed)
