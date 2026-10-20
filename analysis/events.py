"""Load the cache event log, client results and Prometheus snapshots."""
import gzip
import json
from pathlib import Path
from typing import (Any, Dict, Iterator, List, NamedTuple, Optional, Tuple,
                    Union)

PathLike = Union[str, Path]

ENGINE_CHAT_PREFIX = "chatcmpl-"


class Event(NamedTuple):
    """One line of the log (``vllm/core/block/cache_events.py``)."""
    t: float
    ev: str
    hash: int  # never null for block events
    block: Optional[int]
    req: Optional[str]
    op: Optional[str]


class EventLog(NamedTuple):
    header: Dict[str, Any]
    events: List[Event]
    # False when the final {"ev": "end"} line is missing: the engine was
    # killed before closing the log, so its tail may be lost.
    complete: bool


def _lines(path: PathLike) -> Iterator[str]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        yield from f


def load_events(path: PathLike) -> EventLog:
    header: Dict[str, Any] = {}
    events: List[Event] = []
    complete = False
    for line in _lines(path):
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row["ev"]
        if kind == "header":
            header = row
        elif kind == "end":
            complete = True
        else:
            events.append(
                Event(row["t"], kind, row["hash"], row["block"], row["req"],
                      row["op"]))
    return EventLog(header, events, complete)


def load_results(path: PathLike) -> List[Dict[str, Any]]:
    """Read the client simulator's per-request JSONL output."""
    return [json.loads(line) for line in _lines(path) if line.strip()]


def parse_request_id(request_id: Optional[str]) -> Optional[Tuple[str, int]]:
    """``chatcmpl-<conversation_id>:<turn>`` -> (conversation_id, turn).

    The client sends ``<conversation_id>:<turn>`` as X-Request-Id and the
    chat endpoint adds the prefix. Returns None for other ids.
    """
    if request_id is None:
        return None
    if request_id.startswith(ENGINE_CHAT_PREFIX):
        request_id = request_id[len(ENGINE_CHAT_PREFIX):]
    conversation, _, turn = request_id.rpartition(":")
    if not conversation or not turn.isdigit():
        return None
    return conversation, int(turn)


def client_request_id(engine_request_id: str) -> str:
    """Engine request id -> the id the client sent."""
    if engine_request_id.startswith(ENGINE_CHAT_PREFIX):
        return engine_request_id[len(ENGINE_CHAT_PREFIX):]
    return engine_request_id


def read_metric(path: PathLike, name: str) -> Optional[float]:
    """First sample of metric `name` in a Prometheus text snapshot."""
    for line in _lines(path):
        if line.startswith(name) and line[len(name)] in "{ ":
            return float(line.rsplit(" ", 1)[1])
    return None
