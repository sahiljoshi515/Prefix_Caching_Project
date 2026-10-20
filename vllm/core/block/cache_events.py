"""Opt-in log of prefix-cache events (COMP 536 project).

When ``VLLM_CACHE_EVENT_LOG`` names a file, the GPU
``PrefixCachingBlockAllocator`` writes one JSON line per cache event:

    {"t": <time.monotonic()>, "ev": <kind>, "hash": <content hash>,
     "block": <physical block id>, "req": <request id>, "op": <operation>}

``ev`` is ``hit`` or ``miss`` (a lookup of a full block), ``insert`` (a
newly computed block becomes cached), ``dedupe`` (a block filled during
decode already existed), ``free`` (the last user released a cached block,
which enters the evictor), ``revive`` (a freed block is used again) or
``evict``. ``op`` is the block-manager call the event happened in:
``allocate`` (prompt), ``append_slots`` (decode), ``free``, ``swap_in`` or
``swap_out``. The first line is a header with ``time.time()`` and
``time.monotonic()``; ``close()`` writes a final ``{"ev": "end"}`` line.

Unset (the default), no recorder exists and every hook is a single ``is
None`` check. See ``specs/004-m2-prefix-sharing-analysis``.
"""
import atexit
import functools
import json
import os
import time
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

import vllm.envs as envs

# Buffered lines are written every FLUSH_EVENTS events or FLUSH_SECONDS,
# whichever comes first, so a killed engine loses at most about a second.
FLUSH_EVENTS = 4096
FLUSH_SECONDS = 1.0

_Event = Tuple[float, str, Optional[int], Optional[int], Optional[str],
               Optional[str]]
_F = TypeVar("_F", bound=Callable)


class CacheEventRecorder:
    """Buffered JSONL writer for the cache events of one engine process."""

    def __init__(self, path: str, block_size: int) -> None:
        self.path = path
        # Stays open for the engine's lifetime; close() releases it.
        self._file = open(path, "w", encoding="utf-8")  # noqa: SIM115
        self._buffer: List[_Event] = []
        self._last_flush = time.monotonic()
        # Set by the block manager around each call (see `attributed`).
        self.request_id: Optional[str] = None
        self.op: Optional[str] = None
        # Sequences carry no request id, so `allocate` records it here.
        self._seq_requests: Dict[int, str] = {}
        self.closed = False
        header = {
            "ev": "header",
            "time": time.time(),
            "t": self._last_flush,
            "pid": os.getpid(),
            "block_size": block_size,
        }
        self._file.write(json.dumps(header) + "\n")
        self._file.flush()
        atexit.register(self.close)

    def log(self, ev: str, content_hash: Optional[int],
            block_id: Optional[int]) -> None:
        now = time.monotonic()
        self._buffer.append(
            (now, ev, content_hash, block_id, self.request_id, self.op))
        if (len(self._buffer) >= FLUSH_EVENTS
                or now - self._last_flush >= FLUSH_SECONDS):
            self.flush()

    def flush(self) -> None:
        if self.closed:
            return
        self._file.writelines(
            json.dumps({
                "t": t,
                "ev": ev,
                "hash": content_hash,
                "block": block_id,
                "req": req,
                "op": op,
            }) + "\n"
            for t, ev, content_hash, block_id, req, op in self._buffer)
        self._file.flush()
        self._buffer.clear()
        self._last_flush = time.monotonic()

    def close(self) -> None:
        if self.closed:
            return
        self.flush()
        self._file.write(json.dumps({"ev": "end"}) + "\n")
        self._file.close()
        self.closed = True
        if _recorders.get(self.path) is self:
            del _recorders[self.path]

    @contextmanager
    def attribute(self, target, op: str) -> Iterator[None]:
        """Attribute events to `target`'s request while `op` runs.

        `target` is a SequenceGroup (``allocate``, ``swap_in``,
        ``swap_out``) or a Sequence (``append_slots``, ``free``).
        """
        request_id = getattr(target, "request_id", None)
        if request_id is None:
            request_id = self._seq_requests.get(target.seq_id)
        elif op == "allocate":
            for seq in target.get_seqs():
                self._seq_requests[seq.seq_id] = request_id
        previous = self.request_id, self.op
        self.request_id, self.op = request_id, op
        try:
            yield
        finally:
            self.request_id, self.op = previous
            if op == "free":
                self._seq_requests.pop(target.seq_id, None)


# One recorder per path and process: the block manager and its GPU
# allocator are built separately but must share it.
_recorders: Dict[str, CacheEventRecorder] = {}


def recorder_from_env(block_size: int) -> Optional[CacheEventRecorder]:
    path = envs.VLLM_CACHE_EVENT_LOG
    if not path:
        return None
    recorder = _recorders.get(path)
    if recorder is None:
        recorder = _recorders[path] = CacheEventRecorder(path, block_size)
    return recorder


def close_all() -> None:
    for recorder in list(_recorders.values()):
        recorder.close()


def attributed(op: str) -> Callable[[_F], _F]:
    """Decorate a block-manager method so its cache events name a request.

    The method's first argument is the SequenceGroup or Sequence; the
    manager's recorder is ``self._events`` (None when logging is off).
    """

    def decorate(method: _F) -> _F:

        @functools.wraps(method)
        def wrapper(manager, target, *args, **kwargs):
            events = manager._events
            if events is None:
                return method(manager, target, *args, **kwargs)
            with events.attribute(target, op):
                return method(manager, target, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
