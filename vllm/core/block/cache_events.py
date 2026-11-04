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
``swap_out``. A ``free`` line also has ``"la"``, the block's last-access
time as the evictor sees it (the scheduler's ``time.time()``), so the
evictor's order can be replayed from the log. The first line is a header
with ``time.time()`` and ``time.monotonic()``; ``close()`` writes a final
``{"ev": "end"}`` line.

A path ending in ``.gz`` is written gzipped. Unset (the default), no
recorder exists and every hook is a single ``is
None`` check. See ``specs/004-m2-prefix-sharing-analysis``.
"""
import atexit
import functools
import gzip
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
               Optional[str], Optional[float]]
_F = TypeVar("_F", bound=Callable)


class CacheEventRecorder:
    """Buffered JSONL writer for the cache events of one engine process."""

    def __init__(self, path: str, block_size: int) -> None:
        self.path = path
        # Stays open for the engine's lifetime; close() releases it. A path
        # ending in .gz is written gzipped (long runs log hundreds of MB).
        opener = gzip.open if path.endswith(".gz") else open
        self._file = opener(path, "wt", encoding="utf-8")  # noqa: SIM115
        self._buffer: List[_Event] = []
        self._last_flush = time.monotonic()
        # The block manager names the running request in the module's
        # request context (see `attributed`); the recorder reads it.
        enable_request_context()
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

    def log(self,
            ev: str,
            content_hash: Optional[int],
            block_id: Optional[int],
            la: Optional[float] = None) -> None:
        now = time.monotonic()
        self._buffer.append((now, ev, content_hash, block_id,
                             _context.request_id, _context.op, la))
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
                **({} if la is None else {
                       "la": la
                   }),
            }) + "\n"
            for t, ev, content_hash, block_id, req, op, la in self._buffer)
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

    # The request context's fields, kept here for code that sets them on
    # the recorder (the cache simulator).
    @property
    def request_id(self) -> Optional[str]:
        return _context.request_id

    @request_id.setter
    def request_id(self, value: Optional[str]) -> None:
        _context.request_id = value

    @property
    def op(self) -> Optional[str]:
        return _context.op

    @op.setter
    def op(self, value: Optional[str]) -> None:
        _context.op = value


class RequestContext:
    """The request the block manager is serving and the call it is in.

    The event log names each event's request from it, and an evictor that
    labels blocks by workload (spec 007 FR-002) reads it when a block is
    freed. It is off until a recorder or such an evictor enables it, so
    vLLM does nothing extra by default.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.request_id: Optional[str] = None
        self.op: Optional[str] = None
        # Sequences carry no request id, so `allocate` records it here.
        self._seq_requests: Dict[int, str] = {}

    @contextmanager
    def attribute(self, target, op: str) -> Iterator[None]:
        """Name `target`'s request while `op` runs.

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


_context = RequestContext()


def enable_request_context() -> None:
    _context.enabled = True


def current_request() -> Optional[str]:
    """The engine id of the request being served, or None outside a block
    manager call or while the context is off."""
    return _context.request_id


def set_current_request(request_id: Optional[str],
                        op: Optional[str] = None) -> None:
    """Name the running request without a block manager (simulators,
    tests)."""
    _context.enabled = True
    _context.request_id, _context.op = request_id, op


ENGINE_PREFIXES = ("chatcmpl-", "cmpl-")


def workload_label(engine_request_id: Optional[str]) -> Optional[str]:
    """A request's workload tag: the client id's text before its first
    colon (spec 006 FR-002), after the server's ``chatcmpl-``/``cmpl-``
    prefix. None for None."""
    if engine_request_id is None:
        return None
    for prefix in ENGINE_PREFIXES:
        if engine_request_id.startswith(prefix):
            engine_request_id = engine_request_id[len(prefix):]
            break
    return engine_request_id.split(":", 1)[0]


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
    """Decorate a block-manager method so the request context names its
    request while it runs (for the event log and for label-reading
    evictors).

    The method's first argument is the SequenceGroup or Sequence. With the
    context off (no recorder, no such evictor) the method runs unchanged.
    """

    def decorate(method: _F) -> _F:

        @functools.wraps(method)
        def wrapper(manager, target, *args, **kwargs):
            if not _context.enabled:
                return method(manager, target, *args, **kwargs)
            with _context.attribute(target, op):
                return method(manager, target, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
