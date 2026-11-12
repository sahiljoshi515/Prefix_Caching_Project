"""Belady's offline optimum (MIN) as a cache-simulator evictor (spec 007).

MIN evicts the cached block whose next use is farthest in the future. On a
fixed sequence of lookups no eviction policy has fewer misses (Belady 1966;
Mattson et al. 1970), so MIN's reuse at a capacity bounds what any policy
could reuse there: the headroom in the policy figures. It needs the future,
so no server can run it. ``analysis/cachesim.py`` can, because its request
timing does not depend on the policy (compute is skipped), so every prompt
and its block hashes are known before the run. The evictor lives here and
is never shipped in ``vllm/``.

A *lookup* of a content hash is a hit or a miss on it while a request's
prompt is allocated. The simulator allocates prompts by start time, ties in
plan order, and a prompt that waits for free blocks keeps its place (waiting
prompts are allocated first come, first served). So the order of lookups is
fixed even when waits shift their times; the oracle numbers allocations in
that order and keys on the number, not the time, and the simulator checks
each allocation against it. MIN's optimality carries over to blocks in use
and blocks written while decoding, which are the same under every policy,
with one exception: a decode block whose content is already cached
(``dedupe``) takes the cached copy without a lookup, which MIN does not
foresee. The balanced-human-s1 ceiling run has 3 among 267,570 decode
blocks.

Why MIN keeps prefix closure (a cached block's ancestors are cached, so a
prompt's hits form a prefix and are all usable). Let D be a cached
descendant of block A; MIN never evicts A:

1. If D is in use, so is A, and only free blocks are evicted: a request
   holding a block holds its whole chain, and each full block of a block
   table is the cached copy of its hash (a hit, an insert or a dedupe).
2. If D is free, it sorts before A. Allocation looks up every full block of
   a prompt in chain order, and a prompt that contains D contains A, so
   every allocation that looks up D looks up A first: next_lookup(A) <=
   next_lookup(D) at every point. A free block's key, computed when it
   enters the free pool, stays exact while it is there, because the next
   lookup of its hash hits it. A has fewer hashed tokens than D, so D sorts
   first (next use farther or equal, then deeper).

A block is inserted only after its ancestors are in the inserting request's
table, so inserts keep closure too, and by induction over events MIN never
has a hit after a miss.
"""
import bisect
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from analysis.cachesim import Timing, block_hashes
from client_simulator.schema import TokenRequest
from vllm.core.eviction_policies import _FreeBlock, _Key, _KeyedEvictor

# The next lookup of a hash that is never looked up again.
NEVER = math.inf


@dataclass
class FutureLookups:
    """Every prompt allocation of a run, numbered in the order the simulator
    makes them, and the allocations that look up each content hash."""
    order: List[str]  # request ids, in allocation order
    times: List[float]  # the planned start of each
    lookups: Dict[int, List[int]]  # content hash -> sorted allocations
    position: int = 0  # allocations made so far

    def next_lookup(self,
                    content_hash: int,
                    position: Optional[int] = None) -> float:
        """The number of the first allocation from `position` on (default:
        the next one) that looks up `content_hash`; NEVER if none does."""
        start = self.position if position is None else position
        found = self.lookups.get(content_hash, ())
        i = bisect.bisect_left(found, start)
        return found[i] if i < len(found) else NEVER

    def time(self, position: float) -> float:
        """The planned start of allocation `position`; inf for NEVER."""
        return math.inf if position == NEVER else self.times[int(position)]

    def allocated(self, request_id: str) -> None:
        """Count `request_id`'s prompt as allocated, so its lookups are past.

        Raises if it is not the allocation expected next: every key would
        then be wrong.
        """
        expected = (self.order[self.position]
                    if self.position < len(self.order) else None)
        if request_id != expected:
            raise RuntimeError(f"Belady oracle expected {expected!r} to be "
                               f"allocated next, not {request_id!r}")
        self.position += 1


def future_lookups(requests: List[TokenRequest], timing: Dict[str, Timing],
                   block_size: int) -> FutureLookups:
    """The lookups of a simulator run of `requests` with `timing`, hashed as
    the simulator hashes prompts. A request with no timing is never sent and
    has no lookups."""
    sent = [r for r in requests if r.request_id in timing]
    # Stable sort: equal starts stay in plan order, as in cachesim's queue.
    sent.sort(key=lambda r: timing[r.request_id].start)
    lookups: Dict[int, List[int]] = defaultdict(list)
    for position, request in enumerate(sent):
        for content_hash in block_hashes(request.prompt_token_ids, block_size):
            found = lookups[content_hash]
            if not found or found[-1] != position:
                found.append(position)
    return FutureLookups([r.request_id for r in sent],
                         [timing[r.request_id].start for r in sent],
                         dict(lookups))


class BeladyEvictor(_KeyedEvictor):
    """Evicts the free block whose hash is looked up again last (never
    counts as last); ties go to the deeper block, then the larger block id.

    Uses the lazy heap of ``vllm/core/eviction_policies.py``. A block's key
    does not depend on its last access, so ``update`` pushes nothing new.
    """

    def __init__(self, lookups: Optional[FutureLookups] = None) -> None:
        super().__init__()
        self.lookups = lookups

    def _key(self, block_id: int, block: _FreeBlock) -> _Key:
        if self.lookups is None:
            raise RuntimeError("BeladyEvictor needs the run's lookups")
        # The heap pops the smallest key: negate, so the farthest next
        # lookup, then the deepest block, then the largest id goes first.
        return (-self.lookups.next_lookup(block.content_hash),
                -block.num_hashed_tokens, -block_id)
