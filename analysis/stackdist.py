"""Reuse (stack) distances of an access stream, and LRU's hit ratio from them.

An access's reuse distance is the number of distinct other keys accessed
since the previous access of the same key (Mattson et al., 1970). An LRU
cache of ``C`` keys holds exactly the ``C`` most recently used ones, so an
access hits it if and only if its distance is below ``C``. One pass over a
stream therefore gives LRU's hit ratio at every capacity at once, which is
how M3 Task 2 asks how much capacity each workload's reuse needs.

The stream is whatever the caller passes (for example the content hashes of
``analysis/tables.py``'s lookups). vLLM's cache is LRU only over blocks in
the free pool: blocks in use cannot be evicted and count against the
capacity, so this is a guide, not a replay (``analysis/cachesim.py`` is).
"""
from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence, np.ndarray]


def previous_positions(keys: ArrayLike) -> np.ndarray:
    """For each position, the position of the previous access of the same
    key, or -1."""
    keys = np.asarray(keys)
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    previous = np.full(len(keys), -1, dtype=np.int64)
    repeat = np.flatnonzero(ordered[1:] == ordered[:-1]) + 1
    previous[order[repeat]] = order[repeat - 1]
    return previous


def reuse_distances(keys: ArrayLike) -> np.ndarray:
    """Each access's reuse distance: distinct other keys accessed since the
    previous access of its key; -1 for a key's first access.

    A Fenwick tree over positions marks the latest access of every key, so
    the distance is the number of marks after the previous access:
    O(N log N).
    """
    previous = previous_positions(keys).tolist()
    n = len(previous)
    tree = [0] * (n + 1)
    marks = 0
    out = [-1] * n
    for i, p in enumerate(previous):
        if p >= 0:
            # Marks at positions up to p (inclusive), then drop p's mark.
            below, j = 0, p + 1
            while j:
                below += tree[j]
                j &= j - 1
            out[i] = marks - below
            j = p + 1
            while j <= n:
                tree[j] -= 1
                j += j & -j
            marks -= 1
        j = i + 1
        while j <= n:
            tree[j] += 1
            j += j & -j
        marks += 1
    return np.asarray(out, dtype=np.int64)


def lru_hit_ratio(
        distances: ArrayLike,
        capacity: Union[int, ArrayLike]) -> Union[None, float, np.ndarray]:
    """Share of accesses an LRU cache of `capacity` keys would hit: those
    with ``0 <= distance < capacity``. First accesses (-1) always miss.

    `capacity` may be an array, for a whole curve at once (then an array
    is returned). None for an empty stream.
    """
    distances = np.asarray(distances)
    if not len(distances):
        return None
    reused = np.sort(distances[distances >= 0])
    hits = np.searchsorted(reused, np.asarray(capacity), side="left")
    ratio = hits / len(distances)
    return float(ratio) if np.ndim(ratio) == 0 else ratio
