"""RACE-RP: the access history, the RACE sketch and the evictor
(specs/007-m3-mixed-workload-policy).

Checks are written against the policy's mathematical definition, not its
code: features, buckets, sketch counts and scores are worked out by hand
or recomputed in plain Python, and a randomized prefix-closed workload
checks every victim against a brute-force reference that recomputes each
candidate's features from its own full access log and takes the spec's
argmin from scratch.
"""
import json
import math
import random
from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from vllm.core import reuse_evictors
from vllm.core.block import cache_events
from vllm.core.reuse_evictors import (FEATURES, GHOST_S, LONG_S, NUM_FEATURES,
                                      SHORT_S, STD_FLOOR, AccessHistory,
                                      RaceRPEvictor, RaceSketch)

pytestmark = pytest.mark.m3

# The specification's constants, restated so the tests do not read them
# from the module under test.
SPEC_SHORT, SPEC_LONG, SPEC_GHOST, SPEC_ALPHA = 10.0, 120.0, 180.0, 5.0
# A std at or below this counts as zero (a constant column's computed std
# can be ~1e-16 rather than 0), and zero is read as 1.
SPEC_STD_FLOOR = 1e-9
# Feature indices of x = [f10, f120, idle, depth, family] (each log1p).
F10, F120, IDLE, DEPTH, FAMILY = range(5)
A, B, C, D = 101, 102, 103, 104  # content hashes for AccessHistory tests

Path = Tuple[int, ...]


@pytest.fixture(autouse=True)
def _restore_request_context(monkeypatch):
    """Every evictor turns the block manager's request context on; put it
    back so other tests see vLLM's default."""
    monkeypatch.setattr(cache_events._context, "enabled",
                        cache_events._context.enabled)


class FakeClock:

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def log1p_all(*values: float) -> Tuple[float, ...]:
    return tuple(math.log1p(v) for v in values)


# --- The spec's sketch, in plain Python -------------------------------------


def spec_buckets(planes: Sequence,
                 mean: Sequence[float],
                 std: Sequence[float],
                 x: Sequence[float],
                 floor: float = SPEC_STD_FLOOR) -> Tuple[int, ...]:
    """h_l(x) = sum_{j=1..P} 2^(j-1) [(R_l z)_j >= 0] for every table, z the
    standardized x (a std <= `floor` read as 1) scaled to unit length (or
    0)."""
    z = [(xi - m) / (s if s > floor else 1.0)
         for xi, m, s in zip(x, mean, std)]
    norm = math.sqrt(sum(v * v for v in z))
    z = [v / norm for v in z] if norm > 0 else [0.0] * len(z)
    return tuple(
        sum(2**(j - 1) for j, row in enumerate(table, start=1)
            if sum(r * v for r, v in zip(row, z)) >= 0) for table in planes)


class SpecSketch:
    """N_l and W_l built from scratch, and S = (W + alpha r_bar)/(N + alpha)
    with N, W the means over tables at the candidate's buckets."""

    def __init__(self,
                 planes,
                 mean,
                 std,
                 r_bar: float = 0.0,
                 alpha: float = SPEC_ALPHA) -> None:
        self.planes = np.asarray(planes, dtype=np.float64).tolist()
        self.mean = [float(v) for v in mean]
        self.std = [float(v) for v in std]
        self.L, self.P = len(self.planes), len(self.planes[0])
        self.N = [[0.0] * 2**self.P for _ in range(self.L)]
        self.W = [[0.0] * 2**self.P for _ in range(self.L)]
        self.r_bar, self.alpha = r_bar, alpha

    def buckets(self, x: Sequence[float]) -> Tuple[int, ...]:
        return spec_buckets(self.planes, self.mean, self.std, x)

    def add(self, x: Sequence[float], r: float) -> None:
        for table, k in enumerate(self.buckets(x)):
            self.N[table][k] += 1
            self.W[table][k] += r

    def query(self, x: Sequence[float]) -> Tuple[float, float]:
        cells = list(enumerate(self.buckets(x)))
        return (sum(self.N[t][k] for t, k in cells) / self.L,
                sum(self.W[t][k] for t, k in cells) / self.L)

    def score(self, x: Sequence[float]) -> float:
        n, w = self.query(x)
        return (w + self.alpha * self.r_bar) / (n + self.alpha)


def training_data(seed: int, n: int = 300, constant_f10: bool = False):
    """Random feature vectors shaped like real ones, with 0/1 rewards that
    rise with f120 (integer counts and rewards keep sums exact)."""
    rng = np.random.default_rng(seed + 1000)
    f120 = rng.integers(0, 9, n)
    f10 = (np.zeros(n, dtype=int) if constant_f10 else np.minimum(
        f120, rng.integers(0, 4, n)))
    raw = np.stack([
        f10, f120,
        rng.integers(0, 300, n),
        rng.integers(1, 6, n),
        rng.integers(0, 12, n)
    ],
                   axis=1)
    x = np.log1p(raw.astype(np.float64))
    r = (rng.random(n) < 0.15 + 0.1 * f120).astype(np.float64)
    return x, r


def trained(L: int, P: int, seed: int, constant_f10: bool = False):
    """A sketch trained on random data, and the same sketch built by the
    spec (planes drawn independently from the same seed)."""
    x, r = training_data(seed, constant_f10=constant_f10)
    mean, std = x.mean(axis=0), x.std(axis=0)
    r_bar = sum(r.tolist()) / len(r)
    sketch = RaceSketch.random(L, P, seed, mean, std, r_bar=r_bar)
    sketch.add(x, r)
    spec = SpecSketch(
        np.random.default_rng(seed).standard_normal((L, P, 5)), mean, std,
        r_bar)
    for xi, ri in zip(x.tolist(), r.tolist()):
        spec.add(xi, ri)
    return sketch, spec


def family_depth_sketch(seed: int):
    """Scores that depend only on the family count and depth, at many
    levels: each plane tests log1p(family) >= c log1p(depth) for its own
    c, and the counts are random integers. Returns the sketch and the
    same sketch as a SpecSketch."""
    rng = np.random.default_rng(seed)
    L, P = 8, 3
    planes = np.zeros((L, P, NUM_FEATURES))
    planes[:, :, FAMILY] = 1.0
    planes[:, :, DEPTH] = -rng.uniform(0.2, 2.5, (L, P))
    counts = rng.integers(1, 20, (L, 2**P)).astype(np.float64)
    rewards = np.floor(rng.random((L, 2**P)) * (counts + 1))
    mean, std = np.zeros(NUM_FEATURES), np.ones(NUM_FEATURES)
    sketch = RaceSketch(planes, mean, std, counts, rewards, r_bar=0.3)
    spec = SpecSketch(planes, mean, std, 0.3)
    spec.N, spec.W = counts.tolist(), rewards.tolist()
    return sketch, spec


def threshold_sketch(feature: int, value: float) -> RaceSketch:
    """One table, one plane on `feature`: bucket 1 when x[feature] >=
    log1p(value), else bucket 0. Ten observations per bucket with rewards
    0 and 10 and r_bar 0, so S = 0 below the threshold and 2/3 above."""
    planes = np.zeros((1, 1, NUM_FEATURES))
    planes[0, 0, feature] = 1.0
    mean = np.zeros(NUM_FEATURES)
    mean[feature] = math.log1p(value)
    return RaceSketch(planes,
                      mean,
                      np.ones(NUM_FEATURES),
                      counts=np.full((1, 2), 10.0),
                      rewards=np.array([[0.0, 10.0]]))


LOW, HIGH = 0.0, 2.0 / 3.0  # threshold_sketch scores


def f120_levels_sketch() -> RaceSketch:
    """For depth-1 blocks: S = 1/2 with one access in 120 s, 1/6 with two
    and 5/6 with three or more. Plane 1 tests f120 >= 2; plane 2 tests
    f120 >= 3 (the depth term, constant log 2, shifts its threshold)."""
    planes = np.zeros((1, 2, NUM_FEATURES))
    planes[0, 0, F120] = 1.0
    planes[0, 1, F120] = 1.0
    planes[0, 1, DEPTH] = -0.3 / math.log(2)
    mean = np.zeros(NUM_FEATURES)
    mean[F120] = math.log1p(1.5)
    return RaceSketch(planes,
                      mean,
                      np.ones(NUM_FEATURES),
                      counts=np.full((1, 4), 10.0),
                      rewards=np.array([[5.0, 0.0, 0.0, 10.0]]),
                      r_bar=0.5)


def empty_sketch(L: int = 4, P: int = 3, r_bar: float = 0.25) -> RaceSketch:
    """No training data: every block scores alpha r_bar / alpha."""
    return RaceSketch.random(L,
                             P,
                             0,
                             np.zeros(NUM_FEATURES),
                             np.ones(NUM_FEATURES),
                             r_bar=r_bar)


# --- A hand-driven evictor --------------------------------------------------


class Hand:
    """An evictor driven by hand. Blocks are named; `tree` gives each name
    its parent and physical block id. A request uses a block and all its
    ancestors, root first (reviving idle ones, as the allocator does:
    remove, then record_access), and `free` returns them to the evictor."""

    def __init__(self, sketch: RaceSketch,
                 tree: Dict[str, Tuple[Optional[str], int]]) -> None:
        self.clock = FakeClock()
        self.ev = RaceRPEvictor(sketch, clock=self.clock)
        self.tree = tree
        self.hashes = {name: 10_000 + i for i, name in enumerate(tree)}
        self.names = {h: name for name, h in self.hashes.items()}
        self.refs: Dict[str, int] = defaultdict(int)

    def path(self, name: str) -> List[str]:
        names: List[str] = []
        node: Optional[str] = name
        while node is not None:
            names.append(node)
            node = self.tree[node][0]
        return names[::-1]

    def use(self, name: str, t: Optional[float] = None) -> None:
        if t is not None:
            self.clock.t = t
        prev = None
        for depth, node in enumerate(self.path(name), start=1):
            block_id = self.tree[node][1]
            self.refs[node] += 1
            if self.refs[node] == 1 and block_id in self.ev:
                self.ev.remove(block_id)
            self.ev.record_access(block_id, self.hashes[node], depth, prev)
            prev = self.hashes[node]

    def free(self, name: str) -> None:
        for node in self.path(name):
            self.refs[node] -= 1
            if self.refs[node] == 0:
                self.ev.add(self.tree[node][1], self.hashes[node], 16,
                            self.clock.t)

    def request(self, name: str, t: Optional[float] = None) -> None:
        self.use(name, t)
        self.free(name)

    def evict(self, t: Optional[float] = None) -> str:
        if t is not None:
            self.clock.t = t
        block_id, content_hash = self.ev.evict()
        name = self.names[content_hash]
        assert self.tree[name][1] == block_id
        return name

    def features(self, name: str) -> Tuple[float, ...]:
        h = self.hashes[name]
        return self.ev.features(h, self.ev.blocks[h], self.clock.t)


def chain(prefix: str, n: int, first_id: int):
    """Blocks prefix1 -> prefix2 -> ... -> prefix<n>."""
    return {
        f"{prefix}{d}":
        (None if d == 1 else f"{prefix}{d - 1}", first_id + d - 1)
        for d in range(1, n + 1)
    }


# --- Module constants -------------------------------------------------------


def test_constants_match_the_spec():
    assert (SHORT_S, LONG_S, GHOST_S) == (SPEC_SHORT, SPEC_LONG, SPEC_GHOST)
    assert NUM_FEATURES == 5
    assert FEATURES == ("log1p_f10", "log1p_f120", "log1p_idle_s",
                        "log1p_depth", "log1p_family_120")
    assert RaceSketch(np.zeros((1, 1, 5)), np.zeros(5),
                      np.ones(5)).alpha == SPEC_ALPHA
    assert STD_FLOOR == SPEC_STD_FLOOR


# --- AccessHistory ----------------------------------------------------------


def test_count_window_is_half_open():
    hist = AccessHistory()
    for t in (0.0, 5.0, 10.0):
        hist.record(A, t)
    assert hist.count(A, 10.0, 10.0) == 2  # (0, 10]: 0 is exactly 10 s ago
    assert hist.count(A, 15.0, 10.0) == 1  # (5, 15]
    assert hist.count(A, 9.5, 10.0) == 2  # (-0.5, 9.5]
    assert hist.count(A, 130.0, 120.0) == 0  # (10, 130]
    assert hist.count(A, 129.5, 120.0) == 1
    assert hist.count(B, 10.0, 120.0) == 0  # never accessed


def test_record_prunes_to_the_window_before_the_latest_access():
    hist = AccessHistory()
    for t in (0.0, 60.0, 120.0):
        hist.record(A, t)
    # 0 is exactly 120 s before the latest access: it can never be counted.
    assert hist.times[A] == [60.0, 120.0]
    assert hist.last[A] == 120.0
    assert hist.num_times == 2
    assert hist.max_times == 2


@pytest.mark.parametrize("seed", range(3))
def test_pruning_never_changes_a_count(seed):
    rng = random.Random(seed)
    hist = AccessHistory()
    every: Dict[int, List[float]] = defaultdict(list)
    t, max_times = 0.0, 0
    for _ in range(500):
        t += rng.choice((0.0, 0.0, 0.5, 1.0, 3.0, 10.0, 40.0, 120.0))
        h = rng.randrange(4)
        hist.record(h, t)
        every[h].append(t)
        # Stored: the times in the 120 s before the identity's latest access.
        assert hist.times[h] == [s for s in every[h] if s > t - SPEC_LONG]
        assert hist.num_times == sum(map(len, hist.times.values()))
        max_times = max(max_times, hist.num_times)
        for q in (t, t + rng.choice((0.5, 10.0, 119.5, 120.0, 500.0))):
            for g in every:
                for span in (SPEC_SHORT, SPEC_LONG,
                             rng.choice((0.5, 1.0, 37.5, 119.5))):
                    brute = sum(1 for s in every[g] if q - span < s <= q)
                    assert hist.count(g, q, span) == brute
    assert hist.max_times == max_times
    assert hist.max_identities == len(every)


def test_ghost_is_kept_at_exactly_ghost_s_and_forgotten_after():
    hist = AccessHistory()
    hist.record(A, 100.0)
    hist.record(B, 100.0)
    hist.evicted(A)
    hist.expire(280.0, cached={})  # idle exactly 180 s: kept
    assert hist.times[A] == [100.0] and hist.last[A] == 100.0
    hist.expire(280.5, cached={B: None})
    assert A not in hist.times and A not in hist.last
    assert hist.num_times == 1
    # Forgotten: no counts, and its idle time starts from scratch.
    assert hist.features(A, 1, A, 280.5) == log1p_all(0, 0, 0, 1, 0)
    # B was never evicted, so it stays however long it is idle.
    hist.expire(1e6, cached={})
    assert hist.last[B] == 100.0


def test_ghost_accessed_again_outlives_its_old_schedule():
    hist = AccessHistory()
    hist.record(A, 0.0)
    hist.evicted(A)  # to be forgotten after 180 s idle
    hist.expire(100.0, {})
    hist.record(A, 100.0)  # computed again (re-inserted) at 100...
    hist.evicted(A)  # ...and evicted again: now forgotten after 280
    hist.expire(200.0, {})  # the first schedule passes
    assert hist.times[A] == [0.0, 100.0]
    assert hist.count(A, 200.0, SPEC_LONG) == 1
    hist.expire(280.0, {})
    assert A in hist.last
    hist.expire(280.5, {})
    assert A not in hist.last


def test_cached_identity_is_never_forgotten():
    hist = AccessHistory()
    hist.record(A, 0.0)
    hist.evicted(A)
    hist.expire(1000.0, cached={A: object()})  # cached again
    assert hist.last[A] == 0.0 and hist.times[A] == [0.0]


def test_evicting_unknown_or_twice_is_harmless():
    hist = AccessHistory()
    hist.evicted(D)  # never accessed: nothing to forget
    hist.record(A, 0.0)
    hist.evicted(A)
    hist.evicted(A)  # two schedules for one identity
    hist.expire(181.0, {})
    assert hist.last == {} and hist.times == {}
    assert hist.num_times == 0
    hist.expire(1e6, {})
    assert hist.num_times == 0


def test_history_counters():
    hist = AccessHistory()
    hist.record(A, 0.0)
    hist.record(B, 0.0)
    hist.record(A, 50.0)
    hist.record(A, 130.0)  # A's access at 0 is pruned
    assert (hist.num_times, hist.max_times, hist.max_identities) == (3, 3, 2)
    hist.evicted(B)
    hist.expire(181.0, {})  # B forgotten
    assert (hist.num_times, hist.max_times, hist.max_identities) == (2, 3, 2)
    hist.record(C, 181.0)
    assert (hist.num_times, hist.max_times, hist.max_identities) == (3, 3, 2)
    hist.record(D, 181.0)
    assert (hist.num_times, hist.max_times, hist.max_identities) == (4, 4, 3)


def test_history_features_by_hand():
    hist = AccessHistory()
    for t in (10.0, 50.0, 100.0):
        hist.record(B, t)  # the family block
    for t in (0.0, 5.0, 95.0, 100.0):
        hist.record(A, t)
    # (95, 105]: 95 is exactly 10 s ago and not counted.
    assert hist.features(A, 3, B, 105.0) == log1p_all(1, 4, 5, 3, 3)
    # At 130: (120, 130] and (10, 130]; B's access at 10 is 120 s ago.
    assert hist.features(A, 3, B, 130.0) == log1p_all(0, 2, 30, 3, 2)
    # At its latest access: that access counts and it has been idle 0 s.
    assert hist.features(A, 3, B, 100.0) == log1p_all(2, 4, 0, 3, 3)
    assert hist.features(A, 4, None, 105.0) == log1p_all(1, 4, 5, 4, 0)
    assert hist.features(C, 2, C, 105.0) == log1p_all(0, 0, 0, 2, 0)


# --- RaceSketch -------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 4), (2, 2, 6), (1, 2, 2, 5)])
def test_planes_must_be_L_by_P_by_5(shape):
    with pytest.raises(ValueError, match=r"\(L, P, 5\)"):
        RaceSketch(np.zeros(shape), np.zeros(5), np.ones(5))


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan")])
def test_alpha_must_be_positive(alpha):
    # With alpha <= 0 an empty bucket would score 0/0 (or divide by <= 0).
    with pytest.raises(ValueError, match="alpha must be positive"):
        RaceSketch(np.zeros((1, 1, 5)), np.zeros(5), np.ones(5), alpha=alpha)
    data = RaceSketch(np.zeros((1, 1, 5)), np.zeros(5), np.ones(5)).to_json()
    data["alpha"] = alpha
    with pytest.raises(ValueError, match="alpha must be positive"):
        RaceSketch.from_json(data)
    small = RaceSketch(np.zeros((1, 1, 5)),
                       np.zeros(5),
                       np.ones(5),
                       alpha=1e-6)
    assert small.score(np.zeros((1, 5))).tolist() == [0.0]  # r_bar 0


def test_count_arrays_must_be_L_by_2_to_the_P():
    planes = np.zeros((2, 2, 5))
    ok = np.zeros((2, 4))
    for bad in (np.zeros((2, 3)), np.zeros((4, 2)), np.zeros(8)):
        with pytest.raises(ValueError, match="count arrays"):
            RaceSketch(planes, np.zeros(5), np.ones(5), counts=bad)
        with pytest.raises(ValueError, match="count arrays"):
            RaceSketch(planes, np.zeros(5), np.ones(5), ok, rewards=bad)
    sketch = RaceSketch(planes, np.zeros(5), np.ones(5))
    assert (sketch.L, sketch.P) == (2, 2)
    assert np.array_equal(sketch.counts, ok)
    assert np.array_equal(sketch.rewards, ok)
    assert (sketch.r_bar, sketch.alpha, sketch.meta) == (0.0, 5.0, {})


def test_random_planes_are_seeded_standard_normal_draws():
    mean, std = np.arange(5.0), np.full(5, 2.0)
    sketch = RaceSketch.random(16,
                               6,
                               7,
                               mean,
                               std,
                               r_bar=0.3,
                               alpha=2.0,
                               meta={"seed": 7})
    expect = np.random.default_rng(7).standard_normal((16, 6, 5))
    assert np.array_equal(sketch.planes, expect)
    assert np.array_equal(
        RaceSketch.random(16, 6, 7, mean, std).planes, expect)
    assert not np.array_equal(
        RaceSketch.random(16, 6, 8, mean, std).planes, expect)
    assert sketch.counts.shape == sketch.rewards.shape == (16, 64)
    assert not sketch.counts.any() and not sketch.rewards.any()
    assert (sketch.r_bar, sketch.alpha, sketch.meta) == (0.3, 2.0, {"seed": 7})


def test_normalize_standardizes_then_scales_to_unit_length():
    sketch = RaceSketch(np.zeros((1, 1, 5)), [1, 2, 3, 4, 5], [2, 0, 1, 1, 4])
    # std 0 is read as 1: z = [1, 0, 0, 1, 0] / sqrt(2).
    z = sketch.normalize([3, 2, 3, 5, 5])
    assert z.shape == (1, 5)
    np.testing.assert_allclose(z, [[2**-0.5, 0, 0, 2**-0.5, 0]], atol=1e-15)
    rows = sketch.normalize([[1, 2, 3, 4, 5], [1, 7, 3, 4, 5]])
    assert np.array_equal(rows, [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]])


# Hand-picked planes (L = 2, P = 2), mean 0 and std 1:
# table 0: bit 1 = [x0 >= 0], bit 2 = [x1 >= 0];
# table 1: bit 1 = [-x0 >= 0], bit 2 = [x2 + x3 >= 0].
HAND_PLANES = np.array(
    [[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], [[-1, 0, 0, 0, 0], [0, 0, 1, 1, 0]]],
    dtype=np.float64)
X1 = [1, -2, 0, 0, 0]  # buckets (1, 2): x2 + x3 = 0 sets bit 2
X2 = [-1, 3, 0, -1, 0]  # (2, 1)
X0 = [0, 0, 0, 0, 0]  # z = 0: every bit set, bucket 2^P - 1 = 3
X4 = [2, 1, 1, 0, 9]  # (3, 2)
X5 = [-1, -1, -1, 0, 0]  # (0, 1)


def hand_sketch(**kwargs) -> RaceSketch:
    return RaceSketch(HAND_PLANES, np.zeros(5), np.ones(5), **kwargs)


def test_buckets_by_hand():
    buckets = hand_sketch().buckets(np.array([X1, X2, X0, X4, X5]))
    assert buckets.shape == (5, 2)
    assert buckets.tolist() == [[1, 2], [2, 1], [3, 3], [3, 2], [0, 1]]
    assert hand_sketch().buckets(X1).tolist() == [[1, 2]]
    # Standardization comes first: with mean X4 the vector X4 is zero.
    shifted = RaceSketch(HAND_PLANES, X4, np.ones(5))
    assert shifted.buckets(X4).tolist() == [[3, 3]]


def test_buckets_match_the_spec_randomized():
    rng = np.random.default_rng(5)
    mean, std = rng.normal(size=5), np.abs(rng.normal(size=5))
    std[2] = 0.0
    sketch = RaceSketch.random(16, 6, 3, mean, std)
    planes = np.random.default_rng(3).standard_normal((16, 6, 5))
    x = np.vstack([rng.normal(size=(300, 5)) + mean, mean])
    got = sketch.buckets(x)
    assert got.shape == (301, 16)
    assert got.min() >= 0 and got.max() <= 63
    for row, buckets in zip(x.tolist(), got.tolist()):
        assert tuple(buckets) == spec_buckets(planes, mean, std, row)
    assert got[-1].tolist() == [63] * 16  # x = mean


def test_add_query_score_by_hand():
    sketch = hand_sketch(r_bar=0.375)
    sketch.add(np.array([X1, X2, X4]), np.array([1.0, 0.0, 0.5]))
    sketch.add(np.array(X1), 0.0)  # one observation, scalar reward
    assert sketch.counts.tolist() == [[0, 2, 1, 1], [0, 1, 3, 0]]
    assert sketch.rewards.tolist() == [[0, 1, 0, 0.5], [0, 0, 1.5, 0]]
    n, w = sketch.query(np.array([X1, X0, X5]))
    assert n.tolist() == [2.5, 0.5, 0.5]
    assert w.tolist() == [1.25, 0.25, 0.0]
    expect = [(1.25 + 1.875) / 7.5, (0.25 + 1.875) / 5.5, 1.875 / 5.5]
    np.testing.assert_allclose(sketch.score(np.array([X1, X0, X5])),
                               expect,
                               rtol=1e-15)
    sketch.alpha = 2.0
    np.testing.assert_allclose(sketch.score(X1), [(1.25 + 0.75) / 4.5],
                               rtol=1e-15)


def test_untrained_sketch_scores_r_bar():
    sketch = empty_sketch(r_bar=0.25)
    x = np.random.default_rng(0).normal(size=(20, 5))
    assert np.array_equal(sketch.score(x), np.full(20, 0.25))


@pytest.mark.parametrize("L,P,seed", [(16, 6, 1), (2, 2, 2), (3, 4, 3)])
def test_trained_sketch_matches_the_spec(L, P, seed):
    sketch, spec = trained(L, P, seed, constant_f10=(seed == 2))
    assert sketch.counts.tolist() == spec.N
    assert sketch.rewards.tolist() == spec.W
    assert sketch.counts.sum() == 300 * L  # every observation, every table
    queries = training_data(seed + 50)[0]
    queries = np.vstack([queries, spec.mean])
    n, w = sketch.query(queries)
    scores = sketch.score(queries)
    for i, row in enumerate(queries.tolist()):
        assert (n[i], w[i]) == spec.query(row)
        assert scores[i] == spec.score(row)


def test_add_with_real_valued_rewards_matches_the_spec():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(200, 5))
    r = rng.random(200)
    sketch = RaceSketch.random(8, 5, 4, x.mean(0), x.std(0))
    sketch.add(x, r)
    spec = SpecSketch(
        np.random.default_rng(4).standard_normal((8, 5, 5)), x.mean(0),
        x.std(0))
    for xi, ri in zip(x.tolist(), r.tolist()):
        spec.add(xi, ri)
    assert sketch.counts.tolist() == spec.N
    np.testing.assert_allclose(sketch.rewards, spec.W, rtol=1e-12)


def assert_same_sketch(a: RaceSketch, b: RaceSketch) -> None:
    for name in ("planes", "mean", "std", "counts", "rewards"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert (a.L, a.P, a.r_bar, a.alpha, a.meta) == (b.L, b.P, b.r_bar, b.alpha,
                                                    b.meta)
    x = training_data(77)[0]
    assert np.array_equal(a.score(x), b.score(x))


def test_json_round_trip_is_exact(tmp_path):
    sketch, _ = trained(16, 6, 1)
    sketch.meta = {"trained_on": "seed 1", "horizon_s": 120}
    data = sketch.to_json()
    assert data["features"] == list(FEATURES)
    assert (data["L"], data["P"]) == (16, 6)
    assert_same_sketch(RaceSketch.from_json(json.loads(json.dumps(data))),
                       sketch)
    path = tmp_path / "sketch.json"
    sketch.save(path)
    assert_same_sketch(RaceSketch.load(path), sketch)
    assert_same_sketch(RaceSketch.load(str(path)), sketch)
    del data["meta"]
    assert RaceSketch.from_json(data).meta == {}


def test_json_round_trip_keeps_zero_std_as_one():
    sketch, _ = trained(4, 3, 2, constant_f10=True)
    copy = RaceSketch.from_json(json.loads(json.dumps(sketch.to_json())))
    assert_same_sketch(copy, sketch)
    assert copy.std[F10] == 1.0


def test_std_at_or_below_the_floor_counts_as_zero():
    std = [SPEC_STD_FLOOR, 2.2e-16, 2 * SPEC_STD_FLOOR, 0.0, 3.0]
    sketch = RaceSketch(np.zeros((1, 1, 5)), np.zeros(5), std)
    assert sketch.std.tolist() == [1.0, 1.0, 2 * SPEC_STD_FLOOR, 1.0, 3.0]
    # Just above the floor a std is used as it is: 2e-9 / 2e-9 = 1.
    assert sketch.normalize([0, 0, 2e-9, 0, 0]).tolist() == [[0, 0, 1, 0, 0]]
    # At or below it the column is divided by 1, so a 1e-9 offset stays
    # negligible next to the last column's 3 / 3.
    np.testing.assert_allclose(sketch.normalize([1e-9, 1e-9, 0, 1e-9, 3]),
                               [[1e-9, 1e-9, 0, 1e-9, 1]],
                               rtol=1e-9)


def test_constant_training_column_is_not_blown_up():
    # Training where every block had depth 2: that column's computed std
    # is about 2.2e-16, not 0.
    x, r = training_data(1)
    x[:, DEPTH] = math.log1p(2)
    mean, std = x.mean(axis=0), x.std(axis=0)
    assert 0 < std[DEPTH] <= SPEC_STD_FLOOR
    sketch = RaceSketch.random(16, 6, 1, mean, std)
    sketch.add(x, r)
    assert sketch.std[DEPTH] == 1.0
    planes = np.random.default_rng(1).standard_normal((16, 6, 5))
    queries = training_data(51)[0]  # depths 1 to 5
    got = [tuple(b) for b in sketch.buckets(queries).tolist()]
    for row, buckets in zip(queries.tolist(), got):
        assert buckets == spec_buckets(planes, mean, std, row)
    # Dividing by 2.2e-16 instead would leave depth the only coordinate
    # that counts: blocks deeper than 2 would all share one bucket per
    # table, and so would blocks at depth 1.
    raw = [
        spec_buckets(planes, mean, std, row, floor=0.0)
        for row in queries.tolist() if row[DEPTH] != math.log1p(2)
    ]
    assert len(set(raw)) == 2
    assert len(set(got)) > 20
    copy = RaceSketch.from_json(json.loads(json.dumps(sketch.to_json())))
    assert_same_sketch(copy, sketch)


# --- RaceRPEvictor: features ------------------------------------------------


def test_default_clock_is_the_block_managers_event_time():
    assert RaceRPEvictor(empty_sketch()).clock is cache_events.event_time


def test_record_access_features_by_hand():
    tree = {**chain("a", 4, 1), "b2": ("a1", 9)}
    hand = Hand(empty_sketch(), tree)
    hand.request("a4", t=0.0)  # a1 a2 a3 a4
    hand.request("a2", t=100.0)  # a1 a2
    hand.request("b2", t=105.0)  # a1 b2
    hand.clock.t = 110.0
    # x = log1p of [f10, f120, idle, depth, family]; a block's family count
    # is that of the block among its first two (itself at depth 1 or 2).
    assert hand.features("a1") == log1p_all(1, 3, 5, 1, 3)
    assert hand.features("a2") == log1p_all(0, 2, 10, 2, 2)  # 100: 10 s ago
    assert hand.features("a3") == log1p_all(0, 1, 110, 3, 2)
    assert hand.features("a4") == log1p_all(0, 1, 110, 4, 2)
    assert hand.features("b2") == log1p_all(1, 1, 5, 2, 1)
    hand.clock.t = 120.0  # the accesses at 0 are now exactly 120 s ago
    assert hand.features("a3") == log1p_all(0, 0, 120, 3, 1)
    assert hand.features("a1") == log1p_all(0, 2, 15, 1, 2)
    a1 = hand.hashes["a1"]
    assert hand.ev.history.times[a1] == [0.0, 100.0, 105.0]
    assert hand.ev.history.last[a1] == 105.0


def test_block_computed_again_keeps_its_unexpired_history():
    hand = Hand(empty_sketch(), chain("a", 2, 1))
    hand.request("a1", t=0.0)
    hand.request("a2", t=50.0)
    assert hand.evict(t=60.0) == "a2"
    assert hand.evict() == "a1"
    hand.request("a2", t=100.0)  # both recomputed: accesses still count
    assert hand.features("a1") == log1p_all(1, 3, 0, 1, 3)
    assert hand.features("a2") == log1p_all(1, 2, 0, 2, 2)


def test_ghosts_are_forgotten_only_at_accesses():
    hand = Hand(empty_sketch(), {n: (None, i) for i, n in enumerate("abc")})
    for name in "abc":
        hand.request(name, t=0.0)
    a, b, c = (hand.hashes[n] for n in "abc")
    assert hand.evict(t=10.0) == "a"
    assert hand.evict(t=500.0) == "b"  # not an access: nothing forgotten
    assert set(hand.ev.history.last) == {a, b, c}
    hand.request("c", t=500.0)  # a and b idle 500 s; c is cached
    assert set(hand.ev.history.last) == {c}


def test_ghost_exactly_ghost_s_idle_is_kept_in_the_evictor():
    hand = Hand(empty_sketch(), {"a": (None, 1), "b": (None, 2)})
    hand.request("a", t=0.0)
    assert hand.evict(t=10.0) == "a"
    hand.request("b", t=180.0)
    assert hand.ev.history.times[hand.hashes["a"]] == [0.0]
    hand.request("b", t=180.5)
    assert hand.hashes["a"] not in hand.ev.history.times


# --- RaceRPEvictor: choosing the victim -------------------------------------


def lru_disagrees(sketch: RaceSketch) -> Hand:
    """a (accessed at 0 and 1) and b (at 2), both idle."""
    hand = Hand(sketch, {"a": (None, 1), "b": (None, 2), "c": (None, 3)})
    hand.request("a", t=0.0)
    hand.request("a", t=1.0)
    hand.request("b", t=2.0)
    return hand


def test_victim_has_the_lowest_score():
    hand = lru_disagrees(threshold_sketch(F120, 1.5))
    hand.clock.t = 10.0
    assert hand.ev.sketch.score(hand.features("a")).tolist() == [HIGH]
    assert hand.ev.sketch.score(hand.features("b")).tolist() == [LOW]
    assert hand.evict() == "b"  # LRU would evict a


@pytest.mark.parametrize("t,victim", [(119.5, "c"), (120.0, "a"),
                                      (121.0, "a")])
def test_victim_follows_the_120_s_window(t, victim):
    hand = lru_disagrees(threshold_sketch(F120, 1.5))
    assert hand.evict(t=10.0) == "b"
    hand.request("c", t=100.0)
    # Before 120, a has 2 accesses in (t - 120, t] (HIGH); from 120 its
    # access at 0 no longer counts, so it ties with c (LOW) and is older.
    assert hand.evict(t=t) == victim


def test_victim_uses_the_family_count():
    tree = {**chain("a", 3, 1), **chain("b", 3, 11)}
    hand = Hand(threshold_sketch(FAMILY, 1.5), tree)
    hand.request("a3", t=0.0)
    hand.request("a2", t=1.0)  # a second request through a1 a2
    hand.request("b3", t=2.0)
    # a3's family (a2) has 2 requests: HIGH; b's family 1: LOW.
    order = [hand.evict(t=3.0) for _ in range(6)]
    assert order == ["b3", "b2", "b1", "a3", "a2", "a1"]


def test_ties_break_by_depth_then_last_access_then_block_id():
    tree = {
        "a1": (None, 7),
        "a2": ("a1", 3),
        "a3": ("a2", 9),
        "b1": (None, 5),
        "c1": (None, 2),
        "d1": (None, 8),
    }
    hand = Hand(empty_sketch(), tree)
    hand.request("a3", t=0.0)
    hand.request("d1", t=0.0)
    hand.request("b1", t=1.0)
    hand.request("c1", t=1.0)
    order = [hand.evict(t=2.0) for _ in range(6)]
    # Equal scores: deepest first (a3, then a2 once it is a leaf), then the
    # oldest last access (a1 and d1 at 0, by id 7 < 8), then c1 (id 2) and
    # b1 (id 5), both last used at 1.
    assert order == ["a3", "a2", "a1", "d1", "c1", "b1"]


def test_ties_between_equal_features_go_to_the_lower_block_id():
    sketch, _ = trained(16, 6, 1)
    hand = Hand(sketch, {"x": (None, 4), "y": (None, 3), "z": (None, 5)})
    for name in ("x", "y", "z"):
        hand.request(name, t=0.0)
    assert [hand.evict(t=7.0) for _ in range(3)] == ["y", "x", "z"]


def test_only_idle_leaves_are_evicted():
    tree = {**chain("a", 2, 1), "b1": (None, 3)}
    hand = Hand(threshold_sketch(DEPTH, 1.5), tree)  # depth 1 LOW, 2 HIGH
    hand.use("a2", t=0.0)  # a1 a2 stay in use
    hand.request("b1", t=0.0)
    assert hand.evict(t=1.0) == "b1"
    with pytest.raises(ValueError, match="No usable cache memory"):
        hand.ev.evict()
    hand.free("a2")
    # a1 scores lower, but its child is cached: a2 goes first.
    assert hand.evict() == "a2"
    assert hand.evict() == "a1"


def test_idle_parent_of_a_block_in_use_is_not_a_candidate():
    hand = Hand(empty_sketch(), chain("a", 2, 1))
    hand.use("a2", t=0.0)
    hand.ev.add(1, hand.hashes["a1"], 16, 0.0)  # only a1 idle
    with pytest.raises(ValueError, match="No evictable leaf"):
        hand.ev.evict()


# --- RaceRPEvictor: one scan per time ---------------------------------------


def test_one_scan_per_time_and_blocks_freed_meanwhile_are_scored():
    tree = {n: (None, i) for i, n in enumerate("pqsr", start=1)}
    hand = Hand(threshold_sketch(F120, 1.5), tree)
    hand.request("q", t=0.0)
    hand.request("q", t=1.0)  # f120 = 2: HIGH
    hand.request("s", t=2.0)
    hand.request("s", t=3.0)  # HIGH
    hand.request("p", t=4.0)  # LOW
    assert hand.evict(t=10.0) == "p"
    hand.request("r")  # a new first block at 10, f120 = 1: LOW
    assert hand.evict() == "r"
    assert hand.evict() == "q"  # q and s tie; q is older
    assert hand.ev.stats["scans"] == 1
    assert hand.evict(t=11.0) == "s"
    assert hand.ev.stats["scans"] == 2


def family_scenario() -> Hand:
    """Families a, b, c (x1 -> x2 -> x3), two requests each, at 1, 2, 0:
    every leaf x3 has family count 2 (LOW below 3)."""
    tree = {**chain("a", 3, 1), **chain("b", 3, 11), **chain("c", 3, 21)}
    hand = Hand(threshold_sketch(FAMILY, 2.5), tree)
    for t, family in ((0.0, "c"), (1.0, "a"), (2.0, "b")):
        hand.request(family + "3", t=t)
        hand.request(family + "3", t=t)
    assert hand.evict(t=10.0) == "c3"  # all LOW: the oldest
    assert hand.ev.stats["scans"] == 1
    return hand


def test_second_block_access_rescans_its_familys_scores():
    hand = family_scenario()
    hand.request("a2")  # a third request through a1 a2, at 10
    assert hand.features("a3")[FAMILY] == math.log1p(3)  # now HIGH
    assert hand.evict() == "b3"  # not a3, whose heap score is stale
    assert hand.ev.stats["scans"] == 2


def test_first_block_access_keeps_the_heap():
    hand = family_scenario()
    hand.request("a1")  # a1's own counts change, no family count does
    assert hand.evict() == "a3"
    assert hand.ev.stats["scans"] == 1


def stale_entry_scenario() -> Hand:
    """x evicted and recomputed into the same block id at one time."""
    tree = {"x": (None, 7), "y": (None, 9), "z": (None, 4)}
    hand = Hand(f120_levels_sketch(), tree)
    hand.request("x", t=0.0)  # f120 = 1: S = 1/2
    hand.request("y", t=1.0)  # 1/2
    hand.request("z", t=1.0)
    hand.request("z", t=1.0)  # 1/6
    assert hand.evict(t=10.0) == "z"
    hand.request("x")  # revived at 10: f120 = 2, S = 1/6
    assert hand.evict() == "x"
    hand.request("x")  # computed again into block 7: f120 = 3, S = 5/6
    return hand


def test_stale_entry_scenario_scores():
    hand = stale_entry_scenario()
    sketch = hand.ev.sketch
    assert sketch.score(hand.features("x")).tolist() == pytest.approx([5 / 6])
    assert sketch.score(hand.features("y")).tolist() == [0.5]


def test_block_recomputed_into_the_same_id_is_scored_afresh():
    # Regression: versions once restarted at 0 for a new block, so a heap
    # entry of x's evicted predecessor (same id, hash and version) was
    # taken as live and x was evicted with its old score.
    hand = stale_entry_scenario()
    assert hand.evict() == "y"  # 1/2 < 5/6


def test_idle_leaf_accessed_at_the_heap_time_is_rescored():
    """Not the allocator's order (it removes a block before using it), but
    the victim must still be the argmin."""
    tree = {"p": (None, 1), "q": (None, 2), "s": (None, 3)}
    hand = Hand(f120_levels_sketch(), tree)
    for t, name in enumerate("pqs"):
        hand.request(name, t=float(t))  # each f120 = 1: S = 1/2
    assert hand.evict(t=10.0) == "p"
    hand.ev.record_access(3, hand.hashes["s"], 1, None)  # s stays idle
    assert hand.features("s")[F120] == math.log1p(2)  # S = 1/6
    assert hand.evict() == "s"
    assert hand.evict() == "q"


def test_block_under_an_uncached_parent_has_no_family():
    hand = Hand(threshold_sketch(F120, 1.5), chain("a", 3, 1))
    a2, a3 = hand.hashes["a2"], hand.hashes["a3"]
    hand.clock.t = 0.0
    hand.ev.record_access(3, a3, 3, a2)  # a2 is not cached
    assert hand.features("a3") == log1p_all(1, 1, 0, 3, 0)  # no family
    hand.ev.record_access(2, a2, 2, None)  # a2 cached afterwards and
    hand.ev.record_access(2, a2, 2, None)  # used twice: HIGH
    hand.ev.add(2, a2, 32, 0.0)
    hand.ev.add(3, a3, 48, 0.0)
    # Not linked, so both are leaves: a3 (LOW) goes first.
    assert hand.evict(t=1.0) == "a3"
    # At 200, a3 is computed again under the cached a2, and used twice.
    hand.clock.t = 200.0
    hand.ev.remove(2)
    hand.ev.record_access(2, a2, 2, None)  # a2: f120 = 1, LOW
    hand.ev.record_access(3, a3, 3, a2)
    hand.ev.record_access(3, a3, 3, a2)  # a3: f120 = 2, HIGH
    assert hand.features("a3") == log1p_all(2, 2, 0, 3, 1)  # a2's family
    hand.ev.add(2, a2, 32, 200.0)
    hand.ev.add(3, a3, 48, 200.0)
    # Now a2 has a cached child: a3 goes first despite its higher score
    # (had the orphan's eviction counted against a2, a2 would be a leaf).
    assert hand.evict() == "a3"
    assert hand.evict() == "a2"
    assert hand.ev.num_blocks == 0


def test_eviction_never_changes_the_sketch():
    sketch, spec = trained(4, 3, 3)
    before = [
        a.copy() for a in (sketch.planes, sketch.mean, sketch.std,
                           sketch.counts, sketch.rewards)
    ]
    work = Workload(12, sketch, spec)
    work.run(250)
    assert work.evictions > 20
    after = (sketch.planes, sketch.mean, sketch.std, sketch.counts,
             sketch.rewards)
    for old, new in zip(before, after):
        assert np.array_equal(old, new)
    assert (sketch.r_bar, sketch.alpha) == (spec.r_bar, SPEC_ALPHA)


# --- RaceRPEvictor: metadata and settings -----------------------------------


def test_metadata_words_and_summary():
    sketch = empty_sketch(L=16, P=6)
    hand = Hand(sketch, {**chain("a", 2, 1), "b1": (None, 3)})
    hand.request("a2", t=0.0)
    hand.request("a1", t=50.0)
    hand.request("b1", t=200.0)
    # history: a1 [0, 50], a2 [0], b1 [200]: 3 identities + 4 times.
    sketch_words = 2 * 16 * 64 + 16 * 6 * 5 + 2 * 5 + 2
    assert sketch_words == 2540
    words = {"dependency": 12, "family": 3, "history": 7, "sketch": 2540}
    assert hand.ev.metadata_words() == words
    assert hand.evict() == "a2"
    assert hand.ev.metadata_words() == {**words, "dependency": 8, "family": 2}
    hand.request("b1", t=380.0)  # a2 forgotten; b1's access at 200 pruned
    assert hand.ev.metadata_words() == {
        **words, "dependency": 8,
        "family": 2,
        "history": 2 + 3
    }
    summary = hand.ev.summary()
    assert summary["metadata_words"] == hand.ev.metadata_words()
    assert summary["metadata_bytes"] == 8 * (8 + 2 + 5 + 2540)
    assert summary["max_history_times"] == 4
    assert summary["max_history_identities"] == 3
    assert summary["evictions"] == 1


def test_from_params(tmp_path, monkeypatch):
    sketch, _ = trained(16, 6, 1)
    path = tmp_path / "race_sketch.json"
    sketch.save(path)
    ev = RaceRPEvictor.from_params(f"sketch={path}")
    assert_same_sketch(ev.sketch, sketch)
    assert ev.clock is cache_events.event_time
    ev = RaceRPEvictor.from_params(f" sketch = {path} , alpha=2.5")
    assert ev.sketch.alpha == 2.5
    assert np.array_equal(ev.sketch.counts, sketch.counts)
    for text in (None, "", "alpha=1", "sketch="):
        with pytest.raises(ValueError, match="sketch=<file>"):
            RaceRPEvictor.from_params(text)
    for text in (f"sketch={path},bogus=1", f"sketch={path},tau_short=1",
                 f"sketch={path},alpha"):
        with pytest.raises(ValueError, match="VLLM_EVICTION_PARAMS"):
            RaceRPEvictor.from_params(text)
    for alpha in ("0", "-2.5", "nan"):
        with pytest.raises(ValueError, match="alpha must be positive"):
            RaceRPEvictor.from_params(f"sketch={path},alpha={alpha}")
    registered = []
    monkeypatch.setattr(reuse_evictors.atexit, "register",
                        lambda fn, *args: registered.append((fn, args)))
    stats = tmp_path / "stats.json"
    ev = RaceRPEvictor.from_params(f"sketch={path},stats={stats}")
    assert registered == [(ev.write_stats, (str(stats), ))]
    fn, args = registered[0]
    fn(*args)
    written = json.loads(stats.read_text())
    assert written["max_history_times"] == 0
    assert written["metadata_words"]["sketch"] == 2540


# --- Randomized: every victim against a from-scratch reference -------------


class Workload:
    """A prefix-closed cache driven at random beside a from-scratch model
    of the spec.

    Requests use whole prefixes (a block is created only under its cached
    parent, and every block a request holds has its ancestors held too),
    hold them for a while and free them; idle blocks are revived by later
    requests; blocks are evicted when a new block needs a physical block
    and at random, at repeated and advancing times. The model keeps every
    access, applies the 180 s ghost rule itself, and before each eviction
    recomputes all candidates' features and the spec's argmin.
    """

    def __init__(self,
                 seed: int,
                 sketch: RaceSketch,
                 spec: SpecSketch,
                 capacity: int = 36) -> None:
        self.rng = random.Random(seed)
        self.clock = FakeClock(0.0)
        self.ev = RaceRPEvictor(sketch, clock=self.clock)
        self.spec = spec
        self.hashes: Dict[Path, int] = {}
        self.paths: Dict[int, Path] = {}
        self.ids: Dict[int, int] = {}  # cached content hash -> block id
        self.refs: Dict[int, int] = {}
        self.free_ids = list(range(100, 100 + capacity))
        self.rng.shuffle(self.free_ids)
        self.active: List[List[int]] = []
        self.recent: List[Path] = []
        self.accesses: Dict[int, List[float]] = defaultdict(list)
        # [time, path, blocks accessed so far], in time order.
        self.requests: List[list] = []
        self.request_times: List[float] = []
        # The spec's history: identities remembered, their latest access,
        # and when each was last forgotten.
        self.remembered: Dict[int, float] = {}
        self.forgotten: Dict[int, float] = {}
        self.max_identities = self.max_times = 0
        self.evictions = self.ties = self.forgets = 0

    # The model.

    def content_hash(self, path: Path) -> int:
        if path not in self.hashes:
            h = self.rng.getrandbits(48)
            while h in self.paths:
                h = self.rng.getrandbits(48)
            self.hashes[path], self.paths[h] = h, path
        return self.hashes[path]

    def stored_times(self, h: int) -> int:
        times = self.accesses[h]
        return len(times) - bisect_right(times, self.remembered[h] - SPEC_LONG)

    def on_access(self, h: int, now: float) -> None:
        for g, last in list(self.remembered.items()):
            if g not in self.ids and now - last > SPEC_GHOST:
                del self.remembered[g]
                self.forgotten[g] = now
                self.forgets += 1
        self.remembered[h] = now
        self.accesses[h].append(now)
        self.max_identities = max(self.max_identities, len(self.remembered))
        self.max_times = max(self.max_times,
                             sum(map(self.stored_times, self.remembered)))

    def spec_features(self, h: int, t: float) -> Tuple[float, ...]:
        path = self.paths[h]
        since = self.forgotten.get(h, -math.inf)
        times = [s for s in self.accesses[h] if s >= since]
        f10 = sum(1 for s in times if t - SPEC_SHORT < s <= t)
        f120 = sum(1 for s in times if t - SPEC_LONG < s <= t)
        # Requests in (t - 120, t] that used the block's first two blocks
        # (its first, for a first block).
        k = min(len(path), 2)
        start = bisect_right(self.request_times, t - SPEC_LONG)
        family = sum(1 for s, p, done in self.requests[start:]
                     if s <= t and done >= k and p[:k] == path[:k])
        return log1p_all(f10, f120, t - times[-1], len(path), family)

    def spec_victim(self, t: float) -> int:
        cached = set(self.paths[h] for h in self.ids)
        keys = []
        for h, block_id in self.ids.items():
            path = self.paths[h]
            if self.refs[h] or any(q[:-1] == path
                                   for q in cached if len(q) == len(path) + 1):
                continue
            x = self.spec_features(h, t)
            assert self.ev.features(h, self.ev.blocks[h], t) == x
            keys.append(((self.spec.score(x), -len(path), self.accesses[h][-1],
                          block_id), h))
        keys.sort()
        if len(keys) > 1 and keys[0][0][0] == keys[1][0][0]:
            self.ties += 1
        return keys[0][1]

    # Driving the cache.

    def evict_one(self) -> None:
        idle = [h for h in self.ids if not self.refs[h]]
        assert self.ev.num_blocks == len(idle)
        assert all(self.ids[h] in self.ev for h in idle)
        want = self.spec_victim(self.clock.t)
        assert self.ev.evict() == (self.ids[want], want)
        self.free_ids.insert(self.rng.randrange(len(self.free_ids) + 1),
                             self.ids.pop(want))
        del self.refs[want]
        self.evictions += 1

    def request(self, path: Path) -> None:
        t = self.clock.t
        record = [t, path, 0]
        self.requests.append(record)
        self.request_times.append(t)
        held = []
        for depth in range(1, len(path) + 1):
            h = self.content_hash(path[:depth])
            prev = None if depth == 1 else self.content_hash(path[:depth - 1])
            if h in self.ids:
                self.refs[h] += 1
                if self.refs[h] == 1:
                    self.ev.remove(self.ids[h])
            else:
                if not self.free_ids:
                    self.evict_one()
                self.ids[h] = self.free_ids.pop()
                self.refs[h] = 1
            self.ev.record_access(self.ids[h], h, depth, prev)
            self.on_access(h, t)
            record[2] = depth
            held.append(h)
        self.active.append(held)

    def finish(self, i: int) -> None:
        held = self.active.pop(i)
        self.rng.shuffle(held)
        for h in held:
            self.refs[h] -= 1
            if not self.refs[h]:
                self.ev.add(self.ids[h], h, 16 * len(self.paths[h]),
                            self.clock.t)

    def random_path(self) -> Path:
        if self.recent and self.rng.random() < 0.5:
            hot = self.rng.choice(self.recent[-10:])
            return hot[:self.rng.randint(1, len(hot))]
        n = self.rng.choice((1, 2, 3, 3, 4, 4, 5))
        path = (self.rng.randrange(3), ) + tuple(
            self.rng.randrange(2 if depth == 1 else 3)
            for depth in range(1, n))
        self.recent.append(path)
        return path

    def burst(self) -> None:
        """At one time: evict, a short request through the first two
        blocks of a cached deeper prefix (its family count rises), and
        evict again."""
        self.evict_some(1)
        deep = sorted(p for p in self.hashes
                      if len(p) > 2 and self.hashes[p] in self.ids)
        if deep:
            self.request(self.rng.choice(deep)[:2])
            self.finish(len(self.active) - 1)
        self.evict_some(self.rng.randint(1, 3))

    def evict_some(self, n: int) -> None:
        for _ in range(n):
            if self.ev.num_blocks:
                self.evict_one()

    def step_time(self, grid: bool) -> float:
        """Mixed: mostly small steps, sometimes past a window or the ghost
        limit. Grid: multiples of 10 s, so accesses exactly 10, 120 and
        180 s apart are common, and many operations share one time."""
        r = self.rng.random()
        if grid:
            return 0.0 if r < 0.6 else self.rng.choice(
                (10.0, 20.0, 60.0, 120.0, 180.0, 190.0))
        if r < 0.45:
            return 0.0
        if r < 0.93:
            return self.rng.choice((0.5, 1.0, 2.0, 5.0, 10.0))
        return self.rng.choice(
            (60.0, 110.0, 120.0, 170.0, 180.0, 180.5, 250.0))

    def run(self, steps: int, grid: bool = False) -> None:
        for _ in range(steps):
            self.clock.t += self.step_time(grid)
            op = self.rng.random()
            if op < 0.4 and len(self.active) < 4:
                self.request(self.random_path())
            elif op < 0.7 and self.active:
                self.finish(self.rng.randrange(len(self.active)))
            elif op < 0.85 and len(self.active) < 4:
                self.burst()
            else:
                self.evict_some(self.rng.randint(1, 4))
            self.check_history()

    def check_history(self) -> None:
        history = self.ev.history
        assert set(history.last) == set(self.remembered)
        assert history.last == self.remembered
        sketch = self.ev.sketch
        assert self.ev.metadata_words() == {
            "dependency":
            4 * len(self.ids),
            "family":
            len(self.ids),
            "history":
            len(self.remembered) +
            sum(map(self.stored_times, self.remembered)),
            "sketch":
            2 * sketch.L * 2**sketch.P + sketch.L * sketch.P * 5 + 2 * 5 + 2,
        }
        summary = self.ev.summary()
        assert summary["max_history_times"] == self.max_times
        assert summary["max_history_identities"] == self.max_identities


@pytest.mark.parametrize("seed,kind,grid", [
    (1, "coarse", False),
    (2, "coarse", True),
    (3, "coarse", False),
    (4, "fine", False),
    (5, "fine", True),
    (6, "fine", True),
    (7, "untrained", False),
    (8, "untrained", True),
    (9, "family", False),
    (10, "family", True),
    (11, "family", True),
])
def test_victims_match_the_spec_reference(seed, kind, grid):
    if kind == "untrained":  # every score equal: tie-breaking only
        sketch = empty_sketch(L=4, P=3, r_bar=0.3)
        spec = SpecSketch(
            np.random.default_rng(0).standard_normal((4, 3, 5)), np.zeros(5),
            np.ones(5), 0.3)
    elif kind == "coarse":  # 2 tables of 4 buckets: many exact ties
        sketch, spec = trained(2, 2, seed, constant_f10=(seed == 3))
    elif kind == "family":  # family counts move scores: rescans matter
        sketch, spec = family_depth_sketch(seed)
    else:
        sketch, spec = trained(16, 6, seed)
    work = Workload(seed, sketch, spec)
    work.run(600, grid)
    stats = work.ev.stats
    assert work.evictions == stats["evictions"] > 150
    assert stats["scans"] < stats["evictions"]  # heaps were reused
    assert work.forgets > 0  # ghosts expired
    assert work.ties > 0
