"""Sampled LRFU evictor (spec 007).

Tests against the policy's definition, not the implementation. Each cached
block i keeps a score s_i and the logical time tau_i of its last access. An
access at logical time t sets

    s_i <- 2 ** (-(t - tau_i) / H) * s_i + 1,    tau_i <- t,

a newly cached block starts with s_i = 1, tau_i = t, and its score at time T
is S_i(T) = 2 ** (-(T - tau_i) / H) * s_i, which must equal the full sum
sum_j 2 ** (-(T - t_j) / H) over its counted accesses t_j. The logical clock
steps once per counted ``record_access``; evictions do not move it. A request
counts each cached block at most once, even when it comes back after other
requests (each of the last REQUEST_MEMORY requests remembers what it
counted); a block evicted and cached again under the same id is a new block;
a None request always counts. An eviction draws up to k = sample_size
distinct blocks from the idle pool (an array in add order with swap-removal)
with ``random.Random(seed).sample`` and evicts the lowest S_i(T), ties to the
lower block id, also when S_i(T) is too small for a float. An evicted block
forgets its score.

Hand-worked cases, the O(1) score against the full sums, the LFU and LRU
limits, the ranking against S_i(T) to 40 digits, the pool's layout, the
request memory, and a randomized check of every victim against an
independent reference of the spec.
"""
import decimal
import json
import math
import random
from collections import Counter, OrderedDict
from decimal import Decimal
from typing import Optional

import pytest

from vllm.core import sampled_lrfu
from vllm.core.block import cache_events
from vllm.core.sampled_lrfu import DEFAULTS, SampledLRFUEvictor

pytestmark = pytest.mark.m3

LRFU_KEYS = "half_life, sample_size, seed, stats"
SUMMARY_KEYS = {
    "evictions", "scans", "candidates_scored", "max_candidates",
    "evict_seconds", "max_resident_blocks", "evict_us_mean",
    "candidates_per_scan", "metadata_words", "metadata_bytes"
}


@pytest.fixture(autouse=True)
def _restore_request_context(monkeypatch):
    """The evictor switches on the module-wide request context; undo that."""
    monkeypatch.setattr(cache_events._context, "enabled",
                        cache_events._context.enabled)


class Requests:
    """A stand-in for the block manager's current request."""

    def __init__(self, value: Optional[str] = None) -> None:
        self.value = value

    def __call__(self) -> Optional[str]:
        return self.value


def h(block_id):
    """A distinct content hash per block id."""
    return 1000 + block_id


def lrfu(**params):
    requests = Requests()
    return SampledLRFUEvictor(request=requests, **params), requests


def access(evictor, *block_ids):
    """The running request uses each block (lookup hit, insertion or
    dedupe), in order."""
    for block_id in block_ids:
        evictor.record_access(block_id, h(block_id), 1, None)


def free(evictor, *block_ids):
    """The last request using each block lets it go: it becomes a candidate.
    """
    for block_id in block_ids:
        evictor.add(block_id, h(block_id), 16, 0.0)


def full_sum(times, now, half_life):
    """The spec's score: every counted access, halved every H accesses."""
    return sum(2.0**(-(now - t) / half_life) for t in times)


# 40 significant digits and an exponent range no idle gap here leaves.
EXACT = decimal.Context(prec=40, Emin=decimal.MIN_EMIN, Emax=decimal.MAX_EMAX)


def exact_score(s, gap, half_life):
    """S = s * 2 ** (-gap / H) to 40 digits, never underflowing."""
    exponent = EXACT.divide(Decimal(-gap), Decimal(half_life))
    return EXACT.multiply(Decimal(s), EXACT.power(Decimal(2), exponent))


def scaled_score(s, gap, half_life):
    """S = s * 2 ** (-gap / H) as (binary exponent, mantissa in [0.5, 1)),
    which compares as S does: a long idle gap only lowers the exponent."""
    q = gap / half_life
    whole = math.floor(q)
    mantissa, exponent = math.frexp(s * 2.0**(whole - q))
    return exponent - whole, mantissa


def log2_of(scaled):
    exponent, mantissa = scaled
    return exponent + math.log2(mantissa)


def take(pool, block_id):
    """Swap-removal from the candidate array: the last entry fills the gap.
    """
    i = pool.index(block_id)
    last = pool.pop()
    if last != block_id:
        pool[i] = last


def check_pool(evictor, expected):
    """The candidate array is `expected`, and the position map, hashes,
    membership and count agree with it."""
    assert evictor._pool == expected
    assert evictor._pos == {b: i for i, b in enumerate(expected)}
    assert set(evictor._hash) == set(expected)
    assert evictor.num_blocks == len(expected)
    assert all(b in evictor for b in expected)


def watch_ranking(evictor):
    """Record every block an eviction ranks (via `rank`)."""
    ranked = []
    real = evictor.rank

    def rank(block_id):
        ranked.append(block_id)
        return real(block_id)

    evictor.rank = rank
    return ranked


# --- Settings ---------------------------------------------------------------


@pytest.mark.parametrize("text", [None, "", " , "])
def test_from_params_defaults(text):
    evictor = SampledLRFUEvictor.from_params(text)
    assert DEFAULTS == {"half_life": 512.0, "sample_size": 4, "seed": 0}
    assert (evictor.half_life, evictor.sample_size) == (512.0, 4)
    assert evictor.rng.getstate() == random.Random(0).getstate()


def test_from_params_overrides_with_typed_values():
    evictor = SampledLRFUEvictor.from_params(
        "half_life=8, sample_size=2, seed=7")
    assert (evictor.half_life, evictor.sample_size) == (8.0, 2)
    assert type(evictor.half_life) is float
    assert type(evictor.sample_size) is int
    assert evictor.rng.getstate() == random.Random(7).getstate()


def test_from_params_keeps_unnamed_defaults():
    evictor = SampledLRFUEvictor.from_params("sample_size=9")
    assert (evictor.half_life, evictor.sample_size) == (512.0, 9)
    assert evictor.rng.getstate() == random.Random(0).getstate()


@pytest.mark.parametrize(
    "text", ["tau=2", "half_life", "alpha=0.5", "Half_life=2", "=1", "k=4"])
def test_from_params_rejects_unknown_keys(text):
    with pytest.raises(ValueError,
                       match=f"VLLM_EVICTION_PARAMS: .* {LRFU_KEYS} "):
        SampledLRFUEvictor.from_params(text)


@pytest.mark.parametrize("text",
                         ["half_life=slow", "sample_size=2.5", "seed=x"])
def test_from_params_rejects_non_numbers(text):
    with pytest.raises(ValueError):
        SampledLRFUEvictor.from_params(text)


@pytest.mark.parametrize("text, message", [
    ("half_life=0", "half_life must be positive"),
    ("half_life=-1", "half_life must be positive"),
    ("half_life=nan", "half_life must be positive"),
    ("sample_size=0", "sample_size must be at least 1"),
    ("sample_size=-3", "sample_size must be at least 1"),
])
def test_from_params_rejects_out_of_range_settings(text, message):
    with pytest.raises(ValueError, match=message):
        SampledLRFUEvictor.from_params(text)


@pytest.mark.parametrize("kwargs, message", [
    ({
        "half_life": 0.0
    }, "half_life must be positive"),
    ({
        "half_life": -2.0
    }, "half_life must be positive"),
    ({
        "half_life": math.nan
    }, "half_life must be positive"),
    ({
        "sample_size": 0
    }, "sample_size must be at least 1"),
    ({
        "sample_size": -1
    }, "sample_size must be at least 1"),
])
def test_constructor_rejects_out_of_range_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SampledLRFUEvictor(request=Requests(), **kwargs)


def test_constructor_defaults_and_float_half_life():
    evictor = SampledLRFUEvictor(half_life=3, request=Requests())
    assert evictor.half_life == 3.0 and type(evictor.half_life) is float
    assert evictor.sample_size == 4
    assert evictor.rng.getstate() == random.Random(0).getstate()
    assert evictor.now == 0 and evictor.num_blocks == 0


def test_constructing_turns_on_the_request_context(monkeypatch):
    monkeypatch.setattr(cache_events._context, "enabled", False)
    SampledLRFUEvictor()
    assert cache_events._context.enabled


# --- Scores: initialization, decay, clock -----------------------------------


def test_new_block_starts_with_score_one_at_the_current_time():
    evictor, _ = lrfu(half_life=2)
    assert evictor.now == 0
    access(evictor, 1)
    assert evictor.now == 1
    assert (evictor.score[1], evictor.stamp[1]) == (1.0, 1)
    access(evictor, 2)
    assert evictor.now == 2
    assert (evictor.score[2], evictor.stamp[2]) == (1.0, 2)
    # One step after its access, block 1 has decayed by 2 ** (-1/2).
    assert evictor.current(1) == 2.0**-0.5
    assert evictor.current(2) == 1.0
    assert evictor.current(1, now=5) == 2.0**-2.0


def test_decay_updates_worked_by_hand():
    """H = 2: block X accessed at t = 1, 3, 7 (block Y at t = 2, 4, 5, 6).
    """
    evictor, _ = lrfu(half_life=2, sample_size=10)
    X, Y = 1, 2
    access(evictor, X)  # t=1: s = 1
    assert (evictor.score[X], evictor.stamp[X]) == (1.0, 1)
    access(evictor, Y)  # t=2
    access(evictor, X)  # t=3: s = 2^(-2/2) * 1 + 1 = 1.5
    assert (evictor.score[X], evictor.stamp[X]) == (1.5, 3)
    access(evictor, Y, Y, Y)  # t=4, 5, 6 (a None request always counts)
    access(evictor, X)  # t=7: s = 2^(-4/2) * 1.5 + 1 = 1.375
    assert (evictor.score[X], evictor.stamp[X]) == (1.375, 7)
    assert evictor.now == 7
    # S_X(9) = 2^(-2/2) * 1.375 = 0.6875 = 2^-4 + 2^-3 + 2^-1.
    assert evictor.current(X, now=9) == 0.6875 == full_sum([1, 3, 7], 9, 2)
    # Y: 1 at t=2, then 1.5 at t=4, 2^(-1/2) * 1.5 + 1 at t=5, ...
    y5 = 2.0**-0.5 * 1.5 + 1.0
    y6 = 2.0**-0.5 * y5 + 1.0
    assert evictor.stamp[Y] == 6
    assert evictor.score[Y] == pytest.approx(y6, rel=1e-15)
    assert evictor.current(Y) == pytest.approx(2.0**-0.5 * y6, rel=1e-15)
    assert evictor.current(Y) == pytest.approx(full_sum([2, 4, 5, 6], 7, 2),
                                               rel=1e-14)
    # At T = 7: S_X = 1.375 < S_Y = 1.737...: the frequent Y is kept.
    free(evictor, X, Y)
    assert evictor.evict() == (X, h(X))


@pytest.mark.parametrize("half_life", [0.75, 2.0, 37.0, 512.0, 1e9])
@pytest.mark.parametrize("seed", range(3))
def test_scores_equal_the_full_sums_over_counted_accesses(half_life, seed):
    """The O(1) update is the full-sum definition at every access time and
    at later times, across many interleaved blocks."""
    rng = random.Random(seed)
    evictor, _ = lrfu(half_life=half_life)
    times = {}
    for t in range(1, 1501):
        block_id = rng.randrange(12)
        access(evictor, block_id)
        times.setdefault(block_id, []).append(t)
        assert evictor.now == t
        assert evictor.stamp[block_id] == t
        assert evictor.score[block_id] == pytest.approx(full_sum(
            times[block_id], t, half_life),
                                                        rel=1e-12)
        if t % 100 == 0:
            for b, ts in times.items():
                assert evictor.current(b) == pytest.approx(full_sum(
                    ts, t, half_life),
                                                           rel=1e-12,
                                                           abs=1e-300)
    later = evictor.now + rng.randrange(1, 40)
    for b, ts in times.items():
        assert evictor.current(b, now=later) == pytest.approx(full_sum(
            ts, later, half_life),
                                                              rel=1e-12,
                                                              abs=1e-300)


def test_clock_steps_on_accesses_only():
    evictor, _ = lrfu(sample_size=2)
    access(evictor, 1, 2, 3)
    assert evictor.now == 3
    free(evictor, 1, 2, 3)
    evictor.update(1, 99.0)
    evictor.remove(2)
    assert evictor.now == 3
    evictor.evict()
    evictor.evict()
    assert evictor.now == 3  # Evictions do not advance the clock.
    access(evictor, 2)
    assert evictor.now == 4 and evictor.stamp[2] == 4


def test_eviction_does_not_age_or_touch_other_blocks():
    """Decay is lazy: an eviction changes no survivor's s or tau."""
    evictor, _ = lrfu(half_life=3, sample_size=10)
    access(evictor, 1, 2, 3, 1, 4)  # t = 1..5
    free(evictor, 1, 2, 3, 4)
    before = {b: (evictor.score[b], evictor.stamp[b]) for b in (1, 2, 3, 4)}
    # S(5): 2 -> 2^(-3/3) = 0.5, 3 -> 2^(-2/3), 4 -> 1,
    # 1 -> 2^(-1/3) * (2^(-3/3) + 1) = 1.19.
    assert evictor.evict() == (2, h(2))
    del before[2]
    assert {b: (evictor.score[b], evictor.stamp[b])
            for b in (1, 3, 4)} == before
    assert evictor.evict() == (3, h(3))


def test_update_changes_nothing():
    evictor, _ = lrfu(half_life=2, sample_size=10)
    access(evictor, 1, 2, 1)
    free(evictor, 1, 2)
    state = (dict(evictor.score), dict(evictor.stamp), list(evictor._pool),
             evictor.now)
    evictor.update(2, 1e9)
    evictor.update(1, -5.0)
    assert (dict(evictor.score), dict(evictor.stamp), list(evictor._pool),
            evictor.now) == state
    assert evictor.evict()[0] == 2  # Accessed once, at t=2.


# --- The LFU and LRU limits -------------------------------------------------


def test_very_large_half_life_keeps_the_most_used_block_regardless_of_recency(
):
    evictor, _ = lrfu(half_life=1e15, sample_size=8)
    access(evictor, 1, 1, 1, 1)  # The most used, and the least recent.
    access(evictor, 2, 2, 2)
    access(evictor, 3, 3)
    access(evictor, 4)  # The most recent, used once.
    free(evictor, 1, 2, 3, 4)
    assert [evictor.evict()[0] for _ in range(4)] == [4, 3, 2, 1]


def test_very_small_half_life_evicts_the_least_recent_regardless_of_count():
    evictor, _ = lrfu(half_life=0.5, sample_size=8)
    access(evictor, 1, 1, 1, 1)  # The most used, and the least recent.
    access(evictor, 2, 2, 2)
    access(evictor, 3, 3)
    access(evictor, 4)
    free(evictor, 4, 3, 2, 1)
    assert [evictor.evict()[0] for _ in range(4)] == [1, 2, 3, 4]


def run_idle_workload(evictor, rng, steps, on_evict):
    """Random hits, frees and evictions on 30 blocks with every candidate
    scored (sample_size >= pool); `on_evict(victim, idle, history)`."""
    idle, in_use, history = set(), set(), {}
    for _ in range(steps):
        op = rng.random()
        if op < 0.55:
            block_id = rng.randrange(30)
            if block_id in idle:
                evictor.remove(block_id)
                idle.discard(block_id)
            access(evictor, block_id)
            history.setdefault(block_id, []).append(evictor.now)
            in_use.add(block_id)
        elif op < 0.85 and in_use:
            block_id = rng.choice(sorted(in_use))
            in_use.discard(block_id)
            free(evictor, block_id)
            idle.add(block_id)
        elif idle:
            victim, _ = evictor.evict()
            on_evict(victim, idle, history)
            idle.discard(victim)
            del history[victim]
    return idle


@pytest.mark.parametrize("seed", range(5))
def test_very_large_half_life_follows_access_counts(seed):
    evictor, _ = lrfu(half_life=1e12, sample_size=1000, seed=seed)
    victims = []

    def on_evict(victim, idle, history):
        counts = {b: len(history[b]) for b in idle}
        assert counts[victim] == min(counts.values())
        victims.append(victim)

    run_idle_workload(evictor, random.Random(seed), 600, on_evict)
    assert len(victims) > 50


@pytest.mark.parametrize("half_life", [0.25, 0.02])
@pytest.mark.parametrize("seed", range(5))
def test_very_small_half_life_follows_recency(half_life, seed):
    """H < 1 is exact LRU (given distinct access times, as here), also for
    candidates idle so long that S_i(T) is 0.0 as a float (past about
    1075 * H accesses)."""
    evictor, _ = lrfu(half_life=half_life, sample_size=1000, seed=seed)
    victims, underflowed = [], 0

    def on_evict(victim, idle, history):
        nonlocal underflowed
        last = {b: history[b][-1] for b in idle}
        assert last[victim] == min(last.values())
        victims.append(victim)
        underflowed += sum(
            evictor.current(b) == 0.0 for b in idle if b != victim)

    run_idle_workload(evictor, random.Random(seed), 3000, on_evict)
    assert len(victims) > 300
    if half_life < 0.1:
        assert underflowed > 0


@pytest.mark.parametrize(
    "half_life, history, gap, order",
    [
        # Recency: S_2 = 2^-3000 < S_1 = 2^-2000 at T = 4.
        (1e-3, [2, 1], 2, [2, 1]),
        # The default H: block 1, used five times, keeps S_1 / S_2 =
        # 2^(-1/H) * s_1 = 4.98 however long all three sit idle; then recency.
        (512.0, [1, 1, 1, 1, 1, 2, 3], 552_000, [2, 3, 1]),
    ])
def test_long_idle_gaps_keep_the_score_order(half_life, history, gap, order):
    """Even when every S_i(T) is 0.0 as a float, the victims follow the
    spec's (positive) S_i(T), not the block ids."""
    evictor, _ = lrfu(half_life=half_life, sample_size=4)
    access(evictor, *history)
    filler = max(history) + 1
    for _ in range(gap):
        access(evictor, filler)
    assert all(evictor.current(b) == 0.0 for b in order)
    free(evictor, *sorted(order))
    assert [evictor.evict()[0] for _ in order] == order


@pytest.mark.parametrize("half_life", [1e-3, 0.25, 2.0, 512.0, 1e9])
@pytest.mark.parametrize("seed", range(3))
def test_rank_orders_blocks_as_their_scores_do(half_life, seed):
    """On random states (s_i reachable for H, tau_i up to 10^7 accesses
    back, some equal or adjacent), rank(i) < rank(j) exactly when
    S_i(T) < S_j(T), computed to 40 digits, at two times T; including
    blocks whose float S_i(T) has underflowed to 0.0."""
    rng = random.Random(seed)
    evictor, _ = lrfu(half_life=half_life)
    # s_i < 1 / (1 - 2^(-1/H)) when accesses are at distinct times.
    s_max = min(1.0 / (1.0 - 2.0**(-1.0 / half_life)), 1e4)
    for b in range(40):
        evictor.score[b] = (1.0 if rng.random() < 0.3 else 1.0 + rng.random() *
                            (s_max - 1.0))
        evictor.stamp[b] = (rng.randrange(10**7) if rng.random() < 0.6 else
                            10**7 - rng.randrange(4))
    latest = max(evictor.stamp.values())
    # rank = log2(s) + tau / H in a float resolves log2 S to a few ulps of
    # its size; closer scores may come out either way.
    resolution = Decimal(8 * 2.0**-52 * (latest / half_life + 14) * 0.7)
    compared = near = underflowed = 0
    for now in (latest, latest + rng.randrange(1, 10**6)):
        exact = {
            b: exact_score(evictor.score[b], now - evictor.stamp[b], half_life)
            for b in evictor.score
        }
        underflowed += sum(evictor.current(b, now) == 0.0 for b in exact)
        for a in exact:
            for b in exact:
                if a >= b:
                    continue
                if exact[a] == exact[b]:
                    assert evictor.rank(a) == evictor.rank(b)
                    continue
                gap = EXACT.abs(EXACT.subtract(exact[a], exact[b]))
                if gap <= EXACT.multiply(resolution, max(exact[a], exact[b])):
                    near += 1
                    continue
                assert ((evictor.rank(a) < evictor.rank(b)) == (exact[a] <
                                                                exact[b]))
                compared += 1
    assert compared > 1400 and near <= 10
    if half_life <= 512:
        assert underflowed > 0


# --- Victim selection among the sampled candidates --------------------------

# Pool order (the add order) and access counts (scores at H = inf).
POOL = [14, 11, 16, 10, 13, 15, 12]
COUNTS = {10: 5, 11: 2, 12: 7, 13: 1, 14: 4, 15: 6, 16: 3}


def counted_pool(**params):
    evictor, requests = lrfu(half_life=math.inf, **params)
    for block_id, count in COUNTS.items():
        access(evictor, *[block_id] * count)
    free(evictor, *POOL)
    return evictor


def test_victims_worked_by_hand_with_seed_zero():
    """k = 3, seed 0. random.Random(0) draws positions [6, 3, 5] of 7, then
    [0, 2, 3] of 6, then [3, 2, 1] of 5."""
    rng = random.Random(0)
    assert [rng.sample(range(n), 3) for n in (7, 6, 5)] == [[6, 3,
                                                             5], [0, 2, 3],
                                                            [3, 2, 1]]
    evictor = counted_pool(sample_size=3, seed=0)
    check_pool(evictor, [14, 11, 16, 10, 13, 15, 12])
    # Positions 6, 3, 5 = blocks 12, 10, 15 (counts 7, 5, 6): 10 goes,
    # though 13 (count 1) has the lowest score of all; 12 fills its slot.
    assert evictor.evict() == (10, h(10))
    check_pool(evictor, [14, 11, 16, 12, 13, 15])
    # Positions 0, 2, 3 = blocks 14, 16, 12 (counts 4, 3, 7): 16 goes.
    assert evictor.evict() == (16, h(16))
    check_pool(evictor, [14, 11, 15, 12, 13])
    # Positions 3, 2, 1 = blocks 12, 15, 11 (counts 7, 6, 2): 11 goes.
    assert evictor.evict() == (11, h(11))
    check_pool(evictor, [14, 13, 15, 12])


@pytest.mark.parametrize("sample_size", [1, 2, 3, 5])
def test_victim_is_the_lowest_score_among_the_drawn_candidates(sample_size):
    """For many seeds, each victim is the argmin over the blocks at the
    positions random.Random(seed).sample draws from the pool order derived
    here; with k < pool, sampling (not a full scan) sometimes misses the
    global minimum."""
    missed_global_min = 0
    for seed in range(60):
        evictor = counted_pool(sample_size=sample_size, seed=seed)
        rng, pool = random.Random(seed), list(POOL)
        while pool:
            n = len(pool)
            drawn = [
                pool[i] for i in rng.sample(range(n), min(sample_size, n))
            ]
            expected = min(drawn, key=lambda b: (COUNTS[b], b))
            missed_global_min += expected != min(pool, key=COUNTS.get)
            assert evictor.evict() == (expected, h(expected))
            take(pool, expected)
            check_pool(evictor, pool)
    assert missed_global_min > 0


@pytest.mark.parametrize("seed", range(20))
def test_tied_candidates_go_to_the_lower_block_id(seed):
    evictor, _ = lrfu(half_life=math.inf, sample_size=3, seed=seed)
    access(evictor, 9, 5, 7, 3, 8, 6)  # Every score is 1.
    pool = [7, 9, 5, 3, 8, 6]
    free(evictor, *pool)
    rng = random.Random(seed)
    while pool:
        drawn = [
            pool[i] for i in rng.sample(range(len(pool)), min(3, len(pool)))
        ]
        assert evictor.evict()[0] == min(drawn)
        take(pool, min(drawn))


def test_equal_scores_go_to_the_lower_block_id_whatever_the_add_order():
    evictor, _ = lrfu(half_life=math.inf, sample_size=10)
    access(evictor, 9, 5, 7, 7, 3)  # Scores: 9, 5, 3 -> 1; 7 -> 2.
    free(evictor, 7, 9, 5, 3)
    assert [evictor.evict()[0] for _ in range(4)] == [3, 5, 9, 7]


def test_sampling_draws_distinct_candidates_and_scores_no_others():
    """Each eviction evaluates exactly min(k, n) distinct pool blocks."""
    evictor, _ = lrfu(sample_size=4, seed=3)
    access(evictor, *range(20))
    free(evictor, *range(20))
    ranked = watch_ranking(evictor)
    for n in range(20, 0, -1):
        pool = set(evictor._pool)
        ranked.clear()
        scored_before = evictor.stats["candidates_scored"]
        victim, _ = evictor.evict()
        assert len(ranked) == min(4, n) == len(set(ranked))
        assert set(ranked) <= pool and victim in ranked
        assert evictor.stats["candidates_scored"] - scored_before == min(4, n)
    assert evictor.stats["candidates_scored"] == 4 * 17 + 3 + 2 + 1


@pytest.mark.parametrize("pool_size", [1, 2, 3])
def test_fewer_blocks_than_sample_size_scores_the_whole_pool(pool_size):
    evictor, _ = lrfu(half_life=4, sample_size=4)
    access(evictor, *range(pool_size))  # Block 0 is the oldest.
    free(evictor, *range(pool_size))
    ranked = watch_ranking(evictor)
    assert evictor.evict()[0] == 0
    assert sorted(ranked) == list(range(pool_size))
    assert evictor.stats["candidates_scored"] == pool_size
    assert evictor.stats["max_candidates"] == pool_size


def test_sampling_is_uniform_over_the_pool():
    """With k = 1 the victim is the one drawn block; over 2000 seeds each
    of 5 blocks goes first about 400 times."""
    first = Counter()
    for seed in range(2000):
        evictor, _ = lrfu(sample_size=1, seed=seed)
        access(evictor, *range(5))
        free(evictor, *range(5))
        first[evictor.evict()[0]] += 1
    assert set(first) == set(range(5))
    assert all(300 < c < 500 for c in first.values()), first


# --- Blocks in use, the pool, re-insertion -----------------------------------


def test_blocks_in_use_are_never_evicted():
    evictor, _ = lrfu(sample_size=100)
    access(evictor, 1, 2, 3, 4, 5)  # 1 has the lowest score.
    free(evictor, 2, 3, 1)
    evictor.remove(1)  # In use again.
    assert sorted(evictor.evict()[0] for _ in range(2)) == [2, 3]
    with pytest.raises(ValueError, match="No usable cache memory left"):
        evictor.evict()
    assert set(evictor.score) == {1, 4, 5}  # Still cached.


@pytest.mark.parametrize("seed", range(5))
def test_victims_are_always_idle_blocks(seed):
    """Requests take blocks (removing idle ones from the pool) and give all
    of theirs back when they finish; no victim is ever held by one."""
    rng = random.Random(seed)
    evictor, _ = lrfu(half_life=3, sample_size=3, seed=seed)
    refs, running, evicted = Counter(), [], 0
    for _ in range(3000):
        op = rng.random()
        if op < 0.35 and len(running) < 4:
            blocks = [rng.randrange(25) for _ in range(rng.randint(1, 5))]
            for block_id in blocks:
                if block_id in evictor:
                    evictor.remove(block_id)
                access(evictor, block_id)
                refs[block_id] += 1
            running.append(blocks)
        elif op < 0.7 and running:
            for block_id in running.pop(rng.randrange(len(running))):
                refs[block_id] -= 1
                if not refs[block_id]:
                    free(evictor, block_id)
        elif evictor.num_blocks:
            idle = set(evictor._pool)
            victim, _ = evictor.evict()
            assert victim in idle and refs[victim] == 0
            assert all(b not in evictor for b in +refs)
            evicted += 1
    assert evicted > 300


def test_add_of_a_block_in_the_pool_keeps_one_slot_and_the_new_hash():
    evictor, _ = lrfu(sample_size=10)
    access(evictor, 1, 2)
    free(evictor, 1, 2)
    evictor.add(1, 555, 16, 3.0)
    check_pool(evictor, [1, 2])
    assert evictor.evict() == (1, 555)
    check_pool(evictor, [2])


def test_membership_and_count_follow_add_remove_and_evict():
    evictor, _ = lrfu(sample_size=10)
    access(evictor, 1, 2, 3)
    assert evictor.num_blocks == 0 and 1 not in evictor
    free(evictor, 1, 2)
    assert (1 in evictor, 2 in evictor, 3 in evictor) == (True, True, False)
    assert evictor.num_blocks == 2
    evictor.remove(1)
    assert 1 not in evictor and evictor.num_blocks == 1
    assert evictor.evict() == (2, h(2))
    assert 2 not in evictor and evictor.num_blocks == 0


def test_swap_removal_moves_the_last_candidate_into_the_gap():
    evictor, _ = lrfu()
    access(evictor, 1, 2, 3, 4, 5)
    free(evictor, 1, 2, 3, 4, 5)
    check_pool(evictor, [1, 2, 3, 4, 5])
    evictor.remove(2)
    check_pool(evictor, [1, 5, 3, 4])
    evictor.remove(4)  # The last entry: nothing moves.
    check_pool(evictor, [1, 5, 3])
    evictor.remove(1)
    check_pool(evictor, [3, 5])
    free(evictor, 2)  # Re-inserted at the end.
    check_pool(evictor, [3, 5, 2])
    evictor.remove(3)
    evictor.remove(2)
    evictor.remove(5)
    check_pool(evictor, [])


@pytest.mark.parametrize("seed", range(5))
def test_pool_layout_stays_consistent_under_random_operations(seed):
    rng = random.Random(seed)
    evictor, _ = lrfu(half_life=5, sample_size=3, seed=seed)
    draws = random.Random(seed)
    pool, cached = [], set()
    for _ in range(2000):
        op = rng.random()
        block_id = rng.randrange(30)
        if op < 0.35:
            access(evictor, block_id)
            cached.add(block_id)
        elif op < 0.65 and block_id in cached:
            free(evictor, block_id)
            if block_id not in pool:
                pool.append(block_id)
        elif op < 0.85 and block_id in pool:
            evictor.remove(block_id)
            take(pool, block_id)
        elif op >= 0.85 and pool:
            n = len(pool)
            drawn = [pool[i] for i in draws.sample(range(n), min(3, n))]
            victim, _ = evictor.evict()
            assert victim in drawn
            take(pool, victim)
            cached.discard(victim)
        check_pool(evictor, pool)
        assert set(evictor.score) == cached


def test_removed_and_re_added_block_keeps_its_score():
    evictor, _ = lrfu(half_life=2, sample_size=10)
    access(evictor, 1, 1, 2)
    free(evictor, 1)
    state = (evictor.score[1], evictor.stamp[1])
    evictor.remove(1)
    assert (evictor.score[1], evictor.stamp[1]) == state
    free(evictor, 1)
    assert (evictor.score[1], evictor.stamp[1]) == state
    check_pool(evictor, [1])


def test_evicted_block_is_forgotten_and_restarts_at_score_one():
    evictor, requests = lrfu(half_life=1e9, sample_size=10)
    requests.value = "A"
    access(evictor, 1)
    requests.value = None
    access(evictor, 1, 1, 2)  # s_1 is about 3.
    free(evictor, 1, 2)
    evictor.remove(2)
    assert evictor.evict() == (1, h(1))
    assert 1 not in evictor.score and 1 not in evictor.stamp
    with pytest.raises(KeyError):
        evictor.current(1)
    with pytest.raises(ValueError, match="without an access"):
        free(evictor, 1)  # Cached again only through a new access.
    requests.value = "A"  # The request that once counted it counts again.
    access(evictor, 1)
    assert (evictor.score[1], evictor.stamp[1]) == (1.0, 5)
    assert evictor.now == 5
    free(evictor, 1)
    check_pool(evictor, [1])


def test_add_of_a_block_never_accessed_raises():
    evictor, _ = lrfu()
    with pytest.raises(ValueError, match="without an access"):
        free(evictor, 7)
    assert evictor.num_blocks == 0 and 7 not in evictor


def test_remove_and_update_of_a_block_not_in_the_pool_raise():
    evictor, _ = lrfu()
    access(evictor, 1)  # Cached but in use.
    with pytest.raises(ValueError, match="not in the evictor"):
        evictor.remove(1)
    with pytest.raises(ValueError, match="not in the evictor"):
        evictor.update(1, 1.0)
    with pytest.raises(ValueError, match="not in the evictor"):
        evictor.remove(2)
    free(evictor, 1)
    evictor.remove(1)
    with pytest.raises(ValueError, match="not in the evictor"):
        evictor.remove(1)


def test_evict_from_an_empty_pool_raises():
    evictor, _ = lrfu()
    with pytest.raises(ValueError, match="No usable cache memory left"):
        evictor.evict()
    access(evictor, 1)
    free(evictor, 1)
    evictor.evict()
    with pytest.raises(ValueError, match="No usable cache memory left"):
        evictor.evict()
    assert evictor.stats["evictions"] == 1


# --- Determinism -------------------------------------------------------------


def victim_sequence(evictor, rng, steps=400):
    victims = []
    for _ in range(steps):
        op = rng.random()
        block_id = rng.randrange(20)
        if op < 0.5:
            if block_id in evictor:
                evictor.remove(block_id)
            access(evictor, block_id)
            free(evictor, block_id)
        elif evictor.num_blocks:
            victims.append(evictor.evict())
    return victims


def test_same_seed_gives_the_same_victims_and_seeds_differ():
    sequences = {}
    for seed in range(10):
        first = victim_sequence(lrfu(seed=seed)[0], random.Random(99))
        second = victim_sequence(lrfu(seed=seed)[0], random.Random(99))
        assert first == second and len(first) > 50
        sequences[seed] = tuple(first)
    assert len(set(sequences.values())) > 1


def test_default_seed_is_zero():
    default = victim_sequence(SampledLRFUEvictor(request=Requests()),
                              random.Random(5))
    assert default == victim_sequence(lrfu(seed=0)[0], random.Random(5))


def test_policy_draws_from_its_own_generator():
    state = random.getstate()
    try:
        random.seed(123)
        expected = [random.random() for _ in range(3)]
        random.seed(123)
        victims = victim_sequence(lrfu(seed=4)[0], random.Random(1))
        # The global generator is untouched...
        assert [random.random() for _ in range(3)] == expected
        # ...and reseeding it changes nothing.
        random.seed(777)
        assert victim_sequence(lrfu(seed=4)[0], random.Random(1)) == victims
    finally:
        random.setstate(state)


# --- Once-per-request counting -----------------------------------------------


def test_a_request_counts_a_block_once():
    evictor, requests = lrfu(half_life=2)
    requests.value = "A"
    access(evictor, 1, 1, 1)
    assert evictor.now == 1  # Accesses that do not count do not tick.
    assert (evictor.score[1], evictor.stamp[1]) == (1.0, 1)


def test_a_request_counts_each_of_its_blocks_once():
    evictor, requests = lrfu(half_life=2)
    requests.value = "A"
    access(evictor, 1, 2, 1, 2, 3)
    assert evictor.now == 3
    assert evictor.stamp == {1: 1, 2: 2, 3: 3}
    assert evictor.score == {1: 1.0, 2: 1.0, 3: 1.0}


def test_different_requests_each_count():
    evictor, requests = lrfu(half_life=2)
    requests.value = "A"
    access(evictor, 1)
    requests.value = "B"
    access(evictor, 1)
    assert evictor.now == 2
    assert (evictor.score[1], evictor.stamp[1]) == (2.0**-0.5 + 1.0, 2)


def test_a_none_request_always_counts():
    evictor, requests = lrfu(half_life=2)
    access(evictor, 1, 1, 1)
    assert evictor.now == 3
    assert evictor.score[1] == pytest.approx(full_sum([1, 2, 3], 3, 2),
                                             rel=1e-15)
    requests.value = "A"
    access(evictor, 1, 1)  # One count for A after the None accesses.
    assert evictor.now == 4


@pytest.mark.parametrize("other", ["B", None])
def test_a_request_returning_after_another_is_not_counted_again(other):
    evictor, requests = lrfu(half_life=2)
    requests.value = "A"
    access(evictor, 1)  # t=1
    requests.value = other
    access(evictor, 1)  # t=2
    requests.value = "A"  # E.g. A is preempted and recomputed.
    access(evictor, 1)
    assert evictor.now == 2
    assert evictor.score[1] == 2.0**-0.5 + 1.0


def test_a_request_returning_after_many_others_counts_only_new_blocks():
    evictor, requests = lrfu(half_life=2)
    requests.value = "A"
    access(evictor, 1, 2)  # t=1, 2
    for i in range(50):
        requests.value = f"other-{i}"
        access(evictor, 1, 3)  # t=3..102
    requests.value = "A"
    access(evictor, 2, 1, 4, 1, 4)  # Only block 4 is new to A: t=103.
    assert evictor.now == 103
    assert (evictor.stamp[1], evictor.stamp[2], evictor.stamp[4]) == (101, 2,
                                                                      103)


def use(evictor, requests, request, *block_ids):
    requests.value = request
    access(evictor, *block_ids)


def test_re_cached_block_with_the_same_id_counts_afresh_for_every_request():
    evictor, requests = lrfu(half_life=2, sample_size=4)
    use(evictor, requests, "A", 1)  # t=1: block id 1, first caching.
    use(evictor, requests, "B", 1)  # t=2
    free(evictor, 1)
    assert evictor.evict() == (1, h(1))
    use(evictor, requests, "C", 1)  # t=3: the id cached again, s = 1.
    assert (evictor.score[1], evictor.stamp[1]) == (1.0, 3)
    # A and B counted the first caching, not this one.
    use(evictor, requests, "A", 1)  # t=4
    use(evictor, requests, "B", 1)  # t=5
    assert evictor.now == 5
    use(evictor, requests, "A", 1)
    use(evictor, requests, "B", 1)
    use(evictor, requests, "C", 1)
    assert evictor.now == 5
    assert evictor.score[1] == pytest.approx(full_sum([3, 4, 5], 5, 2),
                                             rel=1e-15)


def test_request_memory_defaults_to_4096_requests():
    assert sampled_lrfu.REQUEST_MEMORY == 4096
    evictor, requests = lrfu()
    use(evictor, requests, "A", 1)  # t=1
    for i in range(4095):
        use(evictor, requests, f"r{i}", 2)  # t=2..4096
    use(evictor, requests, "A", 1)  # 4096 requests remembered, A among them.
    assert evictor.now == 4096 and evictor.stamp[1] == 1
    use(evictor, requests, "r-last", 2)  # t=4097: A, the oldest, goes.
    use(evictor, requests, "A", 1)  # t=4098: counted again.
    assert evictor.now == evictor.stamp[1] == 4098


def test_oldest_request_is_forgotten_beyond_the_memory(monkeypatch):
    monkeypatch.setattr(sampled_lrfu, "REQUEST_MEMORY", 2)
    evictor, requests = lrfu(half_life=2)
    use(evictor, requests, "A", 1)  # t=1
    use(evictor, requests, "B", 1)  # t=2
    use(evictor, requests, "A", 1)  # Remembered: not counted.
    assert evictor.now == 2
    # A cached block's time, and one (block, time) pair per request: 2 words.
    assert evictor.metadata_words()["bookkeeping"] == 1 + 2 * 2
    use(evictor, requests, "C", 1)  # t=3: a third request; A is forgotten.
    assert list(evictor._counted) == ["B", "C"]
    assert evictor.metadata_words()["bookkeeping"] == 1 + 2 * 2
    use(evictor, requests, "B", 1)  # Still remembered.
    assert evictor.now == 3
    use(evictor, requests, "A", 1)  # t=4: counted again; now B goes.
    assert evictor.now == 4
    assert list(evictor._counted) == ["C", "A"]
    use(evictor, requests, "B", 1)  # t=5
    assert evictor.now == 5
    assert evictor.score[1] == pytest.approx(full_sum([1, 2, 3, 4, 5], 5, 2),
                                             rel=1e-15)


def test_a_counting_request_becomes_the_most_recent(monkeypatch):
    """Requests are forgotten in the order they last counted a block."""
    monkeypatch.setattr(sampled_lrfu, "REQUEST_MEMORY", 2)
    evictor, requests = lrfu(half_life=2)
    use(evictor, requests, "A", 1)  # t=1
    use(evictor, requests, "B", 2)  # t=2
    use(evictor, requests, "A", 3)  # t=3: A counts again; B is the oldest.
    use(evictor, requests, "C", 4)  # t=4: B is forgotten.
    assert list(evictor._counted) == ["A", "C"]
    # Four cached blocks; pairs: A (1, 1), (3, 3); C (4, 4).
    assert evictor.metadata_words()["bookkeeping"] == 4 + 2 * 3
    use(evictor, requests, "A", 1, 3)  # Remembered: not counted.
    assert evictor.now == 4
    use(evictor, requests, "B", 2)  # t=5: forgotten, so counted again.
    assert evictor.now == 5 and evictor.stamp[2] == 5
    # A did not count since t=3, so it went when B came back.
    assert list(evictor._counted) == ["C", "B"]
    assert evictor.metadata_words()["bookkeeping"] == 4 + 2 * 2


def test_default_request_is_the_block_managers_current_request(monkeypatch):
    monkeypatch.setattr(cache_events._context, "request_id", "A")
    evictor = SampledLRFUEvictor(half_life=2)
    access(evictor, 1, 1)
    assert evictor.now == 1
    monkeypatch.setattr(cache_events._context, "request_id", "B")
    access(evictor, 1)
    assert evictor.now == 2
    assert evictor.score[1] == 2.0**-0.5 + 1.0
    monkeypatch.setattr(cache_events._context, "request_id", None)
    access(evictor, 1, 1)
    assert evictor.now == 4


# --- Counters and overhead --------------------------------------------------


def test_summary_before_any_eviction():
    evictor, _ = lrfu()
    summary = evictor.summary()
    assert set(summary) == SUMMARY_KEYS
    assert (summary["evictions"], summary["scans"],
            summary["candidates_scored"], summary["max_candidates"],
            summary["max_resident_blocks"]) == (0, 0, 0, 0, 0)
    assert summary["evict_seconds"] == 0.0
    assert summary["evict_us_mean"] is None
    assert summary["candidates_per_scan"] is None
    assert summary["metadata_words"] == {"lrfu": 0, "bookkeeping": 0}
    assert summary["metadata_bytes"] == 0


def test_summary_counts_evictions_candidates_and_metadata():
    evictor, _ = lrfu(sample_size=3)
    access(evictor, *range(1, 7))
    free(evictor, *range(1, 6))  # 6 stays in use.
    evictor.evict()
    evictor.evict()
    summary = evictor.summary()
    assert set(summary) == SUMMARY_KEYS
    assert (summary["evictions"], summary["scans"],
            summary["candidates_scored"], summary["max_candidates"],
            summary["max_resident_blocks"]) == (2, 2, 6, 3, 6)
    assert summary["candidates_per_scan"] == 3.0
    assert summary["evict_seconds"] >= 0.0
    assert summary["evict_us_mean"] == pytest.approx(
        1e6 * summary["evict_seconds"] / 2)
    # Four cached blocks (three idle and 6): 2 words each and a caching time
    # each; the pool holds three (3 words each); None requests remember
    # nothing.
    assert summary["metadata_words"] == {"lrfu": 8, "bookkeeping": 9 + 4}
    assert summary["metadata_bytes"] == 8 * 21
    # Draining the pool: 3, then 2, then 1 candidates.
    for _ in range(3):
        evictor.evict()
    summary = evictor.summary()
    assert (summary["evictions"], summary["candidates_scored"],
            summary["max_candidates"]) == (5, 12, 3)
    assert summary["candidates_per_scan"] == 12 / 5
    assert summary["metadata_words"] == {"lrfu": 2, "bookkeeping": 1}


def test_metadata_counts_cached_blocks_and_remembered_pairs():
    """bookkeeping = 3 per pool entry + 1 per cached block (its caching
    time) + 2 per (block id, caching time) pair a remembered request counted,
    pairs of blocks since evicted included."""
    evictor, requests = lrfu(sample_size=4)
    access(evictor, 1)  # A None request: nothing remembered.
    use(evictor, requests, "A", 2, 2)  # One pair.
    assert evictor.metadata_words() == {"lrfu": 4, "bookkeeping": 2 + 2}
    free(evictor, 1)
    assert evictor.metadata_words() == {"lrfu": 4, "bookkeeping": 3 + 2 + 2}
    use(evictor, requests, "B", 2, 3)  # Two more pairs.
    assert evictor.metadata_words() == {"lrfu": 6, "bookkeeping": 3 + 3 + 6}
    assert evictor.evict()[0] == 1
    assert evictor.metadata_words() == {"lrfu": 4, "bookkeeping": 2 + 6}
    free(evictor, 2, 3)
    evictor.evict()
    evictor.evict()
    # Nothing cached; A and B still remember their three pairs.
    assert evictor.metadata_words() == {"lrfu": 0, "bookkeeping": 6}
    assert evictor.summary()["metadata_bytes"] == 8 * 6


def test_max_resident_blocks_is_the_peak():
    evictor, _ = lrfu(sample_size=10)
    access(evictor, 1, 2, 3)
    free(evictor, 1, 2)
    evictor.evict()
    evictor.evict()
    access(evictor, 4)
    assert len(evictor.score) == 2
    assert evictor.stats["max_resident_blocks"] == 3
    access(evictor, 5, 6)
    access(evictor, 6)  # A hit is not a new resident block.
    assert evictor.stats["max_resident_blocks"] == 4


def test_write_stats_writes_the_summary(tmp_path):
    evictor, _ = lrfu()
    access(evictor, 1, 2)
    free(evictor, 1, 2)
    evictor.evict()
    path = tmp_path / "stats.json"
    evictor.write_stats(str(path))
    assert json.loads(path.read_text()) == evictor.summary()


@pytest.mark.parametrize("make", [
    lambda path: SampledLRFUEvictor(stats_path=path, request=Requests()),
    lambda path: SampledLRFUEvictor.from_params(f"half_life=8,stats={path}"),
])
def test_stats_path_writes_the_summary_at_exit(monkeypatch, tmp_path, make):
    hooks = []
    monkeypatch.setattr(sampled_lrfu.atexit, "register",
                        lambda func, *args: hooks.append((func, args)))
    path = tmp_path / "lrfu_stats.json"
    evictor = make(str(path))
    assert len(hooks) == 1
    func, args = hooks[0]
    assert func.__self__ is evictor and args == (str(path), )
    access(evictor, 1)
    free(evictor, 1)
    evictor.evict()
    func(*args)  # What the interpreter runs at exit.
    written = json.loads(path.read_text())
    assert written == evictor.summary() and written["evictions"] == 1


def test_no_exit_hook_without_a_stats_path(monkeypatch):
    hooks = []
    monkeypatch.setattr(sampled_lrfu.atexit, "register",
                        lambda func, *args: hooks.append((func, args)))
    SampledLRFUEvictor(request=Requests())
    SampledLRFUEvictor.from_params("half_life=8")
    SampledLRFUEvictor.from_params("stats=")
    assert hooks == []


# --- Randomized check against an independent reference ----------------------


class Reference:
    """The spec, kept on its own. Per cached block: s_i, tau_i, its counted
    access times, and which caching of its id it is (its life). Per request:
    the (block id, life) pairs it counted; with `memory`, only the `memory`
    requests that counted most recently are remembered (None: all, the spec
    itself). The candidate array (add order, swap-removal) and its own
    random.Random(seed) for the draws. S_i(T) is compared scaled, so no
    score underflows."""

    def __init__(self, half_life, sample_size, seed, memory=None):
        self.half_life, self.sample_size = half_life, sample_size
        self.memory = memory
        self.rng = random.Random(seed)
        self.now = 0
        self.s, self.tau, self.times, self.life = {}, {}, {}, {}
        self.lives = Counter()  # Cachings of each block id so far.
        self.born = {}  # (block id, life) -> the clock when it was cached.
        self.counted = OrderedDict()  # Request -> its pairs, oldest first.
        self.ever = set()  # Every (request, block id, life) counted.
        self.pool, self.hash = [], {}

    def access(self, block_id, request) -> str:
        """What the access is: not counted, or counted (and how)."""
        cached = block_id in self.s
        if (cached and request is not None and
            (block_id, self.life[block_id]) in self.counted.get(request, ())):
            return "not_counted"
        self.now += 1
        if cached:
            self.s[block_id] = (
                2.0**(-(self.now - self.tau[block_id]) / self.half_life) *
                self.s[block_id] + 1.0)
        else:
            self.lives[block_id] += 1
            self.life[block_id] = life = self.lives[block_id]
            self.born[block_id, life] = self.now
            self.s[block_id], self.times[block_id] = 1.0, []
        self.tau[block_id] = self.now
        self.times[block_id].append(self.now)
        if request is None:
            return "counted"
        life = self.life[block_id]
        if (request, block_id, life) in self.ever:
            kind = "recounted_after_forgetting"
        elif any(
            (request, block_id, old) in self.ever for old in range(1, life)):
            kind = "counted_new_life"
        else:
            kind = "counted"
        self.ever.add((request, block_id, life))
        if request in self.counted:
            self.counted.move_to_end(request)
        else:
            self.counted[request] = set()
            if self.memory is not None and len(self.counted) > self.memory:
                self.counted.popitem(last=False)
        self.counted[request].add((block_id, life))
        return kind

    def full_sum(self, block_id, now):
        return full_sum(self.times[block_id], now, self.half_life)

    def add(self, block_id, content_hash):
        if block_id not in self.hash:
            self.pool.append(block_id)
        self.hash[block_id] = content_hash

    def take(self, block_id):
        take(self.pool, block_id)
        return self.hash.pop(block_id)

    def draw(self):
        """The drawn candidates, their scaled S(now), and the spec's victim.
        """
        n = len(self.pool)
        drawn = [
            self.pool[i]
            for i in self.rng.sample(range(n), min(self.sample_size, n))
        ]
        scaled = {
            b: scaled_score(self.s[b], self.now - self.tau[b], self.half_life)
            for b in drawn
        }
        return min(drawn, key=lambda b: (scaled[b], b)), drawn, scaled

    def evict(self, victim):
        content_hash = self.take(victim)
        for table in (self.s, self.tau, self.times, self.life):
            del table[victim]
        return content_hash

    def pairs(self):
        return sum(len(p) for p in self.counted.values())


class Workload:
    """Requests reading prompts of cached blocks: each request makes one
    burst of accesses when it starts (repeats allowed) and gives every
    reference back when it finishes. A finished request often comes back
    later (preempted and recomputed), after others have used, evicted and
    re-cached its blocks. Evictions, updates, re-adds and misuses are mixed
    in."""

    BLOCKS = 40
    MAX_RUNNING = 4

    def __init__(self, seed, params, memory=None):
        self.rng = random.Random(10_000 + seed)
        self.requests = Requests()
        self.evictor = SampledLRFUEvictor(request=self.requests,
                                          seed=seed,
                                          **params)
        self.ref = Reference(params.get("half_life", DEFAULTS["half_life"]),
                             params.get("sample_size",
                                        DEFAULTS["sample_size"]),
                             seed,
                             memory=memory)
        self.refs = {}  # References per cached block (0 = idle).
        self.content = {}  # Content hash per cached block.
        self.running = []  # (request, the blocks it holds) per request.
        self.finished = []  # Requests that may come back.
        self.ever_evicted = set()
        self.next_request = self.next_hash = self.peak = 0
        self.counts = Counter()

    def start(self):
        if len(self.running) >= self.MAX_RUNNING:
            return
        returning = False
        choice = self.rng.random()
        if choice < 0.15:
            request = None
            self.counts["none_request"] += 1
        elif choice < 0.5 and self.finished:
            request = self.finished.pop(self.rng.randrange(len(self.finished)))
            returning = True
            self.counts["returned"] += 1
        else:
            request = f"req-{self.next_request}"
            self.next_request += 1
        self.requests.value = request
        blocks = []
        for _ in range(self.rng.randint(1, 6)):
            b = self.rng.randrange(self.BLOCKS)
            if b in self.ref.hash:  # A hit on an idle block takes it out.
                last = self.ref.pool[-1] == b
                self.counts["remove_last" if last else "remove_middle"] += 1
                self.evictor.remove(b)
                self.ref.take(b)
            if b not in self.content:
                self.next_hash += 1
                self.content[b] = self.next_hash
                self.refs[b] = 0
                self.counts["recached"] += b in self.ever_evicted
            kind = self.ref.access(b, request)
            self.evictor.record_access(b, self.content[b], 1, None)
            self.counts[kind] += 1
            if returning:
                self.counts[f"returned_{kind}"] += 1
            self.refs[b] += 1
            blocks.append(b)
        self.running.append((request, blocks))

    def finish(self):
        if not self.running:
            return
        request, blocks = self.running.pop(
            self.rng.randrange(len(self.running)))
        if request is not None:
            self.finished.append(request)
        for b in blocks:
            self.refs[b] -= 1
            if not self.refs[b]:
                self.evictor.add(b, self.content[b], 16, float(self.ref.now))
                self.ref.add(b, self.content[b])

    def evict(self):
        if not self.ref.pool:
            with pytest.raises(ValueError,
                               match="No usable cache memory left"):
                self.evictor.evict()
            self.counts["empty"] += 1
            return
        n = len(self.ref.pool)
        expected, drawn, scaled = self.ref.draw()
        victim, content_hash = self.evictor.evict()
        if victim != expected:
            # Only a tie beyond float resolution may go either way.
            assert victim in drawn, (victim, drawn)
            assert abs(log2_of(scaled[victim]) -
                       log2_of(scaled[expected])) < 1e-9, (victim, expected,
                                                           scaled)
            self.counts["near_ties"] += 1
        assert content_hash == self.ref.evict(victim)
        assert self.refs[victim] == 0  # Never a block in use.
        self.counts["evictions"] += 1
        self.counts["sampled" if len(drawn) < n else "whole_pool"] += 1
        self.counts["scored"] += len(drawn)
        # Candidates kept although their float S(now) was 0.0.
        self.counts["underflowed"] += sum(
            self.evictor.current(b) == 0.0 for b in drawn if b != victim)
        del self.refs[victim], self.content[victim]
        self.ever_evicted.add(victim)

    def idle_block(self):
        return self.rng.choice(self.ref.pool) if self.ref.pool else None

    def update(self):
        b = self.idle_block()
        if b is not None:
            self.evictor.update(b, self.rng.random() * 1e6)
            self.counts["update"] += 1

    def re_add(self):
        b = self.idle_block()
        if b is not None:
            self.next_hash += 1
            self.content[b] = self.next_hash
            self.evictor.add(b, self.content[b], 16, 0.0)
            self.ref.add(b, self.content[b])
            self.counts["re_add"] += 1

    def misuse(self):
        b = self.rng.randrange(self.BLOCKS)
        if b not in self.content:
            with pytest.raises(ValueError, match="without an access"):
                self.evictor.add(b, 1, 16, 0.0)
        elif b not in self.ref.hash:
            with pytest.raises(ValueError, match="not in the evictor"):
                self.evictor.remove(b)
            with pytest.raises(ValueError, match="not in the evictor"):
                self.evictor.update(b, 0.0)
        self.counts["misuse"] += 1

    def drain(self):
        """Every request finishes and the pool is evicted to empty (the
        last evictions score fewer than k candidates)."""
        while self.running:
            self.finish()
        while self.ref.pool:
            self.evict()
            self.check_pool()
        self.evict()  # Raises: nothing is left.

    def run(self, steps):
        for step in range(steps):
            if step % 750 == 749:
                self.drain()
            op = self.rng.random()
            if op < 0.3:
                self.start()
            elif op < 0.6:
                self.finish()
            elif op < 0.9:
                self.evict()
            elif op < 0.94:
                self.update()
            elif op < 0.97:
                self.re_add()
            else:
                self.misuse()
            self.peak = max(self.peak, len(self.ref.s))
            self.check_pool()
            if step % 50 == 0:
                self.check_scores()
                self.check_requests()
        self.check_scores()
        self.check_requests()

    def check_pool(self):
        ev, ref = self.evictor, self.ref
        check_pool(ev, ref.pool)
        assert ev._hash == ref.hash
        assert ev.now == ref.now
        assert set(ev.score) == set(ev.stamp) == set(ref.s)

    def check_scores(self):
        ev, ref = self.evictor, self.ref
        assert ev.stamp == ref.tau
        later = ref.now + self.rng.randrange(0, 30)
        for b in ref.s:
            assert ev.score[b] == pytest.approx(ref.s[b], rel=1e-12)
            for now in (ref.now, later):
                assert ev.current(b,
                                  now) == pytest.approx(ref.full_sum(b, now),
                                                        rel=1e-9,
                                                        abs=1e-300)

    def check_requests(self):
        """What each remembered request counted, and the overhead words."""
        ev, ref = self.evictor, self.ref
        assert len(ev._counted) <= sampled_lrfu.REQUEST_MEMORY
        assert {r: set(p)
                for r, p in ev._counted.items()} == {
                    r: {(b, ref.born[b, life])
                        for b, life in p}
                    for r, p in ref.counted.items()
                }
        assert ev._born == {b: ref.born[b, ref.life[b]] for b in ref.s}
        assert ev.metadata_words() == {
            "lrfu": 2 * len(ref.s),
            "bookkeeping": 3 * len(ref.pool) + len(ref.s) + 2 * ref.pairs()
        }


CONFIGS = {
    "defaults": {},
    "short": {
        "half_life": 2.0,
        "sample_size": 3
    },
    "lru-like": {
        "half_life": 0.5,
        "sample_size": 2
    },
    "tiny-half-life": {
        "half_life": 0.01,
        "sample_size": 4
    },
    "lfu-like": {
        "half_life": 1e9,
        "sample_size": 6
    },
    "one-candidate": {
        "half_life": 16.0,
        "sample_size": 1
    },
    "whole-pool": {
        "half_life": 8.0,
        "sample_size": 64
    },
    "request-memory-8": {
        "half_life": 4.0,
        "sample_size": 3,
        "request_memory": 8
    },
}


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", range(4))
def test_victims_match_an_independent_reference(config, seed, monkeypatch):
    params = dict(CONFIGS[config])
    memory = params.pop("request_memory", None)
    if memory is not None:
        monkeypatch.setattr(sampled_lrfu, "REQUEST_MEMORY", memory)
    work = Workload(seed, params, memory=memory)
    work.run(3000)
    counts, stats = work.counts, work.evictor.stats
    assert (stats["evictions"] == stats["scans"] == counts["evictions"] > 300)
    assert stats["candidates_scored"] == counts["scored"]
    assert stats["max_resident_blocks"] == work.peak > 30
    assert counts["near_ties"] <= counts["evictions"] // 100
    # The workload exercised every path the spec names.
    for key in ("counted", "not_counted", "none_request", "returned",
                "returned_not_counted", "returned_counted",
                "returned_counted_new_life", "remove_last", "remove_middle",
                "recached", "whole_pool", "empty", "update", "re_add",
                "misuse"):
        assert counts[key] > 0, key
    if params.get("sample_size", 4) < 10:
        assert counts["sampled"] > 0
    if params.get("half_life", 512.0) <= 0.01:
        assert counts["underflowed"] > 100
    if memory is None:
        # No request was forgotten: the spec's once-per-request, exactly.
        assert work.next_request < sampled_lrfu.REQUEST_MEMORY
        assert counts["recounted_after_forgetting"] == 0
    else:
        assert counts["returned_recounted_after_forgetting"] > 0
