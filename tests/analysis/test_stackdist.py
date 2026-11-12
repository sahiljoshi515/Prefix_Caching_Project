"""Reuse (stack) distances and LRU hit ratios (analysis/stackdist.py).

Every expected value is worked out by hand from the short streams below.
"""
import numpy as np
import pytest

from analysis import stackdist as sd

pytestmark = pytest.mark.m3


def test_reuse_distance_counts_distinct_other_keys():
    # b again after {c}; a again after {b, c} (b twice counts once).
    assert sd.reuse_distances(list("abcba")).tolist() == [-1, -1, -1, 1, 2]


def test_reuse_distance_of_an_immediate_repeat_is_zero():
    assert sd.reuse_distances(list("aa")).tolist() == [-1, 0]
    assert sd.reuse_distances(list("abba")).tolist() == [-1, -1, 0, 1]


def test_reuse_distances_of_integer_keys_and_an_empty_stream():
    # Content hashes are 64-bit integers, negative ones included.
    keys = np.array([2**62, -5, 2**62, -5], dtype=np.int64)
    assert sd.reuse_distances(keys).tolist() == [-1, -1, 1, 1]
    assert sd.reuse_distances([]).tolist() == []


def test_previous_positions():
    assert sd.previous_positions(list("abcba")).tolist() == [-1, -1, -1, 1, 0]


def test_lru_hit_ratio_on_a_tiny_stream():
    # a b c b a. Two slots: a, b, c miss (c evicts a), b hits, a misses.
    # Three slots: only the first three miss.
    distances = sd.reuse_distances(list("abcba"))
    assert sd.lru_hit_ratio(distances, 1) == 0.0
    assert sd.lru_hit_ratio(distances, 2) == pytest.approx(1 / 5)
    assert sd.lru_hit_ratio(distances, 3) == pytest.approx(2 / 5)
    assert sd.lru_hit_ratio(distances, [1, 2, 3]).tolist() == pytest.approx(
        [0.0, 0.2, 0.4])


def test_lru_hit_ratio_matches_a_hand_run_lru_cache():
    # a b a c b a, distances -1 -1 1 -1 2 2. Two slots: [a] [a b] hit a
    # [b a], c evicts b [a c], b evicts a [c b], a misses: 1 of 6. Three
    # slots: a, b and c only fill it, so the last three accesses hit.
    distances = sd.reuse_distances(list("abacba"))
    assert distances.tolist() == [-1, -1, 1, -1, 2, 2]
    assert sd.lru_hit_ratio(distances, 2) == pytest.approx(1 / 6)
    assert sd.lru_hit_ratio(distances, 3) == pytest.approx(3 / 6)


def test_lru_hit_ratio_of_an_empty_stream_is_none():
    assert sd.lru_hit_ratio([], 4) is None
