"""TTDF evictor and the leaf-only base it shares with RACE-RP (spec 007).

Tests against the policy's definition, not the implementation. Each cached
block keeps c_short, c_long and last_update; on every access at time t both
counters decay by exp(-(t - last_update) / tau) and grow by 1, which must
equal the full sums C(b, t) = sum_i exp(-(t - t_i) / tau) over the block's
access times t_i. At eviction time `now` the score is

    S = w_short / tau_short * c_short * exp(-(now - last) / tau_short)
      + w_long / tau_long * c_long * exp(-(now - last) / tau_long)

(defaults tau_short=5, tau_long=120, w_short=0.7, w_long=0.3). The victim is
the idle block (in the evictor) with no cached descendant that has the lowest
score, then the greatest depth, then the oldest last access, then the lowest
block id.

Hand-worked cases, the O(1) counters against the full sums, and a randomized
check of every victim, including those served from the heap kept between
evictions at one time, against a brute-force reference that recomputes the
argmin from scratch with its own counters.
"""
import json
import math
import random

import pytest

from vllm.core import reuse_evictors
from vllm.core.block import cache_events
from vllm.core.reuse_evictors import (TTDF_DEFAULTS, LeafScoreEvictor,
                                      TTDFEvictor, _Block, parse_params)

pytestmark = pytest.mark.m3

TTDF_KEYS = "stats, tau_long, tau_short, w_long, w_short"
# w_short / tau_short and w_long / tau_long at the defaults.
A_S, A_L = 0.7 / 5.0, 0.3 / 120.0


@pytest.fixture(autouse=True)
def _restore_request_context(monkeypatch):
    """Evictors switch on the module-wide request context; undo that."""
    monkeypatch.setattr(cache_events._context, "enabled",
                        cache_events._context.enabled)


class Clock:
    """A fake event clock: every read gives the same time until moved."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def h(block_id):
    """A distinct content hash per block id."""
    return 1000 + block_id


def ttdf(t=0.0, **params):
    clock = Clock(t)
    return TTDFEvictor(clock=clock, **params), clock


def access(evictor, block_id, parent=None, depth=None):
    """A request uses block `block_id` (lookup hit, insertion or dedupe)."""
    if depth is None:
        depth = 1 if parent is None else evictor.blocks[h(parent)].depth + 1
    evictor.record_access(block_id, h(block_id), depth,
                          None if parent is None else h(parent))


def chain(evictor, *block_ids):
    """One request computes `block_ids` as a prompt, root to tip."""
    parent = None
    for block_id in block_ids:
        access(evictor, block_id, parent)
        parent = block_id


def free(evictor, *block_ids):
    for block_id in block_ids:
        evictor.add(block_id, h(block_id), 16, 0.0)


def evict_order(evictor):
    order = []
    while evictor.num_blocks:
        block_id, content_hash = evictor.evict()
        assert content_hash == h(block_id)
        order.append(block_id)
    return order


def block(evictor, block_id):
    return evictor.blocks[h(block_id)]


def full_sum(times, now, tau):
    return sum(math.exp(-(now - t) / tau) for t in times)


def spec_score(times,
               now,
               tau_short=5.0,
               tau_long=120.0,
               w_short=0.7,
               w_long=0.3):
    return (w_short / tau_short * full_sum(times, now, tau_short) +
            w_long / tau_long * full_sum(times, now, tau_long))


# --- Settings ---------------------------------------------------------------


def test_parse_params_reads_key_value_pairs():
    assert parse_params("tau_short=2,w_long=0.5", TTDF_DEFAULTS) == {
        "tau_short": "2",
        "w_long": "0.5"
    }


def test_parse_params_strips_spaces_and_skips_empty_items():
    assert parse_params(" tau_short = 2 ,, w_long=0.5 ,", TTDF_DEFAULTS) == {
        "tau_short": "2",
        "w_long": "0.5"
    }


@pytest.mark.parametrize("text", [None, "", " , ,"])
def test_parse_params_of_nothing_is_empty(text):
    assert parse_params(text, TTDF_DEFAULTS) == {}


def test_parse_params_always_allows_stats_and_keeps_later_equals_signs():
    assert parse_params("stats=/x/a=b.json", ()) == {"stats": "/x/a=b.json"}


def test_parse_params_last_value_of_a_repeated_key_wins():
    assert parse_params("w_long=1,w_long=2", TTDF_DEFAULTS) == {"w_long": "2"}


@pytest.mark.parametrize("text", [
    "tau=2", "tau_short", "tau_short=1,alpha=2", "=1", "sketch=s.json",
    "Tau_short=1"
])
def test_parse_params_rejects_unknown_keys_and_items_without_equals(text):
    with pytest.raises(ValueError,
                       match=f"VLLM_EVICTION_PARAMS: .* {TTDF_KEYS} "):
        parse_params(text, TTDF_DEFAULTS)


def test_from_params_defaults():
    evictor = TTDFEvictor.from_params(None)
    assert (evictor.tau_short, evictor.tau_long, evictor.w_short,
            evictor.w_long) == (5.0, 120.0, 0.7, 0.3)
    assert TTDFEvictor.from_params("").tau_short == 5.0


def test_from_params_overrides_some_defaults():
    evictor = TTDFEvictor.from_params("tau_short=2, w_long=0.5")
    assert (evictor.tau_short, evictor.tau_long, evictor.w_short,
            evictor.w_long) == (2.0, 120.0, 0.7, 0.5)
    assert all(
        type(v) is float for v in (evictor.tau_short, evictor.tau_long,
                                   evictor.w_short, evictor.w_long))


@pytest.mark.parametrize(
    "text", ["tau_short=0", "tau_long=0", "tau_long=-1", "tau_short=-0.5"])
def test_from_params_rejects_non_positive_time_constants(text):
    with pytest.raises(ValueError, match="must be positive"):
        TTDFEvictor.from_params(text)


@pytest.mark.parametrize("kwargs", [{
    "tau_short": 0.0
}, {
    "tau_long": 0.0
}, {
    "tau_short": -1.0,
    "tau_long": 5.0
}])
def test_constructor_rejects_non_positive_time_constants(kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        TTDFEvictor(clock=Clock(), **kwargs)


@pytest.mark.parametrize("text", ["sketch=s.json", "alpha=2", "tau=1"])
def test_from_params_rejects_settings_of_other_policies(text):
    with pytest.raises(ValueError, match="VLLM_EVICTION_PARAMS"):
        TTDFEvictor.from_params(text)


def test_from_params_rejects_a_non_number():
    with pytest.raises(ValueError):
        TTDFEvictor.from_params("tau_short=fast")


# --- Clock and initial state ------------------------------------------------


def test_default_clock_is_the_block_managers_event_time(monkeypatch):
    monkeypatch.setattr(cache_events._context, "enabled", False)
    monkeypatch.setattr(cache_events._context, "time", 123.5)
    evictor = TTDFEvictor()
    # The block manager now stamps each call with its start time.
    assert cache_events._context.enabled
    assert evictor.clock is cache_events.event_time
    access(evictor, 1)
    access(evictor, 1)  # A second request in the same call: no decay.
    assert block(evictor, 1).last == 123.5
    assert (block(evictor, 1).c_short, block(evictor, 1).c_long) == (2.0, 2.0)
    free(evictor, 1)
    monkeypatch.setattr(cache_events._context, "time", 128.5)
    assert evictor.score(block(evictor, 1), evictor.clock()) == pytest.approx(
        spec_score([123.5, 123.5], 128.5), rel=1e-12)
    assert evictor.evict() == (1, h(1))


def test_counters_are_zero_before_the_first_access():
    b = _Block(3, 77, 55)
    assert (b.c_short, b.c_long, b.children, b.version, b.idle,
            b.block_id) == (0.0, 0.0, 0, 0, False, None)
    assert (b.depth, b.parent, b.family) == (3, 77, 55)


def test_first_access_sets_both_counters_to_one():
    evictor, _ = ttdf(7.0)
    access(evictor, 1)
    b = block(evictor, 1)
    assert (b.c_short, b.c_long, b.last) == (1.0, 1.0, 7.0)
    assert (b.block_id, b.depth, b.parent, b.idle) == (1, 1, None, False)
    assert 1 not in evictor and evictor.num_blocks == 0


def test_new_blocks_link_to_their_cached_parent():
    evictor, _ = ttdf()
    chain(evictor, 1, 2, 3)
    assert [block(evictor, b).children for b in (1, 2, 3)] == [1, 1, 0]
    assert [block(evictor, b).parent for b in (1, 2, 3)] == [None, h(1), h(2)]
    assert [block(evictor, b).depth for b in (1, 2, 3)] == [1, 2, 3]
    # RACE-RP's family: the block itself at depth 1 or 2, else the second
    # block's.
    assert [block(evictor, b).family for b in (1, 2, 3)] == [h(1), h(2), h(2)]
    # A second access does not add a second child.
    access(evictor, 2, parent=1)
    assert block(evictor, 1).children == 1
    # A block whose parent is not cached has no family and no link.
    access(evictor, 9, parent=8, depth=3)
    assert block(evictor, 9).family is None
    assert block(evictor, 9).parent is None
    assert h(8) not in evictor.blocks


def test_base_evictor_has_no_score():
    evictor = LeafScoreEvictor(clock=Clock())
    access(evictor, 1)
    assert (block(evictor, 1).c_short, block(evictor, 1).c_long) == (0.0, 0.0)
    assert evictor.metadata_words() == {"dependency": 4}
    free(evictor, 1)
    with pytest.raises(NotImplementedError):
        evictor.evict()


# --- O(1) counters equal the full sums --------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_counters_equal_the_full_sums_over_all_accesses(seed):
    rng = random.Random(seed)
    params = {
        "tau_short": rng.uniform(0.5, 20.0),
        "tau_long": rng.uniform(20.0, 600.0),
        "w_short": rng.uniform(0.0, 1.0),
        "w_long": rng.uniform(0.0, 1.0),
    }
    evictor, clock = ttdf(rng.uniform(0.0, 100.0), **params)
    times = {1: [], 2: [], 3: []}
    for _ in range(300):
        gaps = [0.0, rng.uniform(0.0, 2.0), rng.expovariate(1 / 20.0)]
        clock.t += rng.choice(gaps)
        block_id = rng.choice([1, 2, 3])
        access(evictor, block_id)
        times[block_id].append(clock.t)
        b = block(evictor, block_id)
        c_short = full_sum(times[block_id], clock.t, params["tau_short"])
        c_long = full_sum(times[block_id], clock.t, params["tau_long"])
        assert b.last == clock.t
        assert b.c_short == pytest.approx(c_short, rel=1e-12)
        assert b.c_long == pytest.approx(c_long, rel=1e-12)
        # Lazily decayed to a later eviction time, for every block.
        now = clock.t + rng.choice([0.0, rng.uniform(0.0, 200.0)])
        for other, other_times in times.items():
            if other_times:
                expected = spec_score(other_times, now, **params)
                actual = evictor.score(block(evictor, other), now)
                assert actual == pytest.approx(expected, rel=1e-12, abs=1e-300)


# --- Hand-worked victims ----------------------------------------------------


def test_chain_freed_at_once_evicts_tip_first():
    evictor, clock = ttdf()
    chain(evictor, 1, 2, 3)  # One request at t=0: all three have (1, 1).
    clock.t = 1.0
    free(evictor, 1, 2, 3)
    clock.t = 2.0
    scores = {evictor.score(block(evictor, b), 2.0) for b in (1, 2, 3)}
    assert len(scores) == 1
    # Only the tip has no cached child; each parent becomes the only leaf
    # once its child goes. The lowest id (1) never wins over the leaf rule.
    assert evict_order(evictor) == [3, 2, 1]
    assert evictor.stats["scans"] == 1


def test_equal_scores_go_to_the_deeper_leaf_then_the_lower_id():
    evictor, clock = ttdf()
    chain(evictor, 1, 2)
    access(evictor, 0)
    free(evictor, 0, 1, 2)
    clock.t = 1.0
    # Leaves 2 (depth 2) and 0 (depth 1) tie on score: 2 first. Then 0 and 1
    # tie on score, depth and last access: the lower id first.
    assert evict_order(evictor) == [2, 0, 1]


def test_tie_breaks_are_depth_then_oldest_access_then_lowest_id():
    # With both weights 0 every score is 0, so only the tie-breaks decide.
    evictor, clock = ttdf(w_short=0.0, w_long=0.0)
    chain(evictor, 5, 6)  # 5 stays in use; 6 is a depth-2 block.
    clock.t = 2.0
    access(evictor, 12)
    access(evictor, 11)
    clock.t = 3.0
    access(evictor, 10)
    clock.t = 3.5
    access(evictor, 6, parent=5)  # The deep block is also the newest.
    free(evictor, 10, 11, 12, 6)
    clock.t = 4.0
    assert evict_order(evictor) == [6, 11, 12, 10]
    assert h(5) in evictor.blocks  # In use: never a candidate.


def test_more_frequently_used_block_is_kept():
    # A and B are computed at t=0, c = (1, 1) each. B is hit again at t=1
    # and t=2 while in use:
    #   t=1: c_s = e^-0.2 + 1 = 1.81873, c_l = e^(-1/120) + 1 = 1.99170
    #   t=2: c_s = 1.81873 e^-0.2 + 1 = 2.48905,
    #        c_l = 1.99170 e^(-1/120) + 1 = 2.97517
    # At t=3, with w_s / tau_s = 0.14 and w_l / tau_l = 0.0025:
    #   S_A = 0.14 e^-0.6 + 0.0025 e^-0.025 = 0.07927
    #   S_B = 0.14 * 2.48905 e^-0.2 + 0.0025 * 2.97517 e^(-1/120) = 0.29268
    a, b = 1, 2
    evictor, clock = ttdf()
    access(evictor, a)
    access(evictor, b)
    for t in (1.0, 2.0):
        clock.t = t
        access(evictor, b)
    assert block(evictor, b).c_short == pytest.approx(2.48905, abs=1e-5)
    assert block(evictor, b).c_long == pytest.approx(2.97517, abs=1e-5)
    free(evictor, a, b)
    clock.t = 3.0
    scores = [evictor.score(block(evictor, x), 3.0) for x in (a, b)]
    assert scores == pytest.approx([0.07927, 0.29268], abs=1e-5)
    assert evict_order(evictor) == [a, b]


def test_more_recently_used_block_is_kept():
    # A is used once at t=0, B once at t=4. At t=5:
    #   S_A = 0.14 e^-1 + 0.0025 e^(-5/120) = 0.05150 + 0.00240 = 0.05390
    #   S_B = 0.14 e^-0.2 + 0.0025 e^(-1/120) = 0.11462 + 0.00248 = 0.11710
    # B has the lower id and would go first on a tie: the score decides.
    a, b = 2, 1
    evictor, clock = ttdf()
    access(evictor, a)
    clock.t = 4.0
    access(evictor, b)
    free(evictor, a, b)
    clock.t = 5.0
    scores = [evictor.score(block(evictor, x), 5.0) for x in (a, b)]
    assert scores == pytest.approx([0.05390, 0.11710], abs=1e-5)
    assert evict_order(evictor) == [a, b]


@pytest.mark.parametrize("now,victim", [(21.0, "F"), (34.0, "F"), (35.0, "R"),
                                        (100.0, "R")])
def test_recent_block_kept_early_frequent_block_kept_after_a_long_gap(
        now, victim):
    # F (frequent) is used by five requests at t=0: c_s = c_l = 5. R
    # (recent) is used once, at t=20: c_s = c_l = 1.
    # At t=21:  S_F = 0.14*5 e^(-21/5) + 0.0025*5 e^(-21/120)
    #               = 0.0104969 + 0.0104932 = 0.0209901
    #           S_R = 0.14 e^(-1/5) + 0.0025 e^(-1/120)
    #               = 0.1146223 + 0.0024793 = 0.1171016
    #   R's short term keeps it: F is evicted.
    # At t=100: S_F = 0.14*5 e^-20 + 0.0025*5 e^(-100/120)
    #               = 1.4e-9 + 0.0054325 = 0.0054325
    #           S_R = 0.14 e^-16 + 0.0025 e^(-80/120)
    #               = 1.6e-8 + 0.0012835 = 0.0012836
    #   Both short terms are gone and F's larger long counter keeps it: R is
    #   evicted. The scores cross at t = 34.38.
    f, r = 1, 2
    evictor, clock = ttdf()
    for _ in range(5):
        access(evictor, f)
    clock.t = 20.0
    access(evictor, r)
    free(evictor, f, r)
    clock.t = now
    s_f = evictor.score(block(evictor, f), now)
    s_r = evictor.score(block(evictor, r), now)
    expected = {21.0: (0.0209901, 0.1171016), 100.0: (0.0054325, 0.0012836)}
    if now in expected:
        assert (s_f, s_r) == pytest.approx(expected[now], rel=1e-4)
    assert evictor.evict()[0] == {"F": f, "R": r}[victim]


def test_hit_on_a_block_in_use_counts():
    evictor, clock = ttdf()
    access(evictor, 1)  # t=0: (1, 1)
    access(evictor, 1)  # Another request at t=0: (2, 2), no decay.
    assert (block(evictor, 1).c_short, block(evictor, 1).c_long) == (2.0, 2.0)
    clock.t = 5.0
    access(evictor, 1)
    # c_s = 2 e^-1 + 1 = 1.73576, c_l = 2 e^(-5/120) + 1 = 2.91838
    assert block(evictor, 1).c_short == pytest.approx(1.73576, abs=1e-5)
    assert block(evictor, 1).c_long == pytest.approx(2.91838, abs=1e-5)
    assert block(evictor, 1).last == 5.0
    assert 1 not in evictor
    access(evictor, 2)  # Used once, at the same time.
    free(evictor, 1, 2)
    clock.t = 6.0
    assert evict_order(evictor) == [2, 1]


def test_time_is_frozen_within_a_call():
    evictor, clock = ttdf()
    for t, block_id in ((0.0, 1), (1.0, 2), (2.0, 3)):
        clock.t = t
        access(evictor, block_id)
    free(evictor, 1, 2, 3)
    # One allocate call at t=10 makes room for two new blocks: both
    # evictions and both insertions get the call's time.
    clock.t = 10.0
    assert evictor.evict()[0] == 1
    access(evictor, 4)
    assert evictor.evict()[0] == 2
    access(evictor, 5)
    assert block(evictor, 4).last == block(evictor, 5).last == 10.0
    assert evictor.stats["scans"] == 1  # One scan serves the whole call.
    # Freed later at the same time, 4 joins the kept heap. Used at t=10 it
    # outscores 3 (used at t=2).
    free(evictor, 4)
    assert evict_order(evictor) == [3, 4]
    assert evictor.stats["scans"] == 1


# --- Eligibility ------------------------------------------------------------


def test_blocks_in_use_are_never_evicted():
    evictor, clock = ttdf()
    access(evictor, 1)
    access(evictor, 2)
    free(evictor, 2)
    clock.t = 100.0  # 1 has by far the lower score... but is in use.
    assert evictor.evict()[0] == 2
    with pytest.raises(ValueError, match="No usable cache memory left"):
        evictor.evict()
    assert h(1) in evictor.blocks


def test_ancestor_is_kept_while_its_child_is_cached_despite_a_lower_score():
    evictor, clock = ttdf()
    chain(evictor, 1, 2)
    clock.t = 5.0
    access(evictor, 2, parent=1)  # Only the child is used again.
    free(evictor, 1, 2)
    clock.t = 6.0
    assert evictor.score(block(evictor, 1), 6.0) < evictor.score(
        block(evictor, 2), 6.0)
    assert evict_order(evictor) == [2, 1]


def test_new_child_takes_its_parent_out_of_the_candidates():
    evictor, clock = ttdf()
    access(evictor, 1)  # t=0, by a request that keeps running.
    clock.t = 0.5
    access(evictor, 3)
    free(evictor, 3)
    clock.t = 1.0
    access(evictor, 2, parent=1)  # It fills a new block below 1...
    free(evictor, 2, 1)  # ...and finishes.
    assert block(evictor, 1).children == 1
    assert set(evictor.leaves) == {h(2), h(3)}
    clock.t = 2.0
    s1, s2, s3 = (evictor.score(block(evictor, b), 2.0) for b in (1, 2, 3))
    assert s1 < s3 < s2
    # 1 has the lowest score but is not a candidate until 2 is gone.
    assert evict_order(evictor) == [3, 2, 1]


def test_new_child_of_an_idle_candidate_takes_it_out_at_once():
    evictor, _ = ttdf()
    access(evictor, 1)
    free(evictor, 1)
    assert h(1) in evictor.leaves
    access(evictor, 2, parent=1)
    assert h(1) not in evictor.leaves and 1 in evictor


def test_parent_with_another_cached_child_stays_out():
    evictor, clock = ttdf()
    chain(evictor, 1, 2)
    access(evictor, 3, parent=1)
    free(evictor, 1, 2, 3)
    clock.t = 1.0
    assert evictor.evict()[0] == 2
    assert h(1) not in evictor.leaves and block(evictor, 1).children == 1
    assert evictor.evict()[0] == 3
    assert h(1) in evictor.leaves
    assert evictor.evict()[0] == 1


def test_parent_in_use_does_not_become_a_candidate():
    evictor, clock = ttdf()
    chain(evictor, 1, 2)
    free(evictor, 2)
    clock.t = 1.0
    assert evictor.evict()[0] == 2
    assert block(evictor, 1).children == 0
    assert h(1) not in evictor.leaves and evictor.num_blocks == 0
    free(evictor, 1)  # Freed later, it is a leaf at once.
    assert h(1) in evictor.leaves
    assert evictor.evict()[0] == 1


def test_reinserted_block_starts_with_fresh_counters():
    evictor, clock = ttdf()
    access(evictor, 1)
    for t in (1.0, 2.0):
        clock.t = t
        access(evictor, 1)
    assert block(evictor, 1).c_short > 2.0
    free(evictor, 1)
    clock.t = 3.0
    assert evictor.evict()[0] == 1
    assert h(1) not in evictor.blocks
    clock.t = 4.0
    access(evictor, 1)  # Computed again: a new resident block.
    assert (block(evictor, 1).c_short, block(evictor, 1).c_long,
            block(evictor, 1).last) == (1.0, 1.0, 4.0)


def test_reinserted_block_counts_as_its_parents_child_again():
    evictor, clock = ttdf()
    chain(evictor, 1, 2)
    free(evictor, 2)
    clock.t = 1.0
    assert evictor.evict()[0] == 2
    assert block(evictor, 1).children == 0
    access(evictor, 2, parent=1)
    assert block(evictor, 1).children == 1
    free(evictor, 2, 1)
    assert evict_order(evictor) == [2, 1]


# --- Evictor interface ------------------------------------------------------


def test_membership_and_count_follow_add_and_remove():
    evictor, _ = ttdf()
    chain(evictor, 1, 2)
    assert evictor.num_blocks == 0 and 1 not in evictor
    free(evictor, 1, 2)
    assert evictor.num_blocks == 2 and 1 in evictor and 2 in evictor
    assert block(evictor, 1).idle
    evictor.remove(1)
    assert evictor.num_blocks == 1 and 1 not in evictor and 2 in evictor
    assert not block(evictor, 1).idle


def test_add_of_a_block_never_accessed_raises():
    evictor, _ = ttdf()
    with pytest.raises(ValueError, match="freed without an access"):
        free(evictor, 1)
    assert evictor.num_blocks == 0


def test_remove_and_update_of_a_block_not_in_the_evictor_raise():
    evictor, _ = ttdf()
    access(evictor, 1)  # In use.
    with pytest.raises(ValueError, match="not in the evictor"):
        evictor.remove(1)
    with pytest.raises(ValueError, match="not in the evictor"):
        evictor.update(1, 5.0)
    with pytest.raises(ValueError, match="not in the evictor"):
        evictor.remove(99)
    with pytest.raises(ValueError, match="not in the evictor"):
        evictor.update(99, 5.0)


def test_evict_from_an_empty_evictor_raises():
    evictor, _ = ttdf()
    with pytest.raises(ValueError, match="No usable cache memory left"):
        evictor.evict()
    access(evictor, 1)
    with pytest.raises(ValueError, match="No usable cache memory left"):
        evictor.evict()


def test_update_changes_nothing():
    evictor, clock = ttdf()
    for t, block_id in ((0.0, 1), (1.0, 2), (2.0, 3)):
        clock.t = t
        access(evictor, block_id)
    free(evictor, 1, 2, 3)
    clock.t = 5.0
    assert evictor.evict()[0] == 1
    b = block(evictor, 2)
    before = (b.version, b.c_short, b.c_long, b.last)
    evictor.update(2, 1e9)  # A new access stamp on the oldest block...
    assert (b.version, b.c_short, b.c_long, b.last) == before
    assert evict_order(evictor) == [2, 3]  # ...changes no score or order.
    assert evictor.stats["scans"] == 1


# --- The heap kept between evictions at one time ----------------------------


def scan_counts(evictor):
    return evictor.stats["scans"], evictor.stats["candidates_scored"]


def three_idle_blocks():
    """Blocks 1, 2, 3 used once at t=0, 1, 2 and freed: scores 1 < 2 < 3."""
    evictor, clock = ttdf()
    for t, block_id in ((0.0, 1), (1.0, 2), (2.0, 3)):
        clock.t = t
        access(evictor, block_id)
    free(evictor, 1, 2, 3)
    clock.t = 10.0
    return evictor, clock


def test_heap_is_kept_at_one_time_and_rebuilt_when_time_moves():
    evictor, clock = three_idle_blocks()
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 2
    assert scan_counts(evictor) == (1, 3)
    clock.t = 11.0
    assert evictor.evict()[0] == 3
    assert scan_counts(evictor) == (2, 4)


def test_block_freed_at_the_heap_time_is_pushed():
    evictor, clock = three_idle_blocks()
    assert evictor.evict()[0] == 1
    clock.t = 0.0
    access(evictor, 0)  # A block last used long ago...
    clock.t = 10.0
    free(evictor, 0)  # ...freed at the heap's time is scored and pushed.
    assert evictor.stats["candidates_scored"] == 4
    assert evictor.evict()[0] == 0
    assert evictor.stats["scans"] == 1


def test_block_freed_after_time_moves_waits_for_the_rescan():
    evictor, clock = three_idle_blocks()
    assert evictor.evict()[0] == 1
    clock.t = 0.0
    access(evictor, 0)
    clock.t = 10.5
    free(evictor, 0)  # Not pushed: the heap kept for t=10 is dropped.
    assert evictor.stats["candidates_scored"] == 3
    assert evictor._heap_time is None
    assert evictor.evict()[0] == 0  # t=10.5: a fresh scan finds it.
    assert scan_counts(evictor) == (2, 6)


def test_parent_that_becomes_a_leaf_is_pushed():
    evictor, clock = ttdf()
    chain(evictor, 1, 2)
    clock.t = 5.0
    access(evictor, 3)  # Used more recently than the chain.
    free(evictor, 1, 2, 3)
    clock.t = 6.0
    assert evictor.evict()[0] == 2
    assert evictor.stats["candidates_scored"] == 3  # 2, 3, then 1 pushed.
    assert evictor.evict()[0] == 1
    assert evictor.evict()[0] == 3
    assert evictor.stats["scans"] == 1


def test_block_removed_at_the_heap_time_is_skipped():
    evictor, _ = three_idle_blocks()
    assert evictor.evict()[0] == 1
    evictor.remove(2)  # Next in the heap, but now in use.
    assert evictor.evict()[0] == 3
    assert evictor.stats["scans"] == 1
    with pytest.raises(ValueError, match="No usable cache memory left"):
        evictor.evict()


def test_block_used_and_freed_again_at_the_heap_time_is_rescored():
    evictor, clock = three_idle_blocks()
    assert evictor.evict()[0] == 1
    evictor.remove(2)
    for _ in range(3):
        access(evictor, 2)  # Three requests at t=10.
    free(evictor, 2)  # Pushed with its new score, above 3's.
    assert evictor.evict()[0] == 3
    assert evictor.evict()[0] == 2
    assert evictor.stats["scans"] == 1


def test_evicted_id_reused_for_new_content_at_the_heap_time():
    evictor, clock = three_idle_blocks()
    block_id, _ = evictor.evict()
    assert block_id == 1
    # The allocator puts new content (hash 5000) in physical block 1.
    evictor.record_access(1, 5000, 1, None)
    evictor.add(1, 5000, 16, 10.0)
    assert evictor.evict() == (2, h(2))
    assert evictor.evict() == (3, h(3))
    assert evictor.evict() == (1, 5000)


class _RescanOnAccess(TTDFEvictor):
    """A policy whose accesses change other blocks' scores."""

    def _rescan_after(self, block):
        return True


@pytest.mark.parametrize("cls,scans", [(TTDFEvictor, 1), (_RescanOnAccess, 2)])
def test_access_that_changes_other_scores_forces_a_rescan(cls, scans):
    clock = Clock()
    evictor = cls(clock=clock)
    for block_id in (1, 2, 3):
        access(evictor, block_id)
    free(evictor, 1, 2, 3)
    clock.t = 1.0
    evictor.evict()
    access(evictor, 4)
    evictor.evict()
    assert evictor.stats["scans"] == scans


def test_idle_candidate_accessed_at_the_heap_time_is_rescored():
    """Not from the allocator, which removes a block from the evictor before
    its access; the evictor keeps the heap whole anyway."""
    evictor, clock = ttdf()
    access(evictor, 1)  # t=0
    clock.t = 1.0
    access(evictor, 2)
    clock.t = 2.0
    for _ in range(5):
        access(evictor, 3)
    free(evictor, 1, 2, 3)
    clock.t = 3.0
    assert evictor.evict()[0] == 1
    access(evictor, 2)  # Idle, used at t=1 and t=3.
    assert 2 in evictor and h(2) in evictor.leaves
    assert evictor.stats["candidates_scored"] == 4  # Pushed again.
    # S_2 = 0.14 (e^-0.4 + 1) + 0.0025 (e^(-2/120) + 1) = 0.23880
    # S_3 = 0.14 * 5 e^-0.2 + 0.0025 * 5 e^(-1/120) = 0.58551
    scores = [evictor.score(block(evictor, b), 3.0) for b in (2, 3)]
    assert scores == pytest.approx([0.23880, 0.58551], abs=1e-5)
    assert evictor.evict()[0] == 2
    assert evictor.evict()[0] == 3
    assert evictor.stats["scans"] == 1


def test_idle_parent_accessed_at_the_heap_time_is_not_pushed():
    evictor, clock = ttdf()
    chain(evictor, 1, 2)
    free(evictor, 1, 2)
    clock.t = 1.0
    access(evictor, 3)
    free(evictor, 3)
    assert evictor.evict()[0] == 2  # The heap is kept for t=1; 1 is pushed.
    access(evictor, 4, parent=1)  # 1 has a child again...
    access(evictor, 1)  # ...so its access pushes nothing.
    assert evictor.stats["candidates_scored"] == 3
    assert evictor.evict()[0] == 3
    with pytest.raises(ValueError, match="No evictable leaf"):
        evictor.evict()  # 1 waits for its in-use child 4.


def spec_victim(histories, now):
    """The spec's argmin over depth-1 candidates {block id: access times}."""

    def key(block_id):
        times = histories[block_id]
        return spec_score(times, now), times[-1], block_id

    return min(histories, key=key)


@pytest.mark.parametrize("late", ["freed", "used"])
def test_clock_stepping_back_to_the_heap_time_rescans(late):
    """A clock that steps back to the kept heap's time, after a block became
    a candidate (freed) or changed (used while idle) at a later time, must
    not reuse that heap. Event times never step back."""
    evictor, clock = ttdf()
    histories = {1: [0.0], 2: [0.0], 3: [4.0] * 5}
    access(evictor, 1)
    access(evictor, 2)
    clock.t = 4.0
    for _ in range(5):
        access(evictor, 3)
    free(evictor, 1, 3)
    if late == "used":
        free(evictor, 2)
    clock.t = 5.0
    assert evictor.evict()[0] == 1  # The heap is kept for t=5.
    clock.t = 6.0
    if late == "freed":
        free(evictor, 2)
    else:
        access(evictor, 2)
        histories[2].append(6.0)
    assert evictor._heap_time is None  # Dropped, not reused later.
    clock.t = 5.0
    del histories[1]
    # At t=5, S_3 = 0.14 * 5 e^-0.2 + 0.0025 * 5 e^(-1/120) = 0.58551, and
    # S_2 = 0.05390 (freed) or, also used at t=6,
    # 0.14 (e^-1 + e^0.2) + 0.0025 (e^(-5/120) + e^(1/120)) = 0.22742.
    s_2 = {"freed": 0.05390, "used": 0.22742}[late]
    scores = [evictor.score(block(evictor, b), 5.0) for b in (2, 3)]
    assert scores == pytest.approx([s_2, 0.58551], abs=1e-5)
    expected = spec_victim(histories, 5.0)
    assert expected == 2
    assert evictor.evict()[0] == expected
    assert evictor.stats["scans"] == 2


def test_evict_rescans_when_the_kept_heap_misses_a_candidate():
    """The fallback rescan in `evict`. Every way a block becomes a candidate
    (freed, used while idle, or its last child evicted) pushes it onto the
    kept heap or drops the heap, so the heap misses a candidate only if the
    evictor's state is corrupted, as here."""
    evictor, _ = three_idle_blocks()
    assert evictor.evict()[0] == 1  # The heap is kept for t=10.
    evictor._heap.clear()  # Corrupt: 2 and 3 lose their entries.
    assert evictor.evict() == (2, h(2))
    assert evictor.stats["scans"] == 2
    assert evictor.evict() == (3, h(3))  # From the rescanned heap.
    assert evictor.stats["scans"] == 2


class _MostUsedFirst(LeafScoreEvictor):
    """A score that falls when a block is used (RACE-RP's can): minus the
    number of accesses."""

    def _on_access(self, content_hash, block, now):
        block.c_long += 1

    def _scores(self, hashes, blocks, now):
        return [-b.c_long for b in blocks]


def test_stale_entry_of_an_evicted_block_never_matches_its_new_self():
    clock = Clock()
    evictor = _MostUsedFirst(clock=clock)
    for block_id, uses in ((4, 5), (1, 3), (2, 2)):
        for _ in range(uses):
            access(evictor, block_id)
    free(evictor, 4, 1, 2)
    clock.t = 10.0
    assert evictor.evict()[0] == 4  # The heap kept for t=10: 1 (-3), 2 (-2).
    evictor.remove(1)
    access(evictor, 1)
    access(evictor, 1)
    free(evictor, 1)  # Pushed at -5; its -3 entry is stale.
    assert evictor.evict()[0] == 1  # The stale -3 entry stays in the heap.
    # 1 is cached again at the same time (-1), and changes as many times as
    # its earlier self had when scanned.
    access(evictor, 1)
    free(evictor, 1)
    evictor.remove(1)
    free(evictor, 1)
    assert block(evictor, 1).c_long == 1
    # 2 (-2) is the lowest; 1's earlier entry (-3) must not stand for it.
    assert evictor.evict()[0] == 2
    assert evictor.evict()[0] == 1


def test_idle_block_with_only_an_in_use_child_cannot_be_evicted():
    """The last error in `evict`: idle blocks but no leaf. Unreachable in a
    prefix-closed cache (a block in use keeps its ancestors in use), so the
    state is built by hand."""
    evictor, _ = ttdf()
    chain(evictor, 1, 2)
    free(evictor, 1)  # 1 idle while its child 2 is still in use.
    assert evictor.num_blocks == 1 and not evictor.leaves
    with pytest.raises(ValueError,
                       match="No evictable leaf block among the 1 idle"):
        evictor.evict()
    assert evictor.stats["scans"] == 2  # The fallback rescanned once.
    assert h(1) in evictor.blocks and 1 in evictor


# --- A block cached below an uncached parent --------------------------------
# The allocator never does this: a request holds a block's whole prefix.


@pytest.mark.xfail(
    strict=True,
    reason="Not reachable from the allocator (it never caches a block "
    "below an uncached parent): a parent cached after its child is not "
    "linked to it, so it can be evicted while that descendant is cached")
def test_parent_cached_after_its_child_is_kept_while_the_child_is_cached():
    evictor, _ = ttdf()
    access(evictor, 2, parent=1, depth=2)  # 1 is not cached.
    access(evictor, 1)
    free(evictor, 1)  # 2 stays in use.
    # 1 has a cached descendant, so nothing is eligible.
    with pytest.raises(ValueError):
        evictor.evict()


def test_parent_cached_after_its_child_is_evictable_once_the_child_goes():
    evictor, _ = ttdf()
    access(evictor, 2, parent=1, depth=2)  # 1 is not cached.
    access(evictor, 1)
    free(evictor, 2)
    assert evictor.evict()[0] == 2
    assert block(evictor, 1).children == 0  # Never counted, never -1.
    free(evictor, 1)
    assert evictor.evict()[0] == 1


# --- Overhead counters ------------------------------------------------------

SUMMARY_KEYS = {
    "evictions", "scans", "candidates_scored", "max_candidates",
    "evict_seconds", "score_seconds", "max_resident_blocks", "evict_us_mean",
    "candidates_per_scan", "metadata_words", "metadata_bytes"
}


def test_summary_before_any_eviction():
    evictor, _ = ttdf()
    summary = evictor.summary()
    assert set(summary) == SUMMARY_KEYS
    assert summary["evictions"] == summary["scans"] == 0
    assert summary["evict_us_mean"] is None
    assert summary["candidates_per_scan"] is None
    assert summary["metadata_words"] == {"dependency": 0, "counters": 0}
    assert summary["metadata_bytes"] == 0


def test_summary_counts_scans_candidates_and_metadata():
    evictor, clock = ttdf()
    chain(evictor, 1, 2)
    access(evictor, 3)
    access(evictor, 4)
    free(evictor, 1, 2, 3, 4)  # Leaves: 2, 3, 4.
    clock.t = 1.0
    assert evictor.evict()[0] == 2  # Scan of 3; then 1 is pushed.
    assert evictor.evict()[0] == 1  # From the kept heap.
    clock.t = 2.0
    assert evictor.evict()[0] == 3  # Scan of 4 and 3.
    summary = evictor.summary()
    assert set(summary) == SUMMARY_KEYS
    assert (summary["evictions"], summary["scans"],
            summary["candidates_scored"], summary["max_candidates"],
            summary["max_resident_blocks"]) == (3, 2, 6, 3, 4)
    assert summary["candidates_per_scan"] == 3.0
    assert summary["evict_seconds"] >= summary["score_seconds"] >= 0.0
    assert summary["evict_us_mean"] == pytest.approx(
        1e6 * summary["evict_seconds"] / 3)
    # One resident block (4): 4 dependency words and 3 counters.
    assert summary["metadata_words"] == {"dependency": 4, "counters": 3}
    assert summary["metadata_bytes"] == 8 * 7


def test_max_resident_blocks_is_the_peak():
    evictor, clock = ttdf()
    for block_id in (1, 2, 3):
        access(evictor, block_id)
    free(evictor, 1, 2)
    evictor.evict()
    evictor.evict()
    access(evictor, 4)
    assert len(evictor.blocks) == 2
    assert evictor.stats["max_resident_blocks"] == 3
    for block_id in (5, 6):
        access(evictor, block_id)
    access(evictor, 6)  # A hit is not a new resident block.
    assert evictor.stats["max_resident_blocks"] == 4


def test_write_stats_writes_the_summary(tmp_path):
    evictor, clock = three_idle_blocks()
    evictor.evict()
    path = tmp_path / "stats.json"
    evictor.write_stats(str(path))
    assert json.loads(path.read_text()) == evictor.summary()


@pytest.mark.parametrize("make", [
    lambda path: TTDFEvictor(clock=Clock(), stats_path=path),
    lambda path: TTDFEvictor.from_params(f"tau_short=2,stats={path}"),
])
def test_stats_path_writes_the_summary_at_exit(monkeypatch, tmp_path, make):
    hooks = []
    monkeypatch.setattr(reuse_evictors.atexit, "register",
                        lambda func, *args: hooks.append((func, args)))
    path = tmp_path / "ttdf_stats.json"
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
    monkeypatch.setattr(reuse_evictors.atexit, "register",
                        lambda func, *args: hooks.append((func, args)))
    TTDFEvictor(clock=Clock())
    TTDFEvictor.from_params("tau_short=2")
    TTDFEvictor.from_params("stats=")
    assert hooks == []


# --- Randomized check against a brute-force reference -----------------------

NEAR_TIE = 1e-11  # Relative score gap below which either order is right.


class RefBlock:

    def __init__(self, depth, parent, block_id):
        self.depth = depth
        self.parent = parent  # The prompt's previous block (content hash).
        self.block_id = block_id
        self.times = []  # Access times since it was cached.


class Reference:
    """The specification from scratch: full-sum counters over every access
    time, eligibility as idle with no cached descendant, and the argmin."""

    def __init__(self, tau_short, tau_long, w_short, w_long):
        self.tau_short, self.tau_long = tau_short, tau_long
        self.w_short, self.w_long = w_short, w_long
        self.cached = {}  # content hash -> RefBlock
        self.idle = set()  # cached hashes no running request uses

    def access(self, content_hash, block_id, depth, parent, now):
        b = self.cached.get(content_hash)
        if b is None:
            b = self.cached[content_hash] = RefBlock(depth, parent, block_id)
        b.block_id = block_id
        b.times.append(now)

    def evict(self, content_hash):
        del self.cached[content_hash]
        self.idle.discard(content_hash)

    def counters(self, content_hash, now):
        times = self.cached[content_hash].times
        c_short = full_sum(times, now, self.tau_short)
        c_long = full_sum(times, now, self.tau_long)
        return c_short, c_long

    def score(self, content_hash, now):
        c_short, c_long = self.counters(content_hash, now)
        return (self.w_short / self.tau_short * c_short +
                self.w_long / self.tau_long * c_long)

    def eligible(self):
        ancestors = set()  # Blocks with a cached descendant.
        for b in self.cached.values():
            parent = b.parent
            while parent is not None and parent not in ancestors:
                ancestors.add(parent)
                parent = self.cached[parent].parent
        return [x for x in self.idle if x not in ancestors]

    def signature(self, content_hash):
        """Equal for blocks whose scores are equal as real numbers."""
        if self.w_short == self.w_long == 0:
            return ()
        return tuple(self.cached[content_hash].times)

    def victims(self, now):
        """The spec's victim, and the blocks whose scores are too close to
        its own for floating point to order (normally none). The O(1)
        counters and the full sums round differently, by far less than
        NEAR_TIE of the score."""
        keyed = []
        for x in self.eligible():
            b = self.cached[x]
            keyed.append(
                ((self.score(x, now), -b.depth, b.times[-1], b.block_id), x))
        best_key, best = min(keyed)
        best_score, sig = best_key[0], self.signature(best)
        near = {
            x
            for k, x in keyed if self.signature(x) != sig
            and abs(k[0] - best_score) <= NEAR_TIE * best_score
        }
        return best, near


class Workload:
    """Requests sharing prompt prefixes over a small pool of physical blocks,
    driving the evictor the way vLLM's prefix-caching allocator does: a hit
    or dedupe removes an idle block from the evictor and then records the
    access; a miss takes a free block id, else evicts, and records the
    insertion under the cached previous block; a finished request frees its
    blocks, and a block whose last user finishes is added. Every access and
    eviction of one call has the call's time; many calls share a time. A
    few direct accesses to idle blocks (`touch_idle`), which the allocator
    never makes, and a clock that sometimes steps back to the last
    eviction's time (`tick`), which event times never do, check that the
    kept heap survives those too."""

    BLOCKS = 32
    MAX_DEPTH = 6
    NEW_CONTENT = 0.05  # Chance a prompt block is new below its parent.
    MAX_RUNNING = 4
    STEPS = (0.0, ) * 6 + (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

    def __init__(self, seed, params):
        self.rng = random.Random(seed)
        self.clock = Clock()
        self.evictor = TTDFEvictor(clock=self.clock, **params)
        self.ref = Reference(**{**TTDF_DEFAULTS, **params})
        self.tree = {None: []}  # Prompt contents: hash -> next blocks.
        self.depth = {}
        self.next_hash = 1
        self.free_ids = list(range(self.BLOCKS))
        self.refcount = {}  # cached hash -> running requests using it
        self.running = []  # The hashes each running request holds.
        self.changed = False  # Since the last eviction.
        self.counts = dict.fromkeys(
            ("evictions", "near_ties", "reused_heap", "reused_after_changes",
             "pushed_on_add", "parent_became_leaf", "reinserted", "hits",
             "revived", "idle_leaf_touched_at_heap_time", "stepped_back"), 0)
        self.last_evict = 0.0
        self.ever_cached = set()
        self.peak_resident = 0

    # --- vLLM operations ---

    def prompt(self):
        path, parent = [], None
        for depth in range(1, self.rng.randint(1, self.MAX_DEPTH) + 1):
            nexts = self.tree[parent]
            if nexts and self.rng.random() >= self.NEW_CONTENT:
                # Popular prefixes come first.
                content_hash = nexts[int(len(nexts) * self.rng.random()**3)]
            else:
                content_hash = self.next_hash
                self.next_hash += 1
                nexts.append(content_hash)
                self.tree[content_hash] = []
                self.depth[content_hash] = depth
            path.append(content_hash)
            parent = content_hash
        return path

    def allocate(self):
        held = []
        for content_hash in self.prompt():
            parent = held[-1] if held else None
            if content_hash in self.ref.cached:
                b = self.ref.cached[content_hash]
                assert parent == b.parent
                block_id = b.block_id
                if self.refcount[content_hash] == 0:
                    self.evictor.remove(block_id)
                    self.ref.idle.discard(content_hash)
                    self.counts["revived"] += 1
                self.refcount[content_hash] += 1
                self.counts["hits"] += 1
            else:
                if self.free_ids:
                    block_id = self.free_ids.pop()
                elif self.ref.idle:
                    block_id = self.evict()
                else:
                    break  # No room: the request keeps the prefix it has.
                self.refcount[content_hash] = 1
                if content_hash in self.ever_cached:
                    self.counts["reinserted"] += 1
                self.ever_cached.add(content_hash)
            depth = self.depth[content_hash]
            self.evictor.record_access(block_id, content_hash, depth, parent)
            self.ref.access(content_hash, block_id, depth, parent,
                            self.clock.t)
            held.append(content_hash)
            self.changed = True
            self.peak_resident = max(self.peak_resident, len(self.ref.cached))
        if held:
            self.running.append(held)

    def finish(self):
        held = self.running.pop(self.rng.randrange(len(self.running)))
        if self.rng.random() < 0.5:
            held = held[::-1]
        for content_hash in held:
            self.refcount[content_hash] -= 1
            if self.refcount[content_hash] == 0:
                b = self.ref.cached[content_hash]
                if (self.evictor._heap_time == self.clock.t
                        and self.evictor.blocks[content_hash].children == 0):
                    self.counts["pushed_on_add"] += 1
                self.evictor.add(b.block_id, content_hash, 16 * b.depth,
                                 self.clock.t)
                self.ref.idle.add(content_hash)
                self.changed = True

    def evict(self):
        now = self.clock.t
        best, near = self.ref.victims(now)
        if self.evictor._heap_time == now:
            self.counts["reused_heap"] += 1
            self.counts["reused_after_changes"] += self.changed
        block_id, content_hash = self.evictor.evict()
        if near:
            self.counts["near_ties"] += 1
            assert content_hash in near | {best}
        else:
            assert content_hash == best
        assert block_id == self.ref.cached[content_hash].block_id
        parent = self.ref.cached[content_hash].parent
        self.ref.evict(content_hash)
        del self.refcount[content_hash]
        if parent in self.ref.eligible():
            self.counts["parent_became_leaf"] += 1
        self.counts["evictions"] += 1
        self.changed = False
        self.last_evict = now
        self.check(now)
        return block_id

    def update(self):
        if self.ref.idle:
            content_hash = self.rng.choice(sorted(self.ref.idle))
            self.evictor.update(self.ref.cached[content_hash].block_id,
                                self.clock.t)
        in_use = sorted(x for x, n in self.refcount.items() if n)
        if in_use:
            block_id = self.ref.cached[self.rng.choice(in_use)].block_id
            with pytest.raises(ValueError):
                self.evictor.update(block_id, self.clock.t)

    def touch_idle(self):
        """An access to an idle block without removing it first; it stays
        idle (not from the allocator)."""
        if not self.ref.idle:
            return
        content_hash = self.rng.choice(sorted(self.ref.idle))
        b = self.ref.cached[content_hash]
        if (self.evictor._heap_time == self.clock.t
                and content_hash in self.evictor.leaves):
            self.counts["idle_leaf_touched_at_heap_time"] += 1
        self.evictor.record_access(b.block_id, content_hash, b.depth, b.parent)
        self.ref.access(content_hash, b.block_id, b.depth, b.parent,
                        self.clock.t)
        self.changed = True

    def tick(self):
        since = self.clock.t - self.last_evict
        if 0 < since <= 30.0 and self.rng.random() < 0.03:
            # A clock that steps back to the last eviction's time, as event
            # times never do: a heap kept for it must not be reused.
            self.clock.t = self.last_evict
            self.counts["stepped_back"] += 1
        else:
            self.clock.t += self.rng.choice(self.STEPS)

    def run(self, steps):
        for _ in range(steps):
            self.tick()
            op = self.rng.random()
            if op < 0.4 and len(self.running) < self.MAX_RUNNING:
                self.allocate()
            elif op < 0.75 and self.running:
                self.finish()
            elif op < 0.92:
                if self.ref.idle:
                    self.free_ids.append(self.evict())
                else:
                    with pytest.raises(ValueError,
                                       match="No usable cache memory left"):
                        self.evictor.evict()
            elif op < 0.96:
                self.update()
            else:
                self.touch_idle()
            assert self.evictor.num_blocks == len(self.ref.idle)
        self.check(self.clock.t)

    # --- State against the reference ---

    def check(self, now):
        ev, ref = self.evictor, self.ref
        assert set(ev.blocks) == set(ref.cached)
        assert ev.free_table == {ref.cached[x].block_id: x for x in ref.idle}
        assert set(ev.leaves) == set(ref.eligible())
        for x, b in ref.cached.items():
            # Prefix-closed: a cached block's parent is cached, and a block
            # in use keeps its parent in use.
            if b.parent is not None:
                assert b.parent in ref.cached
                if x not in ref.idle:
                    assert b.parent not in ref.idle
            mine = ev.blocks[x]
            assert mine.last == b.times[-1]
            c_short, c_long = ref.counters(x, b.times[-1])
            assert mine.c_short == pytest.approx(c_short, rel=1e-12)
            assert mine.c_long == pytest.approx(c_long, rel=1e-12)
            assert ev.score(mine, now) == pytest.approx(ref.score(x, now),
                                                        rel=1e-12,
                                                        abs=1e-300)


CONFIGS = {
    "defaults": {},
    "fast": {
        "tau_short": 1.0,
        "tau_long": 10.0,
        "w_short": 0.5,
        "w_long": 0.5
    },
    "long-only": {
        "w_short": 0.0,
        "w_long": 1.0
    },
    "ties-only": {
        "w_short": 0.0,
        "w_long": 0.0
    },
}


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", range(4))
def test_victims_match_a_brute_force_reference(config, seed):
    work = Workload(seed, CONFIGS[config])
    work.run(2500)
    counts = work.counts
    assert work.evictor.stats["evictions"] == counts["evictions"]
    # The workload exercised every path the kept heap must get right.
    assert counts["evictions"] > 300
    assert work.evictor.stats["scans"] < counts["evictions"]
    for key in ("reused_after_changes", "pushed_on_add", "parent_became_leaf",
                "reinserted", "revived", "idle_leaf_touched_at_heap_time",
                "stepped_back"):
        assert counts[key] > 0, key
    assert counts["near_ties"] <= counts["evictions"] // 100
    assert (work.evictor.stats["max_resident_blocks"] == work.peak_resident ==
            Workload.BLOCKS)
