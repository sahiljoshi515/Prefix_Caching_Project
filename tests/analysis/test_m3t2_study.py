"""Phase 1 study aggregation (spec 007 US2): each metric on a hand-made
table whose answer is worked out in the comment beside it."""
import math

import numpy as np
import pytest

from analysis import m3t2_study as st

pytestmark = pytest.mark.m3

W = ["chat", "agent"]
OBJ = object


def arr(values, dtype=None):
    return np.array(values, dtype=dtype)


def requests():
    # Three requests planned inside [10, 100) and one before the window.
    return {
        "req": arr(["c:1:0", "c:1:1", "a:9:0", "c:0:0"], OBJ),
        "workload": arr(["chat", "chat", "agent", "chat"], OBJ),
        "planned_s": arr([10.0, 50.0, 20.0, 5.0]),
        "prompt_tokens": arr([40, 100, 64, 32]),
        "usable_hits": arr([0, 5, 4, 2]),
        "prompt_blocks": arr([2, 6, 4, 2]),
    }


def test_usable_by_workload_is_token_weighted_inside_the_window():
    out = st.usable_by_workload(requests(), W, 10.0, 100.0, 16)
    # chat: min(40, 0) + min(100, 80) = 80 of 140 tokens; the request at
    # 5 s is before the window. agent: min(64, 64) = 64 of 64.
    assert out["chat"]["fraction"] == pytest.approx(80 / 140)
    assert out["agent"]["fraction"] == 1.0
    assert out["all"]["fraction"] == pytest.approx(144 / 204)
    assert out["macro"]["fraction"] == pytest.approx((80 / 140 + 1) / 2)
    assert out["all"]["requests"] == 3


def test_overlap_spreads_intervals_over_bins():
    # [0, 3) and [2, 6) over bins [0,1) [1,2) [2,4) [4,8): 1, 1, 1+2, 2.
    got = st.overlap(arr([0.0, 2.0]), arr([3.0, 6.0]), [0, 1, 2, 4, 8])
    assert got.tolist() == [1, 1, 3, 2]


def test_position_profile_counts_ceiling_reuse_and_policy_loss():
    # Two requests of 4 blocks; the ceiling reuses 3 and 2 blocks, the
    # policy 1 and 2. Bins [0,2) [2,4): blocks 4 and 4; ceiling reuse 2+2
    # and 1+0; lost (positions [1,3) of request 1) 1 and 1.
    out = st.position_profile(arr([4, 4]), arr([3, 2]), {"lru": arr([1, 2])},
                              [0, 2, 4])
    assert out["ceiling_share"] == [1.0, 0.25]
    assert out["lru_lost_share"] == [0.25, 0.25]


def lookups():
    # Request c:1:1 looks up 4 blocks: the policy hits 0, misses 1, then
    # hits 2 behind the miss (cut off), and misses 3. Its ceiling usable
    # prefix is 3, the policy's 1, so blocks 1 (evicted) and 2 (cut off)
    # are lost and block 3 is beyond the ceiling's prefix.
    return {
        "req":
        arr(["c:1:1"] * 4 + ["a:9:0"] * 2, OBJ),
        "workload":
        arr(["chat"] * 4 + ["agent"] * 2, OBJ),
        "planned_s":
        arr([50.0] * 4 + [20.0] * 2),
        "prefill":
        arr([0] * 6),
        "index":
        arr([0, 1, 2, 3, 0, 1]),
        "hit":
        arr([True, False, True, False, True, True]),
        "usable":
        arr([True, False, False, False, True, True]),
        "in_use":
        arr([False, False, False, False, False, True]),
        "turn":
        arr([1, 1, 1, 1, 0, 0]),
        "source":
        arr([
            "same_session", "same_session", "other_session", "first",
            "other_workload", "other_workload"
        ], OBJ),
        "interval_s":
        arr([0.5, 40.0, 40.0, math.nan, 2.0, 0.01]),
    }


def test_lost_reuse_splits_evicted_and_cut_off_blocks():
    out = st.lost_reuse(lookups(), {
        "c:1:1": 3,
        "a:9:0": 2
    }, {
        "c:1:1": 1,
        "a:9:0": 2
    }, W, 10.0, 100.0)
    assert out["chat"] == {"kept": 1, "evicted": 1, "cut_off": 1}
    assert out["agent"] == {"kept": 2, "evicted": 0, "cut_off": 0}
    assert out["all"] == {"kept": 3, "evicted": 1, "cut_off": 1}


def test_source_shares_describe_the_reused_blocks():
    out = st.source_shares(lookups(), W, 10.0, 100.0)
    # chat hits: blocks 0 (same session) and 2 (other session).
    assert out["chat"]["hits"] == 2
    assert out["chat"]["same_session"] == 0.5
    assert out["chat"]["other_session"] == 0.5
    assert out["chat"]["first_turn"] == 0.0
    # agent hits: both from another workload, both turn 0, one shared live.
    assert out["agent"]["other_workload"] == 1.0
    assert out["agent"]["first_turn"] == 1.0
    assert out["agent"]["in_use"] == 0.5


def test_hit_by_interval_skips_first_accesses_and_live_shares():
    out = st.hit_by_interval(lookups(), W, [0, 1, 10, 100], 10.0, 100.0)
    # chat re-accesses: 0.5 s usable; 40 s and 40 s not usable.
    assert out["chat"]["n"] == [1, 0, 2]
    assert out["chat"]["usable_share"] == [1.0, None, 0.0]
    # agent: the 2 s re-access counts; the live-shared one does not.
    assert out["agent"]["n"] == [0, 1, 0]


def residencies():
    # chat: two stays revived after 5 s and 20 s, one evicted after 10 s
    # whose hash came back, one censored at 30 s. agent: one evicted by a
    # chat request after 2 s, never looked up again.
    return {
        "owner_workload": arr(["chat"] * 4 + ["agent"], OBJ),
        "owner_planned_s": arr([20.0] * 5),
        "end": arr(["revive", "revive", "evict", "censored", "evict"], OBJ),
        "idle_s": arr([5.0, 20.0, 10.0, 30.0, 2.0]),
        "hits_before": arr([0, 3, 1, 0, 0]),
        "evictor_workload": arr([None, None, "agent", None, "chat"], OBJ),
        "next_lookup_t": arr([1.0, 2.0, 50.0, math.nan, math.nan]),
    }


def test_come_back_is_kaplan_meier_of_the_ceilings_stays():
    rs = residencies()
    out = st.come_back(rs, ["chat"], [4.0, 5.0, 20.0, 30.0], 10.0, 100.0)
    # chat stays (5 r, 20 r, 10 e, 30 c): evictions do not happen in a
    # ceiling run, so here the eviction counts as not coming back
    # (censored). KM: at 5, 1 - 1/4; at 20, x (1 - 1/2) -> 0.375 left.
    assert out["chat"]["came_back"] == pytest.approx([0.0, 0.25, 0.625, 0.625])
    assert out["chat"]["revived_share"] == 0.5


def test_eviction_ages_and_frequency_outcomes():
    rs = residencies()
    ages = st.eviction_ages(rs, W, 10.0, 100.0)
    assert ages["chat"]["evicted"] == 1
    assert ages["chat"]["evicted_share"] == pytest.approx(1 / 3)
    assert ages["all"]["quantiles"]["p50"] == pytest.approx(6.0)
    freq = st.frequency_outcomes(rs, W, 10.0, 100.0)
    # chat stays by hits before: 0 -> revived; 1 -> evicted; 3 -> revived.
    assert freq["chat"]["0"] == {"stays": 1, "revived_share": 1.0}
    assert freq["chat"]["1"] == {"stays": 1, "revived_share": 0.0}
    assert freq["chat"]["2+"] == {"stays": 1, "revived_share": 1.0}


def test_hit_density_and_who_evicts_whom():
    rs = residencies()
    dens = st.hit_density(rs, W, 10.0, 100.0)
    # Pool time: chat 65 s, agent 2 s; revives: chat 2, agent 0.
    assert dens["chat"]["pool_time_share"] == pytest.approx(65 / 67)
    assert dens["chat"]["revive_share"] == 1.0
    assert dens["chat"]["revives_per_block_minute"] == pytest.approx(2 /
                                                                     (65 / 60))
    mat = st.evictions_matrix(rs, W, 10.0, 100.0)
    # rows: evictor chat, agent; columns: victim chat, agent.
    assert mat["counts"] == [[0, 1], [1, 0]]
    assert mat["regretted_share"] == {"chat": 1.0, "agent": 0.0}


def test_cdf_points_and_merge_seeds():
    cdf = st.cdf_points(arr([1.0, 2.0, 3.0, 4.0, 100.0]), 5)
    assert cdf["n"] == 5 and cdf["x"][0] == 1.0 and cdf["x"][-1] == 100.0
    assert cdf["within"]["15"] == 0.8
    merged = st.merge_seeds([{
        "a": 1.0,
        "b": [1, None],
        "c": "x"
    }, {
        "a": 3.0,
        "b": [3, 2],
        "c": "y"
    }])
    assert merged == {"a": 2.0, "b": [2.0, None], "c": "x"}


def test_run_names_follow_the_grid():
    assert st.run_name("balanced", "human", 2, "lru", 131072) == \
        "balanced-human-s2-lru-cap131072"
    assert st.run_name("balanced", "human", 2, "ceiling", 131072) == \
        "balanced-human-s2-ceiling"
    assert st.run_name("balanced", "human", 2, "lru", 131072, 1) == \
        "balanced-human-s2-lru-cap131072-r1"


def test_come_back_stops_where_few_stays_are_still_observed():
    # 200 stays: 199 revived after 1 s, one censored at 50 s. At 50 s only
    # 1 of 200 (0.5%) is still observed, below the 1% floor: no value.
    rs = {
        "owner_workload": arr(["chat"] * 200, OBJ),
        "owner_planned_s": arr([20.0] * 200),
        "end": arr(["revive"] * 199 + ["censored"], OBJ),
        "idle_s": arr([1.0] * 199 + [50.0]),
    }
    out = st.come_back(rs, ["chat", "agent"], [1.0, 50.0], 10.0, 100.0)
    assert out["chat"]["came_back"] == [pytest.approx(199 / 200), None]
    # A workload with no stays is present, with no values.
    assert out["agent"]["came_back"] == [None, None]


def test_merge_seeds_gives_none_when_a_seed_lacks_a_value():
    assert st.merge_seeds([{"q": {"p50": 1.0}}, {"q": None}]) == {"q": None}
    assert st.merge_seeds([{"q": None}, {"q": {"p50": 1.0}}]) == {"q": None}
    assert st.merge_seeds([{"n": 2.0}, {"n": None}]) == {"n": None}


def test_usable_counts_only_completed_requests():
    req = requests()
    req["status"] = arr(["completed", "failed", "completed", "completed"], OBJ)
    out = st.usable_by_workload(req, W, 10.0, 100.0, 16)
    # The failed request (c:1:1) is left out: chat is 0 of 40 tokens.
    assert out["chat"]["fraction"] == 0.0
    assert out["chat"]["requests"] == 1
