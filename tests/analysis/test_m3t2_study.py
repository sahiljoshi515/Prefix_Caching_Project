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


def test_timeline_is_token_weighted_per_slice_with_the_warm_up():
    # Slices [0, 30) and [30, 60) s, every workload together. [0, 30):
    # requests at 10, 20 and 5 s serve min(40, 0) + min(64, 64) +
    # min(32, 32) = 96 of 136 tokens. [30, 60): min(100, 80) = 80 of 100.
    out = st.timeline(requests(), 30.0, 60.0, 16)
    assert out["edges"] == [0.0, 30.0, 60.0]
    assert out["usable"] == [pytest.approx(96 / 136), pytest.approx(0.8)]
    assert out["requests"] == [3, 1]
    # A request that did not complete is left out of its slice.
    req = requests()
    req["status"] = arr(["completed", "failed", "completed", "completed"], OBJ)
    assert st.timeline(req, 30.0, 60.0, 16)["usable"][1] is None


def test_capacity_curve_averages_the_reported_seeds(tmp_path):
    # Two reported seeds, LRU at two sizes plus the ceiling; seed 1 (not
    # reported) and an engine-timed row are left out. At 64 tokens the
    # curve is (0.2 + 0.4) / 2 = 0.3; at 128, (0.5 + 0.7) / 2 = 0.6.
    import json as _json

    def row(seed, cap, value, timing="model", policy="lru"):
        return {
            "plan": f"data/processed/m3/balanced-human-s{seed}",
            "policy": policy,
            "capacity": cap,
            "capacity_tokens": 0 if cap == "ceiling" else cap,
            "timing": timing,
            "summary": {
                "usable_reused_fraction": value,
                "block_hit_rate": value + 0.01
            },
        }

    rows = [
        row(2, 64, 0.2),
        row(3, 64, 0.4),
        row(2, 128, 0.5),
        row(3, 128, 0.7),
        row(1, 64, 0.9),
        row(2, 64, 0.99, timing="engine:x"),
        row(2, "ceiling", 0.8),
        row(3, "ceiling", 0.9),
    ]
    sweep = tmp_path / "sweep.jsonl"
    sweep.write_text("\n".join(_json.dumps(r) for r in rows))
    runs = tmp_path / "runs"
    runs.mkdir()
    cfg = {
        "mix": "balanced",
        "pacing": "human",
        "reported_seeds": [2, 3],
        "policies": ["lru"],
        "workloads": W,
        "block_size": 16,
        "capacity_tokens": 128
    }
    out = st.capacity_curve(runs, sweep, cfg)
    assert out["capacities"] == [64, 128]
    assert out["curve"]["lru"]["hit_rate"] == [
        pytest.approx(0.3), pytest.approx(0.6)
    ]
    assert out["curve"]["lru"]["seeds"][0] == [0.2, 0.4]
    assert out["ceiling"] == pytest.approx(0.85)
    assert out["engine"] == []


# --- The six report questions ----------------------------------------------


def test_grouped_usable_sums_the_bailian_traces():
    row = {
        "sharegpt": {
            "requests": 2,
            "prompt_tokens": 100,
            "usable_tokens": 50
        },
        "bailian-chat": {
            "requests": 1,
            "prompt_tokens": 60,
            "usable_tokens": 30
        },
        "bailian-api": {
            "requests": 3,
            "prompt_tokens": 40,
            "usable_tokens": 40
        },
        "all": {
            "requests": 6,
            "prompt_tokens": 200,
            "usable_tokens": 120,
            "fraction": 0.6
        },
    }
    out = st.grouped_usable(row)
    # Bailian: (30 + 40) of (60 + 40) tokens.
    assert out["bailian"] == {
        "requests": 4,
        "prompt_tokens": 100,
        "usable_tokens": 70,
        "fraction": 0.7
    }
    assert out["sharegpt"]["fraction"] == 0.5
    assert out["agentbank"]["fraction"] is None
    assert out["all"]["fraction"] == 0.6


def test_verdict_needs_every_seed_and_half_a_point():
    assert st.verdict([0.01, 0.02, 0.006]) == "higher"
    assert st.verdict([-0.01, -0.02, -0.03]) == "lower"
    assert st.verdict([0.01, -0.001, 0.03]) == "tie"  # one seed disagrees
    assert st.verdict([0.001, 0.002, 0.003]) == "tie"  # under half a point
    assert st.verdict([0.01, None]) == "tie"


def test_reaccess_intervals_and_the_share_served_per_bin():
    # Re-accesses (not "first") of the first prefill: 0.5 s (served), 40 s
    # twice (lost) and 2 s (served, agent); the 0.01 s one is a block a
    # running request still held, which no policy could have evicted.
    offered, served = st.reaccess_intervals(lookups(), 10.0, 100.0)
    assert sorted(offered.tolist()) == [0.5, 2.0, 40.0, 40.0]
    assert sorted(served.tolist()) == [0.5, 2.0]
    agent = st.reaccess_intervals(lookups(), 10.0, 100.0, ["agent"])
    assert agent[1].tolist() == [2.0]
    view = st.interval_view([offered], [served], [0, 1, 10], 5)
    shares = [(b["offered"], b["served"], b["share"]) for b in view["by_bin"]]
    assert shares == [(1, 1, 1.0), (1, 1, 1.0), (2, 0, 0.0)]
    assert view["by_bin"][-1]["hi"] is None
    assert view["served"]["n"] == 2 and view["offered"] == 4
    assert "curve" not in view
    fine = st.interval_view([offered], [served], [0, 10], 5, [0, 1, 3, 30])
    counts = [(b["offered"], b["served"]) for b in fine["curve"]]
    assert counts == [(1, 1), (1, 1), (0, 0), (2, 0)]


def write_client(path, rows):
    import json as _json
    path.mkdir(parents=True, exist_ok=True)
    (path / "client.jsonl").write_text("\n".join(_json.dumps(r) for r in rows))


def test_latency_by_group_is_the_median_inside_the_window(tmp_path):
    write_client(tmp_path, [
        {
            "workload": "bailian-api",
            "planned_s": 10,
            "latency_s": 1.0,
            "status": "completed"
        },
        {
            "workload": "bailian-chat",
            "planned_s": 20,
            "latency_s": 3.0,
            "status": "completed"
        },
        {
            "workload": "sharegpt",
            "planned_s": 20,
            "latency_s": 9.0,
            "status": "failed"
        },
        {
            "workload": "sharegpt",
            "planned_s": 5,
            "latency_s": 7.0,
            "status": "completed"
        },
        {
            "workload": "sharegpt",
            "planned_s": 30,
            "latency_s": 0.5,
            "status": "completed"
        },
    ])
    assert st.latency_by_group(tmp_path, 10.0, 100.0) == {
        "bailian": 2.0,
        "sharegpt": 0.5
    }


def test_capacity_curve_over_the_window_with_belady(tmp_path):
    import json as _json

    def row(seed, policy, cap, whole, inside, window=(600.0, 1800.0)):
        return {
            "plan": f"data/processed/m3/balanced-human-s{seed}",
            "policy": policy,
            "capacity": cap,
            "capacity_tokens": 0 if cap == "ceiling" else cap,
            "timing": "model",
            "window": list(window) if window else None,
            "summary": {
                "usable_reused_fraction": whole,
                "window": {
                    "usable_reused_fraction": inside
                } if window else None,
            },
        }

    rows = [
        row(2, "lru", 64, 0.1, 0.2),
        row(3, "lru", 64, 0.1, 0.4),
        row(2, "belady", 64, 0.1, 0.5),
        row(3, "belady", 64, 0.1, 0.7),
        row(2, "lru", "ceiling", 0.1, 0.8),
        row(3, "lru", "ceiling", 0.1, 0.9),
        # A whole-run row of the same plan and size is not windowed.
        {
            **row(2, "lru", 64, 0.99, 0.99, window=None)
        },
    ]
    sweep = tmp_path / "sweep.jsonl"
    sweep.write_text("\n".join(_json.dumps(r) for r in rows))
    runs = tmp_path / "runs"
    runs.mkdir()
    cfg = {
        "mix": "balanced",
        "pacing": "human",
        "reported_seeds": [2, 3],
        "policies": ["lru"],
        "workloads": W,
        "block_size": 16,
        "capacity_tokens": 64
    }
    out = st.capacity_curve(runs, sweep, cfg, (600, 1800), ["lru", "belady"])
    assert out["window_s"] == [600.0, 1800.0]
    assert out["curve"]["lru"]["hit_rate"] == [pytest.approx(0.3)]
    assert out["curve"]["belady"]["hit_rate"] == [pytest.approx(0.6)]
    assert out["curve"]["lru"]["block_hit_rate"] == [None]
    assert out["ceiling"] == pytest.approx(0.85)
    # Without the window only the whole-run row counts.
    whole = st.capacity_curve(runs, sweep, cfg)
    assert whole["curve"]["lru"]["hit_rate"] == [None]


def write_tables(run, requests, lookups_):
    import pyarrow as pa
    import pyarrow.parquet as pq
    (run / "tables").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(requests), run / "tables" / "requests.parquet")
    pq.write_table(pa.table(lookups_), run / "tables" / "lookups.parquet")


def toy_run(run, rid, workload, usable_hits, intervals, usable, latency):
    """One request of 2 blocks (32 tokens) planned at 20 s, whose two
    lookups are re-accesses with `intervals`, `usable` of them served."""
    write_tables(
        run, {
            "req": [rid],
            "workload": [workload],
            "session": ["x"],
            "turn": [1],
            "planned_s": [20.0],
            "status": ["completed"],
            "prompt_tokens": [32],
            "prompt_blocks": [2],
            "hits": [usable_hits],
            "usable_hits": [usable_hits],
            "hits_after_miss": [0],
            "cached_tokens": [16 * usable_hits],
            "preempted": [0]
        }, {
            "req": [rid] * 2,
            "workload": [workload] * 2,
            "turn": [1, 1],
            "planned_s": [20.0, 20.0],
            "prefill": [0, 0],
            "index": [0, 1],
            "hit": usable,
            "usable": usable,
            "in_use": [False, False],
            "source": ["same_session"] * 2,
            "interval_s": intervals,
            "requests_between": [0, 0],
            "evicted_before": [False, False]
        })
    write_client(run, [{
        "workload": workload,
        "planned_s": 20.0,
        "latency_s": latency,
        "status": "completed"
    }])


def test_isolation_compares_each_workload_alone_and_in_the_mix(
        tmp_path, monkeypatch):
    monkeypatch.setattr(st, "GROUPS", ("sharegpt", "bailian"))
    cfg = {
        "w0_s": 0.0,
        "stop_s": 100.0,
        "block_size": 16,
        "workloads": ["sharegpt", "bailian-chat"],
        "capacity_tokens": 64,
        "pacing": "human",
        "reported_seeds": [2],
        "policies": ["lru", "fifo"],
        "served_interval_edges_s": [0, 10],
        "cdf_points": 3,
        "interval_bins_s": [0, 10]
    }
    runs = tmp_path / "runs"
    # Alone, ShareGPT keeps both blocks under LRU and one under FIFO;
    # Bailian keeps one under either.
    for policy, sg in (("lru", 2), ("fifo", 1)):
        toy_run(runs / f"alone-sharegpt-human-s2-{policy}-cap64",
                "sharegpt:x:1", "sharegpt", sg, [5.0, 50.0], [True, sg == 2],
                0.5)
        toy_run(runs / f"alone-bailian-human-s2-{policy}-cap64",
                "bailian-chat:y:1", "bailian-chat", 1, [5.0, 50.0],
                [True, False], 1.0)

    def usable(sg, bl):
        return {
            "sharegpt": {
                "requests": 1,
                "prompt_tokens": 32,
                "usable_tokens": 16 * sg
            },
            "bailian-chat": {
                "requests": 1,
                "prompt_tokens": 32,
                "usable_tokens": 16 * bl
            },
            "all": {
                "requests": 2,
                "prompt_tokens": 64,
                "usable_tokens": 16 * (sg + bl),
                "fraction": None
            }
        }

    # In the mix both keep one block each; the ceiling keeps both.
    headline = {
        "lru": {
            2: usable(1, 1)
        },
        "fifo": {
            2: usable(1, 1)
        },
        "ceiling": {
            2: usable(2, 2)
        }
    }
    latency = {
        2: {
            "lru": {
                "sharegpt": 0.8,
                "bailian": 1.5
            },
            "fifo": {
                "sharegpt": 0.9,
                "bailian": 1.6
            }
        }
    }
    lookups_ = {2: {"sharegpt": 2, "bailian": 2}}
    out = st.isolation_study(runs, cfg, headline, latency, lookups_)
    lru = out["rows"]["sharegpt"]["lru"]
    assert lru["alone"]["mean"] == 1.0 and lru["mixed"]["mean"] == 0.5
    assert lru["alone_minus_mixed"]["mean"] == 0.5
    assert lru["verdict"] == "higher"
    assert lru["mixed_efficiency"]["mean"] == 0.5
    assert lru["latency_alone_s"]["mean"] == 0.5
    assert lru["latency_mixed_s"]["mean"] == 0.8
    assert out["rows"]["sharegpt"]["fifo"]["verdict"] == "tie"
    assert out["rows"]["bailian"]["lru"]["alone_minus_mixed"]["mean"] == 0
    assert out["rows"]["sharegpt"]["requests"]["mean"] == 1
    assert out["rows"]["sharegpt"]["prompt_tokens_per_s"]["mean"] == 0.32
    # Alone, FIFO serves ShareGPT's 5 s reuse and loses the 50 s one.
    bins = out["intervals"]["sharegpt"]["fifo"]["by_bin"]
    assert [(b["offered"], b["served"]) for b in bins] == [(1, 1), (1, 0)]
    assert out["vs_lru"]["sharegpt"]["fifo"]["verdict"] == "lower"
    assert all(r["lookups"] == r["mixed_lookups"] == 2
               for r in out["integrity"])
    # A missing run's tables: no comparison yet.
    import shutil
    shutil.rmtree(runs / "alone-bailian-human-s2-fifo-cap64" / "tables")
    assert st.isolation_study(runs, cfg, headline, latency, lookups_) is None
