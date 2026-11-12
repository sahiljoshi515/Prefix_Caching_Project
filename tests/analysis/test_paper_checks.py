"""The ATC'25 paper scorecard (analysis/paper_checks.py).

A four-request trace and a seven-lookup mix, with every expected value
worked out by hand below. Blocks are 16 tokens; hash ids are small ints.
"""
import gzip
import json
import math

import numpy as np
import pytest

from analysis import paper_checks as pc

pytestmark = pytest.mark.m3


def rec(chat, t, ids, turn=1, parent=-1, kind="text", tokens=None, out=10):
    return {
        "chat_id": chat,
        "parent_chat_id": parent,
        "timestamp": t,
        "input_length": 16 * len(ids) if tokens is None else tokens,
        "output_length": out,
        "type": kind,
        "turn": turn,
        "hash_ids": ids,
    }


# t=0   r0 (session 0) reads blocks 1, 2, 3: all new.
# t=10  r1, r0's next turn, reads 1, 2, 3 (hits, same session) and 4, 5.
# t=25  r2 (session 2) reads 1, 2 (hits, from r1: another session); its
#       third hash id covers a partial block, which is never cached.
# t=30  r3 (session 3, a search) reads 7, 2, 3: new prefixes, though ids
#       2 and 3 were seen (the content view counts them).
RECORDS = [
    rec(3, 30.0, [7, 2, 3], kind="search"),  # out of order on purpose
    rec(0, 0.0, [1, 2, 3]),
    rec(1, 10.0, [1, 2, 3, 4, 5], turn=2, parent=0),
    rec(2, 25.0, [1, 2, 9], tokens=40),
]


@pytest.fixture(scope="module")
def analysis():
    return pc.RawAnalysis(pc.trace_from_records(RECORDS, "bailian-chat"))


@pytest.fixture(scope="module")
def metrics(analysis):
    return pc.raw_metrics(analysis, w0=0.0, horizon=0.0)


def test_trace_is_in_time_order_with_full_blocks_only(analysis):
    tr = analysis.trace
    assert tr.chat_id.tolist() == [0, 1, 2, 3]
    assert tr.lengths.tolist() == [3, 5, 2, 3]  # r2's partial block dropped
    assert tr.hashes.tolist() == [1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 7, 2, 3]
    assert tr.root.tolist() == [0, 0, 2, 3]
    assert tr.has_child.tolist() == [True, False, False, False]


def test_prefix_keys_follow_whole_prefixes(analysis):
    key = analysis.key
    assert key[3:6].tolist() == key[0:3].tolist()  # r1 extends r0
    assert key[8:10].tolist() == key[0:2].tolist()  # r2 shares 1, 2
    # r3's "2" follows "7", not "1": a different block.
    assert key[11] not in key[:11] and key[12] not in key[:12]
    assert len(set(key.tolist())) == 8


def test_neighbours_by_hand():
    near = pc.neighbours(np.array([5, 3, 5, 5, 3]))
    assert near.prev.tolist() == [-1, -1, 0, 2, 1]
    assert near.next.tolist() == [2, 4, 3, -1, -1]
    assert near.rank.tolist() == [0, 0, 1, 2, 1]
    assert near.count.tolist() == [2, 3]  # key 3, then key 5
    assert near.first.tolist() == [1, 0] and near.last.tolist() == [4, 3]
    assert near.group.tolist() == [1, 0, 1, 1, 0]


def test_ideal_hit_ratio(metrics):
    # 13 blocks; hits: r1's first three, r2's two. Content view adds r3's
    # 2 and 3.
    assert metrics[1]["prompt_blocks"] == 13
    assert metrics[1]["ideal_hit_ratio"] == pytest.approx(5 / 13)
    assert metrics[1]["content_hit_ratio"] == pytest.approx(7 / 13)
    # A warm-up drops r0 and r1 (before 20 s): r2's 2 hits of r2 and r3's
    # 5 blocks.
    warm = pc.raw_metrics(pc.RawAnalysis(
        pc.trace_from_records(RECORDS, "bailian-chat")),
                          w0=20.0,
                          horizon=0.0)
    assert warm[1]["ideal_hit_ratio"] == pytest.approx(2 / 5)


def test_same_session_share_and_skew(metrics):
    # r1's 3 hits follow r0 (same session); r2's 2 follow r1 (another).
    assert metrics[2]["same_session_share"] == pytest.approx(3 / 5)
    # Reuses per block: [1]: 2, [1,2]: 2, [1,2,3]: 1, five others 0. The top
    # 10% of 8 blocks is one block: 2 of 5 reuses.
    assert metrics[2]["top10_block_share_of_reuses"] == pytest.approx(2 / 5)
    assert metrics[2]["multi_request_sessions"]
    # r2 (turn 1) reads 2 of the 5 hits.
    assert metrics[3]["turn1_share_of_hits"] == pytest.approx(2 / 5)


def test_reuse_time_quantiles(metrics):
    # r1's hits wait 10 s, r2's 15 s: [10, 10, 10, 15, 15].
    q = metrics[4]
    assert q["n"] == 5 and q["mean"] == pytest.approx(12.0)
    assert q["p50"] == 10.0 and q["p80"] == 15.0


def test_next_turn_probability(metrics):
    # Turn 1: r0 has a child, r2 and r3 do not; turn 2: r1 does not.
    v = metrics[7]
    assert v["single_turn"] == pytest.approx(1 / 3)
    assert v["multi_turn"] == 0.0
    assert v["by_turn"] == {"1": pytest.approx(1 / 3), "2": 0.0}


def test_position_profile(metrics):
    # Reused later: r0's three blocks, r1's first two. Tenth = index * 10
    # // blocks: tenth 0 holds r0, r1, r2, r3's first blocks (2 of 4).
    p = metrics[8]["prefix"]
    assert p["by_decile"][0] == 0.5
    assert p["by_decile"][2] == 1.0  # r1 index 1
    assert p["by_decile"][3] == 0.5  # r0 and r3 index 1
    assert p["by_decile"][6] == pytest.approx(1 / 3)  # r0 2, r1 3, r3 2
    assert p["by_decile"][1] is None
    assert p["by_index"]["0"] == 0.5
    assert p["by_index"]["1-3"] == pytest.approx(3 / 8)
    assert p["by_index"]["4-15"] == 0.0 and p["by_index"]["16-63"] is None
    # The content view: r3's 2 and 3 are not read again either.
    assert metrics[8]["content"]["by_decile"][0] == 0.5


def test_frequency_and_lifespan(metrics):
    # Past accesses 0: r0's three (all reused later), r1's 4 and 5, r3's
    # three (none): 3 of 8. Past 1: r1's first three, two reused later.
    # Past 2: r2's two, neither.
    f = metrics[9]["future_reuse_by_past_accesses"]
    assert f == {"0": 3 / 8, "1": pytest.approx(2 / 3), "2": 0.0, "3+": None}
    # Lifespans: [1] and [1,2] 25 s, [1,2,3] 10 s, five blocks 0.
    life = metrics[9]["lifespan_s"]
    assert life["p50"] == 0.0 and life["p90"] == 25.0
    assert metrics[9]["blocks_never_reused"] == pytest.approx(5 / 8)


def test_live_working_set(metrics):
    # Needed: r0's blocks over [0, 10), r1's first two over [10, 25).
    # Grid 0, 10, 20 s: 3, 2, 2.
    live = metrics[10]["live_blocks"]
    assert live["max"] == 3 and live["mean"] == pytest.approx(7 / 3)
    assert live["share_of_time_above_capacity"] == 0.0


def test_live_counts_half_open():
    grid = np.array([0.0, 1.0, 2.0])
    counts = pc.live_counts(np.array([0.0, 1.0, 1.0]),
                            np.array([1.0, 1.0, 3.0]), grid)
    assert counts.tolist() == [1, 1, 1]


def test_category_fits_and_mixture():
    durations = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 40.0])
    observed = np.array([1, 1, 0, 1, 1, 0], dtype=bool)
    category = np.array([0, 0, 0, 1, 1, 1])
    other = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
    out = pc.category_fits(durations,
                           observed,
                           category, ["fast", "slow"],
                           other,
                           min_n=3)
    # Plain rates: 2 / 6 and 2 / 70.
    assert out["categories"]["fast"]["rate"] == pytest.approx(1 / 3)
    assert out["rate_ratio_max"] == pytest.approx(70 / 6)
    assert out["rate_ratio"]["fast/slow"] == pytest.approx(70 / 6)
    fast = out["categories"]["fast"]
    assert fast["reuse_by_other_session_share"] == 0.5
    assert fast["median_s_other_session"] == 1.0
    assert fast["median_s_same_session"] == 2.0
    assert pc.category_fits(durations, observed, category, ["a", "b"],
                            min_n=4)["categories"] == {}


def test_history_check_on_a_steady_process():
    # One category, the same exponential before and after the split.
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 200, 20000))
    gap = rng.exponential(5.0, len(t))
    next_t = np.minimum(t + gap, 250.0)
    has_next = t + gap <= 250.0
    out = pc.history_check(t, next_t, has_next, np.zeros(len(t), int), ["c"],
                           100.0)
    row = out["categories"]["c"]
    assert row["rate_ratio"] == pytest.approx(1.0, abs=0.05)
    assert row["ks_own_fit"] < 0.03
    assert out["rate_within_1_5x_share"] == 1.0


def test_thinning_by_hand(analysis):
    # Window [0, 26): sessions 0 (r0, r1) and 2 (r2); r3 starts later.
    manifest = {
        "duration_s": 26.0,
        "stop_s": 26.0,
        "max_model_len": 32768,
        "workloads": {
            "bailian-chat": {
                "window_start_s": 0.0,
                "kept_session_fraction": 0.5
            }
        },
    }
    out = pc.thinning_for(analysis, manifest, np.array([0, 1]), w0=0.0)
    full, kept = out["full_rate"], out["kept"]
    # Full rate: 10 blocks, 5 hits (3 same-session, 2 cross-session).
    assert full["requests"] == 3 and full["sessions"] == 2
    assert full["ideal_hit_ratio"] == 0.5
    assert full["same_session_hit_ratio"] == pytest.approx(0.3)
    assert full["cross_session_hit_ratio"] == pytest.approx(0.2)
    # Kept: session 0 alone, 8 blocks, r1's 3 hits.
    assert kept["requests"] == 2 and kept["sessions"] == 1
    assert kept["ideal_hit_ratio"] == pytest.approx(3 / 8)
    assert kept["cross_session_hit_ratio"] == 0.0
    assert out["kept_over_full_hit_ratio"] == pytest.approx(0.75)
    assert out["kept_over_full_cross_session"] == 0.0
    assert out["plan_requests_outside_window"] == 0


def test_window_mask_drops_late_turns_and_long_requests(analysis):
    tr = analysis.trace
    # A 5 s stop keeps r0 only. A request needs prompt + output + 1 tokens:
    # r0 59, r1 91, r2 51, so a 52-token model keeps r2 (r3 starts late).
    assert pc.window_mask(tr, 0.0, 26.0, 5.0,
                          32768).tolist() == [True, False, False, False]
    assert pc.window_mask(tr, 0.0, 26.0, 26.0,
                          52).tolist() == [False, False, True, False]


# --------------------------------------------------------------------------
# The mix side: tables with the frozen schema's columns
# --------------------------------------------------------------------------

NAN = math.nan
# (t, req, workload, session, turn, planned_s, prefill, index, hash, hit,
#  prev_t, source, interval_s). wa:s1:0 runs before the window (10 s).
# Block 3 was written while wa:s1:0 decoded (at 12 s), so its first lookup
# is a hit. The last row is a re-prefill, which the analyses skip.
LOOKUPS = [
    (5, "wa:s1:0", "wa", "wa:s1", 0, 5, 0, 0, 1, False, NAN, "first", NAN),
    (5, "wa:s1:0", "wa", "wa:s1", 0, 5, 0, 1, 2, False, NAN, "first", NAN),
    (20, "wa:s1:1", "wa", "wa:s1", 1, 20, 0, 0, 1, True, 5, "same_session",
     15),
    (20, "wa:s1:1", "wa", "wa:s1", 1, 20, 0, 1, 2, True, 5, "same_session",
     15),
    (20, "wa:s1:1", "wa", "wa:s1", 1, 20, 0, 2, 3, True, 12, "same_session",
     8),
    (30, "wb:s2:0", "wb", "wb:s2", 0, 30, 0, 0, 1, True, 20, "other_workload",
     10),
    (30, "wb:s2:0", "wb", "wb:s2", 0, 30, 0, 1, 9, False, NAN, "first", NAN),
    (40, "wa:s3:0", "wa", "wa:s3", 0, 40, 0, 0, 1, True, 30, "other_workload",
     10),
    (40, "wa:s3:0", "wa", "wa:s3", 0, 40, 0, 1, 2, True, 20, "other_session",
     20),
    (50, "wa:s3:0", "wa", "wa:s3", 0, 40, 1, 0, 1, True, 40, "same_session",
     10),
]
LOOKUP_NAMES = ("t", "req", "workload", "session", "turn", "planned_s",
                "prefill", "index", "hash", "hit", "prev_t", "source",
                "interval_s")
# (t_free, t_end, end, hits_before, owner_workload, next_lookup_t)
RESIDENCIES = [
    (6, 20, "revive", 0, "wa", 20),
    (25, 40, "revive", 1, "wa", 40),
    (35, 200, "censored", 2, "wb", NAN),
]
REQUESTS = [("wa:s1:0", "wa", "wa:s1", 0, 5.0),
            ("wa:s1:1", "wa", "wa:s1", 1, 20.0),
            ("wb:s2:0", "wb", "wb:s2", 0, 30.0),
            ("wa:s3:0", "wa", "wa:s3", 0, 40.0)]
RUN = {"w0_s": 10.0, "stop_s": 100.0, "censor_t": 200.0, "plan_dir": "p/x"}


def write_tables(run_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    meta = {b"run": json.dumps(RUN).encode()}

    def write(name, names, rows, types):
        columns = {
            n: pa.array([r[i] for r in rows], type=types[i])
            for i, n in enumerate(names)
        }
        table = pa.table(columns).replace_schema_metadata(meta)
        pq.write_table(table, tables / f"{name}.parquet")

    s, f, i32, i64 = pa.string(), pa.float64(), pa.int32(), pa.int64()
    write("lookups", LOOKUP_NAMES, LOOKUPS,
          [f, s, s, s, i32, f, i32, i32, i64,
           pa.bool_(), f, s, f])
    write("residencies", ("t_free", "t_end", "end", "hits_before",
                          "owner_workload", "next_lookup_t"), RESIDENCIES,
          [f, f, s, i32, s, f])
    write("requests", ("req", "workload", "session", "turn", "planned_s"),
          REQUESTS, [s, s, s, i32, f])


@pytest.fixture(scope="module")
def mix(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("ceiling")
    write_tables(run_dir)
    return pc.mix_metrics(pc.load_mix(run_dir), horizon=0.0)


def test_mix_hit_ratio_and_sources(mix):
    # Window rows (planned in [10, 100), first prefill): wa reads 5 hits;
    # wb 1 hit and 1 miss.
    assert mix["wa"][1]["ideal_hit_ratio"] == 1.0
    assert mix["wb"][1]["ideal_hit_ratio"] == 0.5
    assert mix["all"][1]["prompt_blocks"] == 7
    wa = mix["wa"][2]
    assert wa["same_session_share"] == pytest.approx(0.6)
    assert wa["other_session_share"] == pytest.approx(0.2)
    assert wa["other_workload_share"] == pytest.approx(0.2)
    # wa's hits per block: [1] 2, [2] 2, [3] 1 -> the top block has 2 of 5.
    assert wa["top10_block_share_of_reuses"] == pytest.approx(0.4)
    assert mix["wb"][2]["top10_block_share_of_reuses"] == 1.0
    assert mix["wa"][3]["turn1_share_of_hits"] == pytest.approx(0.4)
    q = mix["wa"][4]
    assert q["p50"] == 15.0 and q["p80"] == 20.0


def test_mix_position_frequency_and_lifespan(mix):
    # wa's window lookups, reused later? 1@20 yes, 2@20 yes, 3@20 no,
    # 1@40 no, 2@40 no.
    p = mix["wa"][8]
    assert p["by_decile"][0] == 0.5 and p["by_decile"][3] == 1.0
    assert p["by_index"]["1-3"] == pytest.approx(1 / 3)
    # Past accesses: block 3's first lookup follows its decode write, so it
    # has 1. 1@40 has 3, 2@40 has 2.
    f = mix["wa"][9]["future_reuse_by_past_accesses"]
    assert f == {"0": None, "1": pytest.approx(2 / 3), "2": 0.0, "3+": 0.0}
    # Lifespans: blocks 1 and 2 from 5 to 40 s; block 3 from its write at
    # 12 s to 20 s; wb's block 9 never comes back.
    assert mix["wa"][9]["lifespan_s"]["p50"] == 35.0
    assert mix["wb"][9]["lifespan_s"]["p90"] == 0.0
    lfu = mix["all"][9]["revived_by_hits_before"]["all"]
    assert lfu == {"0": 1.0, "1": 1.0, "2": 0.0, "3+": None}


def test_mix_next_turn_and_live_set(mix):
    nt = mix["all"][7]
    assert nt["wa"]["single_turn"] == 0.5 and nt["wa"]["multi_turn"] == 0.0
    assert nt["wa"]["multi_request_sessions"]
    assert not nt["wb"]["multi_request_sessions"]
    # Grid 20, 30 s. Hits need blocks over [prev_t, t): at 20 s, 1@30 and
    # 2@40; at 30 s, 1@40 and 2@40.
    live = mix["all"][10]
    assert live["live_blocks"]["max"] == 2
    assert live["free_pool_live_blocks"]["mean"] == 0.5
    assert live["free_pool_blocks"]["max"] == 1
    assert mix["all"][5]["categories"] == {}  # too few accesses


# --------------------------------------------------------------------------
# Rules, the sweep hook, and the whole command
# --------------------------------------------------------------------------


def test_rules():
    h, p, n, u = pc.HOLDS, pc.PARTLY, pc.NO, pc.UNTESTABLE
    one = lambda a, b: {
        "A": {
            "ideal_hit_ratio": a
        },
        "B": {
            "ideal_hit_ratio": b
        }
    }
    assert pc.rule_1(one(0.60, 0.54)) == h
    assert pc.rule_1(one(0.60, 0.58)) == p
    assert pc.rule_1(one(0.85, 0.84)) == n
    q = lambda a, b: {"bailian-chat": {"p80": a}, "bailian-api": {"p80": b}}
    assert pc.rule_4(q(300, 5)) == h
    assert pc.rule_4(q(300, 26)) == p
    assert pc.rule_4(q(300, 100)) == n
    skewed = {"top10_block_share_of_reuses": 0.7, "same_session_share": 0.0}
    assert pc.rule_2(skewed) == h  # single-request sessions: skew only
    assert pc.rule_2(dict(skewed, multi_request_sessions=True)) == p
    assert pc._across([h, h]) == h and pc._across([h, n]) == p
    assert pc._across([n, u, n]) == n and pc._across([u]) == u
    assert pc.rule_7({"single_turn": 0.4, "multi_turn": 0.57}) == h
    assert pc.rule_7({"single_turn": 0.4, "multi_turn": 0.45}) == p
    assert pc.rule_7({"single_turn": 0.0, "multi_turn": None}) == u
    assert pc.rule_8({"by_decile": [0.8, 0.7, None, 0.5]}) == h
    assert pc.rule_8({"by_decile": [0.8, 0.85, 0.5]}) == p
    life = {"p90": 100.0}
    flat = {"0": 0.1, "1": 0.5, "2": 0.5, "3+": 0.52}
    steep = dict(flat, **{"3+": 0.9})
    assert pc.rule_9({
        "lifespan_s": life,
        "future_reuse_by_past_accesses": flat
    }) == h
    assert pc.rule_9({
        "lifespan_s": life,
        "future_reuse_by_past_accesses": steep
    }) == p


def test_policy_spread_hook():
    sweep = [
        {
            "capacity_blocks": 1000,
            "policy": "lru",
            "hit_ratio": 0.5
        },
        {
            "capacity_blocks": 1000,
            "policy": "lfu",
            "hit_ratio": 0.3
        },
        {
            "capacity_tokens": 80000,
            "policy": "lru",
            "hit_ratio": 0.700
        },
        {
            "capacity_tokens": 80000,
            "policy": "lfu",
            "hit_ratio": 0.695
        },
    ]
    out = pc.policy_spread(sweep, live_p95=4000)
    assert out["spread_by_capacity_blocks"] == {
        1000: pytest.approx(0.2),
        5000: pytest.approx(0.005)
    }
    assert out["verdict"] == pc.HOLDS
    assert pc.policy_spread([], 4000)["verdict"] == pc.UNTESTABLE


def test_jsonable():
    out = pc.jsonable({1: np.float64(0.5), "x": [math.nan, np.int64(3)]})
    assert out == {"1": 0.5, "x": [None, 3]}


def test_command_end_to_end(tmp_path, analysis):
    raw = tmp_path / "raw"
    raw.mkdir()
    with (raw / "qwen_traceA_blksz_16.jsonl").open("w") as f:
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")
    plan = tmp_path / "x"
    plan.mkdir()
    (plan / "manifest.json").write_text(
        json.dumps({
            "duration_s": 26.0,
            "stop_s": 26.0,
            "workloads": {
                "bailian-chat": {
                    "window_start_s": 0.0,
                    "kept_session_fraction": 0.5
                }
            },
        }))
    with gzip.open(plan / "plan.jsonl.gz", "wt") as f:
        for i, chat in enumerate([0, 1]):
            f.write(
                json.dumps({
                    "request_id": f"bailian-chat:0:{i}",
                    "meta": {
                        "chat_id": chat
                    }
                }) + "\n")
        f.write(json.dumps({"request_id": "sharegpt:a:0", "meta": {}}) + "\n")
    assert pc.plan_chat_ids(plan)["bailian-chat"].tolist() == [0, 1]
    run_dir = tmp_path / "ceiling"
    write_tables(run_dir)
    (run_dir / "sim.json").write_text("{}")
    out = tmp_path / "card.json"
    pc.main([
        "--raw-dir",
        str(raw), "--traces", "bailian-chat", "--ceiling",
        str(run_dir), "--plan",
        str(plan), "--out",
        str(out)
    ])
    card = json.loads(out.read_text())
    assert [r["claim"] for r in card["rows"]] == list(range(1, 11))
    first = card["rows"][0]
    # The command skips the first 600 s, longer than this whole trace.
    assert first["raw"]["bailian-chat"]["prompt_blocks"] == 0
    assert first["mix"]["ceiling"]["wb"]["ideal_hit_ratio"] == 0.5
    assert first["mix_verdict"]["ceiling"] == pc.PARTLY  # 1.0 and 0.5
    assert card["inputs"]["guide_only_runs"] == ["ceiling"]
    thin = card["thinning"]["x"]["bailian-chat"]
    assert thin["kept"]["requests"] == 2 and thin["full_rate"]["requests"] == 3
    # The run's plan is "p/x", so the run's value attaches to plan "x"; its
    # workloads are not Bailian, so the value is null.
    assert thin["mix_ceiling_hit_ratio"] == {"ceiling": None}
