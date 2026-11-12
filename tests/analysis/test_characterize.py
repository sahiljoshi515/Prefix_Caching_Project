"""Per-workload reuse profiles (specs/007-m3-mixed-workload-policy, US2).

One handcrafted log with two workloads (``wa``, ``wb``); every expected
value below is worked out by hand from it.
"""
import pytest

from analysis import characterize as ch
from analysis import events as ev

pytestmark = pytest.mark.m3

A0 = "cmpl-wa:s1:0-0"
A1 = "cmpl-wa:s1:1-0"
B0 = "cmpl-wb:s2:0-0"
B1 = "cmpl-wb:s2:1-0"
C0 = "cmpl-wa:s3:0-0"


def E(t, kind, h, req, op="allocate"):  # noqa: N802
    return ev.Event(t, kind, h, None, req, op)


# t=0  A0 computes blocks 1, 2; t=1 it writes block 3 while decoding.
# t=3  B0 reads block 1 (computed by wa) and computes 9.
# t=5  A1 reads 1, 2, 3 and computes 4.
# t=6  C0 computes 5, evicting 9 (last held by B0).
# t=9  B1 needs 9 again: a miss on an evicted block.
LOG = [
    E(0, "miss", 1, A0),
    E(0, "insert", 1, A0),
    E(0, "miss", 2, A0),
    E(0, "insert", 2, A0),
    E(1, "insert", 3, A0, "append_slots"),
    E(2, "free", 1, A0, "free"),
    E(2, "free", 2, A0, "free"),
    E(2, "free", 3, A0, "free"),
    E(3, "hit", 1, B0),
    E(3, "revive", 1, B0),
    E(3, "miss", 9, B0),
    E(3, "insert", 9, B0),
    E(4, "free", 1, B0, "free"),
    E(4, "free", 9, B0, "free"),
    E(5, "hit", 1, A1),
    E(5, "revive", 1, A1),
    E(5, "hit", 2, A1),
    E(5, "revive", 2, A1),
    E(5, "hit", 3, A1),
    E(5, "revive", 3, A1),
    E(5, "miss", 4, A1),
    E(5, "insert", 4, A1),
    E(6, "miss", 5, C0),
    E(6, "evict", 9, C0),
    E(6, "insert", 5, C0),
    E(7, "free", 1, A1, "free"),
    E(7, "free", 2, A1, "free"),
    E(7, "free", 3, A1, "free"),
    E(7, "free", 4, A1, "free"),
    E(8, "free", 5, C0, "free"),
    E(9, "miss", 9, B1),
    E(9, "insert", 9, B1),
]


def test_hits_misses_and_their_sources():
    p = ch.reuse_profile(LOG)
    wa, wb = p["wa"], p["wb"]
    # wa: A1's three hits, all on blocks its own session computed (one of
    # them written while decoding); misses: A0 x2, A1, C0.
    assert (wa["hits"], wa["misses"]) == (3, 4)
    assert wa["block_hit_rate"] == pytest.approx(3 / 7)
    assert wa["hit_share_same_session"] == 1.0
    assert wa["hit_share_on_decode_blocks"] == pytest.approx(1 / 3)
    assert (wa["inserts"], wa["decode_inserts"]) == (5, 1)
    # wb: B0's hit is on a block wa computed.
    assert (wb["hits"], wb["misses"]) == (1, 2)
    assert wb["hit_share_cross_workload"] == 1.0
    assert wb["misses_on_evicted"] == 1


def test_free_pool_times_and_requests_in_between():
    p = ch.reuse_profile(LOG)
    # wa revives: block 1 freed by B0 at 4, back at 5 (1 s); blocks 2 and
    # 3 freed at 2, back at 5 (3 s), with B0 started at 3 in between.
    assert p["wa"]["free_pool_s_until_revive"]["n"] == 3
    assert p["wa"]["free_pool_s_until_revive"]["p50"] == 3
    assert p["wa"]["free_pool_s_until_revive"]["mean"] == pytest.approx(7 / 3)
    assert p["wa"]["requests_between_free_and_revive"]["mean"] == (
        pytest.approx(2 / 3))
    # wb: block 1 freed at 2, back at 3; B0 itself starts at 3.
    assert p["wb"]["free_pool_s_until_revive"]["mean"] == 1
    assert p["wb"]["requests_between_free_and_revive"]["mean"] == 0
    assert p["wa"]["revives"] == 3 and p["wa"]["frees"] == 8
    assert p["wa"]["revive_share_of_frees"] == pytest.approx(3 / 8)


def test_evictions_count_against_the_blocks_last_holder():
    p = ch.reuse_profile(LOG)
    # Block 9 was evicted during C0's (wa) allocation but last held by B0.
    assert p["wb"]["evicted"] == 1
    assert "evicted" not in p["wa"]
    assert p["wb"]["free_pool_s_until_evict"]["mean"] == 2


def test_access_gaps_since_the_previous_access():
    p = ch.reuse_profile(LOG)
    # Block 1: inserted 0, hit 3 (wb, 3 s), hit 5 (wa, 2 s); block 2:
    # 0 -> 5; block 3: 1 -> 5.
    assert p["wa"]["access_gap_s"]["n"] == 3
    assert p["wa"]["access_gap_s"]["mean"] == pytest.approx(11 / 3)


def test_access_gaps_span_an_eviction():
    # Regression: block 9 is inserted by B0 at 3, evicted at 6 and inserted
    # again by B1 at 9, a 6 s gap (004's interval). The gap used to be lost
    # with the eviction, leaving wb only block 1's 3 s.
    gaps = ch.reuse_profile(LOG)["wb"]["access_gap_s"]
    assert (gaps["n"], gaps["mean"], gaps["p99"]) == (2, 4.5, 6)


def test_blocks_per_workload_and_their_hits():
    p = ch.reuse_profile(LOG)
    # wa computed blocks 1-5: block 1 hit twice, 2 and 3 once.
    assert p["wa"]["blocks"] == 5
    assert p["wa"]["blocks_ever_hit_share"] == pytest.approx(3 / 5)
    assert p["wb"]["blocks"] == 1  # block 9


def test_prefix_families_and_their_purity():
    families = ch.prefix_families(LOG, depths=(1, 2))
    assert families["1"] == {
        "wa:s1:0": 1,
        "wb:s2:0": 1,
        "wa:s1:1": 1,
        "wa:s3:0": 5,
        "wb:s2:1": 9
    }
    # Depth 2: C0 has one block, so its family is that block.
    assert families["2"]["wa:s1:1"] == 2 and families["2"]["wa:s3:0"] == 5
    one = ch.family_purity(families["1"])
    assert (one["families"], one["purity"]) == (3, pytest.approx(4 / 5))
    two = ch.family_purity(families["2"])
    assert (two["families"], two["purity"],
            two["singleton_families"]) == (3, 1.0, 1)


def test_family_purity_without_singletons():
    # Depth 1: family 1 holds wa:s1:0, wa:s1:1 and wb:s2:0 (majority wa);
    # families 5 (C0) and 9 (B1) have one request each.
    one = ch.family_purity(ch.prefix_families(LOG, depths=(1, ))["1"])
    assert one["singleton_families"] == 2
    assert one["singleton_family_share"] == pytest.approx(2 / 3)
    assert one["singleton_request_share"] == pytest.approx(2 / 5)
    assert one["purity_without_singletons"] == pytest.approx(2 / 3)
    # Only singletons: nothing is left to score.
    alone = ch.family_purity({"wa:x:0": 1, "wb:y:0": 2})
    assert (alone["purity"], alone["purity_without_singletons"]) == (1.0, None)
    assert alone["singleton_family_share"] == 1.0


def test_family_classifier_learned_on_one_run_scored_on_another():
    table = ch.family_classifier(ch.prefix_families(LOG, depths=(2, ))["2"])
    test = {"wa:x:0": 2, "wb:y:0": 9, "wb:z:0": 77}
    result = ch.classify(table, test)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["unseen_share"] == pytest.approx(1 / 3)
    assert result["confusion"] == {
        "wa": {
            "wa": 1
        },
        "wb": {
            "wb": 1,
            "unknown": 1
        }
    }


def test_occupancy_follows_the_last_holder():
    rows = ch.occupancy(LOG, window_s=5.0)
    # At t=5 (end of the first window): blocks 2 and 3 for wa; block 1,
    # last freed by B0, and block 9 for wb (block 1 returns to wa when A1
    # frees it at 7).
    assert rows[0] == {"t_s": 5.0, "wa": 2, "wb": 2}
    # At the end: blocks 1-5 for wa, 9 again for wb (B1 inserted it).
    assert {k: v
            for k, v in rows[-1].items() if k != "t_s"} == {
                "wa": 5,
                "wb": 1
            }
