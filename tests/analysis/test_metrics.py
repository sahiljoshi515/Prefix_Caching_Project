"""Shared prefix-sharing metrics (specs/004-m2-prefix-sharing-analysis).

Every expected value is worked out by hand from the small logs below.
"""
import gzip
import json
from typing import Dict, List

import pytest

from analysis import events as ev
from analysis import metrics as m

pytestmark = pytest.mark.m2


def E(t, kind, h, req=None, op="allocate", block=0):  # noqa: N802
    return ev.Event(t, kind, h, block, req, op)


A0 = "chatcmpl-sharegpt:A:0"
A1 = "chatcmpl-sharegpt:A:1"
B0 = "chatcmpl-sharegpt:B:0"

# Hash 1: inserted by A0 at t=1, hit by A1 at t=4 and by B0 at t=10.
# Hash 2: inserted by A0 at t=1, never hit.
# Hash 3: written while A0 decoded (t=2), hit by A1 at t=4.
# Hash 4: inserted by B0 at t=5, freed, evicted at t=7, inserted again by
#         B0 at t=8 and hit by A1 at t=9.
LOG = [
    E(1, "miss", 1, A0),
    E(1, "insert", 1, A0),
    E(1, "miss", 2, A0),
    E(1, "insert", 2, A0),
    E(2, "insert", 3, A0, op="append_slots"),
    E(3, "free", 1, A0, op="free"),
    E(4, "hit", 1, A1),
    E(4, "revive", 1, A1),
    E(4, "hit", 3, A1),
    E(5, "miss", 4, B0),
    E(5, "insert", 4, B0),
    E(6, "free", 4, B0, op="free"),
    E(7, "evict", 4, A1),
    E(8, "miss", 4, B0),
    E(8, "insert", 4, B0),
    E(9, "hit", 4, A1),
    E(10, "hit", 1, B0),
]

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def write_log(path, lines, end=True, compress=False):
    header = {
        "ev": "header",
        "time": 100.0,
        "t": 0.5,
        "pid": 1,
        "block_size": 16
    }
    rows = [header] + [dict(e._asdict(), hash=e.hash) for e in lines]
    if end:
        rows.append({"ev": "end"})
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if compress:
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)
    return path


@pytest.mark.parametrize("compress", [False, True])
def test_load_events_reads_header_events_and_end(tmp_path, compress):
    name = "events.jsonl.gz" if compress else "events.jsonl"
    log = ev.load_events(write_log(tmp_path / name, LOG, compress=compress))
    assert log.header["block_size"] == 16
    assert log.complete
    assert log.events == LOG


def test_load_events_flags_a_missing_end_line(tmp_path):
    log = ev.load_events(write_log(tmp_path / "e.jsonl", LOG[:3], end=False))
    assert not log.complete
    assert len(log.events) == 3


def test_parse_request_id_strips_prefix_and_splits_turn():
    assert ev.parse_request_id("chatcmpl-sharegpt:QWJhYvA:12") == (
        "sharegpt:QWJhYvA", 12)
    assert ev.parse_request_id("sharegpt:x:0") == ("sharegpt:x", 0)
    assert ev.parse_request_id(None) is None
    assert ev.parse_request_id("chatcmpl-no-turn") is None


def test_read_metric_takes_the_value_from_prometheus_text(tmp_path):
    path = tmp_path / "metrics.txt"
    path.write_text(
        "# HELP vllm:gpu_prefix_cache_hit_rate GPU prefix cache hit rate.\n"
        "# TYPE vllm:gpu_prefix_cache_hit_rate gauge\n"
        'vllm:gpu_prefix_cache_hit_rate{model_name="m"} 0.4375\n')
    assert ev.read_metric(path, "vllm:gpu_prefix_cache_hit_rate") == 0.4375
    assert ev.read_metric(path, "vllm:absent") is None


# --------------------------------------------------------------------------
# Per-request cached fraction (client results)
# --------------------------------------------------------------------------

RESULTS = [
    {
        "request_id": "sharegpt:A:0",
        "conversation_id": "sharegpt:A",
        "turn_index": 0,
        "status": "completed",
        "prompt_tokens": 40,
        "cached_tokens": 0
    },
    {
        "request_id": "sharegpt:A:1",
        "conversation_id": "sharegpt:A",
        "turn_index": 1,
        "status": "completed",
        "prompt_tokens": 100,
        "cached_tokens": 48
    },
    {
        "request_id": "sharegpt:B:0",
        "conversation_id": "sharegpt:B",
        "turn_index": 0,
        "status": "completed",
        "prompt_tokens": 60,
        "cached_tokens": 32
    },
    {
        "request_id": "sharegpt:C:0",
        "conversation_id": "sharegpt:C",
        "turn_index": 0,
        "status": "skipped_context",
        "prompt_tokens": 9000,
        "cached_tokens": None
    },
]


def test_cached_fraction_counts_no_hit_as_zero_and_drops_skipped():
    rows, excluded = m.cached_fractions(RESULTS)
    assert [r["cached_fraction"] for r in rows] == [0.0, 0.48, 32 / 60]
    assert excluded == {"skipped_context": 1}


def test_token_level_reused_fraction():
    rows, _ = m.cached_fractions(RESULTS)
    assert m.token_reused_fraction(rows) == pytest.approx(80 / 200)


def test_mean_cached_fraction_by_turn():
    rows, _ = m.cached_fractions(RESULTS)
    assert m.mean_by_turn(rows) == {
        0: pytest.approx((0.0 + 32 / 60) / 2),
        1: pytest.approx(0.48),
    }


# --------------------------------------------------------------------------
# Block metrics (event log)
# --------------------------------------------------------------------------


def test_hits_per_block_includes_zero_hit_blocks():
    assert m.hits_per_block(LOG) == {1: 2, 2: 0, 3: 1, 4: 1}


def test_hits_per_residency_restarts_after_eviction():
    # Hash 4: first residency (t=5..7) has no hit, the second has one.
    assert sorted(m.hits_per_residency(LOG)) == [0, 0, 1, 1, 2]


def test_reuse_intervals_between_consecutive_accesses():
    rows = m.reuse_intervals(LOG)
    by_hash: Dict[int, List[float]] = {}
    for row in rows:
        by_hash.setdefault(row["hash"], []).append(row["interval_s"])
    # Hash 1: insert 1, hits 4 and 10 -> 3 and 6. Hash 2: one access only.
    # Hash 3: insert 2, hit 4 -> 2. Hash 4: insert 5, insert 8, hit 9.
    assert by_hash == {1: [3, 6], 3: [2], 4: [3, 1]}


def test_reuse_intervals_count_requests_started_in_between():
    rows = {(r["hash"], r["interval_s"]): r for r in m.reuse_intervals(LOG)}
    # Requests start (first allocate event): A0 t=1, A1 t=4, B0 t=5.
    # Hash 1, 1 -> 4: A1 starts at 4 but is the accessing request -> 0.
    assert rows[(1, 3)]["requests_between"] == 0
    # Hash 1, 4 -> 10: B0 started at 5 but is the accessing request -> 0.
    assert rows[(1, 6)]["requests_between"] == 0
    # Hash 4, 5 -> 8: nobody new started in (5, 8).
    assert rows[(4, 3)]["requests_between"] == 0


def test_requests_between_counts_other_requests():
    log = [
        E(0, "insert", 9, "chatcmpl-x:X:0"),
        E(1, "miss", 7, "chatcmpl-y:Y:0"),
        E(2, "miss", 8, "chatcmpl-z:Z:0"),
        E(3, "hit", 9, "chatcmpl-x:X:1"),
    ]
    (row, ) = m.reuse_intervals(log)
    assert row["interval_s"] == 3
    assert row["requests_between"] == 2


def test_block_hit_rate_counts_lookups_only():
    # Lookups: hits at 4, 4, 9, 10; misses at 1, 1, 5, 8.
    assert m.block_hit_rate(LOG) == 0.5
    assert m.block_hit_rate([]) is None


def test_windowed_hit_rate():
    windows = m.windowed_hit_rate(LOG, window_s=5)
    # [1, 6): 2 hits / 5 lookups; [6, 11): 2 hits / 3 lookups.
    assert [(w["start_s"], w["lookups"], w["hit_rate"])
            for w in windows] == [(0, 5, 0.4), (5, 3, pytest.approx(2 / 3))]


def test_hit_sources_split_intra_cross_and_decode_written():
    sources = m.hit_sources(LOG)
    # Hash 1 hit by A1 (same conversation as inserter A0) and B0 (other).
    # Hash 3 was written by A0's decode, hit by A1. Hash 4 by B0, hit by A1.
    assert sources == {
        ("intra", "allocate"): 1,
        ("intra", "append_slots"): 1,
        ("cross", "allocate"): 2,
    }


def test_hit_sources_unknown_when_insert_is_not_in_the_log():
    assert m.hit_sources([E(1, "hit", 5, A0)]) == {("unknown", None): 1}


# --------------------------------------------------------------------------
# Self-checks (SC-003, SC-004)
# --------------------------------------------------------------------------


def test_cached_token_gap_per_request():
    rows, _ = m.cached_fractions(RESULTS)
    gaps = {
        g["request_id"]: g["gap_tokens"]
        for g in m.cached_token_gaps(rows, LOG, block_size=16)
    }
    # A1: 48 cached, 3 allocate hits -> 0. B0: 32 cached, 1 hit -> 16.
    # A0: no hit, 0 cached -> 0.
    assert gaps == {"sharegpt:A:0": 0, "sharegpt:A:1": 0, "sharegpt:B:0": 16}


def test_cdf_points():
    xs, ys = m.cdf([3, 1, 2, 2])
    assert xs == [1, 2, 2, 3]
    assert ys == [0.25, 0.5, 0.75, 1.0]
