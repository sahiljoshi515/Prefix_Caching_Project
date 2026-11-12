"""Per-run lookup, residency and request tables (analysis/tables.py).

One handcrafted run with two workloads (``wa``, ``wb``) and three sessions;
every expected value below is worked out by hand from its log. The header
pairs wall time 1000 with log time 50, so a free's ``la`` of 1002.5 is
52.5 on the log's clock.
"""
import gzip
import json
import math
from pathlib import Path

import pytest

from analysis import tables as tb

pytestmark = pytest.mark.m3

A0, A1, C0 = "wa:s1:0", "wa:s1:1", "wa:s2:0"
B0, B1 = "wb:s3:0", "wb:s3:1"
STOPPED = "wa:s2:1"


def row(t, kind, h, req, op="allocate", block=None, la=None):
    out = {
        "t": t,
        "ev": kind,
        "hash": h,
        "block": block,
        "req": f"cmpl-{req}-0",
        "op": op
    }
    if la is not None:
        out["la"] = la
    return out


# Hashes 1-8 stand for blocks; residencies r0-r12 are numbered by free.
# t=51 A0 computes 1, 2; decodes 3 (completing its partial block), 4.
# t=53 A0 frees all four: r0-r3.
# t=54 C0 (other session) hits 1 (revive r0), computes 7 evicting 2 (r1).
# t=55 B0 (other workload) hits 1 while C0 holds it (no revive), computes
#      5 evicting 4 (r3). t=56 C0 frees 7 (r4); t=57 B0 frees 1 (r5), 5
#      (r6).
# t=58 A1 hits 1 (r5), misses 2 (evicted; 5 is evicted for it, r6), hits
#      3 behind that miss (r2), misses 4 (evicted) and 6; decodes 8.
# t=61 A1 frees 1, 2, 3, 4, 6, 8: r7-r12.
# t=62 B1 hits 1 (r7) and misses 5 (evicted at 58).
LOG = [
    row(51, "miss", 1, A0),
    row(51, "insert", 1, A0, block=0),
    row(51, "miss", 2, A0),
    row(51, "insert", 2, A0, block=1),
    row(52, "insert", 3, A0, "append_slots", block=2),
    row(52.5, "insert", 4, A0, "append_slots", block=3),
    row(53, "free", 1, A0, "free", block=0, la=1002.5),
    row(53, "free", 2, A0, "free", block=1, la=1002.5),
    row(53, "free", 3, A0, "free", block=2, la=1002.5),
    row(53, "free", 4, A0, "free", block=3, la=1002.5),
    row(54, "hit", 1, C0, block=0),
    row(54, "revive", 1, C0, block=0),
    row(54, "miss", 7, C0),
    row(54, "evict", 2, C0, block=1),
    row(54, "insert", 7, C0, block=1),
    row(55, "hit", 1, B0, block=0),
    row(55, "miss", 5, B0),
    row(55, "evict", 4, B0, block=3),
    row(55, "insert", 5, B0, block=3),
    row(56, "free", 7, C0, "free", block=1, la=1005.5),
    row(57, "free", 1, B0, "free", block=0, la=1006.5),
    row(57, "free", 5, B0, "free", block=3, la=1006.5),
    row(58, "hit", 1, A1, block=0),
    row(58, "revive", 1, A1, block=0),
    row(58, "miss", 2, A1),
    row(58, "evict", 5, A1, block=3),
    row(58, "insert", 2, A1, block=3),
    row(58, "hit", 3, A1, block=2),
    row(58, "revive", 3, A1, block=2),
    row(58, "miss", 4, A1),
    row(58, "insert", 4, A1, block=4),
    row(58, "miss", 6, A1),
    row(58, "insert", 6, A1, block=5),
    row(59, "insert", 8, A1, "append_slots", block=6),
] + [
    row(61, "free", h, A1, "free", block=b, la=1010.5)
    for h, b in ((1, 0), (2, 3), (3, 2), (4, 4), (6, 5), (8, 6))
] + [
    row(62, "hit", 1, B1, block=0),
    row(62, "revive", 1, B1, block=0),
    row(62, "miss", 5, B1),
    row(62, "insert", 5, B1, block=7),
]

# (request, prompt tokens, arrival_s, after_s, planned_s)
PLAN = [
    (A0, 40, 1.0, None, 1.0),
    (A1, 90, None, 2.0, 5.0),
    (C0, 40, 3.0, None, 3.0),
    (STOPPED, 50, None, 1.0, 9.0),
    (B0, 40, 4.0, None, 4.0),
    (B1, 40, None, 1.0, 7.0),
]

HEADER = {
    "ev": "header",
    "time": 1000.0,
    "t": 50.0,
    "pid": 1,
    "block_size": 16
}


def write_log(path, log, header=HEADER):
    with gzip.open(path, "wt") as f:
        for line in [header, *log, {"ev": "end"}]:
            f.write(json.dumps(line) + "\n")


def write_run(tmp_path,
              log,
              plan=PLAN,
              manifest=None,
              stopped=(STOPPED, ),
              header=HEADER):
    """A simulator-style run folder next to its plan (``plan_dir`` is
    relative to a folder above the run)."""
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    with gzip.open(plan_dir / "plan.jsonl.gz", "wt") as f:
        for rid, prompt, arrival, after, planned in plan:
            session, _, turn = rid.rpartition(":")
            f.write(
                json.dumps({
                    "request_id": rid,
                    "session_id": session,
                    "turn_index": int(turn),
                    "prompt_token_ids": list(range(prompt)),
                    "output_token_ids": [1] * 30,
                    "arrival_s": arrival,
                    "after_s": after,
                    "meta": {
                        "planned_s": planned
                    },
                }) + "\n")
    (plan_dir / "manifest.json").write_text(json.dumps({"stop_s": 1800.0}))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_log(run_dir / "events.jsonl.gz", log, header)
    with (run_dir / "client.jsonl").open("w") as f:
        for rid, prompt, *_ in plan:
            session, _, turn = rid.rpartition(":")
            status = "stopped" if rid in stopped else "completed"
            client = {
                "request_id": rid,
                "conversation_id": session,
                "turn_index": int(turn),
                "workload": rid.split(":")[0],
                "status": status,
            }
            if status == "completed":  # as analysis/cachesim.py writes
                client.update(prompt_tokens=prompt, cached_tokens=16)
            f.write(json.dumps(client) + "\n")
    manifest = manifest or {
        "plan_dir": "plan",
        "policy": "lru",
        "capacity_tokens": 128,
        "block_size": 16,
    }
    (run_dir / "sim.json").write_text(json.dumps(manifest))
    return run_dir


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    run_dir = write_run(tmp_path_factory.mktemp("tables"), LOG)
    paths = tb.extract_run(run_dir)
    assert sorted(paths) == sorted(tb.TABLES)
    return {name: tb.load_table(run_dir, name).to_pylist() for name in paths}


def column(rows, name):
    return [r[name] for r in rows]


def test_lookups_follow_the_log(run):
    lookups = run["lookups"]
    assert column(lookups, "hash") == [1, 2, 1, 7, 1, 5, 1, 2, 3, 4, 6, 1, 5]
    assert column(
        lookups, "req") == [A0] * 2 + [C0] * 2 + [B0] * 2 + [A1] * 5 + [B1] * 2
    a1 = lookups[6]
    assert (a1["workload"], a1["session"], a1["turn"],
            a1["planned_s"]) == ("wa", "wa:s1", 1, 5.0)


def test_in_use_hits_are_the_ones_without_a_revive(run):
    # B0's hit at 55 shares block 1 with C0; every other hit revives.
    hits = [r for r in run["lookups"] if r["hit"]]
    assert [(r["req"], r["in_use"]) for r in hits] == [(C0, False), (B0, True),
                                                       (A1, False),
                                                       (A1, False),
                                                       (B1, False)]
    assert not any(r["in_use"] for r in run["lookups"] if not r["hit"])


def test_sources_compare_the_previous_accessor(run):
    assert column(run["lookups"], "source") == [
        "first",
        "first",
        "other_session",  # C0 after A0's insert of 1
        "first",
        "other_workload",  # B0 after C0's hit
        "first",
        "other_workload",  # A1 after B0's hit
        "same_session",  # A1 after A0's insert of 2
        "same_session",  # ... of 3 (written while A0 decoded)
        "same_session",
        "first",
        "other_workload",  # B1 after A1's hit
        "same_session",  # B1 after B0's insert of 5
    ]


def test_previous_access_and_its_interval(run):
    lookups = run["lookups"]
    first = lookups[0]
    assert (math.isnan(first["prev_t"]), first["prev_req"], first["prev_kind"],
            math.isnan(first["interval_s"]),
            first["requests_between"]) == (True, None, None, True, -1)
    # B0's hit at 55 follows C0's hit at 54.
    assert (lookups[4]["prev_t"], lookups[4]["prev_req"],
            lookups[4]["prev_kind"], lookups[4]["interval_s"]) == (54, C0,
                                                                   "hit", 1)
    # A1's miss of 2 at 58 follows A0's insert at 51; C0 (54) and B0 (55)
    # started in between, A1 itself at 58.
    assert (lookups[7]["prev_t"], lookups[7]["prev_kind"],
            lookups[7]["interval_s"],
            lookups[7]["requests_between"]) == (51, "insert", 7, 2)
    # B1's miss of 5 at 62 follows B0's insert at 55: only A1 in between.
    assert [r["requests_between"]
            for r in lookups] == [-1, -1, 0, -1, 0, -1, 0, 2, 2, 2, -1, 0, 1]


def test_usable_stops_at_the_first_miss(run):
    a1 = [r for r in run["lookups"] if r["req"] == A1]
    assert column(a1, "hit") == [True, False, True, False, False]
    assert column(a1, "usable") == [True, False, False, False, False]
    assert column(a1, "index") == [0, 1, 2, 3, 4]
    assert all(r["prefill"] == 0 for r in run["lookups"])


def test_evicted_before_means_evicted_since_the_previous_access(run):
    # 2 (evicted 54), 4 (55) and 5 (58) are missed after their evictions;
    # A0's misses and 6 were never cached.
    assert [i for i, r in enumerate(run["lookups"])
            if r["evicted_before"]] == [7, 9, 12]


def test_residencies_end_in_revive_evict_or_censored(run):
    res = run["residencies"]
    assert column(res, "hash") == [1, 2, 3, 4, 7, 1, 5, 1, 2, 3, 4, 6, 8]
    assert column(res, "end") == [
        "revive", "evict", "revive", "evict", "censored", "revive", "evict",
        "revive"
    ] + ["censored"] * 5
    assert column(res, "t_end") == [54, 54, 58, 55, 62, 58, 58, 62] + [62] * 5
    assert column(res, "idle_s") == [1, 1, 5, 2, 6, 1, 1, 1] + [1] * 5
    assert column(
        res,
        "evictor_req") == [None, C0, None, B0, None, None, A1] + [None] * 6
    assert column(res, "evictor_workload")[1] == "wa"
    # The first lookup of the hash at or after its free, by anyone; none
    # for 7 or after A1's frees but block 1's.
    assert [
        None if math.isnan(t) else t for t in column(res, "next_lookup_t")
    ] == [54, 58, 58, 58, None, 58, 62, 62] + [None] * 5


def test_residency_owner_and_inserter(run):
    r5 = run["residencies"][5]  # block 1, freed by B0, computed by A0
    assert (r5["owner_req"], r5["owner_workload"], r5["owner_session"],
            r5["owner_turn"], r5["owner_planned_s"],
            r5["inserter_workload"]) == (B0, "wb", "wb:s3", 0, 4.0, "wa")


def test_what_each_policy_sees(run):
    res = run["residencies"]
    # LRU: la, converted to the log's clock.
    assert column(res, "la")[:7] == [52.5] * 4 + [55.5, 56.5, 56.5]
    # FIFO: when this copy first entered the pool. Block 1's copy dates
    # from 51 (first freed at 53); 2 and 4 were computed again at 58.
    assert column(res, "first_free_t") == [
        53, 53, 53, 53, 56, 53, 57, 53, 61, 53, 61, 61, 61
    ]
    assert column(res, "first_free_la")[5] == 52.5
    # LFU: hits since the copy was computed (C0, B0, then A1 on block 1).
    assert column(res,
                  "hits_before") == [0, 0, 0, 0, 0, 2, 0, 3, 0, 1, 0, 0, 0]


def test_index_runs_across_prompt_and_decode_blocks(run):
    # A0's prompt blocks are 0 and 1, its decode blocks 2 and 3; A1's
    # decode block 8 comes after its five prompt blocks.
    assert column(run["residencies"],
                  "index") == [0, 1, 2, 3, 1, 0, 1, 0, 1, 2, 3, 4, 5]


def test_requests_carry_the_first_prefill_and_004_usable_fraction(run):
    req = {r["req"]: r for r in run["requests"]}
    a1 = req[A1]
    assert (a1["prompt_blocks"], a1["hits"], a1["usable_hits"],
            a1["hits_after_miss"]) == (5, 2, 1, 1)
    assert a1["usable_fraction"] == pytest.approx(16 / 90)
    assert (a1["planned_s"], math.isnan(a1["sent_s"]), a1["t_start"],
            a1["preempted"]) == (5.0, True, 58, False)
    assert req[C0]["usable_fraction"] == pytest.approx(16 / 40)
    assert req[A0]["usable_fraction"] == 0.0
    # Never sent: its prompt length comes from the plan.
    stopped = req[STOPPED]
    assert (stopped["status"], stopped["prompt_tokens"],
            stopped["prompt_blocks"], stopped["cached_tokens"],
            math.isnan(stopped["t_start"])) == ("stopped", 50, 0, None, True)
    assert [r["req"] for r in run["requests"]] == [p[0] for p in PLAN]


def test_run_metadata(tmp_path):
    run_dir = write_run(tmp_path, LOG)
    tb.extract_run(run_dir, w0_s=30.0)
    info = tb.run_info(tb.load_table(run_dir, "residencies"))
    assert {
        k: info[k]
        for k in ("policy", "capacity_tokens", "block_size", "header_time",
                  "header_t", "censor_t", "w0_s", "stop_s", "complete",
                  "checks")
    } == {
        "policy": "lru",
        "capacity_tokens": 128,
        "block_size": 16,
        "header_time": 1000.0,
        "header_t": 50.0,
        "censor_t": 62,
        "w0_s": 30.0,
        "stop_s": 1800.0,
        "complete": True,
        "checks": {},
    }
    assert Path(info["plan_dir"]).resolve() == (tmp_path / "plan").resolve()
    with pytest.raises(ValueError, match="Unknown table"):
        tb.load_table(run_dir, "blocks")


# P is preempted after decoding a block and frees it: its next allocate
# is its second prefill. Q's blocks are all held by R, so its preemption
# logs nothing; it looks up its first block again instead.
P, Q, R = "wa:s5:0", "wa:s6:0", "wa:s7:0"
PREEMPT_LOG = [
    row(10, "miss", 1, P),
    row(10, "insert", 1, P, block=0),
    row(10, "miss", 2, P),
    row(10, "insert", 2, P, block=1),
    row(11, "insert", 3, P, "append_slots", block=2),
    row(12, "free", 1, P, "free", block=0, la=1011.0),
    row(12, "free", 2, P, "free", block=1, la=1011.0),
    row(12, "free", 3, P, "free", block=2, la=1011.0),
    row(13, "hit", 1, P, block=0),
    row(13, "revive", 1, P, block=0),
    row(13, "hit", 2, P, block=1),
    row(13, "revive", 2, P, block=1),
    row(13, "hit", 3, P, block=2),
    row(13, "revive", 3, P, block=2),
    row(20, "miss", 9, R),
    row(20, "insert", 9, R, block=3),
    row(21, "hit", 9, Q, block=3),
    row(22, "hit", 9, Q, block=3),
]


def test_reprefills_after_preemption(tmp_path):
    plan = [(P, 40, 1.0, None, 1.0), (Q, 20, 2.0, None, 2.0),
            (R, 20, 3.0, None, 3.0)]
    manifest = {
        "plan_dir": str(tmp_path / "plan"),
        "eviction_policy": "fifo",
        "capacity_tokens": 64,
        "ceiling": False,
    }
    run_dir = write_run(tmp_path, PREEMPT_LOG, plan, manifest, stopped=())
    tb.extract_run(run_dir)
    lookups = tb.load_table(run_dir, "lookups").to_pylist()
    assert [(r["req"], r["prefill"], r["index"])
            for r in lookups] == [(P, 0, 0), (P, 0, 1), (P, 1, 0), (P, 1, 1),
                                  (P, 1, 2), (R, 0, 0), (Q, 0, 0), (Q, 1, 0)]
    req = {r["req"]: r for r in tb.load_table(run_dir, "requests").to_pylist()}
    assert [req[x]["preempted"] for x in (P, Q, R)] == [True, True, False]
    # The first prefill only: P's re-prefill hits are not counted.
    assert (req[P]["prompt_blocks"], req[P]["hits"]) == (2, 0)
    assert (req[Q]["prompt_blocks"], req[Q]["usable_hits"]) == (1, 1)
    info = tb.run_info(tb.load_table(run_dir, "lookups"))
    assert (info["policy"], info["ceiling"]) == ("fifo", False)


def test_la_of_a_simulator_run_timed_from_an_engine_run(tmp_path):
    # cachesim's engine timing: the log runs on the engine's monotonic clock
    # and stamps the engine's wall-clock la, but its header pairs 0 with 0.
    engine = tmp_path / "engine"
    engine.mkdir()
    write_log(engine / "events.jsonl.gz", [], {
        **HEADER, "time": 2000.0,
        "t": 10.0
    })
    log = [
        row(13, "miss", 1, A0),
        row(13, "insert", 1, A0, block=0),
        row(14, "free", 1, A0, "free", block=0, la=2003.5),
    ]
    manifest = {"plan_dir": "plan", "policy": "lru", "timing": "engine:engine"}
    run_dir = write_run(tmp_path,
                        log,
                        manifest=manifest,
                        header={
                            **HEADER, "time": 0.0,
                            "t": 0.0
                        })
    tb.extract_run(run_dir)
    res = tb.load_table(run_dir, "residencies")
    assert res["la"].to_pylist() == [13.5]
    assert tb.run_info(res)["la_clock"].startswith("engine run")
    # Without that run, la cannot be put on the log's clock.
    (engine / "events.jsonl.gz").unlink()
    tb.extract_run(run_dir)
    res = tb.load_table(run_dir, "residencies")
    assert math.isnan(res["la"][0].as_py())
    assert tb.run_info(res)["la_clock"] == "unknown"
