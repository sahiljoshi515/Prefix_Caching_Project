"""The fast cache simulator (specs/007-m3-mixed-workload-policy, plan).

Toy plans with 4-token blocks, so every hit and victim is derived by hand.
The simulator drives vLLM's own allocator and evictors, so these also check
that it calls them the way the block manager does.
"""
import json

import pytest

from analysis import cachesim
from analysis import run_m2_task2 as task2
from analysis.cachesim import CacheSim, Timing
from client_simulator.schema import TokenRequest

pytestmark = pytest.mark.m3

BS = 4


def req(rid, prompt, output=(), arrival=0.0, after=None):
    return TokenRequest(request_id=rid,
                        session_id=rid.rsplit(":", 1)[0],
                        turn_index=int(rid.rsplit(":", 1)[1]),
                        prompt_token_ids=list(prompt),
                        output_token_ids=list(output),
                        arrival_s=None if after is not None else arrival,
                        after_s=after)


def run(requests, times, blocks=64, policy="lru"):
    sim = CacheSim(blocks, BS, policy)
    timing = {r: Timing(s, f, la=f) for r, (s, f) in times.items()}
    result = sim.run(requests, timing)
    return result.summary, {row["request_id"]: row for row in result.rows}


def test_shared_prefix_and_decode_blocks_are_reused():
    a0 = req("w:a:0", range(1, 9), output=[9, 10, 11, 12, 13])
    # The next turn reads the first turn's prompt and output: blocks
    # [1-4], [5-8] from the prompt and [9-12] written while decoding.
    a1 = req("w:a:1", list(range(1, 14)) + [60, 61, 62], after=1.0)
    b0 = req("v:b:0", list(range(1, 9)) + [50, 51, 52, 53], arrival=0.5)
    summary, rows = run([a0, b0, a1], {
        "w:a:0": (0.0, 0.1),
        "v:b:0": (0.5, 0.6),
        "w:a:1": (1.1, 1.2)
    })
    assert (rows["w:a:0"]["hits"], rows["w:a:0"]["lookups"]) == (0, 2)
    assert (rows["v:b:0"]["hits"], rows["v:b:0"]["usable_cached_tokens"],
            rows["v:b:0"]["lookups"]) == (2, 8, 3)
    # 16 tokens: four full blocks, the first three cached.
    assert (rows["w:a:1"]["hits"], rows["w:a:1"]["usable_cached_tokens"],
            rows["w:a:1"]["lookups"]) == (3, 12, 4)
    assert summary["hits"] == 5 and summary["lookups"] == 9
    assert summary["block_hit_rate"] == pytest.approx(5 / 9)
    assert summary["hit_rate_metric"] == pytest.approx(5 / 9)
    assert summary["usable_reused_fraction"] == pytest.approx(20 / 36)
    assert summary["workloads"]["v"]["usable_reused_fraction"] == 8 / 12
    assert summary.get("evictions", 0) == 0


def lru_fifo_trace(probe):
    """Four blocks. R1 (tokens 1-8) and R2 (101-108) fill the cache, R3
    reads R1's blocks again after R2 is freed, R4 needs two new blocks, and
    one probe then reads `probe` (no second probe: it would evict too)."""
    return [
        req("w:r1:0", range(1, 9)),
        req("w:r2:0", range(101, 109), arrival=1.0),
        req("w:r3:0", range(1, 9), arrival=2.5),
        req("w:r4:0", range(201, 209), arrival=3.0),
        req("w:p:0", probe, arrival=4.0),
    ], {
        "w:r1:0": (0.0, 0.1),
        "w:r2:0": (1.0, 1.1),
        "w:r3:0": (2.5, 2.6),
        "w:r4:0": (3.0, 3.1),
        "w:p:0": (4.0, 4.1),
    }


@pytest.mark.parametrize(
    "policy,r1_hits,r2_hits",
    [
        # LRU: R3 refreshed R1's blocks (last access 2.6), so R4 evicts R2's
        # (1.1).
        ("lru", 2, 0),
        # FIFO: order of first free (R1's at 0.1), so R4 evicts R1's although
        # they were just read.
        ("fifo", 0, 2),
    ])
def test_r4_evicts_what_the_policy_says(policy, r1_hits, r2_hits):
    for probe, expected in ((range(1, 9), r1_hits), (range(101,
                                                           109), r2_hits)):
        requests, times = lru_fifo_trace(probe)
        summary, rows = run(requests, times, blocks=4, policy=policy)
        assert rows["w:r3:0"]["hits"] == 2
        assert rows["w:p:0"]["hits"] == expected
        assert summary["evictions"] == 2 + (2 - expected)


def test_a_prompt_that_does_not_fit_waits_for_a_free():
    big = req("w:a:0", range(1, 13))  # three blocks, held until 2.0
    other = req("w:b:0", range(51, 59), arrival=1.0)  # needs two of four
    summary, rows = run([big, other], {
        "w:a:0": (0.0, 2.0),
        "w:b:0": (1.0, 1.5)
    },
                        blocks=4)
    assert summary["deferred"] == 1
    assert rows["w:b:0"]["start_s"] == 2.0
    assert summary["requests"] == 2


def test_output_appended_at_the_free_is_its_requests_decode(tmp_path):
    """Three blocks. R2's prompt fills a block and its two output tokens
    fill no decode block, so they are appended when R2 is freed. That needs
    a new block, and the only one left is R1's cached [1-4]: its eviction is
    R2's append_slots, although R3 allocated in between."""
    log = tmp_path / "events.jsonl"
    sim = CacheSim(3, BS, "lru", event_log=log)
    requests = [
        req("w:r1:0", range(1, 5)),
        req("w:r2:0", range(11, 15), output=[21, 22]),
        req("w:r3:0", range(31, 35)),
    ]
    times = {"w:r1:0": (0.0, 0.1), "w:r2:0": (1.0, 1.5), "w:r3:0": (1.2, 2.0)}
    sim.run(requests, {r: Timing(s, f, la=f) for r, (s, f) in times.items()})
    events = [json.loads(line) for line in log.read_text().splitlines()]
    evicts = [(e["hash"], e["req"], e["op"], e["t"]) for e in events
              if e["ev"] == "evict"]
    r1_block = sim._hashes(list(range(1, 5)))[0]
    assert evicts == [(r1_block, "cmpl-w:r2:0-0", "append_slots", 1.5)]


def test_engine_timing_reads_starts_frees_and_last_access(tmp_path):
    events = [
        {
            "ev": "header",
            "t": 100.0
        },
        {
            "t": 101.0,
            "ev": "miss",
            "hash": 1,
            "block": None,
            "req": "cmpl-w:a:0-0",
            "op": "allocate"
        },
        {
            "t": 101.5,
            "ev": "insert",
            "hash": 2,
            "block": 1,
            "req": "cmpl-w:a:0-0",
            "op": "append_slots"
        },
        {
            "t": 102.0,
            "ev": "free",
            "hash": 1,
            "block": 0,
            "req": "cmpl-w:a:0-0",
            "op": "free",
            "la": 5001.9
        },
        {
            "ev": "end"
        },
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
    client = [
        {
            "request_id": "w:a:0",
            "sent_s": 1.0,
            "completed_s": 2.0
        },
        {
            "request_id": "w:b:0",
            "sent_s": 3.0,
            "completed_s": 3.5
        },  # no events: placed on the log's clock by the offset (100)
        {
            "request_id": "w:c:0",
            "sent_s": None,
            "completed_s": 4.0
        },  # never sent
    ]
    (tmp_path / "client.jsonl").write_text("\n".join(map(json.dumps, client)))
    plan = [
        req("w:a:0", range(4)),
        req("w:b:0", range(4)),
        req("w:c:0", range(4))
    ]
    timing = cachesim.engine_timing(tmp_path, plan)
    assert timing["w:a:0"] == Timing(101.0, 102.0, 5001.9, [101.5])
    b = timing["w:b:0"]
    assert (b.start, b.finish) == (103.0, 103.5)
    # vLLM's wall clock is 4,899.9 s ahead of the log's (5001.9 - 102).
    assert b.la == pytest.approx(103.5 + 4899.9)
    assert "w:c:0" not in timing


def test_run_folder_is_read_by_the_task2_analysis(tmp_path, monkeypatch):
    a0 = req("w:a:0", range(1, 9), output=[9, 10, 11, 12, 13])
    a1 = req("w:a:1", list(range(1, 14)) + [60, 61, 62], after=1.0)
    plan_dir = tmp_path / "plan"
    from client_simulator.token_plan import write_plan
    write_plan([a0, a1], plan_dir / "plan.jsonl.gz",
               plan_dir / "responses.jsonl")
    out = tmp_path / "run"
    result = cachesim.simulate(plan_dir, "lru", 64 * BS, BS, out=out)
    summary = task2.analyze_run(out, 10.0)
    assert summary["log_complete"]
    assert summary["sc003_pass"]
    assert summary["block_hit_rate_log"] == pytest.approx(
        result.summary["block_hit_rate"])
    assert summary["usable_reused_fraction"] == pytest.approx(
        result.summary["usable_reused_fraction"])


def test_window_counts_only_requests_planned_inside_it():
    early = req("w:a:0", range(1, 9), output=[9, 10], arrival=0.0)
    later = req("w:a:1", range(1, 13), after=1.0)
    later.meta["planned_s"] = 5.0
    other = req("v:b:0", range(1, 9), arrival=6.0)
    rows = [
        {
            "request_id": "w:a:0",
            "workload": "w",
            "prompt_tokens": 8,
            "usable_cached_tokens": 0
        },
        {
            "request_id": "w:a:1",
            "workload": "w",
            "prompt_tokens": 12,
            "usable_cached_tokens": 8
        },
        {
            "request_id": "v:b:0",
            "workload": "v",
            "prompt_tokens": 8,
            "usable_cached_tokens": 8
        },
    ]
    window = cachesim.window_summary(rows, [early, later, other], 5.0, 6.0)
    # Planned at 5 s (the closed-loop turn's planned time) is in [5, 6);
    # 0 s and 6 s are not.
    assert window["requests"] == 1 and window["prompt_tokens"] == 12
    assert window["usable_reused_fraction"] == pytest.approx(8 / 12)
    assert set(window["workloads"]) == {"w"}
    assert (window["w0_s"], window["stop_s"]) == (5.0, 6.0)


def test_simulate_adds_the_window_to_its_summary(tmp_path):
    from client_simulator.token_plan import write_plan
    a0 = req("w:a:0", range(1, 9), output=[9, 10, 11, 12, 13])
    a1 = req("w:a:1", list(range(1, 14)) + [60, 61, 62], after=1.0)
    a1.meta["planned_s"] = 2.0
    plan_dir = tmp_path / "plan"
    write_plan([a0, a1], plan_dir / "plan.jsonl.gz",
               plan_dir / "responses.jsonl")
    whole = cachesim.simulate(plan_dir, "lru", 64 * BS, BS).summary
    assert "window" not in whole
    summary = cachesim.simulate(plan_dir,
                                "lru",
                                64 * BS,
                                BS,
                                window=(1.0, 10.0)).summary
    w = summary["window"]
    assert w["requests"] == 1 and w["prompt_tokens"] == 16
    assert w["usable_tokens"] == summary["usable_tokens"]
