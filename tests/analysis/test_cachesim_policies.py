"""TTDF and RACE-RP in the cache simulator (specs/007-m3-mixed-workload-policy).

Toy plans with 4-token blocks (``test_cachesim.req``), so every hit,
victim and eviction time is worked out by hand. They check that

- a policy string may carry the evictor's settings after a colon
  (``ttdf:tau_short=2``), and a policy without one clears them;
- the simulated clock reaches the evictor (``cache_events.event_time``);
- each eviction is recorded with its simulated time and the workload of the
  evicted block's last request, and window summaries count them;
- the summary times the evict calls and carries TTDF's and RACE-RP's own
  counters.
"""
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import cachesim
from analysis.cachesim import CacheSim, Timing, block_hashes
from analysis.oracle import BeladyEvictor
from client_simulator.token_plan import write_plan
from vllm.core.block import cache_events
from vllm.core.eviction_policies import FIFOEvictor, LFUEvictor
from vllm.core.evictor import LRUEvictor
from vllm.core.reuse_evictors import (NUM_FEATURES, TTDF_DEFAULTS,
                                      RaceRPEvictor, RaceSketch, TTDFEvictor)

from .test_cachesim import BS, req

pytestmark = pytest.mark.m3

ENV = ("VLLM_EVICTION_POLICY", "VLLM_EVICTION_PARAMS", "VLLM_CACHE_EVENT_LOG")


@pytest.fixture(autouse=True)
def clean_env_and_context(monkeypatch):
    """No policy settings from the caller's shell, and the module's request
    context as it was before the test (the simulator names every call)."""
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    ctx = cache_events._context
    saved = ctx.enabled, ctx.request_id, ctx.op, ctx.time
    yield
    ctx.enabled, ctx.request_id, ctx.op, ctx.time = saved


@pytest.fixture
def sketch_file(tmp_path):
    sketch = RaceSketch.random(4,
                               3,
                               seed=5,
                               mean=np.zeros(NUM_FEATURES),
                               std=np.ones(NUM_FEATURES),
                               r_bar=0.5)
    rng = np.random.default_rng(2)
    sketch.add(rng.standard_normal((16, NUM_FEATURES)), rng.random(16))
    path = tmp_path / "sketch.json"
    sketch.save(path)
    return path, sketch


def replay(requests, times, blocks, policy="lru"):
    """Run `requests` with `times` ({id: Timing or (start, finish)})."""
    sim = CacheSim(blocks, BS, policy)
    timing = {
        r: t if isinstance(t, Timing) else Timing(t[0], t[1], la=t[1])
        for r, t in times.items()
    }
    return sim, sim.run(requests, timing)


def ttdf_settings(evictor):
    return {k: getattr(evictor, k) for k in TTDF_DEFAULTS}


# --- Settings after a colon --------------------------------------------------


@pytest.mark.parametrize("policy,expected", [
    ("ttdf", {}),
    ("ttdf:tau_short=2", {
        "tau_short": 2.0
    }),
    ("TTDF:tau_short=2,tau_long=60,w_short=0.5,w_long=0.25", {
        "tau_short": 2.0,
        "tau_long": 60.0,
        "w_short": 0.5,
        "w_long": 0.25
    }),
])
def test_ttdf_settings_after_a_colon_reach_the_evictor(policy, expected):
    evictor = CacheSim(8, BS, policy).gpu.evictor
    assert type(evictor) is TTDFEvictor
    assert ttdf_settings(evictor) == {**TTDF_DEFAULTS, **expected}
    assert "VLLM_EVICTION_PARAMS" not in os.environ  # only while building


def test_race_rp_settings_after_a_colon_reach_the_evictor(sketch_file):
    path, saved = sketch_file
    evictor = CacheSim(8, BS, f"race_rp:sketch={path},alpha=2").gpu.evictor
    assert type(evictor) is RaceRPEvictor
    assert evictor.sketch.alpha == 2.0
    assert np.array_equal(evictor.sketch.planes, saved.planes)
    assert np.array_equal(evictor.sketch.counts, saved.counts)


@pytest.mark.parametrize("policy,error", [
    ("race_rp", "race_rp needs VLLM_EVICTION_PARAMS=sketch="),
    ("race_rp:alpha=2", "race_rp needs VLLM_EVICTION_PARAMS=sketch="),
    ("ttdf:tau_shrt=2", "'tau_shrt=2' is not one of"),
    ("ttdf:tau_short=0", "TTDF time constants must be positive"),
])
def test_bad_settings_stop_the_simulator(policy, error):
    with pytest.raises(ValueError, match=error):
        CacheSim(8, BS, policy)


def test_a_policy_without_a_colon_clears_the_settings(monkeypatch):
    # Would fail if TTDF read it.
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", "tau_shrt=9")
    monkeypatch.setenv("VLLM_EVICTION_POLICY", "fifo")
    evictor = CacheSim(8, BS, "ttdf").gpu.evictor
    assert ttdf_settings(evictor) == TTDF_DEFAULTS
    # The caller's variables are back once the allocator is built.
    assert os.environ["VLLM_EVICTION_PARAMS"] == "tau_shrt=9"
    assert os.environ["VLLM_EVICTION_POLICY"] == "fifo"


@pytest.mark.parametrize("policy,cls", [("lru", LRUEvictor),
                                        ("fifo", FIFOEvictor),
                                        ("lfu", LFUEvictor),
                                        ("belady", BeladyEvictor),
                                        ("Belady", BeladyEvictor)])
def test_the_other_policies_are_unchanged(monkeypatch, policy, cls):
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", "tau_shrt=9")
    assert type(CacheSim(8, BS, policy).gpu.evictor) is cls


# --- The simulated clock reaches the evictor ---------------------------------


def test_ttdf_counts_accesses_at_simulated_times():
    """w:a:0 computes [1-4] and [5-8] at 1 s and its decode fills [9-12] at
    1.25 s; v:b:0 reads [1-8] again at 4 s."""
    a = req("w:a:0", range(1, 9), output=[9, 10, 11, 12], arrival=1.0)
    b = req("v:b:0", range(1, 9), arrival=4.0)
    sim, _ = replay([a, b], {
        "w:a:0": Timing(1.0, 2.0, la=2.0, decode=[1.25]),
        "v:b:0": (4.0, 4.5)
    },
                    blocks=16,
                    policy="ttdf")
    ttdf = sim.gpu.evictor
    h = block_hashes(list(range(1, 13)), BS)
    for content_hash in h[:2]:
        block = ttdf.blocks[content_hash]
        assert block.last == 4.0
        assert block.c_short == pytest.approx(math.exp(-3 / 5) + 1)
        assert block.c_long == pytest.approx(math.exp(-3 / 120) + 1)
    decoded = ttdf.blocks[h[2]]
    assert (decoded.last, decoded.c_short, decoded.c_long) == (1.25, 1.0, 1.0)
    # The real clock is back after the run.
    assert cache_events.time is time


def x_y_z():
    """Four blocks. w reads [1-4] at 0, 1 and 2 s, v reads [101-104] at
    100 s, and u's three-block prompt at 101 s needs one eviction."""
    requests = [
        req("w:x1:0", range(1, 5), arrival=0.0),
        req("w:x2:0", range(1, 5), arrival=1.0),
        req("w:x3:0", range(1, 5), arrival=2.0),
        req("v:y:0", range(101, 105), arrival=100.0),
        req("u:z:0", range(201, 213), arrival=101.0),
    ]
    return requests, {
        r.request_id: (r.arrival_s, r.arrival_s + 0.1)
        for r in requests
    }


def test_ttdf_scores_by_simulated_time():
    """On the simulated clock w's counters have decayed for 99 s at the
    eviction (score ~0.0033) and v's for 1 s (~0.117), so w's block goes.
    Had TTDF read the wall clock, on which the replay takes milliseconds,
    w's three accesses (~0.36) would have kept its block and v's (~0.14)
    would have gone."""
    requests, times = x_y_z()
    sim, result = replay(requests, times, blocks=4, policy="ttdf")
    (hx, ), (hy, ) = block_hashes(list(range(1, 5)),
                                  BS), block_hashes(list(range(101, 105)), BS)
    assert result.evictions == [(101.0, "w")]
    assert hx not in sim.gpu._cached_blocks and hy in sim.gpu._cached_blocks


def test_race_rp_records_accesses_at_simulated_times(sketch_file):
    requests, times = x_y_z()
    sim, _ = replay(requests[:4],
                    times,
                    blocks=4,
                    policy=f"race_rp:sketch={sketch_file[0]}")
    history = sim.gpu.evictor.history
    (hx, ), (hy, ) = block_hashes(list(range(1, 5)),
                                  BS), block_hashes(list(range(101, 105)), BS)
    assert (history.times[hx], history.last[hx]) == ([0.0, 1.0, 2.0], 2.0)
    assert (history.times[hy], history.last[hy]) == ([100.0], 100.0)


# --- Evictions by workload ---------------------------------------------------


def five_requests():
    """Four blocks and five two-block prompts, one a second, each freed at
    once. From the third on each needs two evictions, and every policy here
    gives up the oldest pair: w's at 2 s, v's at 3 s, u's at 4 s."""
    return [
        req("w:r1:0", range(1, 9), arrival=0.0),
        req("v:r2:0", range(101, 109), arrival=1.0),
        req("u:r3:0", range(201, 209), arrival=2.0),
        req("v:r4:0", range(301, 309), arrival=3.0),
        req("u:r5:0", range(401, 409), arrival=4.0),
    ]


FIVE_EVICTIONS = [(2.0, "w"), (2.0, "w"), (3.0, "v"), (3.0, "v"), (4.0, "u"),
                  (4.0, "u")]


def replay_five(policy="lru"):
    requests = five_requests()
    sim, result = replay(
        requests,
        {r.request_id: (r.arrival_s, r.arrival_s + 0.1)
         for r in requests},
        blocks=4,
        policy=policy)
    return requests, sim, result


@pytest.mark.parametrize("policy", ["lru", "fifo", "lfu", "ttdf"])
def test_evictions_name_the_workload_of_the_blocks_last_request(policy):
    _, sim, result = replay_five(policy)
    assert result.evictions == FIVE_EVICTIONS
    assert result.summary["evictions"] == 6


def test_race_rp_evictions_are_recorded(sketch_file):
    _, sim, result = replay_five(f"race_rp:sketch={sketch_file[0]}")
    s = result.summary
    assert len(
        result.evictions) == s["evictions"] == s["evictor"]["evictions"] == 6
    assert sorted(
        t for t, _ in result.evictions) == [2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    assert {w for _, w in result.evictions} <= {"w", "v", "u"}


def test_a_hit_hands_the_blocks_to_the_reading_workload():
    """Four blocks. w computes [1-8]; v reads it and adds [51-54]; u's
    four-block prompt then evicts all three, last used by v."""
    a = req("w:a:0", range(1, 9), arrival=0.0)
    b = req("v:b:0", list(range(1, 9)) + [51, 52, 53, 54], arrival=1.0)
    c = req("u:c:0", range(201, 217), arrival=2.0)
    sim, result = replay([a, b, c], {
        "w:a:0": (0.0, 0.1),
        "v:b:0": (1.0, 1.1),
        "u:c:0": (2.0, 2.1)
    },
                         blocks=4)
    assert result.evictions == [(2.0, "v")] * 3
    assert {sim.owner[h]
            for h in block_hashes(b.prompt_token_ids, BS)} == {"v"}
    assert {sim.owner[h]
            for h in block_hashes(c.prompt_token_ids, BS)} == {"u"}


def test_decode_blocks_belong_to_their_request_from_its_free():
    """Three blocks. w:a:0's prompt [1-4] is one block, its decode fills
    [5-8] at 0.5 s and its last output token, appended at its free, starts a
    partial block (no content hash). v's three-block prompt then needs two
    evictions: w's prompt block and the decode block, which got its owner
    when w:a:0 was freed."""
    a = req("w:a:0", range(1, 5), output=[5, 6, 7, 8, 9])
    b = req("v:b:0", range(101, 113), arrival=2.0)
    sim, result = replay([a, b], {
        "w:a:0": Timing(0.0, 1.0, la=1.0, decode=[0.5]),
        "v:b:0": (2.0, 2.1)
    },
                         blocks=3)
    assert result.evictions == [(2.0, "w"), (2.0, "w")]
    w_blocks = block_hashes(list(range(1, 9)), BS)
    assert set(
        sim.owner) == set(w_blocks) | set(block_hashes(b.prompt_token_ids, BS))
    assert [sim.owner[h] for h in w_blocks] == ["w", "w"]


def test_a_run_without_evictions_records_none():
    a = req("w:a:0", range(1, 9))
    _, result = replay([a], {"w:a:0": (0.0, 0.1)}, blocks=8)
    assert result.evictions == []
    assert result.summary.get("evictions", 0) == 0
    assert result.summary.get("evict_seconds", 0) == 0


# --- Window counts -----------------------------------------------------------


def test_window_counts_evictions_by_workload():
    requests, _, result = replay_five()
    rows, ev = result.rows, result.evictions
    window = cachesim.window_summary(rows, requests, 2.0, 4.0, ev)
    # u:r3:0 (2 s) and v:r4:0 (3 s) are planned in [2, 4); w's and v's
    # pairs are evicted at 2 and 3 s, u's at 4 s is not.
    assert (window["requests"], window["prompt_tokens"],
            window["evictions"]) == (2, 16, 4)
    per = window["workloads"]
    assert set(per) == {"u", "v", "w"}
    assert (per["v"]["requests"], per["v"]["evictions"]) == (1, 2)
    assert per["u"]["requests"] == 1 and "evictions" not in per["u"]
    # w has evictions but no request in the window.
    assert per["w"]["evictions"] == 2
    assert per["w"].get("requests", 0) == 0
    assert per["w"].get("prompt_tokens", 0) == 0
    assert per["w"]["usable_reused_fraction"] is None


@pytest.mark.parametrize("w0,stop,total,per", [
    (2.5, 4.0, 2, {
        "v": 2
    }),
    (3.0, 4.0, 2, {
        "v": 2
    }),
    (4.0, 5.0, 2, {
        "u": 2
    }),
    (2.0, 4.5, 6, {
        "w": 2,
        "v": 2,
        "u": 2
    }),
])
def test_window_counts_evictions_in_w0_to_stop(w0, stop, total, per):
    requests, _, result = replay_five()
    window = cachesim.window_summary(result.rows, requests, w0, stop,
                                     result.evictions)
    assert window["evictions"] == total
    assert {
        w: c["evictions"]
        for w, c in window["workloads"].items() if "evictions" in c
    } == per


def test_a_window_without_evictions_counts_none():
    requests, _, result = replay_five()
    window = cachesim.window_summary(result.rows, requests, 0.0, 2.0,
                                     result.evictions)
    assert window["requests"] == 2 and "evictions" not in window
    assert all("evictions" not in c for c in window["workloads"].values())
    # Without evictions passed (the earlier signature) nothing is counted.
    assert "evictions" not in cachesim.window_summary(result.rows, requests,
                                                      2.0, 4.0)


def test_evictions_of_unknown_owner_count_in_the_total_only():
    requests, _, result = replay_five()
    evictions = result.evictions + [(2.5, None), (5.0, None)]
    window = cachesim.window_summary(result.rows, requests, 2.0, 4.0,
                                     evictions)
    assert window["evictions"] == 5  # 4 by workload, 1 unknown; 5.0 is out
    assert set(window["workloads"]) == {"u", "v", "w"}
    assert sum(c.get("evictions", 0)
               for c in window["workloads"].values()) == 4
    alone = cachesim.window_summary(result.rows, requests, 2.5, 2.9,
                                    [(2.6, None)])
    assert alone["evictions"] == 1 and alone["workloads"] == {}
    assert alone.get("requests", 0) == 0


@pytest.fixture
def five_plan(tmp_path):
    plan_dir = tmp_path / "plan"
    write_plan(five_requests(), plan_dir / "plan.jsonl.gz",
               plan_dir / "responses.jsonl")
    return plan_dir


@pytest.mark.parametrize("policy", ["lru", "ttdf"])
def test_simulate_counts_the_windows_evictions(tmp_path, five_plan, policy):
    """The model timing starts each request at its arrival, so the
    evictions are those of `five_requests`."""
    out = tmp_path / "run"
    result = cachesim.simulate(five_plan,
                               policy,
                               4 * BS,
                               BS,
                               out=out,
                               window=(2.0, 4.5))
    assert result.evictions == FIVE_EVICTIONS
    window = result.summary["window"]
    assert (window["requests"], window["evictions"]) == (3, 6)
    assert {w: c["evictions"]
            for w, c in window["workloads"].items()} == {
                "w": 2,
                "v": 2,
                "u": 2
            }
    written = json.loads((out / "sim.json").read_text())
    assert written["policy"] == policy
    assert written["summary"]["window"]["evictions"] == 6
    assert ("evictor" in written["summary"]) == (policy == "ttdf")


# --- Summary: evict time and the evictor's counters --------------------------


def test_evict_seconds_times_every_evict_call(monkeypatch):
    """A clock that ticks once per read: each evict call takes one tick."""
    ticks = iter(range(1000))
    monkeypatch.setattr(
        cachesim, "time",
        SimpleNamespace(perf_counter=lambda: float(next(ticks))))
    _, _, result = replay_five()
    assert result.summary["evict_seconds"] == 6.0


@pytest.mark.parametrize("policy", ["lru", "ttdf"])
def test_evict_seconds_is_a_duration(policy):
    _, _, result = replay_five(policy)
    s = result.summary
    assert 0.0 <= s["evict_seconds"] < 10.0
    if policy == "ttdf":
        # TTDF's own timer runs inside the simulator's.
        assert s["evictor"]["evict_seconds"] <= s["evict_seconds"]


@pytest.mark.parametrize("policy", ["ttdf", "race_rp"])
def test_summary_carries_the_new_evictors_counters(sketch_file, policy):
    name = policy if policy == "ttdf" else f"race_rp:sketch={sketch_file[0]}"
    _, sim, result = replay_five(name)
    s = result.summary
    counters = s["evictor"]
    assert counters == sim.gpu.evictor.summary()
    assert counters["evictions"] == s["evictions"] == 6
    assert counters["scans"] >= 1 and counters["metadata_bytes"] > 0
    assert ("max_history_times" in counters) == (policy == "race_rp")
    json.dumps(s)  # a run folder's sim.json holds it


@pytest.mark.parametrize("policy", ["lru", "fifo", "lfu", "belady"])
def test_summary_has_no_evictor_counters_for_other_policies(policy):
    _, _, result = replay_five(policy)
    assert "evictor" not in result.summary
    assert result.summary["evictions"] == 6


def test_the_command_line_takes_settings_after_a_colon(five_plan, capsys):
    cachesim.main([
        "--plan",
        str(five_plan), "--policy", "ttdf:tau_short=2", "--capacity",
        str(4 * BS), "--block-size",
        str(BS)
    ])
    text = capsys.readouterr().out
    summary = json.loads(text[text.index("{\n"):])
    assert summary["evictions"] == summary["evictor"]["evictions"] == 6
