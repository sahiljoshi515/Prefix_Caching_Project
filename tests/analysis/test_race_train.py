"""RACE-RP's training (analysis/race_train.py, spec 007, Milestone 3).

The toy trace below is built by hand, so every observation, feature and
target is listed and derived in the comments. Plans replayed through the
cache simulator use 4-token blocks (as tests/analysis/test_cachesim.py
does), except where `train` needs vLLM's 16. Random traces check the
offline features against what the online RACE-RP evictor computes, and the
observations against a direct reading of the specification.
"""
import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from analysis import cachesim
from analysis import race_train as rt
from analysis import run_m2_task2, run_m3_task2
from analysis.race_train import Trace, TraceEvictor
from client_simulator.plan_replay import group_sessions
from client_simulator.token_plan import write_plan
from tests.analysis.test_cachesim import req
from vllm.core.block import cache_events
from vllm.core.evictor import LRUEvictor
from vllm.core.reuse_evictors import STD_FLOOR, RaceRPEvictor, RaceSketch

pytestmark = pytest.mark.m3

BS = 4
AGES = (0.0, 5.0, 30.0, 120.0)
H = 180.0
TAU = 60.0
CLEAN = {"commit": "abc123", "dirty": False}
# The online evictor only needs a sketch to exist; features ignore it.
DUMMY_SKETCH = RaceSketch.random(1, 1, 0, np.zeros(5), np.ones(5))


@pytest.fixture(autouse=True)
def restore_request_context(monkeypatch):
    """The simulator and the evictors set the block manager's request
    context (its time among it); put it back after each test."""
    context = cache_events._context
    for name in ("enabled", "request_id", "op", "time"):
        monkeypatch.setattr(context, name, getattr(context, name))


def x(f10, f120, idle_s, depth, family):
    """A feature vector from its raw values."""
    return tuple(math.log1p(v) for v in (f10, f120, idle_s, depth, family))


def r(delta):
    """The target of a reuse `delta` seconds later."""
    return math.exp(-delta / TAU) if delta <= H else 0.0


def idle_times(trace, h):
    """A block's idle start times (without their access counts)."""
    return [t for t, _ in trace.idle_starts[h]]


def rows(obs):
    """`observations` output as sorted (age, r, x) rows: the specification
    fixes the set of observations, not their order."""
    x_, r_, age = obs
    return sorted(
        zip(age.tolist(), r_.tolist(), (tuple(v) for v in x_.tolist())))


# --- The toy trace, by hand -------------------------------------------------

# Blocks: hash -> (depth, parent, accesses, idle starts).
TOY = {
    1: (1, None, [0, 100], [70]),
    2: (2, 1, [0, 110, 112, 300], [20, 115]),
    3: (3, 2, [100, 300], [120]),
    4: (4, 3, [100], [130]),
    5: (1, None, [780, 960], [790]),
    6: (1, None, [1000], [1000]),
}
# The trace ends at 1000 s (block 6), so observations stop at 1000 - 180 =
# 820 s. Each row: age, delay to the next access (inf: none), features
# (f10, f120, idle, depth, family).
NEVER = math.inf
TOY_OBSERVATIONS = [
    # Block 2, depth 2: its own family. Idle at 20, next access 110: ages 0,
    # 5, 30 (t = 20, 25, 50); age 120 (t = 140) is cut by the re-access.
    (0.0, 90, x(0, 1, 20, 2, 1)),
    (5.0, 85, x(0, 1, 25, 2, 1)),
    (30.0, 60, x(0, 1, 50, 2, 1)),
    # Idle at 115 after accesses at 110 and 112 (in use in between, so no
    # observation there); next access 300.
    # t = 115: reused 185 s later, beyond H: r = 0.
    (0.0, 185, x(2, 3, 3, 2, 3)),
    # t = 120: reused exactly H later, r = exp(-3). Windows are (t - w, t]:
    # 110 is not in (110, 120] and 0 is not in (0, 120].
    (5.0, 180, x(1, 2, 8, 2, 2)),
    (30.0, 155, x(0, 2, 33, 2, 2)),
    # t = 235: the access at 300 is in the future and not counted.
    (120.0, 65, x(0, 0, 123, 2, 0)),
    # Block 1, depth 1. Idle at 70, next access 100: t = 70, 75; age 30
    # (t = 100) is the access itself, and t must come strictly before.
    (0.0, 30, x(0, 1, 70, 1, 1)),
    (5.0, 25, x(0, 1, 75, 1, 1)),
    # Block 3, depth 3: its family is block 2 (accesses 0, 110, 112, 300).
    # Idle at 120, next access 300: t = 120 is reused exactly H later.
    (0.0, 180, x(0, 1, 20, 3, 2)),
    (5.0, 175, x(0, 1, 25, 3, 2)),
    (30.0, 150, x(0, 1, 50, 3, 2)),
    (120.0, 60, x(0, 0, 140, 3, 0)),
    # Block 4, depth 4: never used again (r = 0); its family is block 2,
    # two levels up.
    (0.0, NEVER, x(0, 1, 30, 4, 2)),
    (5.0, NEVER, x(0, 1, 35, 4, 2)),
    (30.0, NEVER, x(0, 1, 60, 4, 2)),
    (120.0, NEVER, x(0, 0, 150, 4, 0)),
    # Block 5, depth 1. Idle at 790, next access 960. 780 is not in
    # (780, 790]. t = 820 is the last time kept (t <= end - H); age 120
    # (t = 910) falls in the trace's final H seconds.
    (0.0, 170, x(0, 1, 10, 1, 1)),
    (5.0, 165, x(0, 1, 15, 1, 1)),
    (30.0, 140, x(0, 1, 40, 1, 1)),
    # Block 6 becomes idle at 1000 s, inside the final H: nothing.
]


def toy_rows(horizon=H, tau=TAU, ages=AGES):
    """TOY_OBSERVATIONS as sorted (age, r, x) rows."""
    return sorted((age, math.exp(-delta / tau) if delta <= horizon else 0.0, f)
                  for age, delta, f in TOY_OBSERVATIONS if age in ages)


def toy_trace():
    """`TOY` as the replay records it: every call in time order."""
    calls = []
    for h, (depth, parent, accesses, idles) in TOY.items():
        calls += [(t, 0, h, depth, parent) for t in accesses]
        calls += [(t, 1, h, None, None) for t in idles]
    trace = Trace()
    for t, kind, h, depth, parent in sorted(calls, key=lambda c: c[:2]):
        if kind == 0:
            trace.access(h, depth, parent, float(t))
        else:
            trace.idle(h, float(t))
    return trace


def test_toy_trace_observations_are_the_hand_derived_ones():
    trace = toy_trace()
    assert trace.end == 1000.0
    obs = rt.observations(trace)
    assert obs[0].shape == (20, 5)
    assert rows(obs) == toy_rows()
    # The two targets at exactly H.
    assert Counter(obs[1].tolist())[math.exp(-3)] == 2
    ages = Counter(obs[2].tolist())
    assert ages == {0.0: 6, 5.0: 6, 30.0: 5, 120.0: 3}


def test_family_is_the_depth_two_ancestor():
    trace = toy_trace()
    assert [trace.family(h) for h in (1, 2, 3, 4, 5)] == [1, 2, 2, 2, 5]
    orphan = Trace()
    orphan.access(9, 3, None, 0.0)
    with pytest.raises(AssertionError, match="without a parent"):
        orphan.family(9)


def test_trace_records_times_in_order_and_its_end():
    trace = Trace()
    assert trace.end == 0.0 and not trace.accesses and not trace.idle_starts
    trace.access(7, 1, None, 2.0)
    trace.idle(7, 3.5)
    trace.access(8, 2, 7, 3.0)  # an earlier time does not move the end
    trace.access(7, 1, None, 9.0)
    assert dict(trace.accesses) == {7: [2.0, 9.0], 8: [3.0]}
    # The idle start keeps how many accesses came before it.
    assert dict(trace.idle_starts) == {7: [(3.5, 1)]}
    assert trace.depth == {7: 1, 8: 2}
    assert trace.parent == {7: None, 8: 7}
    assert trace.end == 9.0


def test_an_access_at_an_idle_starts_instant_is_ordered_as_recorded():
    """Block 1 is freed at 10 s and hit again at 10 s, after the free: it
    is not idle at 10 s, and its next access is that hit. Block 2's request
    accesses it at 10 s and then frees it: idle from 10 s, never used
    again."""
    trace = Trace()
    trace.access(1, 1, None, 0.0)
    trace.access(2, 1, None, 0.0)
    trace.idle(1, 10.0)
    trace.access(1, 1, None, 10.0)
    trace.access(2, 1, None, 10.0)
    trace.idle(2, 10.0)
    trace.idle(1, 12.0)
    trace.access(3, 1, None, 1000.0)  # ends the trace
    assert trace.idle_starts[1] == [(10.0, 1), (12.0, 2)]
    assert trace.idle_starts[2] == [(10.0, 2)]
    # Block 1: nothing from 10 s (t = 10 is not before the next access at
    # 10); from 12 s all four ages, r = 0. Block 2: all four ages from 10 s.
    # Both have accesses at 0 and 10.
    assert rows(rt.observations(trace)) == sorted([
        (0.0, 0.0, x(1, 2, 2, 1, 2)),
        (5.0, 0.0, x(1, 2, 7, 1, 2)),
        (30.0, 0.0, x(0, 2, 32, 1, 2)),
        (120.0, 0.0, x(0, 0, 122, 1, 0)),
        (0.0, 0.0, x(1, 2, 0, 1, 2)),
        (5.0, 0.0, x(1, 2, 5, 1, 2)),
        (30.0, 0.0, x(0, 2, 30, 1, 2)),
        (120.0, 0.0, x(0, 0, 120, 1, 0)),
    ])


def test_features_use_only_the_past():
    trace = toy_trace()
    # Block 2 at t = 115 (accesses 0, 110, 112, 300): the access at 300
    # changes nothing, and a window's left edge is open.
    assert rt.observation_features(trace, 2, 2, 115.0) == x(2, 3, 3, 2, 3)
    assert rt.observation_features(trace, 2, 2, 120.0) == x(1, 2, 8, 2, 2)
    # At an access's own time it is counted, and the block is not idle.
    assert rt.observation_features(trace, 2, 2, 110.0) == x(1, 2, 0, 2, 2)
    # Block 3 with its family (block 2) at t = 130.
    assert rt.observation_features(trace, 3, 2, 130.0) == x(0, 1, 30, 3, 2)


def test_target_is_discounted_reuse_within_the_horizon():
    assert rt.HORIZON_S == H and rt.TAU_S == TAU and rt.AGES == AGES
    assert rt.target(0.0) == 1.0
    assert rt.target(30.0) == math.exp(-0.5)
    assert rt.target(H) == math.exp(-3)
    assert rt.target(H + 1e-9) == 0.0
    assert rt.target(math.inf) == 0.0  # never used again
    assert rt.target(60.0, horizon=50.0) == 0.0
    assert rt.target(30.0, tau=30.0) == math.exp(-1)


def test_observation_ages_are_taken_in_order_whatever_is_given():
    trace = toy_trace()
    obs = rt.observations(trace, ages=(5.0, 0.0))
    assert rows(obs) == toy_rows(ages=(0.0, 5.0))


def test_horizon_and_tau_are_parameters():
    # H = 100 s keeps observations up to 900 s, which adds none here (the
    # next candidates are block 5 at 910 s and block 6 at 1000 s), and only
    # reuses within 100 s count.
    obs = rt.observations(toy_trace(), horizon=100.0, tau=30.0)
    assert rows(obs) == toy_rows(horizon=100.0, tau=30.0)
    assert sum(obs[1] > 0) == 7  # block 2: 4, block 1: 2, block 3: 1


def test_an_empty_trace_has_no_observations():
    x_, r_, age = rt.observations(Trace())
    assert x_.shape == (0, 5) and r_.shape == (0, ) and age.shape == (0, )


# --- The online evictor and a direct reading of the spec --------------------


def random_trace(seed, requests=40):
    """Requests over prefix trees of up to four blocks, as the replay
    records them. Starts are multiples of 10 s and finishes end in 5, so no
    idle start shares its time with an access, while observation times (an
    idle start plus 5 s) and window edges land on access times."""
    rng = random.Random(seed)
    ids = {}

    def block(path):
        return ids.setdefault(path, len(ids) + 1)

    plans = []
    for _ in range(requests):
        depth = rng.randint(1, 4)
        path = (rng.randrange(3), ) + tuple(
            rng.randrange(2) for _ in range(depth - 1))
        start = 10 * rng.randrange(60)
        plans.append((path, start, start + rng.choice((5, 15, 25))))
    timeline = sorted([(s, 1, i) for i, (_, s, _) in enumerate(plans)] +
                      [(f, 0, i) for i, (_, _, f) in enumerate(plans)])
    trace = Trace()
    refs: Counter = Counter()
    for t, allocate, i in timeline:
        path = plans[i][0]
        hashes = [block(path[:k]) for k in range(1, len(path) + 1)]
        for k, h in enumerate(hashes):
            if allocate:  # one access per block, in prefix order
                trace.access(h, k + 1, hashes[k - 1] if k else None, float(t))
                refs[h] += 1
            else:
                refs[h] -= 1
                if refs[h] == 0:
                    trace.idle(h, float(t))
    return trace


def online_features(trace, queries):
    """x_b(t) for each (t, hash) in `queries` from a RACE-RP evictor that has
    seen the trace's accesses up to t, and nothing later."""
    now = [0.0]
    evictor = RaceRPEvictor(DUMMY_SKETCH, clock=lambda: now[0])
    events = sorted((t, trace.depth[h], h)
                    for h, times in trace.accesses.items() for t in times)
    found = {}
    i = 0
    for t, h in sorted(set(queries)):
        while i < len(events) and events[i][0] <= t:
            now[0], depth, g = events[i]
            evictor.record_access(g, g, depth, trace.parent[g])
            i += 1
        block = evictor.blocks[h]
        assert block.family == trace.family(h)
        found[(t, h)] = evictor.features(h, block, t)
    return [found[q] for q in queries]


def spec_observations(trace):
    """The specification read directly: for each idle start s and age a,
    t = s + a is kept if t < the first access after s and t <= end - H. (No
    access in the traces it reads shares an idle start's time, so "after"
    is "later", and the accesses before the free are those up to s.)"""
    every = [t for times in trace.accesses.values() for t in times]
    every += [t for starts in trace.idle_starts.values() for t, _ in starts]
    last_t = max(every) - H
    kept = []
    for h, starts in trace.idle_starts.items():
        for s, before in starts:
            assert before == sum(a <= s for a in trace.accesses[h])
            following = min((a for a in trace.accesses[h] if a > s),
                            default=math.inf)
            for age in AGES:
                t = s + age
                if t < following and t <= last_t:
                    kept.append((t, h, age, r(following - t)))
    features = online_features(trace, [(t, h) for t, h, _, _ in kept])
    return sorted(
        (age, target, f) for (_, _, age, target), f in zip(kept, features))


@pytest.mark.parametrize("seed", range(5))
def test_offline_features_equal_the_online_evictors(seed):
    trace = random_trace(seed)
    end = int(trace.end)
    queries = [(float(t), h) for h, times in trace.accesses.items()
               for t in range(int(times[0]), end + 1, 5)]
    online = online_features(trace, queries)
    offline = [
        rt.observation_features(trace, h, trace.family(h), t)
        for t, h in queries
    ]
    assert offline == online


@pytest.mark.parametrize("seed", range(5))
def test_observations_follow_the_spec_on_random_traces(seed):
    trace = random_trace(seed)
    expected = spec_observations(trace)
    assert len(expected) > 50  # enough to mean something
    assert any(target == 0.0 for _, target, _ in expected)
    assert any(0.0 < target < 1.0 for _, target, _ in expected)
    assert rows(rt.observations(trace)) == expected


# --- Recording the trace through the cache simulator -----------------------


def toy_plan(bs):
    """Four requests in blocks of `bs` tokens:

    - w:a:0 at 0 s: prompt P1 P2, output that fills D3 plus one token;
    - v:b:0 at 30 s: P1 P2, then its own B3;
    - w:a:1, 100 s after w:a:0 finishes: P1 P2 D3, then A4;
    - u:c:0 at 400 s: C1 C2, shared with nothing; it ends the trace.
    """
    prompt = list(range(1, 2 * bs + 1))
    output = list(range(2 * bs + 1, 3 * bs + 2))
    return [
        req("w:a:0", prompt, output=output),
        req("v:b:0", prompt + list(range(1000, 1000 + bs)), arrival=30.0),
        req("w:a:1",
            prompt + output + list(range(2000, 2000 + bs - 1)),
            after=100.0),
        req("u:c:0", range(5000, 5000 + 2 * bs), arrival=400.0),
    ]


def write_toy_plan(path, requests):
    write_plan(requests, path / "plan.jsonl.gz", path / "responses.jsonl")
    return path


def toy_hashes(plan, bs):
    a0, b0, a1, c0 = plan
    p1, p2, d3 = cachesim.block_hashes(
        a0.prompt_token_ids + a0.output_token_ids, bs)
    b3 = cachesim.block_hashes(b0.prompt_token_ids, bs)[2]
    a4 = cachesim.block_hashes(a1.prompt_token_ids, bs)[3]
    c1, c2 = cachesim.block_hashes(c0.prompt_token_ids, bs)
    return p1, p2, d3, b3, a4, c1, c2


def test_record_trace_follows_the_model_timing(tmp_path):
    plan = toy_plan(BS)
    path = write_toy_plan(tmp_path, plan)
    timing = cachesim.model_timing(group_sessions(plan))
    a0, b0, a1, c0 = (timing[q.request_id] for q in plan)
    # The model: a latency of 5 ms + 0.2 ms per generated token (+1).
    assert (a0.start, b0.start, c0.start) == (0.0, 30.0, 400.0)
    assert a0.finish == pytest.approx(0.005 + 0.0002 * 6)
    assert a1.start == a0.finish + 100.0
    assert b0.finish == pytest.approx(30.0052)
    # w:a:0's one decode block is filled halfway through its decode.
    d3_time = a0.start + (a0.finish - a0.start) / 2
    trace = rt.record_trace(path, "model", block_size=BS)
    p1, p2, d3, b3, a4, c1, c2 = toy_hashes(plan, BS)
    # Accesses: the prompt's blocks at each start (inserted, then hit),
    # D3 when it fills and when w:a:1 hits it.
    expected_access = {
        p1: [0.0, 30.0, a1.start],
        p2: [0.0, 30.0, a1.start],
        d3: [d3_time, a1.start],
        b3: [30.0],
        a4: [a1.start],
        c1: [400.0],
        c2: [400.0],
    }
    assert set(trace.accesses) == set(expected_access)
    for h, times in expected_access.items():
        assert trace.accesses[h] == pytest.approx(times)
    # Every block becomes idle when the request holding it finishes (the
    # partial block holding w:a:0's last token is never cached).
    expected_idle = {
        p1: [a0.finish, b0.finish, a1.finish],
        p2: [a0.finish, b0.finish, a1.finish],
        d3: [a0.finish, a1.finish],
        b3: [b0.finish],
        a4: [a1.finish],
        c1: [c0.finish],
        c2: [c0.finish],
    }
    assert set(trace.idle_starts) == set(expected_idle)
    for h, times in expected_idle.items():
        assert idle_times(trace, h) == pytest.approx(times)
        # Each request accesses a block once before freeing it.
        assert [i for _, i in trace.idle_starts[h]
                ] == list(range(1,
                                len(times) + 1))
    assert trace.depth == {p1: 1, p2: 2, d3: 3, b3: 3, a4: 4, c1: 1, c2: 2}
    assert trace.parent == {
        p1: None,
        p2: p1,
        d3: p2,
        b3: p2,
        a4: d3,
        c1: None,
        c2: c1
    }
    assert trace.end == c0.finish
    # Observations stop at c0.finish - 180 = 220.0052 s:
    #   P1 and P2: idle at 0.0062 (next 30: ages 0, 5), at 30.0052 (next
    #   100.0062: ages 0, 5, 30), at 100.0114 (never again: 0, 5, 30;
    #   220.0114 is too late) -> 8 each;
    #   D3: at 0.0062 (next 100.0062: 0, 5, 30), at 100.0114 (0, 5, 30) -> 6;
    #   B3: at 30.0052, never again: all four ages -> 4; A4: 0, 5, 30 -> 3.
    obs = rt.observations(trace)
    assert len(obs[0]) == 29
    assert Counter(obs[2].tolist()) == {0.0: 10, 5.0: 10, 30.0: 8, 120.0: 1}
    assert rows(obs) == spec_observations(trace)
    # D3 idle at a0.finish, reused by w:a:1 100 s later.
    assert any(v == pytest.approx(r(100.0)) for v in obs[1])


def test_a_deduplicated_decode_block_is_an_access(tmp_path):
    """w:d:0's output completes [5-8], already cached by w:a:0: the
    allocator deduplicates it, and that is an access."""
    plan = [
        req("w:a:0", range(1, 9)),
        req("v:d:0", range(1, 7), output=[7, 8, 80, 81], arrival=1.0),
    ]
    path = write_toy_plan(tmp_path, plan)
    timing = cachesim.model_timing(group_sessions(plan))
    a0, d0 = timing["w:a:0"], timing["v:d:0"]
    trace = rt.record_trace(path, "model", block_size=BS)
    p1, p2 = cachesim.block_hashes(list(range(1, 9)), BS)
    fill = d0.start + (d0.finish - d0.start) / 2
    assert trace.accesses[p1] == pytest.approx([0.0, 1.0])
    assert trace.accesses[p2] == pytest.approx([0.0, fill])
    assert idle_times(trace, p2) == pytest.approx([a0.finish, d0.finish])
    assert [i for _, i in trace.idle_starts[p2]] == [1, 2]
    assert trace.depth[p2] == 2 and trace.parent[p2] == p1


def write_engine_run(run_dir, starts_frees):
    """An engine run folder: each request's allocate and free events and
    its client row."""
    events = [{"ev": "header", "t": 0.0}]
    client = []
    for rid, (start, free) in starts_frees.items():
        engine_id = f"cmpl-{rid}-0"
        events.append({
            "t": start,
            "ev": "miss",
            "hash": 1,
            "block": None,
            "req": engine_id,
            "op": "allocate"
        })
        events.append({
            "t": free,
            "ev": "free",
            "hash": 1,
            "block": 0,
            "req": engine_id,
            "op": "free",
            "la": free
        })
        client.append({
            "request_id": rid,
            "sent_s": start,
            "completed_s": free
        })
    events.append({"ev": "end"})
    events.sort(key=lambda e: e.get("t", math.inf))
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
    (run_dir / "client.jsonl").write_text("\n".join(map(json.dumps, client)))


def test_record_trace_takes_an_engine_runs_timing(tmp_path):
    """Times come from the engine run's log. v:b:0 runs inside w:a:0's
    lifetime, so the shared blocks become idle only when w:a:0 finishes.
    u:c:0's four blocks give the ceiling room for v:b:0's whole prompt (the
    simulator, like the block manager, admits a prompt only if all its
    blocks would fit, hits included)."""
    plan = [
        req("w:a:0", range(1, 9)),
        req("v:b:0", list(range(1, 9)) + [50, 51, 52, 53], arrival=0.5),
        req("u:c:0", range(100, 116), arrival=5.0),
    ]
    path = write_toy_plan(tmp_path / "plan", plan)
    write_engine_run(tmp_path / "run", {
        "w:a:0": (10.0, 14.0),
        "v:b:0": (11.0, 12.0),
        "u:c:0": (20.0, 21.0),
    })
    trace = rt.record_trace(path, f"engine:{tmp_path / 'run'}", block_size=BS)
    p1, p2, b3 = cachesim.block_hashes(plan[1].prompt_token_ids, BS)
    c_blocks = cachesim.block_hashes(plan[2].prompt_token_ids, BS)
    assert dict(trace.accesses) == {
        p1: [10.0, 11.0],
        p2: [10.0, 11.0],
        b3: [11.0],
        **{h: [20.0]
           for h in c_blocks}
    }
    assert dict(trace.idle_starts) == {
        p1: [(14.0, 2)],
        p2: [(14.0, 2)],
        b3: [(12.0, 1)],
        **{h: [(21.0, 1)]
           for h in c_blocks}
    }
    assert trace.end == 21.0


def test_record_trace_refuses_an_unknown_timing(tmp_path):
    path = write_toy_plan(tmp_path, toy_plan(BS))
    for timing in ("bogus", "Model", "engine", ""):
        with pytest.raises(ValueError, match="Unknown timing"):
            rt.record_trace(path, timing, block_size=BS)


def test_record_trace_has_room_for_every_block(tmp_path, monkeypatch):
    plan = toy_plan(BS)
    path = write_toy_plan(tmp_path, plan)
    sizes = []

    class SpySim(cachesim.CacheSim):

        def __init__(self, num_blocks, block_size, policy, **kwargs):
            sizes.append((num_blocks, block_size, policy))
            super().__init__(num_blocks, block_size, policy, **kwargs)

    monkeypatch.setattr(cachesim, "CacheSim", SpySim)
    rt.record_trace(path, "model", block_size=BS)
    # Seven distinct full blocks, with the ceiling's 1.25 margin: 9.
    assert run_m3_task2.ceiling_capacity(path, BS) == 9 * BS
    assert sizes == [(9, BS, "lru")]


def test_a_cache_that_must_evict_stops_the_trace(tmp_path, monkeypatch):
    """At four blocks w:a:1's fourth block needs an eviction, and the
    training trace refuses to make one: it would be the policy's, not the
    requests'."""
    path = write_toy_plan(tmp_path, toy_plan(BS))
    monkeypatch.setattr(run_m3_task2, "ceiling_capacity",
                        lambda plan, block_size: 4 * block_size)
    with pytest.raises(RuntimeError, match="must hold every block"):
        rt.record_trace(path, "model", block_size=BS)


def test_a_replay_that_defers_a_prompt_stops_the_trace(tmp_path):
    """Three distinct blocks give a ceiling of four. v:b:0 arrives while
    w:a:0 holds two, and the replay admits a prompt only if all its blocks
    fit, hits included: it would wait until 14 s, so its times would be the
    cache's, not the run's."""
    plan = [
        req("w:a:0", range(1, 9)),
        req("v:b:0", list(range(1, 9)) + [50, 51, 52, 53], arrival=0.5),
    ]
    path = write_toy_plan(tmp_path / "plan", plan)
    write_engine_run(tmp_path / "run", {
        "w:a:0": (10.0, 14.0),
        "v:b:0": (11.0, 12.0)
    })
    with pytest.raises(RuntimeError,
                       match=re.escape("too small: {'deferred': 1}")):
        rt.record_trace(path, f"engine:{tmp_path / 'run'}", block_size=BS)


def test_a_replay_out_of_decode_blocks_stops_the_trace(tmp_path, monkeypatch):
    """Two blocks hold the prompt; the first decode block finds neither a
    free block nor an idle one, so the request would stop growing."""
    plan = [req("w:a:0", range(1, 9), output=range(9, 17))]
    path = write_toy_plan(tmp_path, plan)
    monkeypatch.setattr(run_m3_task2, "ceiling_capacity",
                        lambda plan, block_size: 2 * block_size)
    with pytest.raises(
            RuntimeError,
            match=re.escape("too small: {'decode_out_of_blocks': 1}")):
        rt.record_trace(path, "model", block_size=BS)


def test_a_block_reused_the_instant_it_became_idle_is_not_observed_then(
        tmp_path):
    plan = [
        req("w:a:0", range(1, 9)),
        # Open loop, recorded at 0 s: it starts when w:a:0 finishes.
        req("w:a:1", range(1, 13), arrival=0.0),
        req("u:c:0", range(70, 78), arrival=400.0),
    ]
    path = write_toy_plan(tmp_path, plan)
    timing = cachesim.model_timing(group_sessions(plan))
    a0, a1 = timing["w:a:0"], timing["w:a:1"]
    assert a1.start == a0.finish
    trace = rt.record_trace(path, "model", block_size=BS)
    p1, p2, p3 = cachesim.block_hashes(list(range(1, 13)), BS)
    # P1 became idle at a0.finish and w:a:1 hit it then, after the free:
    # one access came before that idle start.
    assert trace.accesses[p1] == [0.0, a1.start]
    assert trace.idle_starts[p1] == [(a0.finish, 1), (a1.finish, 2)]
    # So P1 and P2 are first observed when w:a:1 frees them, like P3 (its
    # own third block); none is used again: 3 blocks x 4 ages, all r = 0.
    x_, r_, age = rt.observations(trace)
    assert Counter(age.tolist()) == {0.0: 3, 5.0: 3, 30.0: 3, 120.0: 3}
    assert not r_.any()


# --- TraceEvictor ------------------------------------------------------------


def test_trace_evictor_records_and_never_evicts():
    trace = Trace()
    now = [5.0]
    evictor = TraceEvictor(trace, clock=lambda: now[0])
    assert isinstance(evictor, LRUEvictor)
    evictor.record_access(7, 101, 1, None)
    now[0] = 6.0
    evictor.record_access(8, 102, 2, 101)
    now[0] = 9.0
    evictor.add(7, 101, 4, 8.5)
    # Still an LRU evictor holding the idle block.
    assert evictor.num_blocks == 1 and 7 in evictor
    assert dict(trace.accesses) == {101: [5.0], 102: [6.0]}
    assert dict(trace.idle_starts) == {101: [(9.0, 1)]}
    assert trace.depth == {101: 1, 102: 2}
    assert trace.parent == {101: None, 102: 101}
    assert trace.end == 9.0
    with pytest.raises(RuntimeError, match="must hold every block"):
        evictor.evict()
    assert evictor.num_blocks == 1


def test_trace_evictor_uses_the_block_managers_event_time():
    assert TraceEvictor(Trace()).clock is cache_events.event_time


# --- Sketches ----------------------------------------------------------------


def expected_tables(planes, x_, r_):
    """N and W of each table, computed directly: standardize with the
    observations' mean and population std (0 -> 1), scale to unit length,
    and read P hyperplane signs as the bits of a bucket."""
    L, P, _ = planes.shape
    std = x_.std(axis=0)
    z = (x_ - x_.mean(axis=0)) / np.where(std == 0, 1.0, std)
    counts = np.zeros((L, 2**P))
    rewards = np.zeros((L, 2**P))
    for row, target in zip(z, r_):
        norm = np.linalg.norm(row)
        unit = row / norm if norm > 0 else row
        for table in range(L):
            bucket = sum(2**j for j in range(P)
                         if float(planes[table, j] @ unit) >= 0)
            counts[table, bucket] += 1
            rewards[table, bucket] += target
    return counts, rewards


def test_build_sketch_fills_both_tables_from_the_observations():
    rng = np.random.default_rng(3)
    x_ = rng.normal(size=(200, 5))
    x_[:, 0] = 0.0  # never accessed in the last 10 s: std 0 -> 1
    r_ = rng.uniform(size=200)
    r_[:50] = 0.0
    sketch = rt.build_sketch(x_, r_, 3, 4, seed=11, alpha=2.5, meta={"k": 1})
    assert (sketch.L, sketch.P) == (3, 4)
    np.testing.assert_array_equal(
        sketch.planes,
        np.random.default_rng(11).standard_normal((3, 4, 5)))
    np.testing.assert_array_equal(sketch.mean, x_.mean(axis=0))
    std = x_.std(axis=0)
    assert std[0] == 0.0 and sketch.std[0] == 1.0
    np.testing.assert_array_equal(sketch.std[1:], std[1:])
    assert sketch.r_bar == pytest.approx(r_.mean())
    assert sketch.alpha == 2.5 and sketch.meta == {"k": 1}
    # Every observation lands once in every table.
    np.testing.assert_array_equal(sketch.counts.sum(axis=1), [200] * 3)
    np.testing.assert_allclose(sketch.rewards.sum(axis=1), [r_.sum()] * 3)
    counts, rewards = expected_tables(sketch.planes, x_, r_)
    np.testing.assert_array_equal(sketch.counts, counts)
    np.testing.assert_allclose(sketch.rewards, rewards)


def test_a_constant_feature_is_not_scaled_up():
    rng = np.random.default_rng(3)
    x_ = rng.normal(size=(200, 5))
    x_[:, 3] = math.log1p(2)  # every observation at depth 2
    assert 0 < x_.std(axis=0)[3] <= STD_FLOOR  # rounding in the mean
    sketch = rt.build_sketch(x_, rng.uniform(size=200), 16, 6)
    assert sketch.std[3] == 1.0
    # So a block at another depth is hashed by all its features, not by
    # its depth alone: depth-3 queries do not all share one bucket.
    queries = rng.normal(size=(1000, 5))
    queries[:, 3] = math.log1p(3)
    assert len({tuple(b) for b in sketch.buckets(queries)}) > 100


def test_a_std_at_or_below_the_floor_counts_as_zero():
    std = [0.0, STD_FLOOR, 2 * STD_FLOOR, 0.5, 3.0]
    sketch = RaceSketch(np.zeros((1, 1, 5)), np.zeros(5), std)
    assert sketch.std.tolist() == [1.0, 1.0, 2 * STD_FLOOR, 0.5, 3.0]


def test_build_sketch_defaults_and_seeds():
    x_ = np.random.default_rng(0).normal(size=(10, 5))
    r_ = np.linspace(0, 1, 10)
    a = rt.build_sketch(x_, r_, 2, 3)
    assert a.alpha == 5.0 and a.meta == {}
    np.testing.assert_array_equal(
        a.planes,
        np.random.default_rng(rt.SEED).standard_normal((2, 3, 5)))
    b = rt.build_sketch(x_, r_, 2, 3, seed=1)
    assert not np.array_equal(a.planes, b.planes)


def test_build_sketch_needs_observations():
    with pytest.raises(ValueError, match="No training observations"):
        rt.build_sketch(np.zeros((0, 5)), np.zeros(0), 2, 3)


@pytest.mark.parametrize("text, size", [("16x6", (16, 6)), ("16X6", (16, 6)),
                                        ("1x1", (1, 1)), (" 4 x 8 ", (4, 8))])
def test_parse_size(text, size):
    assert rt.parse_size(text) == size


@pytest.mark.parametrize("text, message", [
    ("16", "is not LxP"),
    ("16x", "is not LxP"),
    ("axb", "is not LxP"),
    ("1x2x3", "is not LxP"),
    ("16x6.5", "is not LxP"),
    ("0x6", "must be positive"),
    ("16x0", "must be positive"),
    ("-1x6", "must be positive"),
])
def test_parse_size_refuses(text, message):
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        rt.parse_size(text)


def test_sketch_path_names_its_size(tmp_path):
    assert rt.sketch_path(tmp_path, 16, 6) == tmp_path / "sketch-L16-P6.json"


# --- train and the command line ------------------------------------------


@pytest.fixture
def clean_git(monkeypatch):
    monkeypatch.setattr(run_m2_task2, "git_state", lambda: dict(CLEAN))


def test_train_writes_one_sketch_per_size_and_the_observations(
        tmp_path, clean_git):
    # train replays with vLLM's 16-token blocks.
    plan_dir = write_toy_plan(tmp_path / "plan", toy_plan(16))
    out = tmp_path / "race" / "nested"
    paths = rt.train(plan_dir, [(2, 3), (1, 2)],
                     out,
                     timing="model",
                     seed=7,
                     alpha=2.0)
    assert paths == [out / "sketch-L2-P3.json", out / "sketch-L1-P2.json"]
    trace = rt.record_trace(plan_dir, "model")
    x_, r_, age = rt.observations(trace)
    # The same shape of trace as with 4-token blocks: 29 observations.
    assert len(x_) == 29
    saved = np.load(out / "observations.npz")
    np.testing.assert_array_equal(saved["x"], x_)
    np.testing.assert_array_equal(saved["r"], r_)
    np.testing.assert_array_equal(saved["age"], age)
    for path, (L, P) in zip(paths, [(2, 3), (1, 2)]):
        sketch = RaceSketch.load(path)
        assert (sketch.L, sketch.P) == (L, P)
        np.testing.assert_array_equal(
            sketch.planes,
            np.random.default_rng(7).standard_normal((L, P, 5)))
        np.testing.assert_allclose(sketch.mean, x_.mean(axis=0))
        std = x_.std(axis=0)
        np.testing.assert_allclose(sketch.std, np.where(std == 0, 1.0, std))
        assert sketch.r_bar == pytest.approx(r_.mean())
        assert sketch.alpha == 2.0
        counts, rewards = expected_tables(sketch.planes, x_, r_)
        np.testing.assert_array_equal(sketch.counts, counts)
        np.testing.assert_allclose(sketch.rewards, rewards)
        assert sketch.meta == {
            "plan": str(plan_dir),
            "timing": "model",
            "observations": 29,
            "observations_by_age": {
                "0.0": 10,
                "5.0": 10,
                "30.0": 8,
                "120.0": 1
            },
            "reused_within_horizon": pytest.approx(float((r_ > 0).mean())),
            "ages_s": [0.0, 5.0, 30.0, 120.0],
            "horizon_s": 180.0,
            "tau_s": 60.0,
            "seed": 7,
            "trace_end_s": trace.end,
            **CLEAN,
        }
    # Some reused within H and some not.
    assert 0 < sketch.meta["reused_within_horizon"] < 1


def test_train_refuses_a_trace_with_no_observations(tmp_path, clean_git):
    # Without u:c:0 the trace ends near 100 s: all of it is in the final H.
    plan_dir = write_toy_plan(tmp_path / "plan", toy_plan(16)[:3])
    with pytest.raises(ValueError, match="No training observations"):
        rt.train(plan_dir, [(2, 3)], tmp_path / "out", timing="model")
    # Nothing written: no folder, no observations.npz.
    assert not (tmp_path / "out").exists()


def test_main_trains_and_prints_each_sketch(tmp_path, clean_git, capsys):
    plan_dir = write_toy_plan(tmp_path / "plan", toy_plan(16))
    out = tmp_path / "out"
    rt.main([
        "--plan",
        str(plan_dir), "--timing", "model", "--sketch", "2x3", "1X2", "--seed",
        "7", "--alpha", "2", "--out",
        str(out)
    ])
    assert capsys.readouterr().out.splitlines() == [
        f"{out / 'sketch-L2-P3.json'}: 29 observations",
        f"{out / 'sketch-L1-P2.json'}: 29 observations",
    ]
    sketch = RaceSketch.load(out / "sketch-L2-P3.json")
    assert sketch.alpha == 2.0 and sketch.meta["seed"] == 7


def test_main_defaults_make_the_sketch_the_runner_loads(
        tmp_path, monkeypatch, capsys):
    calls = []
    sketch_file = tmp_path / "s.json"
    sketch_file.write_text(json.dumps({"meta": {"observations": 5}}))

    def fake_train(*args):
        calls.append(args)
        return [sketch_file]

    monkeypatch.setattr(rt, "train", fake_train)
    rt.main([])
    assert calls == [
        (Path("data/processed/m3/balanced-human-s1"), [(16, 6)],
         Path("results/m3-race"),
         "engine:results/m3-task2/balanced-human-s1-ceiling", 0, 5.0)
    ]
    assert capsys.readouterr().out == f"{sketch_file}: 5 observations\n"
    # The default size and folder give the file the grid's race_rp runs load.
    assert rt.sketch_path(Path("results/m3-race"), 16,
                          6) == run_m3_task2.RACE_SKETCH


def test_main_refuses_a_bad_size(capsys):
    with pytest.raises(SystemExit) as exit_info:
        rt.main(["--sketch", "16by6"])
    assert exit_info.value.code == 2
    assert "is not LxP" in capsys.readouterr().err
