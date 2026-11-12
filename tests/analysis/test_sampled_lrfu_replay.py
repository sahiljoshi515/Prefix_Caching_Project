"""Sampled LRFU in the cache simulator and the Milestone 3 runner (specs/007).

Toy plans with 4-token blocks (``test_cachesim.req``), so every hit and
victim is worked out by hand. They check that

- ``sampled_lrfu`` and ``sampled_lrfu:<settings>`` replay through
  ``analysis.cachesim`` (``CacheSim``, ``simulate``, the runner's
  ``run_sims``), with the same output metrics as the baselines (the
  summary's keys are LRU's plus the evictor's own ``evictor`` counters),
  window summaries, and the same result on every run;
- with a sample covering the pool, a huge half-life evicts in LFU order and
  a tiny one in LRU order, as the ``lfu`` and ``lru`` policies do;
- the runner's ``policies`` grid has its 15 runs, and a Sampled LRFU run's
  server gets only its counters' file (``policy_env``), which the evictor
  accepts.
"""
import contextlib
import json
import os
import random
from pathlib import Path

import pytest

from analysis import cachesim
from analysis import run_m3_task2 as m3
from analysis.cachesim import ENGINE_ID, CacheSim, Timing, block_hashes
from analysis.run_m3_task2 import MixConfig
from client_simulator.token_plan import write_plan
from vllm.core import sampled_lrfu
from vllm.core.block import cache_events
from vllm.core.eviction_policies import LFUEvictor
from vllm.core.evictor import LRUEvictor
from vllm.core.sampled_lrfu import SampledLRFUEvictor

from .test_cachesim import BS, req
from .test_cachesim_policies import FIVE_EVICTIONS, five_requests
from .test_run_m3_task2 import CLEAN, FakeReplay, make_plan, run_toy_grid

pytestmark = pytest.mark.m3

ENV = ("VLLM_EVICTION_POLICY", "VLLM_EVICTION_PARAMS", "VLLM_CACHE_EVENT_LOG")
SMALL = "sampled_lrfu:half_life=4,sample_size=2"
STATS = "evictor-stats.json"
# The evictor's own counters, in every Sampled LRFU summary.
COUNTERS = {
    "evictions", "scans", "candidates_scored", "max_candidates",
    "evict_seconds", "max_resident_blocks", "evict_us_mean",
    "candidates_per_scan", "metadata_words", "metadata_bytes"
}
# Wall-clock durations, which differ between otherwise equal runs.
TIMERS = {"evict_seconds", "evict_us_mean"}


@pytest.fixture(autouse=True)
def clean_env_and_context(monkeypatch):
    """No policy settings from the caller's shell, the module's request
    context as it was before the test (the simulator names every call),
    and a runner that sees a clean tree, free ports and no caffeinate."""
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    ctx = cache_events._context
    saved = ctx.enabled, ctx.request_id, ctx.op, ctx.time
    monkeypatch.setattr(m3, "git_state", lambda: dict(CLEAN))
    monkeypatch.setattr(m3, "port_in_use", lambda port: False)
    monkeypatch.setattr(m3, "keep_awake", contextlib.nullcontext)
    yield
    ctx.enabled, ctx.request_id, ctx.op, ctx.time = saved


@pytest.fixture
def exit_hooks(monkeypatch):
    """The `stats` setting registers an exit hook; record it instead."""
    registered = []
    monkeypatch.setattr(sampled_lrfu.atexit, "register",
                        lambda fn, *args: registered.append((fn, args)))
    return registered


def replay(requests, times, blocks, policy="lru", env=None):
    """Run `requests` with `times` ({id: (start, finish)} or Timing)."""
    sim = CacheSim(blocks, BS, policy, env)
    timing = {
        r: t if isinstance(t, Timing) else Timing(t[0], t[1], la=t[1])
        for r, t in times.items()
    }
    return sim, sim.run(requests, timing)


def replay_five(policy):
    requests = five_requests()
    return replay(
        requests,
        {r.request_id: (r.arrival_s, r.arrival_s + 0.1)
         for r in requests},
        blocks=4,
        policy=policy)


def without_timers(summary):
    out = {k: v for k, v in summary.items() if k not in TIMERS}
    if "evictor" in out:
        out["evictor"] = {
            k: v
            for k, v in out["evictor"].items() if k not in TIMERS
        }
    return out


# --- Settings after a colon --------------------------------------------------


@pytest.mark.parametrize("policy,half_life,sample_size,seed", [
    ("sampled_lrfu", 512.0, 4, 0),
    (SMALL, 4.0, 2, 0),
    ("SAMPLED_LRFU:seed=3,half_life=1e9", 1e9, 4, 3),
])
def test_settings_after_a_colon_reach_the_evictor(policy, half_life,
                                                  sample_size, seed):
    evictor = CacheSim(8, BS, policy).gpu.evictor
    assert type(evictor) is SampledLRFUEvictor
    assert (evictor.half_life, evictor.sample_size) == (half_life, sample_size)
    assert evictor.rng.getstate() == random.Random(seed).getstate()
    assert "VLLM_EVICTION_PARAMS" not in os.environ  # only while building
    assert "VLLM_EVICTION_POLICY" not in os.environ


@pytest.mark.parametrize("policy,error", [
    ("sampled_lrfu:half_life=0", "half_life must be positive"),
    ("sampled_lrfu:sample_size=0", "sample_size must be at least 1"),
    ("sampled_lrfu:halflife=4", "'halflife=4' is not one of"),
    ("sampled_lrfu:tau_short=2", "'tau_short=2' is not one of"),
])
def test_bad_settings_stop_the_simulator(policy, error):
    with pytest.raises(ValueError, match=error):
        CacheSim(8, BS, policy)


def test_a_policy_without_a_colon_clears_the_settings(monkeypatch):
    # Would fail if Sampled LRFU read it.
    monkeypatch.setenv("VLLM_EVICTION_PARAMS", "half_life=0")
    evictor = CacheSim(8, BS, "sampled_lrfu").gpu.evictor
    assert (evictor.half_life, evictor.sample_size) == (512.0, 4)
    assert os.environ["VLLM_EVICTION_PARAMS"] == "half_life=0"


# --- The simulator drives the logical clock ----------------------------------


def test_each_request_counts_its_blocks_once():
    """w:a:0 computes [1-4] and [5-8] and its decode fills [9-12] (three
    steps); v:b:0 reads [1-8] (two more). Each access is named with its
    request's engine id, as the server names it."""
    a = req("w:a:0", range(1, 9), output=[9, 10, 11, 12], arrival=1.0)
    b = req("v:b:0", range(1, 9), arrival=4.0)
    sim = CacheSim(16, BS, "sampled_lrfu:half_life=1")
    lrfu, gpu = sim.gpu.evictor, sim.gpu
    accesses = []
    record = lrfu.record_access

    def named(block_id, content_hash, depth, prev_hash):
        accesses.append((cache_events.current_request(), content_hash))
        record(block_id, content_hash, depth, prev_hash)

    lrfu.record_access = named
    result = sim.run(
        [a, b], {
            "w:a:0": Timing(1.0, 2.0, la=2.0, decode=[1.25]),
            "v:b:0": Timing(4.0, 4.5, la=4.5),
        })
    h = block_hashes(list(range(1, 13)), BS)
    ids = [gpu._cached_blocks[x] for x in h]
    assert lrfu.now == 5
    assert [lrfu.stamp[i] for i in ids] == [4, 5, 3]
    # Prompt blocks: inserted at 1 and 2, read at 4 and 5 (3 steps later).
    assert [lrfu.score[i] for i in ids] == [2**-3 + 1, 2**-3 + 1, 1.0]
    w, v = ENGINE_ID.format("w:a:0"), ENGINE_ID.format("v:b:0")
    assert accesses == [(w, h[0]), (w, h[1]), (w, h[2]), (v, h[0]), (v, h[1])]
    assert set(lrfu._pool) == set(ids)
    assert result.summary["hits"] == 2


# --- The same output metrics as the baselines --------------------------------


@pytest.mark.parametrize("policy", ["sampled_lrfu", SMALL])
def test_the_summary_has_the_baselines_metrics_plus_its_counters(policy):
    _, base = replay_five("lru")
    sim, result = replay_five(policy)
    s, lru = result.summary, base.summary
    assert set(s) == set(lru) | {"evictor"}
    for name in ("lfu", "fifo"):
        assert set(replay_five(name)[1].summary) == set(lru)
    # TTDF and RACE-RP report their counters the same way.
    assert set(replay_five("ttdf")[1].summary) == set(s)
    assert set(s["workloads"]) == set(lru["workloads"]) == {"u", "v", "w"}
    for w, counts in s["workloads"].items():
        assert set(counts) == set(lru["workloads"][w]), w
    assert [set(row) for row in result.rows] == [set(row) for row in base.rows]
    assert set(s["evictor"]) == COUNTERS
    assert s["evictor"] == sim.gpu.evictor.summary()
    assert s["evictions"] == s["evictor"]["evictions"] == 6
    assert s["evictor"]["scans"] == 6
    assert s["evictor"]["max_candidates"] <= sim.gpu.evictor.sample_size
    assert 0.0 <= s["evict_seconds"] < 10.0
    # Its own timer runs inside the simulator's.
    assert s["evictor"]["evict_seconds"] <= s["evict_seconds"]
    json.dumps(s)  # a run folder's sim.json holds it


def test_with_the_whole_pool_sampled_it_evicts_the_oldest_pairs():
    """Each of the five requests' blocks is used once, so the scores fall
    with age and every eviction (sample 4 of at most 4 candidates) takes
    the oldest block: the same victims as LRU."""
    _, result = replay_five("sampled_lrfu")
    assert result.evictions == FIVE_EVICTIONS
    assert result.summary["evictor"]["candidates_scored"] == 4 + 3 + 4 + 3 + \
        4 + 3


@pytest.fixture
def five_plan(tmp_path):
    plan_dir = tmp_path / "plan"
    write_plan(five_requests(), plan_dir / "plan.jsonl.gz",
               plan_dir / "responses.jsonl")
    return plan_dir


@pytest.mark.parametrize("policy", ["sampled_lrfu", SMALL])
def test_simulate_writes_the_same_run_folder_and_window_as_lru(
        tmp_path, five_plan, policy):
    runs = {}
    for name in ("lru", policy):
        out = tmp_path / name.replace(":", "_").replace(",", "_")
        result = cachesim.simulate(five_plan,
                                   name,
                                   4 * BS,
                                   BS,
                                   out=out,
                                   window=(2.0, 4.5))
        runs[name] = out, result
    (lru_out, lru), (out, result) = runs["lru"], runs[policy]
    assert sorted(p.name
                  for p in out.iterdir()) == sorted(p.name
                                                    for p in lru_out.iterdir())
    window, lru_window = result.summary["window"], lru.summary["window"]
    assert set(window) == set(lru_window)
    assert (window["requests"], window["evictions"]) == (3, 6)
    per = {w: c.get("evictions", 0) for w, c in window["workloads"].items()}
    assert sum(per.values()) == 6
    if policy == "sampled_lrfu":
        # The whole pool is sampled: LRU's victims (see above).
        assert per == {"w": 2, "v": 2, "u": 2}
        assert window == lru_window
    written = json.loads((out / "sim.json").read_text())
    assert set(written) == set(json.loads((lru_out / "sim.json").read_text()))
    assert written["policy"] == policy
    assert written["summary"]["window"]["evictions"] == 6
    assert written["summary"]["evictor"]["evictions"] == 6
    rows = [json.loads(x) for x in (out / "client.jsonl").open()]
    lru_rows = [json.loads(x) for x in (lru_out / "client.jsonl").open()]
    assert [set(r) for r in rows] == [set(r) for r in lru_rows]
    assert len(rows) == 5
    # A window computed afterwards gives the same counts.
    again = cachesim.window_summary(result.rows, five_requests(), 2.0, 4.5,
                                    result.evictions)
    assert again == window


def busy_plan():
    """24 requests over 12 s, drawn from five two-block prefixes with a
    third block of their own and a decode block, so a 6-block cache keeps
    evicting from a pool larger than a sample of 2."""
    rng = random.Random(4)
    requests = []
    for i in range(24):
        start = 100 * rng.randrange(5)
        prompt = list(range(start, start + 8)) + [1000 + i] * 3
        requests.append(
            req(f"{'wvu'[i % 3]}:r{i}:0",
                prompt,
                output=[2000 + i] * 5,
                arrival=0.5 * i))
    return requests


def run_busy(policy, blocks=6):
    requests = busy_plan()
    times = {
        r.request_id: Timing(r.arrival_s,
                             r.arrival_s + 0.4,
                             la=r.arrival_s + 0.4,
                             decode=[r.arrival_s + 0.1, r.arrival_s + 0.2])
        for r in requests
    }
    return replay(requests, times, blocks=blocks, policy=policy)


def test_replays_are_the_same_every_time(tmp_path):
    first_sim, first = run_busy(SMALL)
    second_sim, second = run_busy(SMALL)
    assert first.evictions == second.evictions
    assert first.rows == second.rows
    assert without_timers(first.summary) == without_timers(second.summary)
    assert set(first_sim.gpu._cached_blocks) == set(
        second_sim.gpu._cached_blocks)
    counters = first.summary["evictor"]
    assert counters["evictions"] >= 10
    assert counters["candidates_scored"] == 2 * counters["evictions"]
    # The plan written to disk replays the same way, twice.
    plan_dir = tmp_path / "plan"
    write_plan(busy_plan(), plan_dir / "plan.jsonl.gz",
               plan_dir / "responses.jsonl")
    a, b = (cachesim.simulate(plan_dir, SMALL, 6 * BS, BS) for _ in range(2))
    assert a.evictions == b.evictions and a.rows == b.rows
    assert without_timers(a.summary) == without_timers(b.summary)


def test_the_seed_picks_the_sample():
    """Same plan, five seeds: the sample is the policy's own generator, so
    the seed changes what survives (each seed's result is fixed)."""
    kept = set()
    for seed in range(5):
        sim, _ = run_busy(f"sampled_lrfu:half_life=4,sample_size=2,"
                          f"seed={seed}")
        kept.add(frozenset(sim.gpu._cached_blocks))
    assert len(kept) > 1


# --- The LFU and LRU ends ----------------------------------------------------


def frequency_recency_trace():
    """Four one-block prompts, each read by its own workload: D four times
    (0-3 s), A three times (4-6 s), B twice (7-8 s), C once (9 s). The
    least frequent is the most recent. At 10 s x's three-block prompt
    needs three of the four blocks."""
    requests = []
    t = 0.0
    for name, start, uses in (("d", 31, 4), ("a", 1, 3), ("b", 11, 2),
                              ("c", 21, 1)):
        for k in range(uses):
            requests.append(
                req(f"{name}:{name}{k}:0", range(start, start + 4), arrival=t))
            t += 1.0
    requests.append(req("x:x:0", range(101, 113), arrival=t))
    return requests, {
        r.request_id: (r.arrival_s, r.arrival_s + 0.1)
        for r in requests
    }


@pytest.mark.parametrize(
    "policy,baseline,victims",
    [
        # Half-life 1e9 accesses: the counts decide (4, 3, 2, 1), so C, B, A.
        ("sampled_lrfu:half_life=1e9,sample_size=8", "lfu", ["c", "b", "a"]),
        # Half-life 0.1 accesses: the last access decides, so D, A, B.
        ("sampled_lrfu:half_life=0.1,sample_size=8", "lru", ["d", "a", "b"]),
    ])
def test_the_half_life_moves_it_between_lfu_and_lru(policy, baseline, victims):
    requests, times = frequency_recency_trace()
    sim, result = replay(requests, times, blocks=4, policy=policy)
    assert result.evictions == [(10.0, w) for w in victims]
    assert sim.gpu.evictor.now == 10 + 3
    # The baseline it becomes evicts the same blocks.
    base_sim, base = replay(requests, times, blocks=4, policy=baseline)
    assert type(base_sim.gpu.evictor) is {
        "lfu": LFUEvictor,
        "lru": LRUEvictor
    }[baseline]
    assert base.evictions == result.evictions
    assert set(base_sim.gpu._cached_blocks) == set(sim.gpu._cached_blocks)
    # The one block kept, and x's three.
    (kept, ) = set("dabc") - set(victims)
    survivors = {
        sim.owner[h]
        for h in sim.gpu._cached_blocks if sim.owner.get(h)
    }
    assert survivors == {kept, "x"}


def test_a_tiny_half_life_is_lru_even_where_the_score_underflows():
    """P, Q, R, S are computed at 0-3 s (in that block order) and P and Q
    read again at 4 and 5 s; x's three-block prompt at 6 s then needs
    three of them. LRU gives up R, S, P. With a half-life of 0.001
    accesses every idle score but the newest is below the smallest double
    (2^-1000 per step), so comparing scores would tie S with P and give up
    P, the lower block id; ranking by log2(s) + tau / H keeps LRU's
    order."""
    requests = [
        req(f"{name}:{name}{k}:0", range(start, start + 4), arrival=t)
        for t, (name, start,
                k) in enumerate([("p", 1, 0), ("q", 11,
                                               0), ("r", 21,
                                                    0), ("s", 31,
                                                         0), ("p", 1,
                                                              1), ("q", 11,
                                                                   1)])
    ]
    requests.append(req("x:x:0", range(101, 113), arrival=6.0))
    times = {r.request_id: (r.arrival_s, r.arrival_s + 0.1) for r in requests}
    policy = "sampled_lrfu:half_life=0.001,sample_size=8"
    # The cache just before x: S's block id is above P's, and at x's
    # second eviction (t=7) both scores are 0.0.
    before, _ = replay(requests[:-1], times, blocks=4, policy=policy)
    lrfu, cached = before.gpu.evictor, before.gpu._cached_blocks
    (hp, ), (hs, ) = block_hashes([1, 2, 3, 4],
                                  BS), block_hashes([31, 32, 33, 34], BS)
    assert cached[hp] < cached[hs]
    assert lrfu.current(cached[hp], 7) == lrfu.current(cached[hs], 7) == 0.0
    sim, result = replay(requests, times, blocks=4, policy=policy)
    assert result.evictions == [(6.0, "r"), (6.0, "s"), (6.0, "p")]
    _, lru = replay(requests, times, blocks=4, policy="lru")
    assert lru.evictions == result.evictions


# --- The runner --------------------------------------------------------------


def test_the_policies_grid_has_the_three_new_policies():
    assert m3.NEW_POLICIES == ("ttdf", "race_rp", "sampled_lrfu")
    configs = m3.grid_configs("policies")
    main = [("balanced", "human", seed, policy, 131072) for seed in (2, 3, 4)
            for policy in ("ttdf", "race_rp", "sampled_lrfu")]
    checks = [("balanced", "human", 2, policy, capacity)
              for capacity in (32768, 393216)
              for policy in ("ttdf", "race_rp", "sampled_lrfu")]
    assert [(c.mix, c.pacing, c.seed, c.policy, c.capacity_tokens)
            for c in configs] == main + checks
    assert len(configs) == len({c.name for c in configs}) == 15
    lrfu = [c.name for c in configs if c.policy == "sampled_lrfu"]
    assert lrfu == [
        "balanced-human-s2-sampled_lrfu-cap131072",
        "balanced-human-s3-sampled_lrfu-cap131072",
        "balanced-human-s4-sampled_lrfu-cap131072",
        "balanced-human-s2-sampled_lrfu-cap32768",
        "balanced-human-s2-sampled_lrfu-cap393216",
    ]
    assert [c.capacity_tokens for c in m3.grid_configs("policies", 65536)
            ] == [65536] * 9 + [32768] * 3 + [393216] * 3


def test_policy_env_gives_sampled_lrfu_only_its_counters(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no RACE-RP sketch here: not needed
    run_dir = tmp_path / "run"
    env = m3.policy_env("sampled_lrfu", run_dir)
    assert env == {
        "VLLM_EVICTION_PARAMS": f"stats={(run_dir / STATS).resolve()}"
    }
    assert m3.policy_env("sampled_lrfu", run_dir,
                         tmp_path / "missing.json") == env
    # A relative run folder is made absolute (the server runs elsewhere).
    relative = m3.policy_env("sampled_lrfu", Path("runs/x"))
    assert relative == {
        "VLLM_EVICTION_PARAMS":
        f"stats={tmp_path.resolve() / 'runs/x' / STATS}"
    }


def test_the_runners_settings_are_ones_the_evictor_takes(tmp_path, exit_hooks):
    """A replay with the server settings the runner gives a Sampled LRFU
    run writes the evictor's counters to the run's evictor-stats.json."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    env = m3.policy_env("sampled_lrfu", run_dir)
    sim, result = replay_five_with_env(env)
    assert (sim.gpu.evictor.half_life, sim.gpu.evictor.sample_size) == (512.0,
                                                                        4)
    (fn, args), = exit_hooks
    assert args == (str((run_dir / STATS).resolve()), )
    fn(*args)
    stats = json.loads((run_dir / STATS).read_text())
    assert stats == json.loads(json.dumps(result.summary["evictor"]))
    assert stats["evictions"] == 6


def replay_five_with_env(env):
    requests = five_requests()
    return replay(
        requests,
        {r.request_id: (r.arrival_s, r.arrival_s + 0.1)
         for r in requests},
        blocks=4,
        policy="sampled_lrfu",
        env=env)


def test_a_grid_run_passes_its_settings_to_the_replay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no RACE-RP sketch here: not needed
    make_plan(tmp_path / "data", "toy-human-s1")
    lru, lrfu = (MixConfig("toy", "human", 1, policy, 64, block_size=BS)
                 for policy in ("lru", "sampled_lrfu"))
    fake = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, [lru, lrfu], fake)
    assert [(e["name"], e["status"], e["error"]) for e in entries] == [
        (lru.name, "finished", None),
        (lrfu.name, "finished", None),
    ]
    calls = {c["run_dir"].name: c for c in fake.calls}
    assert calls[lrfu.name]["policy"] == "sampled_lrfu"
    out = (tmp_path / "out").resolve()
    assert calls[lrfu.name]["extra_env"] == {
        "VLLM_EVICTION_PARAMS": f"stats={out / lrfu.name / STATS}"
    }
    assert "extra_env" not in calls[lru.name]
    run = json.loads((tmp_path / "out" / lrfu.name / "run.json").read_text())
    assert (run["config"], run["eviction_policy"]) == (lrfu.name,
                                                       "sampled_lrfu")
    assert not {"race_sketch", "race_sketch_sha256"} & set(run)


def test_the_simulator_sweep_runs_sampled_lrfu(tmp_path, five_plan):
    """`run_sims` passes the policy strings through to the simulator; each
    Sampled LRFU row has the baseline's metrics plus its counters."""
    out = tmp_path / "sims.jsonl"
    policies = ["lru", "sampled_lrfu", SMALL]
    rows = m3.run_sims([five_plan],
                       policies, [4 * BS],
                       out,
                       block_size=BS,
                       jobs=1,
                       window=(2.0, 4.5),
                       log=lambda line: None)
    assert [r["policy"] for r in rows] == policies
    assert all("error" not in r for r in rows), rows
    lru = rows[0]["summary"]
    for row in rows[1:]:
        assert set(row["summary"]) == set(lru) | {"evictor"}
        assert set(row["summary"]["window"]) == set(lru["window"])
        assert row["summary"]["evictions"] == 6
    written = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["policy"] for r in written] == policies
    # Done runs are not run again.
    assert m3.run_sims([five_plan],
                       policies, [4 * BS],
                       out,
                       block_size=BS,
                       jobs=1,
                       window=(2.0, 4.5),
                       log=lambda line: None) == []
