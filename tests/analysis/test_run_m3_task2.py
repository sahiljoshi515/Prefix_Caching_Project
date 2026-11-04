"""The Milestone 3 Task 2 runner without a server (specs/007, FR-009).

Grid names and shards, resuming with a fake replay, the no-eviction
ceiling's capacity on a hand-made plan, the dirty-tree refusal, the
client's --stop-at guard, and the simulator sweep's resuming with a fake
simulate.
"""
import argparse
import contextlib
import gzip
import json
import os
import random
import subprocess
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import pytest

from analysis import run_m3_task2 as m3
from analysis.run_m3_task2 import MixConfig
from client_simulator.schema import TokenRequest
from client_simulator.token_plan import write_plan

pytestmark = pytest.mark.m3

CLEAN = {"commit": "abc123", "dirty": False}
BS = 4  # small blocks, so block counts can be worked out by hand
KEEP_AWAKE = m3.keep_awake  # before the fixture below replaces it


@pytest.fixture(autouse=True)
def quiet_machine(monkeypatch):
    """A clean tree, free ports and no caffeinate, whatever the machine."""
    monkeypatch.setattr(m3, "git_state", lambda: dict(CLEAN))
    monkeypatch.setattr(m3, "port_in_use", lambda port: False)
    monkeypatch.setattr(m3, "keep_awake", contextlib.nullcontext)


def req(rid, prompt, output=()):
    session, turn = rid.rsplit(":", 1)
    return TokenRequest(request_id=rid,
                        session_id=session,
                        turn_index=int(turn),
                        prompt_token_ids=list(prompt),
                        output_token_ids=list(output),
                        arrival_s=0.0)


def make_plan(data, name, requests=None, stop_s=100.0, max_model_len=16):
    path = data / "processed" / "m3" / name
    write_plan(requests or [req("w:a:0", range(1, 9))], path / "plan.jsonl.gz",
               path / "responses.jsonl")
    (path / "manifest.json").write_text(
        json.dumps({
            "mix": name,
            "stop_s": stop_s,
            "max_model_len": max_model_len
        }))
    return path


# Five requests in 4-token blocks. Distinct full blocks, counted in order:
#   A: prompt 1-8, output 9-11: [1-4], [5-8]; 9-11 is partial.        2
#   B: A's prompt + 20-23, output 24-25: shares both, adds [20-23].    3
#   C: 1-4 then 30-33: shares [1-4], adds [30-33].                     4
#   D: 40-43 then 5-8: after another first block, [5-8] is another
#      block (chained hashes), so both are new.                        6
#   E: prompt 1-6, output 7, 8, 50, 51: the output completes A's
#      [5-8], so both blocks are shared; 50-51 is partial.             6
CEILING_PLAN = [
    req("w:a:0", range(1, 9), output=[9, 10, 11]),
    req("w:b:0", list(range(1, 9)) + [20, 21, 22, 23], output=[24, 25]),
    req("v:c:0", [1, 2, 3, 4, 30, 31, 32, 33]),
    req("v:d:0", [40, 41, 42, 43, 5, 6, 7, 8]),
    req("u:e:0", range(1, 7), output=[7, 8, 50, 51]),
]
# ceil(1.25 x 6) = 8 blocks of 4 tokens.
CEILING_TOKENS = 32


def toy_configs():
    return [
        MixConfig("toy", "human", 1, "lru", 64, block_size=BS),
        MixConfig("toy", "human", 1, "fifo", 64, block_size=BS),
        MixConfig("toy", "human", 1, "lru", 64, block_size=BS, ceiling=True),
    ]


def write_events(path, evictions=0, end=True):
    rows = [{"ev": "header", "block_size": BS}]
    rows += [{
        "t": 1.0,
        "ev": "insert",
        "hash": 1,
        "block": 0,
        "req": "cmpl-w:a:0-0",
        "op": "allocate"
    }]
    rows += [{
        "t": 2.0,
        "ev": "evict",
        "hash": 1,
        "block": 0,
        "req": "cmpl-w:b:0-0",
        "op": "allocate"
    }] * evictions
    rows += [{"ev": "end"}] if end else []
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)


class FakeReplay:
    """Stands in for replay_plan: writes a finished run folder whose event
    log has `evictions` evictions, and checks no two runs share a port."""

    def __init__(self, evictions=0, barrier=None):
        self.calls = []
        self.evictions = evictions
        self.barrier = barrier
        self.lock = threading.Lock()
        self.ports_in_use = set()

    def __call__(self, plan_dir, run_dir, **kwargs):
        port = kwargs["port"]
        with self.lock:
            self.calls.append({
                "plan_dir": plan_dir,
                "run_dir": run_dir,
                **kwargs
            })
            assert port not in self.ports_in_use
            self.ports_in_use.add(port)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=10)
            # As replay_plan does, which refuses a folder that exists.
            run_dir.mkdir(parents=True)
            write_events(run_dir / "events.jsonl.gz", self.evictions)
            manifest = {
                "capacity_tokens": kwargs["capacity_tokens"],
                **kwargs["manifest_updates"], "finished_at_utc": "now"
            }
            (run_dir / "run.json").write_text(json.dumps(manifest))
            return manifest
        finally:
            with self.lock:
                self.ports_in_use.discard(port)


def run_toy_grid(tmp_path, configs, replay, **kwargs):
    logs: List[str] = []
    kwargs.setdefault("jobs", 2)
    entries = m3.run_grid(configs,
                          tmp_path / "out",
                          data=tmp_path / "data",
                          grid="toy",
                          replay=replay,
                          log=logs.append,
                          **kwargs)
    return entries, logs


# --- Configurations and grids ----------------------------------------------


def test_run_names_give_plan_policy_and_capacity():
    config = MixConfig("balanced", "human", 2, "lfu", 131072)
    assert config.plan_name == "balanced-human-s2"
    assert config.name == "balanced-human-s2-lfu-cap131072"
    assert MixConfig("balanced", "human", 2, "lru", 65536,
                     replicate=1).name == "balanced-human-s2-lru-cap65536-r1"
    ceiling = MixConfig("balanced", "fast", 2, "lru", 131072, ceiling=True)
    assert ceiling.name == "balanced-fast-s2-ceiling"
    # A ceiling never evicts, so one serves every capacity.
    assert MixConfig("balanced", "fast", 2, "lru", 65536,
                     ceiling=True).name == ceiling.name
    assert ceiling.record() == {
        "config": "balanced-fast-s2-ceiling",
        "mix": "balanced",
        "pacing": "fast",
        "seed": 2,
        "replicate": 0,
        "ceiling": True
    }


def test_invalid_configurations_are_refused():
    with pytest.raises(ValueError, match="LRU"):
        MixConfig("balanced", "human", 2, "lfu", 131072, ceiling=True)
    with pytest.raises(ValueError, match="multiple of block size"):
        MixConfig("balanced", "human", 2, "lru", 1000)
    with pytest.raises(ValueError):
        MixConfig("balanced", "human", 2, "lru", 131072, replicate=-1)


def test_phase1_has_21_runs_in_four_groups():
    configs = m3.grid_configs("phase1")
    assert len(configs) == 21
    assert len({c.name for c in configs}) == 21
    groups = Counter((c.pacing, c.seed, c.replicate) for c in configs)
    assert groups == {
        ("human", 2, 0): 4,  # reported
        ("human", 3, 0): 4,
        ("human", 4, 0): 4,
        ("human", 1, 0): 4,  # train
        ("fast", 2, 0): 4,  # fast pacing
        ("human", 2, 1): 1,  # noise
    }
    # Each plan: LRU, LFU and FIFO at the capacity, and its ceiling.
    for plan in {c.plan_name for c in configs}:
        runs = sorted((c.policy, c.ceiling) for c in configs
                      if c.plan_name == plan and not c.replicate)
        assert runs == [("fifo", False), ("lfu", False), ("lru", False),
                        ("lru", True)]
    assert all(c.mix == "balanced" and c.block_size == 16 for c in configs)
    assert {c.capacity_tokens for c in configs if not c.ceiling} == {131072}
    assert [c for c in configs if c.replicate] == [
        MixConfig("balanced", "human", 2, "lru", 131072, replicate=1)
    ]
    # The reported seeds come first.
    assert [c.seed for c in configs[:12]] == [2] * 4 + [3] * 4 + [4] * 4


def test_grid_capacity_changes_only_the_capacity_runs():
    default = m3.grid_configs("phase1")
    smaller = m3.grid_configs("phase1", capacity_tokens=65536)
    assert {c.capacity_tokens for c in smaller if not c.ceiling} == {65536}
    ceilings = [c.name for c in default if c.ceiling]
    assert [c.name for c in smaller if c.ceiling] == ceilings


def test_smoke_grid_and_unknown_grids():
    assert m3.grid_configs("smoke") == [
        MixConfig("smoke", "human", 101, "lru", 131072)
    ]
    assert m3.grid_configs("smoke")[0].plan_name == "smoke-human-s101"
    with pytest.raises(ValueError, match="phase1"):
        m3.grid_configs("nope")


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_shards_are_disjoint_complete_and_keep_plans_together(count):
    configs = m3.grid_configs("phase1")
    shards = [
        m3.shard_configs(configs, index, count)
        for index in range(1, count + 1)
    ]
    names = [c.name for shard in shards for c in shard]
    assert sorted(names) == sorted(c.name for c in configs)
    assert len(set(names)) == len(names)
    for shard in shards:
        assert shard, "no part is empty"
        assert shard == [c for c in configs if c in shard]  # grid order
        # Every policy of a plan, its ceiling and its replicate together.
        plans = {c.plan_name for c in shard}
        assert shard == [c for c in configs if c.plan_name in plans]
    # The same on every machine.
    assert shards == [
        m3.shard_configs(m3.grid_configs("phase1"), index, count)
        for index in range(1, count + 1)
    ]


def test_two_shards_balance_phase1():
    # Plans of 5 runs (seed 2 and its replicate) and 4 runs, largest first
    # to the emptier part: s2 | s3, s4 | s1 | fast -> 5+4 and 4+4+4.
    first, second = (m3.shard_configs(m3.grid_configs("phase1"), i, 2)
                     for i in (1, 2))
    plans = sorted({c.plan_name for c in first})
    assert plans == ["balanced-human-s1", "balanced-human-s2"]
    assert (len(first), len(second)) == (9, 12)
    with pytest.raises(ValueError, match="at most 5 parts"):
        m3.shard_configs(m3.grid_configs("phase1"), 1, 6)


def test_shard_argument():
    assert m3.parse_shard("2/3") == (2, 3)
    assert m3.parse_shard(" 1 / 1 ") == (1, 1)
    for bad in ("0/2", "3/2", "1/0", "a/b", "1"):
        with pytest.raises(argparse.ArgumentTypeError):
            m3.parse_shard(bad)
    with pytest.raises(ValueError):
        m3.shard_configs(m3.grid_configs("smoke"), 0, 1)


# --- Plans -----------------------------------------------------------------


def test_plans_are_built_once_each(tmp_path):
    data = tmp_path / "data"
    commands = []

    def fake_run(command, check):
        commands.append(command)
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    built = m3.build_plans(m3.grid_configs("phase1"),
                           data,
                           run=fake_run,
                           log=lambda s: None)
    assert [p.name for p in built] == [
        "balanced-human-s2", "balanced-human-s3", "balanced-human-s4",
        "balanced-human-s1", "balanced-fast-s2"
    ]
    assert commands[0][1:] == [
        "-m", "analysis.mix", "--mix", "balanced", "--pacing", "human",
        "--seed", "2", "--data",
        str(data), "--out",
        str(data / "processed" / "m3" / "balanced-human-s2")
    ]
    # Built plans are kept.
    assert m3.build_plans(m3.grid_configs("phase1"),
                          data,
                          run=fake_run,
                          log=lambda s: None) == []
    # The smoke plan is the balanced mix over 180 s.
    m3.build_plans(m3.grid_configs("smoke"),
                   data,
                   run=fake_run,
                   log=lambda s: None)
    assert commands[-1][3:9] == [
        "--mix", "balanced", "--pacing", "human", "--seed", "101"
    ]
    assert commands[-1][-2:] == ["--duration", "180"]


def test_forced_rebuild_removes_the_manifest_first(tmp_path):
    data = tmp_path / "data"
    plan = make_plan(data, "smoke-human-s101")
    manifest_seen = []

    def failing_run(command, check):
        manifest_seen.append((plan / "manifest.json").exists())
        return SimpleNamespace(returncode=2)

    with pytest.raises(m3.RunnerError, match="exited with 2"):
        m3.build_plans(m3.grid_configs("smoke"),
                       data,
                       force=True,
                       run=failing_run,
                       log=lambda s: None)
    # So the half-built plan is not taken for a built one next time.
    assert manifest_seen == [False]
    assert not (plan / "manifest.json").exists()


# --- The no-eviction ceiling -----------------------------------------------


def test_block_hashes_are_vllms():
    from vllm.core.block.prefix_caching_block import PrefixCachingBlock
    tokens = random.Random(0).choices(range(150000), k=16 * 5 + 7)
    expected, prev = [], None
    for i in range(len(tokens) // 16):
        prev = PrefixCachingBlock.hash_block_tokens(
            i == 0, prev, tokens[i * 16:(i + 1) * 16])
        expected.append(prev)
    assert len(expected) == 5  # the 7-token tail is not a full block
    assert m3.block_hashes(tokens, 16) == expected


def test_ceiling_counts_distinct_full_blocks(tmp_path):
    plan = make_plan(tmp_path, "toy-human-s1", CEILING_PLAN)
    assert m3.distinct_full_blocks(plan, BS) == 6
    assert m3.ceiling_capacity(plan, BS) == CEILING_TOKENS
    # 8-token blocks: [1-8] (A, B and E share it), C's and D's one block
    # each; every other token is in a partial block.
    assert m3.distinct_full_blocks(plan, 8) == 3


def test_count_evictions_streams_the_log(tmp_path):
    write_events(tmp_path / "events.jsonl.gz", evictions=2)
    assert m3.count_evictions(tmp_path / "events.jsonl.gz") == (2, True)
    write_events(tmp_path / "cut.jsonl", evictions=0, end=False)
    assert m3.count_evictions(tmp_path / "cut.jsonl") == (0, False)


# --- The engine grid -------------------------------------------------------


def test_grid_skips_finished_runs_and_replays_interrupted_ones(tmp_path):
    make_plan(tmp_path / "data", "toy-human-s1", CEILING_PLAN, stop_s=100.0)
    lru, fifo, ceiling = toy_configs()
    out = tmp_path / "out"
    (out / lru.name).mkdir(parents=True)
    (out / lru.name / "run.json").write_text('{"finished_at_utc": "x"}')
    (out / fifo.name).mkdir(parents=True)  # interrupted: not finished
    (out / fifo.name / "run.json").write_text('{"started_at_utc": "x"}')
    (out / fifo.name / "stale.txt").write_text("from the interrupted run")
    replay = FakeReplay()
    entries, logs = run_toy_grid(tmp_path, [lru, fifo, ceiling],
                                 replay,
                                 stop_slack=600.0)
    statuses = [e["status"] for e in entries]
    assert statuses == ["skipped", "finished", "finished"]
    calls = {c["run_dir"].name: c for c in replay.calls}
    assert sorted(calls) == sorted([fifo.name, ceiling.name])
    assert not (out / fifo.name / "stale.txt").exists()
    # The client stops sending stop_slack after the plan's stop_s.
    assert {c["stop_at"] for c in replay.calls} == {700.0}
    assert calls[fifo.name]["capacity_tokens"] == 64
    assert calls[fifo.name]["policy"] == "fifo"
    assert calls[ceiling.name]["capacity_tokens"] == CEILING_TOKENS
    assert calls[ceiling.name]["policy"] == "lru"
    assert calls[ceiling.name]["max_model_len"] == 16
    # run.json records the configuration and the ceiling's evictions.
    run = json.loads((out / ceiling.name / "run.json").read_text())
    assert {
        k: run[k]
        for k in ("grid", "config", "mix", "pacing", "seed", "replicate",
                  "ceiling", "ceiling_distinct_blocks", "ceiling_evictions")
    } == {
        "grid": "toy",
        "config": ceiling.name,
        "mix": "toy",
        "pacing": "human",
        "seed": 1,
        "replicate": 0,
        "ceiling": True,
        "ceiling_distinct_blocks": 6,
        "ceiling_evictions": 0
    }
    status = json.loads((out / "grid.json").read_text())
    assert status["grid"] == "toy" and status["commit"] == "abc123"
    recorded = [(r["name"], r["status"]) for r in status["runs"]]
    assert recorded == [(e["name"], e["status"]) for e in entries]
    assert status["runs"][1]["wall_s"] is not None
    assert status["runs"][1]["error"] is None
    # One line per start and per finish.
    for name in (fifo.name, ceiling.name):
        assert sum(name in line and "started" in line for line in logs) == 1
        assert sum(name in line and "finished in" in line
                   for line in logs) == 1
    # Run again: nothing is replayed.
    again = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, [lru, fifo, ceiling], again)
    assert [e["status"] for e in entries] == ["skipped"] * 3
    assert again.calls == []


def test_ceiling_is_at_least_max_model_len(tmp_path):
    # vLLM refuses a cache smaller than one longest request.
    make_plan(tmp_path / "data",
              "toy-human-s1",
              CEILING_PLAN,
              max_model_len=48)
    replay = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, toy_configs()[2:], replay)
    assert entries[0]["status"] == "finished"
    assert replay.calls[0]["capacity_tokens"] == 48


def test_a_ceiling_that_evicts_fails_and_stays_failed(tmp_path):
    make_plan(tmp_path / "data", "toy-human-s1", CEILING_PLAN)
    ceiling = toy_configs()[2]
    entries, _ = run_toy_grid(tmp_path, [ceiling], FakeReplay(evictions=2))
    assert entries[0]["status"] == "failed"
    assert "evicted 2 blocks" in entries[0]["error"]
    run_dir = tmp_path / "out" / ceiling.name
    assert json.loads((run_dir / "run.json").read_text())["ceiling_evictions"] \
        == 2
    # Not replayed (the same capacity would evict again), still failed.
    again = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, [ceiling], again)
    assert entries[0]["status"] == "failed" and again.calls == []


def test_ceiling_evictions_are_counted_for_a_run_finished_without_them(
        tmp_path):
    # Interrupted after the run finished but before it was checked.
    make_plan(tmp_path / "data", "toy-human-s1", CEILING_PLAN)
    ceiling = toy_configs()[2]
    run_dir = tmp_path / "out" / ceiling.name
    run_dir.mkdir(parents=True)
    write_events(run_dir / "events.jsonl.gz")
    (run_dir / "run.json").write_text('{"finished_at_utc": "x"}')
    replay = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, [ceiling], replay)
    assert entries[0]["status"] == "skipped" and replay.calls == []
    run = json.loads((run_dir / "run.json").read_text())
    assert run["ceiling_evictions"] == 0 and run["ceiling_log_complete"]


def test_grid_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(m3, "git_state", lambda: {
        "commit": "abc123",
        "dirty": True
    })
    make_plan(tmp_path / "data", "toy-human-s1")
    replay = FakeReplay()
    with pytest.raises(m3.RunnerError, match="uncommitted"):
        run_toy_grid(tmp_path, toy_configs()[:1], replay)
    assert replay.calls == []
    entries, _ = run_toy_grid(tmp_path,
                              toy_configs()[:1],
                              replay,
                              allow_dirty=True)
    assert entries[0]["status"] == "finished"
    assert json.loads((tmp_path / "out" / "grid.json").read_text())["dirty"]


def test_concurrent_runs_each_get_their_own_port(tmp_path):
    make_plan(tmp_path / "data", "toy-human-s1")
    # Both runs must be in replay at once to pass the barrier.
    replay = FakeReplay(barrier=threading.Barrier(2))
    entries, _ = run_toy_grid(tmp_path,
                              toy_configs()[:2],
                              replay,
                              jobs=2,
                              base_port=9100)
    assert [e["status"] for e in entries] == ["finished", "finished"]
    assert sorted(c["port"] for c in replay.calls) == [9100, 9101]


def test_grid_refuses_busy_ports(tmp_path, monkeypatch):
    monkeypatch.setattr(m3, "port_in_use", lambda port: port == 9101)
    make_plan(tmp_path / "data", "toy-human-s1")
    replay = FakeReplay()
    with pytest.raises(m3.RunnerError, match=r"\[9101\]"):
        run_toy_grid(tmp_path, toy_configs()[:2], replay, base_port=9100)
    assert replay.calls == []


def test_a_run_whose_port_is_taken_fails_alone(tmp_path, monkeypatch):
    make_plan(tmp_path / "data", "toy-human-s1")
    taken = iter([False, True])  # free at the start, taken at the run
    monkeypatch.setattr(m3, "port_in_use", lambda port: next(taken))
    replay = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, toy_configs()[:1], replay, jobs=1)
    assert entries[0]["status"] == "failed"
    assert "in use" in entries[0]["error"] and replay.calls == []


def test_one_grid_per_folder(tmp_path):
    make_plan(tmp_path / "data", "toy-human-s1")
    with m3.grid_lock(tmp_path / "out"), pytest.raises(m3.RunnerError,
                                                       match="Another grid"):
        run_toy_grid(tmp_path, toy_configs()[:1], FakeReplay())


def test_grid_needs_its_plans_and_a_feasible_capacity(tmp_path):
    with pytest.raises(m3.RunnerError, match="plans --grid"):
        run_toy_grid(tmp_path, toy_configs(), FakeReplay())
    make_plan(tmp_path / "data", "toy-human-s1", max_model_len=16)
    too_small = MixConfig("toy", "human", 1, "lru", 8, block_size=BS)
    with pytest.raises(m3.RunnerError, match="max_model_len"):
        run_toy_grid(tmp_path, [too_small], FakeReplay())


# --- One replay ------------------------------------------------------------


class FakeServer:
    """Stands in for SimulatorServer; closes an empty event log on exit,
    as the engine does when it stops."""

    def __init__(self,
                 model,
                 trace,
                 log_path,
                 event_log=None,
                 args=(),
                 env=None,
                 port=8000):
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.event_log = event_log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.event_log is not None:
            write_events(self.event_log)

    def kv_cache_blocks(self):
        return {"num_gpu_blocks": 4, "block_size": 16}


def fake_client(commands):
    """Stands in for the plan_replay subprocess: records its command and
    writes the manifest the client writes when it finishes."""

    def run(command, **kwargs):
        commands.append(command)
        output = command[command.index("--output") + 1]
        Path(output + ".run.json").write_text(
            json.dumps({
                "status": "finished",
                "summary": {
                    "completed": 1
                }
            }))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return run


def test_replay_passes_stop_at_to_the_client(tmp_path, monkeypatch):
    commands: List[List[str]] = []
    monkeypatch.setattr(m3, "SimulatorServer", FakeServer)
    monkeypatch.setattr(m3.subprocess, "run", fake_client(commands))
    plan = make_plan(tmp_path / "data", "toy-human-s1")
    m3.replay_plan(plan,
                   tmp_path / "guarded",
                   policy="lru",
                   capacity_tokens=64,
                   port=8123,
                   stop_at=2400.0,
                   manifest_updates={
                       "mix": "toy",
                       "seed": 1
                   })
    command = commands[-1]
    assert command[command.index("--stop-at") + 1] == "2400.0"
    assert command[command.index("--base-url") + 1] == \
        "http://127.0.0.1:8123/v1"
    run = json.loads((tmp_path / "guarded" / "run.json").read_text())
    assert run["stop_at"] == 2400.0
    assert (run["mix"], run["seed"], run["commit"]) == ("toy", 1, "abc123")
    assert run["finished_at_utc"] and run["client_summary"] == {"completed": 1}
    m3.replay_plan(plan, tmp_path / "open", policy="lru", capacity_tokens=64)
    assert "--stop-at" not in commands[-1]


def test_grid_drives_the_real_replay_plan(tmp_path, monkeypatch):
    # FakeReplay takes any arguments; this checks replay_plan takes the
    # grid's, and what the server and client are then given.
    commands: List[List[str]] = []
    monkeypatch.setattr(m3, "SimulatorServer", FakeServer)
    monkeypatch.setattr(m3.subprocess, "run", fake_client(commands))
    make_plan(tmp_path / "data", "toy-human-s1", CEILING_PLAN, stop_s=50.0)
    lru, _, ceiling = toy_configs()
    entries, _ = run_toy_grid(tmp_path, [lru, ceiling],
                              m3.replay_plan,
                              jobs=1,
                              stop_slack=10.0)
    assert [e["status"] for e in entries] == ["finished", "finished"]
    assert all(c[c.index("--stop-at") + 1] == "60.0" for c in commands)
    run_dir = tmp_path / "out" / ceiling.name
    run = json.loads((run_dir / "run.json").read_text())
    assert (run["capacity_tokens"], run["num_gpu_blocks"]) == (32, 8)
    assert run["server_args"][-6:] == [
        "--max-model-len", "16", "--block-size", "4",
        "--num-gpu-blocks-override", "8"
    ]
    assert (run["ceiling"], run["ceiling_evictions"], run["stop_at"]) == \
        (True, 0, 60.0)
    assert run["plan"]["stop_s"] == 50.0


# --- The cache-simulator sweep ---------------------------------------------


def fake_simulate(calls):

    def simulate(plan_dir, policy, capacity_tokens, block_size, timing):
        calls.append(
            (plan_dir.name, policy, capacity_tokens, block_size, timing))
        if policy == "belady":
            raise ValueError(f"Unknown policy {policy!r}")
        return SimpleNamespace(summary={"usable_reused_fraction": 0.5})

    return simulate


def test_sim_resumes_and_retries_failed_runs(tmp_path):
    plan = make_plan(tmp_path / "data", "toy-human-s1", CEILING_PLAN)
    out = tmp_path / "sim" / "rows.jsonl"
    calls: List[Tuple] = []
    rows = m3.run_sims([plan], ["lru", "belady"], [64, "ceiling"],
                       out,
                       block_size=BS,
                       jobs=1,
                       simulate=fake_simulate(calls),
                       log=lambda s: None)
    assert len(rows) == 4
    assert sorted(c[2] for c in calls) == [CEILING_TOKENS] * 2 + [64] * 2
    failed = [r for r in rows if "error" in r]
    assert {r["policy"] for r in failed} == {"belady"} and len(failed) == 2
    assert "Unknown policy" in failed[0]["error"]
    assert {(r["capacity"], r["capacity_tokens"])
            for r in rows if "error" not in r} == {(64, 64),
                                                   ("ceiling", CEILING_TOKENS)}
    assert all(r["commit"] == "abc123" and r["timing"] == "model"
               and r["block_size"] == BS and "seconds" in r for r in rows)
    # With FIFO added: FIFO runs, and the failed runs are tried again.
    calls.clear()
    m3.run_sims([plan], ["lru", "belady", "fifo"], [64, "ceiling"],
                out,
                block_size=BS,
                jobs=1,
                simulate=fake_simulate(calls),
                log=lambda s: None)
    assert sorted((c[1], c[2]) for c in calls) == [("belady", CEILING_TOKENS),
                                                   ("belady", 64),
                                                   ("fifo", CEILING_TOKENS),
                                                   ("fifo", 64)]
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 8
    # Nothing left to do.
    calls.clear()
    assert m3.run_sims([plan], ["lru", "fifo"], [64, "ceiling"],
                       out,
                       block_size=BS,
                       jobs=1,
                       simulate=fake_simulate(calls),
                       log=lambda s: None) == []
    assert calls == []


def test_sim_starts_a_new_line_after_a_cut_one(tmp_path):
    plan = make_plan(tmp_path / "data", "toy-human-s1")
    out = tmp_path / "rows.jsonl"
    out.write_text('{"plan": "x", "policy": "lru", "capa')  # interrupted
    m3.run_sims([plan], ["lru"], [64],
                out,
                block_size=BS,
                jobs=1,
                simulate=fake_simulate([]),
                log=lambda s: None)
    lines = out.read_text().splitlines()
    assert lines[0] == '{"plan": "x", "policy": "lru", "capa'
    assert json.loads(lines[1])["policy"] == "lru"


def test_sim_runs_a_repeated_value_once(tmp_path):
    plan = make_plan(tmp_path / "data", "toy-human-s1")
    calls: List[Tuple] = []
    rows = m3.run_sims([plan, plan], ["lru", "lru"], [64, 64],
                       tmp_path / "rows.jsonl",
                       block_size=BS,
                       jobs=1,
                       simulate=fake_simulate(calls),
                       log=lambda s: None)
    assert len(rows) == len(calls) == 1


def test_sim_refuses_a_capacity_that_is_not_whole_blocks(tmp_path):
    with pytest.raises(ValueError, match="multiple of block size"):
        m3.run_sims([tmp_path], ["lru"], [30],
                    tmp_path / "rows.jsonl",
                    block_size=BS,
                    jobs=1,
                    simulate=fake_simulate([]),
                    log=lambda s: None)


# --- Keeping the Mac awake -------------------------------------------------


def test_keep_awake_runs_caffeinate_on_macos_only(monkeypatch):
    events = []

    class FakePopen:

        def __init__(self, args):
            events.append(args)

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self):
            return 0

    monkeypatch.setattr(m3.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(m3.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(m3, "sys", SimpleNamespace(platform="darwin"))
    with KEEP_AWAKE():
        assert events == [["caffeinate", "-i", "-w", str(os.getpid())]]
    assert events[-1] == "terminate"
    events.clear()
    monkeypatch.setattr(m3, "sys", SimpleNamespace(platform="linux"))
    with KEEP_AWAKE():
        pass
    assert events == []
