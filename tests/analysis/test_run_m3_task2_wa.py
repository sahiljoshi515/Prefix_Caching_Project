"""The runner's wa runs (analysis/run_m3_task2.py, spec 007 Phase 2): the
``wa`` grid, wa's server settings, and finding a finished wa run again.

Grids run with the fake replay of tests/analysis/test_run_m3_task2.py,
which records the arguments each run's replay got and writes its run.json.
"""
import contextlib
import json

import pytest

from analysis import run_m3_task2 as m3
from analysis.run_m3_task2 import MixConfig
from tests.analysis.test_run_m3_task2 import (BS, CLEAN, FakeReplay, make_plan,
                                              run_toy_grid)
from vllm.core.reuse_evictors import parse_params
from vllm.core.wa_evictor import DEFAULTS as WA_DEFAULTS

pytestmark = pytest.mark.m3

STATS = "evictor-stats.json"


@pytest.fixture(autouse=True)
def quiet_machine(monkeypatch):
    """A clean tree, free ports and no caffeinate, whatever the machine."""
    monkeypatch.setattr(m3, "git_state", lambda: dict(CLEAN))
    monkeypatch.setattr(m3, "port_in_use", lambda port: False)
    monkeypatch.setattr(m3, "keep_awake", contextlib.nullcontext)


# --- The grid ----------------------------------------------------------------


def test_the_wa_grid_runs_wa_beside_lru_and_race_rp():
    assert "wa" in m3.GRIDS
    configs = m3.grid_configs("wa")
    policies = ("lru", "race_rp", "wa_oracle", "wa_none")
    assert [(c.seed, c.policy) for c in configs] == [(seed, policy)
                                                     for seed in (2, 3, 4)
                                                     for policy in policies]
    assert {(c.mix, c.pacing, c.capacity_tokens, c.ceiling, c.replicate)
            for c in configs} == {("balanced", "human", 131072, False, 0)}
    assert configs[2].name == "balanced-human-s2-wa_oracle-cap131072"
    assert [c.capacity_tokens
            for c in m3.grid_configs("wa", 65536)] == [65536] * 12


def test_its_lru_and_race_rp_runs_are_the_other_grids_runs():
    """In a folder where phase1 and policies ran, they are skipped."""
    others = {
        c.name
        for grid in ("phase1", "policies") for c in m3.grid_configs(grid)
    }
    for config in m3.grid_configs("wa"):
        assert (config.name in others) == (config.policy in ("lru", "race_rp"))


# --- Settings ----------------------------------------------------------------


def test_the_settings_are_those_seed_1_picked():
    """plan.md, Phase 2 experiments: buckets on, half-life 600 s with
    oracle labels and 1,800 s with none."""
    assert m3.WA_RUNS == {
        "wa_oracle": "labels=oracle,buckets=1,half_life=600",
        "wa_none": "labels=none,buckets=1,half_life=1800",
    }
    for text in m3.WA_RUNS.values():
        parse_params(text, WA_DEFAULTS)  # a misspelling would stop the server


@pytest.mark.parametrize("policy", ["wa_oracle", "wa_none"])
def test_wa_runs_get_their_label_modes_settings(tmp_path, policy):
    run_dir = tmp_path / "run"
    # wa needs no sketch, so a missing one does not matter.
    env = m3.policy_env(policy, run_dir, tmp_path / "no-sketch.json")
    assert env == {
        "VLLM_EVICTION_PARAMS":
        f"{m3.WA_RUNS[policy]},stats={(run_dir / STATS).resolve()}"
    }
    assert m3.engine_policy(policy) == "wa"


@pytest.mark.parametrize("policy",
                         ["lru", "lfu", "fifo", "ttdf", "race_rp", "wa"])
def test_other_policies_run_under_their_own_name(policy):
    assert m3.engine_policy(policy) == policy


# --- The grid's replays ------------------------------------------------------


def test_a_wa_run_records_its_grid_policy_and_is_found_again(tmp_path):
    make_plan(tmp_path / "data", "toy-human-s1")
    lru, oracle, none = (MixConfig("toy",
                                   "human",
                                   1,
                                   policy,
                                   64,
                                   block_size=BS)
                         for policy in ("lru", "wa_oracle", "wa_none"))
    replay = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, [lru, oracle, none], replay)
    assert [e["status"] for e in entries] == ["finished"] * 3
    calls = {c["run_dir"].name: c for c in replay.calls}
    out = (tmp_path / "out").resolve()
    # Both label modes run wa on the server, each with its settings.
    for config in (oracle, none):
        assert calls[config.name]["policy"] == "wa"
        assert calls[config.name]["extra_env"] == {
            "VLLM_EVICTION_PARAMS": (f"{m3.WA_RUNS[config.policy]},"
                                     f"stats={out / config.name / STATS}")
        }
    assert calls[lru.name]["policy"] == "lru"
    assert "extra_env" not in calls[lru.name]
    # run.json names the grid policy only where the server's differs.
    runs = {
        name: json.loads((tmp_path / "out" / name / "run.json").read_text())
        for name in calls
    }
    assert runs[oracle.name]["run_policy"] == "wa_oracle"
    assert runs[none.name]["run_policy"] == "wa_none"
    assert "run_policy" not in runs[lru.name]
    assert m3.run_config(tmp_path / "out" / oracle.name) == oracle
    # The next grid skips all three, rather than taking the wa runs for
    # runs of another configuration.
    again = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, [lru, oracle, none], again)
    assert [e["status"] for e in entries] == ["skipped"] * 3
    assert again.calls == []


# --- One replay --------------------------------------------------------------


def test_replay_passes_the_policys_settings(tmp_path, monkeypatch):
    seen = {}

    def fake_replay(plan_dir, run_dir, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return {"client_summary": {}}

    monkeypatch.setattr(m3, "replay_plan", fake_replay)
    base = [
        "replay", "--plan",
        str(tmp_path / "plan"), "--out",
        str(tmp_path / "run"), "--capacity", "131072"
    ]
    m3.main(base + ["--policy", "wa", "--params", "labels=none,buckets=1"])
    assert seen["policy"] == "wa"
    assert seen["extra_env"] == {
        "VLLM_EVICTION_PARAMS": "labels=none,buckets=1"
    }
    m3.main(base + ["--policy", "lru"])
    assert seen["policy"] == "lru"
    assert seen["extra_env"] is None
