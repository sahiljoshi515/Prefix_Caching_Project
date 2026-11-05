"""The runner's Milestone 3 policy runs (analysis/run_m3_task2.py): the
``policies`` grid and the server settings of TTDF and RACE-RP runs.

Grids run with the fake replay of tests/analysis/test_run_m3_task2.py,
which records the arguments each run's replay got and writes its run.json.
"""
import contextlib
import hashlib
import json
from pathlib import Path

import pytest

from analysis import run_m3_task2 as m3
from analysis.run_m3_task2 import MixConfig
from tests.analysis.test_run_m3_task2 import (BS, CLEAN, FakeReplay, make_plan,
                                              run_toy_grid)

pytestmark = pytest.mark.m3

STATS = "evictor-stats.json"
# RACE-RP's server runs its NumPy scoring on one math thread.
ONE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


@pytest.fixture(autouse=True)
def quiet_machine(monkeypatch):
    """A clean tree, free ports and no caffeinate, whatever the machine."""
    monkeypatch.setattr(m3, "git_state", lambda: dict(CLEAN))
    monkeypatch.setattr(m3, "port_in_use", lambda port: False)
    monkeypatch.setattr(m3, "keep_awake", contextlib.nullcontext)


# --- The grid ------------------------------------------------------------


def test_the_policies_grid_is_known():
    assert "policies" in m3.GRIDS
    assert m3.NEW_POLICIES == ("ttdf", "race_rp")
    assert set(m3.NEW_POLICIES).isdisjoint(m3.BASELINES)


def test_policies_grid_runs_ttdf_and_race_rp_on_the_reported_seeds():
    configs = m3.grid_configs("policies")
    assert [(c.mix, c.pacing, c.seed, c.policy, c.capacity_tokens)
            for c in configs] == [
                ("balanced", "human", 2, "ttdf", 131072),
                ("balanced", "human", 2, "race_rp", 131072),
                ("balanced", "human", 3, "ttdf", 131072),
                ("balanced", "human", 3, "race_rp", 131072),
                ("balanced", "human", 4, "ttdf", 131072),
                ("balanced", "human", 4, "race_rp", 131072),
                # Seed 2 at the capacity sweep's two ends.
                ("balanced", "human", 2, "ttdf", 32768),
                ("balanced", "human", 2, "race_rp", 32768),
                ("balanced", "human", 2, "ttdf", 393216),
                ("balanced", "human", 2, "race_rp", 393216),
            ]
    assert len({c.name for c in configs}) == 10
    assert all(not c.ceiling and c.replicate == 0 and c.block_size == 16
               for c in configs)
    assert configs[1].name == "balanced-human-s2-race_rp-cap131072"


def test_policies_grid_capacity_moves_only_the_main_runs():
    configs = m3.grid_configs("policies", 65536)
    assert [c.capacity_tokens
            for c in configs] == [65536] * 6 + [32768, 32768, 393216, 393216]


def test_an_unknown_grid_names_the_policies_grid():
    with pytest.raises(ValueError, match="policies"):
        m3.grid_configs("nope")


# --- policy_env ------------------------------------------------------------


@pytest.mark.parametrize("policy", ["lru", "lfu", "fifo"])
def test_baselines_need_no_settings(tmp_path, policy):
    missing = tmp_path / "no-sketch.json"
    assert m3.policy_env(policy, tmp_path / "run", missing) == {}


def test_ttdf_writes_its_counters_to_the_run(tmp_path):
    run_dir = tmp_path / "run"
    # TTDF needs no sketch, so a missing one does not matter.
    env = m3.policy_env("ttdf", run_dir, tmp_path / "no-sketch.json")
    assert env == {
        "VLLM_EVICTION_PARAMS": f"stats={(run_dir / STATS).resolve()}"
    }


def test_race_rp_loads_its_sketch(tmp_path):
    sketch = tmp_path / "sketch.json"
    sketch.write_text("{}")
    run_dir = tmp_path / "run"
    env = m3.policy_env("race_rp", run_dir, sketch)
    assert env == {
        "VLLM_EVICTION_PARAMS":
        f"sketch={sketch.resolve()},stats={(run_dir / STATS).resolve()}",
        **ONE_THREAD,
    }
    # A copy: changing one run's settings changes no other's.
    env["OMP_NUM_THREADS"] = "8"
    assert m3.ONE_THREAD == ONE_THREAD
    assert m3.policy_env("race_rp", run_dir, sketch)["OMP_NUM_THREADS"] == "1"


def test_policy_env_paths_are_absolute(tmp_path, monkeypatch):
    """The server does not run in the runner's folder: relative paths are
    made absolute against the runner's working directory."""
    monkeypatch.chdir(tmp_path)
    Path("sk.json").write_text("{}")
    env = m3.policy_env("race_rp", Path("runs/x"), Path("sk.json"))
    here = Path.cwd().resolve()
    assert env["VLLM_EVICTION_PARAMS"] == (f"sketch={here / 'sk.json'},"
                                           f"stats={here / 'runs/x' / STATS}")
    for item in env["VLLM_EVICTION_PARAMS"].split(","):
        assert Path(item.split("=", 1)[1]).is_absolute()


def test_race_rp_without_its_sketch_is_refused(tmp_path):
    missing = tmp_path / "no-sketch.json"
    with pytest.raises(m3.RunnerError,
                       match="race_rp needs its sketch .*race_train"):
        m3.policy_env("race_rp", tmp_path / "run", missing)


def test_policy_env_defaults_to_the_trained_sketch(tmp_path, monkeypatch):
    assert Path("results/m3-race/sketch-L16-P6.json") == m3.RACE_SKETCH
    monkeypatch.chdir(tmp_path)
    with pytest.raises(m3.RunnerError, match="sketch-L16-P6.json"):
        m3.policy_env("race_rp", tmp_path / "run")
    m3.RACE_SKETCH.parent.mkdir(parents=True)
    m3.RACE_SKETCH.write_text("{}")
    env = m3.policy_env("race_rp", tmp_path / "run")
    assert env["VLLM_EVICTION_PARAMS"].startswith(
        f"sketch={(tmp_path / m3.RACE_SKETCH).resolve()},")


# --- The grid's replays ----------------------------------------------------


def policy_configs():
    return [
        MixConfig("toy", "human", 1, policy, 64, block_size=BS)
        for policy in ("lru", "ttdf", "race_rp")
    ]


def calls_and_runs(tmp_path, replay):
    calls = {c["run_dir"].name: c for c in replay.calls}
    runs = {
        name: json.loads((tmp_path / "out" / name / "run.json").read_text())
        for name in calls
    }
    return calls, runs


def test_grid_passes_each_policys_settings_and_records_the_sketch(
        tmp_path, monkeypatch):
    # The sketch at its default place, relative to the working directory.
    monkeypatch.chdir(tmp_path)
    m3.RACE_SKETCH.parent.mkdir(parents=True)
    m3.RACE_SKETCH.write_bytes(b'{"L": 16, "P": 6}')
    make_plan(tmp_path / "data", "toy-human-s1")
    lru, ttdf, race = policy_configs()
    replay = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, [lru, ttdf, race], replay)
    assert [e["status"] for e in entries] == ["finished"] * 3
    calls, runs = calls_and_runs(tmp_path, replay)
    # The baseline's replay gets no extra server settings at all.
    assert "extra_env" not in calls[lru.name]
    out = (tmp_path / "out").resolve()
    assert calls[ttdf.name]["extra_env"] == {
        "VLLM_EVICTION_PARAMS": f"stats={out / ttdf.name / STATS}"
    }
    # TTDF's settings are its counters only; RACE-RP's add one thread.
    assert calls[race.name]["extra_env"] == {
        "VLLM_EVICTION_PARAMS":
        (f"sketch={tmp_path.resolve() / m3.RACE_SKETCH}"
         f",stats={out / race.name / STATS}"),
        **ONE_THREAD,
    }
    assert [calls[c.name]["policy"]
            for c in (lru, ttdf, race)] == ["lru", "ttdf", "race_rp"]
    # Only RACE-RP's run.json names its sketch, with the file's hash.
    assert runs[race.name]["race_sketch"] == str(m3.RACE_SKETCH)
    assert runs[race.name]["race_sketch_sha256"] == hashlib.sha256(
        b'{"L": 16, "P": 6}').hexdigest()
    for config in (lru, ttdf):
        assert not {"race_sketch", "race_sketch_sha256"} & set(
            runs[config.name])
    # The grid's own fields are still there.
    assert runs[race.name]["config"] == race.name
    assert runs[race.name]["grid"] == "toy"


def test_a_race_rp_run_without_its_sketch_fails_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no results/m3-race here
    make_plan(tmp_path / "data", "toy-human-s1")
    lru, ttdf, race = policy_configs()
    replay = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, [lru, ttdf, race], replay)
    status = {e["name"]: (e["status"], e["error"]) for e in entries}
    assert status[lru.name] == ("finished", None)
    assert status[ttdf.name] == ("finished", None)
    assert status[race.name][0] == "failed"
    assert status[race.name][1].startswith(
        "RunnerError: race_rp needs its sketch")
    # It never reached the server.
    assert race.name not in {c["run_dir"].name for c in replay.calls}


def test_the_sketch_recorded_is_the_sketch_loaded(tmp_path, monkeypatch):
    """With RACE_SKETCH moved, the server loads the file whose path and
    hash run.json records, not the default one."""
    monkeypatch.chdir(tmp_path)  # no sketch at the default place
    sketch = tmp_path / "elsewhere" / "sketch.json"
    sketch.parent.mkdir()
    sketch.write_bytes(b'{"P": 4}')
    monkeypatch.setattr(m3, "RACE_SKETCH", sketch)
    make_plan(tmp_path / "data", "toy-human-s1")
    race = policy_configs()[2]
    replay = FakeReplay()
    entries, _ = run_toy_grid(tmp_path, [race], replay)
    assert entries[0]["status"] == "finished", entries[0]["error"]
    calls, runs = calls_and_runs(tmp_path, replay)
    params = calls[race.name]["extra_env"]["VLLM_EVICTION_PARAMS"]
    assert params.startswith(f"sketch={sketch.resolve()},")
    assert runs[race.name]["race_sketch"] == str(sketch)
    assert runs[race.name]["race_sketch_sha256"] == hashlib.sha256(
        b'{"P": 4}').hexdigest()
