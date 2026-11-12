"""TTDF, RACE-RP and Sampled LRFU in the mixed-workload study
(analysis/m3t2_study.py): which engine runs join, the per-group views, the
windowed capacity curve by workload group, and the whole study on a tiny
hand-made run grid."""
import json
import math
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from analysis import isolate
from analysis import m3t2_policies as mp
from analysis import m3t2_study as st

pytestmark = pytest.mark.m3

RACE = mp.SPEC["race_rp"]
TT_A = "ttdf:tau_short=1,tau_long=60"
# One workload of each group, in the order the evictions matrix uses.
WL = ["sharegpt", "agentbank", "ccbench", "bailian-chat"]
CAP = 64


def write_parquet(run, name, cols):
    (run / "tables").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), run / "tables" / f"{name}.parquet")


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def touch_requests(run):
    """A run folder with only an (empty) requests table."""
    write_parquet(run, "requests", {"req": pa.array([], pa.string())})


# --- Which new policies join -----------------------------------------------


def grid_cfg(**kw):
    return {
        "mix": "balanced",
        "pacing": "human",
        "capacity_tokens": CAP,
        "reported_seeds": [2, 3],
        "policies": ["lru"],
        "new_policies": ["ttdf", "race_rp"],
        **kw
    }


def test_new_policies_run_needs_tables_for_every_reported_seed(tmp_path):
    cfg = grid_cfg()
    for x in (2, 3):
        touch_requests(tmp_path /
                       st.run_name("balanced", "human", x, "ttdf", CAP))
    # RACE-RP: seed 3 has a folder but no tables; another size is no help.
    touch_requests(tmp_path /
                   st.run_name("balanced", "human", 2, "race_rp", CAP))
    (tmp_path / st.run_name("balanced", "human", 3, "race_rp", CAP)).mkdir()
    touch_requests(tmp_path /
                   st.run_name("balanced", "human", 3, "race_rp", 2 * CAP))
    assert st.new_policies_run(tmp_path, cfg) == ["ttdf"]
    touch_requests(tmp_path /
                   st.run_name("balanced", "human", 3, "race_rp", CAP))
    assert st.new_policies_run(tmp_path, cfg) == ["ttdf", "race_rp"]
    # No new policies configured: none.
    del cfg["new_policies"]
    assert st.new_policies_run(tmp_path, cfg) == []


# --- Per-group view of the mixed runs --------------------------------------


def usable_row(**hits):
    """usable_by_workload's row: 32 prompt tokens per workload, 16 per
    usable block."""
    row = {
        w: {
            "requests": 1,
            "prompt_tokens": 32,
            "usable_tokens": 16 * hits[w]
        }
        for w in WL
    }
    used = sum(16 * hits[w] for w in WL)
    row["all"] = {
        "requests": len(WL),
        "prompt_tokens": 32 * len(WL),
        "usable_tokens": used,
        "fraction": used / (32 * len(WL))
    }
    return row


HITS = {
    "ceiling": dict(sharegpt=2, agentbank=2, ccbench=2, **{"bailian-chat": 2}),
    "lru": dict(sharegpt=1, agentbank=2, ccbench=0, **{"bailian-chat": 1}),
    "fifo": dict(sharegpt=0, agentbank=2, ccbench=0, **{"bailian-chat": 1}),
    "ttdf": dict(sharegpt=2, agentbank=2, ccbench=1, **{"bailian-chat": 1}),
}


def test_mixed_group_view_defaults_to_the_baselines():
    cfg = grid_cfg(policies=["lru", "fifo"])
    headline = {p: {x: usable_row(**HITS[p]) for x in (2, 3)} for p in HITS}
    base = st.mixed_group_view(headline, cfg)
    assert set(base["sharegpt"]) == {"lru", "fifo", "ceiling", "prompt_tokens"}
    assert st.mixed_group_view(headline, cfg, []) == base
    out = st.mixed_group_view(headline, cfg, ["lru", "fifo", "ttdf"])
    assert list(out) == list(isolate.GROUPS) + ["all"]
    sg = out["sharegpt"]
    # ShareGPT: LRU 16 of 32 tokens, TTDF 32 of 32, in both seeds.
    assert sg["ttdf"]["usable"]["mean"] == 1.0
    assert sg["ttdf"]["vs_lru"]["seeds"] == [0.5, 0.5]
    assert sg["ttdf"]["verdict"] == "higher"
    assert sg["fifo"]["verdict"] == "lower"
    assert "vs_lru" not in sg["lru"] and "verdict" not in sg["lru"]
    assert "vs_lru" not in sg["ceiling"]
    assert sg["prompt_tokens"] == 32.0
    # Bailian: both keep one block; the whole mix: 96 against 64 tokens.
    assert out["bailian"]["ttdf"]["verdict"] == "tie"
    assert out["all"]["ttdf"]["vs_lru"]["mean"] == pytest.approx(0.25)
    assert out["all"]["prompt_tokens"] == 128.0
    # The baselines' entries are those of the default view.
    assert out["sharegpt"]["fifo"] == base["sharegpt"]["fifo"]


# --- The windowed capacity curve by workload group -------------------------

PLAN = "data/processed/m3/balanced-human-s{}"


def wrow(seed,
         policy,
         cap,
         usable,
         workloads,
         evictions=None,
         window=(0.0, 100.0)):
    """A windowed replay row; `workloads`: name -> (prompt, usable[,
    evictions])."""
    inside = {
        "usable_reused_fraction": usable,
        "workloads": {
            w: {
                "prompt_tokens": v[0],
                "usable_tokens": v[1],
                **({
                    "evictions": v[2]
                } if len(v) > 2 else {})
            }
            for w, v in workloads.items()
        },
    }
    if evictions is not None:
        inside["evictions"] = evictions
    return {
        "plan": PLAN.format(seed),
        "policy": policy,
        "capacity": cap,
        "capacity_tokens": 0 if cap == "ceiling" else cap,
        "timing": "model",
        "window": list(window) if window else None,
        "summary": {
            "usable_reused_fraction": 0.0,
            "window": inside
        },
    }


def capacity_rows():
    # Per seed (2, 3) and capacity (64, 128), workload: (prompt tokens,
    # usable tokens, evictions in the 100 s window).
    return [
        wrow(
            2, "lru", 64, 0.5, {
                "sharegpt": (100, 50, 100),
                "bailian-chat": (40, 10, 60),
                "bailian-api": (60, 30, 40)
            }, 200),
        wrow(3, "lru", 64, 0.7, {
            "sharegpt": (100, 70, 50),
            "bailian-chat": (50, 25, 50)
        }, 100),
        # At 128 tokens the replay did not count evictions.
        wrow(2, "lru", 128, 0.8, {"sharegpt": (100, 80)}),
        wrow(3, "lru", 128, 0.9, {"sharegpt": (100, 90)}),
        # RACE-RP's spec string, filed under race_rp; seed 3 lacks 64.
        wrow(2, RACE, 64, 0.3, {"sharegpt": (100, 30, 5)}, 5),
        wrow(2, RACE, 128, 0.6, {"sharegpt": (100, 60, 10)}, 10),
        wrow(3, RACE, 128, 0.6, {"sharegpt": (100, 60, 10)}, 10),
        wrow(2, "lru", "ceiling", 0.9, {
            "sharegpt": (100, 100),
            "bailian-chat": (40, 40)
        }),
        wrow(3, "lru", "ceiling", 0.95, {
            "sharegpt": (100, 90),
            "bailian-chat": (50, 50)
        }),
        # Not counted: unreported seed 1, a whole-run row, engine timing.
        wrow(1, "lru", 64, 0.0, {"sharegpt": (100, 0, 999)}, 999),
        wrow(2, "lru", 64, 0.0, {"sharegpt": (100, 0)}, window=None),
        {
            **wrow(2, "lru", 64, 0.0, {"sharegpt": (100, 0)}), "timing":
            "engine:x"
        },
    ]


def engine_run(run, usable_hits, planned=20.0):
    """An engine run with one 32-token ShareGPT request."""
    write_parquet(
        run, "requests", {
            "req": ["sharegpt:a:1"],
            "workload": ["sharegpt"],
            "session": ["a"],
            "turn": [1],
            "planned_s": [planned],
            "status": ["completed"],
            "prompt_tokens": [32],
            "prompt_blocks": [2],
            "hits": [usable_hits],
            "usable_hits": [usable_hits],
            "hits_after_miss": [0],
            "cached_tokens": [16 * usable_hits],
            "preempted": [0]
        })


def capacity_cfg():
    return {
        **grid_cfg(new_policies=["race_rp"]), "workloads": ["sharegpt"],
        "block_size": 16
    }


def test_capacity_curve_files_spec_rows_and_splits_by_group(tmp_path):
    sweep = tmp_path / "sweep.jsonl"
    write_lines(sweep, capacity_rows())
    runs = tmp_path / "runs"
    engine_run(runs / "balanced-human-s2-race_rp-cap64", 1)
    engine_run(runs / "balanced-human-s2-lru-cap128", 2)
    # Not a configured policy, an unreported seed, a run without tables.
    engine_run(runs / "balanced-human-s2-ttdf-cap64", 2)
    engine_run(runs / "balanced-human-s1-lru-cap64", 2)
    (runs / "balanced-human-s3-lru-cap64").mkdir()
    out = st.capacity_curve(runs, sweep, capacity_cfg(), (0, 100),
                            ["lru", "race_rp"], {"race_rp": RACE})
    assert out["capacities"] == [64, 128]
    assert out["curve"]["race_rp"]["hit_rate"] == [None, pytest.approx(0.6)]
    assert out["curve"]["lru"]["hit_rate"] == [
        pytest.approx(0.6), pytest.approx(0.85)
    ]
    assert set(out["groups"]) == {"lru", "race_rp"}
    lru = out["groups"]["lru"]
    assert list(lru) == list(isolate.GROUPS)
    # ShareGPT: 50/100 and 70/100 at 64; 80/100 and 90/100 at 128.
    assert lru["sharegpt"] == [pytest.approx(0.6), pytest.approx(0.85)]
    # Bailian sums its traces: (10 + 30) / (40 + 60) and 25 / 50; at 128
    # neither seed has a Bailian request.
    assert lru["bailian"] == [pytest.approx(0.45), None]
    assert out["group_seeds"]["lru"]["bailian"] == [[0.4, 0.5], [None, None]]
    assert lru["agentbank"] == [None, None]
    # Evicted blocks per second of the 100 s window: ShareGPT 100 and 50;
    # Bailian 60 + 40 and 50; the mix 200 and 100. None at 128 (not
    # counted).
    ev = out["evictions"]["lru"]
    assert list(ev) == list(isolate.GROUPS) + ["all"]
    assert ev["sharegpt"] == [pytest.approx(0.75), None]
    assert ev["bailian"] == [pytest.approx(0.75), None]
    assert ev["agentbank"] == [0.0, None]
    assert ev["all"] == [pytest.approx(1.5), None]
    # RACE-RP lacks seed 3 at 64 tokens: nothing there.
    race = out["groups"]["race_rp"]
    assert race["sharegpt"] == [None, pytest.approx(0.6)]
    assert out["group_seeds"]["race_rp"]["sharegpt"] == [None, [0.6, 0.6]]
    assert out["evictions"]["race_rp"]["sharegpt"] == [
        None, pytest.approx(0.1)
    ]
    assert out["evictions"]["race_rp"]["all"] == [None, pytest.approx(0.1)]
    # The ceiling per group: ShareGPT 100/100 and 90/100, Bailian 1 and 1.
    assert out["group_ceiling"]["sharegpt"] == pytest.approx(0.95)
    assert out["group_ceiling"]["bailian"] == pytest.approx(1.0)
    assert out["group_ceiling"]["ccbench"] is None
    assert out["ceiling"] == pytest.approx(0.925)
    # Engine runs of baseline and new policies, beside the replay's value.
    assert [(e["seed"], e["policy"], e["capacity"])
            for e in out["engine"]] == [(2, "lru", 128), (2, "race_rp", 64)]
    by = {e["policy"]: e for e in out["engine"]}
    assert by["race_rp"]["engine"] == 0.5 and by["race_rp"]["replay"] == 0.3
    assert by["lru"]["engine"] == 1.0 and by["lru"]["replay"] == 0.8


def test_capacity_curve_group_ceiling_needs_every_seed(tmp_path):
    rows = [
        r for r in capacity_rows()
        if not (r["capacity"] == "ceiling" and r["plan"].endswith("s3"))
    ]
    sweep = tmp_path / "sweep.jsonl"
    write_lines(sweep, rows)
    runs = tmp_path / "runs"
    runs.mkdir()
    out = st.capacity_curve(runs, sweep, capacity_cfg(), (0, 100), ["lru"],
                            {"race_rp": RACE})
    assert out["group_ceiling"] == {}
    assert out["ceiling"] is None
    assert out["engine"] == []


def test_capacity_curve_without_specs_or_window(tmp_path):
    sweep = tmp_path / "sweep.jsonl"
    write_lines(sweep, capacity_rows())
    runs = tmp_path / "runs"
    runs.mkdir()
    # Without `specs` the spec string is its own policy, not race_rp.
    out = st.capacity_curve(runs, sweep, capacity_cfg(), (0, 100),
                            ["lru", "race_rp", RACE])
    assert out["curve"]["race_rp"]["hit_rate"] == [None, None]
    assert out["curve"][RACE]["hit_rate"] == [None, pytest.approx(0.6)]
    # Over the whole run (no window) there is no per-group view.
    whole = st.capacity_curve(runs, sweep, capacity_cfg(), None, ["lru"],
                              {"race_rp": RACE})
    assert whole["window_s"] is None
    for key in ("groups", "group_seeds", "evictions", "group_ceiling"):
        assert key not in whole
    # Only seed 2's whole-run row at 64 tokens: no seed mean.
    assert whole["capacities"] == [64]
    assert whole["curve"]["lru"]["hit_rate"] == [None]


def test_capacity_groups_tolerates_summaries_without_workloads():
    cell = {"usable_reused_fraction": 0.5, "evictions": 30}
    mean_, seeds_, evict = st._capacity_groups([[cell, cell], [cell, None]],
                                               10.0)
    for g in isolate.GROUPS:
        assert mean_[g] == [None, None]
        assert seeds_[g] == [[None, None], None]
        # The total was counted but not split by workload: no per-group
        # value (not 0).
        assert evict[g] == [None, None]
    assert evict["all"] == [pytest.approx(3.0), None]
    # One seed with workloads, one without: still no per-group value.
    full = {
        **cell, "workloads": {
            "sharegpt": {
                "prompt_tokens": 10,
                "usable_tokens": 5,
                "evictions": 30
            }
        }
    }
    mean_, _, evict = st._capacity_groups([[full, cell]], 10.0)
    assert evict["sharegpt"] == [None]
    assert mean_["sharegpt"] == [None]
    assert evict["all"] == [pytest.approx(3.0)]
    _, _, evict = st._capacity_groups([[full, full]], 10.0)
    assert evict["sharegpt"] == [pytest.approx(3.0)]
    assert evict["agentbank"] == [0.0]


def test_capacity_groups_evictions_none_when_a_seed_did_not_count():
    a = {
        "usable_reused_fraction": 0.5,
        "evictions": 20,
        "workloads": {
            "agentbank": {
                "prompt_tokens": 10,
                "usable_tokens": 5,
                "evictions": 20
            }
        }
    }
    b = {k: v for k, v in a.items() if k != "evictions"}
    mean_, seeds_, evict = st._capacity_groups([[a, b]], 10.0)
    assert mean_["agentbank"] == [0.5]
    assert seeds_["agentbank"] == [[0.5, 0.5]]
    assert evict["agentbank"] == [None]
    assert evict["all"] == [None]
    _, _, evict = st._capacity_groups([[a, a]], 10.0)
    assert evict["agentbank"] == [2.0] and evict["sharegpt"] == [0.0]


# --- The whole study on a tiny grid ----------------------------------------
#
# Two reported seeds; one baseline (LRU), TTDF with engine runs for both
# seeds, RACE-RP with a run for seed 2 only (so it stays out). Each mixed run
# has one 32-token request per workload, planned at 20 s, inside the 100 s
# window [10, 110).

NAN = math.nan
# Free-pool stays: (owner, end, idle s, evictor, next lookup t).
STAYS: Dict[str, List[Tuple[Any, ...]]] = {
    "ceiling": [("sharegpt", "revive", 5.0, None, 50.0),
                ("agentbank", "revive", 2.0, None, 30.0),
                ("ccbench", "censored", 80.0, None, NAN),
                ("bailian-chat", "revive", 3.0, None, 40.0)],
    # LRU evicts a ShareGPT block for AgentBank (looked up again) and a
    # Bailian block for ShareGPT (never looked up again).
    "lru": [("sharegpt", "evict", 5.0, "agentbank", 50.0),
            ("bailian-chat", "evict", 3.0, "sharegpt", NAN),
            ("agentbank", "revive", 2.0, None, 30.0),
            ("ccbench", "censored", 80.0, None, NAN)],
    # TTDF evicts one CC-Bench block, looked up again.
    "ttdf": [("ccbench", "evict", 4.0, "ccbench", 60.0),
             ("sharegpt", "revive", 1.0, None, 25.0)],
}
STAYS["race_rp"] = STAYS["ttdf"]
HITS["race_rp"] = HITS["ttdf"]


def mixed_run(run, policy, workloads=WL):
    hits = HITS[policy]
    reqs = [f"{w}:s:1" for w in workloads]
    n = len(workloads)
    write_parquet(
        run, "requests", {
            "req": reqs,
            "workload": list(workloads),
            "session": ["s"] * n,
            "turn": [1] * n,
            "planned_s": [20.0] * n,
            "status": ["completed"] * n,
            "prompt_tokens": [32] * n,
            "prompt_blocks": [2] * n,
            "hits": [hits[w] for w in workloads],
            "usable_hits": [hits[w] for w in workloads],
            "hits_after_miss": [0] * n,
            "cached_tokens": [16 * hits[w] for w in workloads],
            "preempted": [0] * n
        })
    # Two re-accesses per request, 5 s and 50 s after the previous one.
    write_parquet(
        run, "lookups", {
            "req": [r for r in reqs for _ in (0, 1)],
            "workload": [w for w in workloads for _ in (0, 1)],
            "turn": [1] * 2 * n,
            "planned_s": [20.0] * 2 * n,
            "prefill": [0] * 2 * n,
            "index": [0, 1] * n,
            "hit": [i < hits[w] for w in workloads for i in (0, 1)],
            "usable": [i < hits[w] for w in workloads for i in (0, 1)],
            "in_use": [False] * 2 * n,
            "source": ["same_session", "other_session"] * n,
            "interval_s": [5.0, 50.0] * n,
            "requests_between": [0] * 2 * n,
            "evicted_before": [False] * 2 * n
        })
    stays = [s for s in STAYS.get(policy, []) if s[0] in workloads]
    write_parquet(
        run, "residencies", {
            "owner_workload": [s[0] for s in stays],
            "owner_planned_s": [20.0] * len(stays),
            "end": [s[1] for s in stays],
            "idle_s": [s[2] for s in stays],
            "hits_before": [0] * len(stays),
            "evictor_workload": pa.array([s[3] for s in stays], pa.string()),
            "next_lookup_t": [s[4] for s in stays],
            "index": [0] * len(stays)
        })
    write_lines(run / "client.jsonl", [{
        "workload": w,
        "planned_s": 20.0,
        "latency_s": 1.0,
        "status": "completed"
    } for w in workloads])
    if policy == "ceiling":
        write_lines(run / "events.jsonl", [{
            "ev": "insert",
            "hash": 1,
            "req": "sharegpt:s:0"
        }, {
            "ev": "insert",
            "hash": 2,
            "req": "sharegpt:s:0"
        }, {
            "ev": "hit",
            "hash": 1,
            "req": "chatcmpl-sharegpt:s:1"
        }])


def study_cfg(**kw):
    cfg = {
        "block_size": 16,
        "capacity_tokens": CAP,
        "policies": ["lru"],
        # As in the real config; Sampled LRFU has no engine runs here.
        "new_policies": ["ttdf", "race_rp", "sampled_lrfu"],
        "mix": "balanced",
        "pacing": "human",
        "sensitivity_pacing": "fast",
        "train_seeds": [],
        "reported_seeds": [2, 3],
        "noise_replicate": {
            "seed": 2,
            "policy": "lru",
            "replicate": 1
        },
        "w0_s": 10.0,
        "w0_sensitivity_s": [],
        "stop_s": 110.0,
        "workloads": WL,
        "interval_bins_s": [0, 1, 10, 100],
        "index_bins": [0, 1, 2, 4],
        "cdf_points": 3,
        "timeline_step_s": 50,
        "served_interval_edges_s": [0, 10],
        "sim_policies": ["lru"],
        "km_grid_s": [1, 10],
    }
    cfg.update(kw)
    return cfg


STATS = {
    2:
    dict(evict_us_mean=2.0,
         candidates_per_scan=10.0,
         max_candidates=40,
         metadata_bytes=2048,
         evictions=100),
    3:
    dict(evict_us_mean=4.0,
         candidates_per_scan=20.0,
         max_candidates=30,
         metadata_bytes=4096,
         evictions=300),
}


def build_grid(root):
    runs, plans = root / "runs", root / "plans"
    for x in (2, 3):
        for policy in ("ceiling", "lru", "ttdf"):
            mixed_run(runs / st.run_name("balanced", "human", x, policy, CAP),
                      policy)
        (runs / st.run_name("balanced", "human", x, "ttdf", CAP) /
         mp.STATS_FILE).write_text(json.dumps(STATS[x]))
        # Each group alone, under the baseline only.
        for group, members in isolate.GROUPS.items():
            alone = [w for w in WL if w in members]
            mixed_run(
                runs / st.run_name(isolate.isolated_mix(group), "human", x,
                                   "lru", CAP), "lru", alone)
        write_lines(plans / f"balanced-human-s{x}" / "manifest.json",
                    [{
                        "requests": 4,
                        "sessions": 4,
                        "prompt_tokens": 128,
                        "workloads": {w: {
                            "requests": 1
                        }
                                      for w in WL}
                    }])
    mixed_run(runs / st.run_name("balanced", "human", 2, "race_rp", CAP),
              "race_rp")
    return runs, plans


def tuning_rows():
    # The tuning sweep: TT_A beats TTDF's spec; RACE-RP keeps its spec;
    # Sampled LRFU was not tuned (no rows), so it has no pick.
    rows = []
    for policy, v in (("lru", 0.1), ("ttdf", 0.4), (TT_A, 0.5),
                      ("race_rp:sketch=t/sketch-L16-P6.json", 0.3)):
        rows += [{
            "plan": "data/processed/m3/balanced-human-s1",
            "policy": policy,
            "capacity_tokens": c,
            "window": [600.0, 1800.0],
            "summary": {
                "window": {
                    "usable_reused_fraction": v
                }
            }
        } for c in mp.TUNING_CAPACITIES]
    return rows


def window_rows():
    # The capacity sweep over [10, 110): every page name at 64 tokens.
    rows = []
    for x in (2, 3):
        for policy, v in (("lru", 0.5), ("ttdf", 0.7), (RACE, 0.6),
                          ("sampled_lrfu", 0.65), (TT_A, 0.8)):
            rows.append(
                wrow(x,
                     policy,
                     CAP,
                     v, {"sharegpt": (32, 32 * v, 10)},
                     10,
                     window=(10.0, 110.0)))
        rows.append(
            wrow(x,
                 "lru",
                 "ceiling",
                 1.0, {"sharegpt": (32, 32)},
                 window=(10.0, 110.0)))
    return rows


@pytest.fixture
def grid(tmp_path, monkeypatch):
    runs, plans = build_grid(tmp_path)
    sketch = tmp_path / "sketch-L16-P6.json"
    sketch.write_text(json.dumps({"L": 16, "P": 6, "meta": {"seed": 0}}))
    monkeypatch.setattr(mp, "SKETCH", str(sketch))
    tune = tmp_path / "tune.jsonl"
    write_lines(tune, tuning_rows())
    overhead = tmp_path / "overhead.jsonl"
    write_lines(overhead, [{
        "policy": RACE,
        "capacity_tokens": CAP,
        "summary": {
            "evictions": 100,
            "evict_seconds": 0.0003,
            "evictor": {
                "metadata_bytes": 2048,
                "candidates_per_scan": 4.0
            }
        }
    }, {
        "policy": "lru",
        "capacity_tokens": CAP,
        "summary": {
            "evictions": 0,
            "evict_seconds": 0.0
        }
    }])
    window = tmp_path / "capacity-window.jsonl"
    write_lines(window, window_rows())
    return {
        "runs": runs,
        "plans": plans,
        "sketch": sketch,
        "tune": tune,
        "overhead": overhead,
        "window": window
    }


def test_study_adds_the_new_policies_to_the_mixed_views_only(grid):
    cfg = study_cfg()
    out = st.study(grid["runs"], cfg, None, None, grid["plans"],
                   grid["window"], grid["tune"], grid["overhead"])
    # TTDF has runs for both seeds; RACE-RP only for seed 2 and Sampled
    # LRFU none, so they are out.
    assert list(out["headline"]) == ["lru", "ttdf", "ceiling"]
    assert list(out["mixed_intervals"]) == ["lru", "ttdf", "ceiling"]
    assert "curve" in out["mixed_intervals"]["ttdf"]
    assert set(out["policy_views"]) == {"lru", "ttdf"}
    # ShareGPT: TTDF keeps both blocks, LRU one, in both seeds.
    sg = out["mixed_groups"]["sharegpt"]
    assert sg["ttdf"]["usable"]["mean"] == 1.0
    assert sg["ttdf"]["vs_lru"]["seeds"] == [0.5, 0.5]
    assert sg["ttdf"]["verdict"] == "higher"
    assert "race_rp" not in sg
    # The alone study stays the baselines'.
    iso = out["isolation"]
    assert iso is not None
    assert "lru" in iso["rows"]["sharegpt"]
    assert "ttdf" not in iso["rows"]["sharegpt"]
    assert list(iso["intervals"]["sharegpt"]) == ["lru"]
    assert iso["vs_lru"]["sharegpt"] == {}

    new = out["new_policies"]
    assert new["engine"] == ["ttdf"]
    assert new["specs"] == {**mp.SPEC, "ttdf_tuned": TT_A}
    assert new["sketch"] == {"seed": 0}
    assert new["tuning"]["selected"]["ttdf"]["policy"] == TT_A
    assert new["tuning"]["selected"]["race_rp"]["is_spec"] is True
    assert set(new["tuning"]["selected"]) == {"ttdf", "race_rp"}
    assert set(new["tuning"]["candidates"]) == {"ttdf", "race_rp"}
    # Evictions over the 100 s window, both seeds alike. LRU: one ShareGPT
    # block (looked up again) and one Bailian block (not); TTDF: one
    # CC-Bench block (looked up again).
    ev = new["evictions"]
    assert set(ev) == {"lru", "ttdf"}
    assert ev["lru"]["sharegpt"] == {
        "per_s": pytest.approx(0.01),
        "seeds": [pytest.approx(0.01)] * 2,
        "regretted": 1.0
    }
    assert ev["lru"]["bailian"]["per_s"] == pytest.approx(0.01)
    assert ev["lru"]["bailian"]["regretted"] == 0.0
    assert ev["lru"]["ccbench"]["regretted"] is None
    assert ev["lru"]["all"]["per_s"] == pytest.approx(0.02)
    assert ev["lru"]["all"]["regretted"] == pytest.approx(0.5)
    assert ev["ttdf"]["ccbench"]["per_s"] == pytest.approx(0.01)
    assert ev["ttdf"]["ccbench"]["regretted"] == 1.0
    assert ev["ttdf"]["sharegpt"] == {
        "per_s": 0.0,
        "seeds": [0.0, 0.0],
        "regretted": None
    }
    # Overhead: TTDF's two engine runs; the timed replay of RACE-RP's spec
    # (the LRU row has no evictions).
    eng = new["overhead"]["engine"]
    assert list(eng) == ["ttdf"]
    assert eng["ttdf"]["runs"] == 2
    assert eng["ttdf"]["evict_us"] == pytest.approx(3.0)
    assert eng["ttdf"]["metadata_kib"] == pytest.approx(3.0)
    assert eng["ttdf"]["max_candidates"] == 40
    rep = new["overhead"]["replay"]
    assert list(rep) == ["race_rp"]
    assert rep["race_rp"][str(CAP)]["evict_us"] == pytest.approx(3.0)
    assert rep["race_rp"][str(CAP)]["metadata_kib"] == pytest.approx(2.0)
    # The windowed capacity curve has every page name, tuned TTDF too.
    cw = out["capacity_window"]
    assert list(cw["curve"]) == [
        "lru", "ttdf", "race_rp", "sampled_lrfu", "ttdf_tuned"
    ]
    assert cw["curve"]["race_rp"]["hit_rate"] == [pytest.approx(0.6)]
    assert cw["curve"]["sampled_lrfu"]["hit_rate"] == [pytest.approx(0.65)]
    assert cw["curve"]["ttdf_tuned"]["hit_rate"] == [pytest.approx(0.8)]
    assert cw["groups"]["ttdf"]["sharegpt"] == [pytest.approx(0.7)]
    assert cw["evictions"]["race_rp"]["all"] == [pytest.approx(0.1)]
    assert cw["group_ceiling"]["sharegpt"] == 1.0
    # Engine runs of TTDF and RACE-RP at 64 tokens sit beside the replay.
    pairs = {(e["seed"], e["policy"]): e["replay"] for e in cw["engine"]}
    assert pairs[(2, "race_rp")] == pytest.approx(0.6)
    assert pairs[(3, "ttdf")] == pytest.approx(0.7)
    # The per-workload CSVs come out of this summary.
    names = [p.name for p in mp.write_csvs(out, grid["runs"].parent / "csv")]
    assert "engine_128k_evictions.csv" in names
    assert "capacity_by_workload.csv" in names


def test_study_without_tuning_overhead_sweeps_or_sketch(grid):
    grid["sketch"].rename(grid["sketch"].with_suffix(".moved"))
    cfg = study_cfg()
    out = st.study(grid["runs"], cfg, None, None, grid["plans"])
    new = out["new_policies"]
    assert new["sketch"] is None
    assert new["tuning"] is None
    assert new["specs"] == mp.SPEC
    assert new["overhead"]["replay"] == {}
    assert list(new["overhead"]["engine"]) == ["ttdf"]
    assert "capacity_window" not in out
    # Sweep paths that do not exist: no tuning, no replay overhead, no
    # windowed curve.
    out = st.study(grid["runs"], cfg, None, None, grid["plans"],
                   grid["window"].with_name("missing.jsonl"),
                   grid["tune"].with_name("missing.jsonl"),
                   grid["overhead"].with_name("missing.jsonl"))
    assert out["new_policies"]["tuning"] is None
    assert out["new_policies"]["overhead"]["replay"] == {}
    assert "capacity_window" not in out


def test_study_without_new_policies_is_the_baselines_alone(grid):
    cfg = study_cfg()
    del cfg["new_policies"]
    out = st.study(grid["runs"], cfg, None, None, grid["plans"],
                   grid["window"], grid["tune"], grid["overhead"])
    assert list(out["headline"]) == ["lru", "ceiling"]
    assert out["new_policies"]["engine"] == []
    assert list(out["new_policies"]["evictions"]) == ["lru"]
    assert out["new_policies"]["overhead"]["engine"] == {}
    # The replay still compares the specified settings with the baselines.
    assert list(out["capacity_window"]["curve"]) == [
        "lru", "ttdf", "race_rp", "sampled_lrfu", "ttdf_tuned"
    ]


# --- Command line ----------------------------------------------------------


def test_main_passes_the_tuning_and_overhead_sweeps_to_study(
        tmp_path, monkeypatch, capsys):
    import analysis.run_m2_task2 as m2
    seen: Dict[Any, Any] = {}

    def fake_study(runs, cfg, sweep, reports, plans, capacity_sweep,
                   tuning_sweep, overhead_sweep):
        seen.update(tuning=tuning_sweep,
                    overhead=overhead_sweep,
                    capacity=capacity_sweep,
                    cfg=cfg)
        return {"x": 1.0}

    monkeypatch.setattr(st, "study", fake_study)
    monkeypatch.setattr(m2, "git_state", lambda: {"commit": "abc"})
    out = tmp_path / "summary.json"
    st.main([
        "--tuning-sweep",
        str(tmp_path / "t.jsonl"), "--overhead-sweep",
        str(tmp_path / "o.jsonl"), "--out",
        str(out)
    ])
    assert seen["tuning"] == tmp_path / "t.jsonl"
    assert seen["overhead"] == tmp_path / "o.jsonl"
    assert seen["cfg"]["new_policies"] == ["ttdf", "race_rp", "sampled_lrfu"]
    assert json.loads(out.read_text()) == {
        "x": 1.0,
        "provenance": {
            "commit": "abc"
        }
    }
    assert "Wrote" in capsys.readouterr().out
    # The defaults are the sweeps the policies module reads.
    st.main(["--out", str(out)])
    assert seen["tuning"] == mp.TUNING_SWEEP
    assert seen["overhead"] == mp.OVERHEAD_SWEEP
