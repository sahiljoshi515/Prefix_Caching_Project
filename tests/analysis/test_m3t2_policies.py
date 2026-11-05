"""TTDF and RACE-RP analysis (analysis/m3t2_policies.py): policy strings,
seed-1 tuning, per-workload evictions, overhead and the CSV export, each on
a hand-made input whose answer is worked out beside it."""
import csv
import json

import pytest

from analysis import m3t2_policies as mp

pytestmark = pytest.mark.m3

S1 = "data/processed/m3/balanced-human-s1"
S2 = "data/processed/m3/balanced-human-s2"
# The specified RACE-RP setting as the tuning sweep names it: a sketch in
# the tuning folder, L 16 and P 6 (the spec is matched by settings).
RR_SPEC = "race_rp:sketch=results/m3-race/tune/sketch-L16-P6.json"
RR_L64 = "race_rp:sketch=results/m3-race/tune/sketch-L64-P3.json"
RR_P1 = "race_rp:sketch=results/m3-race/tune/sketch-L16-P1.json"
TT_A = "ttdf:tau_short=1,tau_long=60"
TT_B = "ttdf:tau_short=30"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.reader(f))


# --- Policy strings --------------------------------------------------------


def test_family_is_the_text_before_the_first_colon():
    assert mp.family("ttdf:tau_short=2") == "ttdf"
    assert mp.family("race_rp:sketch=a:b.json") == "race_rp"
    assert mp.family("lru") == "lru"


def test_settings_fill_ttdf_defaults():
    assert mp.settings("ttdf") == {
        "tau_short": 5.0,
        "tau_long": 120.0,
        "w_short": 0.7,
        "w_long": 0.3
    }
    # Given values replace the defaults; an item without "=" is ignored.
    got = mp.settings("ttdf:tau_short=2,junk,w_long=0.5")
    assert got == {
        "tau_short": 2.0,
        "tau_long": 120.0,
        "w_short": 0.7,
        "w_long": 0.5
    }
    assert all(isinstance(v, float) for v in got.values())


def test_settings_read_race_rp_tables_and_bits_from_the_sketch_name():
    assert mp.settings(RR_L64) == {"L": 64, "P": 3}
    assert mp.settings(mp.SPEC["race_rp"]) == mp.SPEC_SETTINGS["race_rp"]
    for bad in ("race_rp", "race_rp:sketch=results/sketch.json",
                "race_rp:sketch=sketch-L16-P6.json.bak"):
        with pytest.raises(ValueError, match="sketch-L<L>-P<P>"):
            mp.settings(bad)


def test_settings_of_other_policies_are_empty():
    assert mp.settings("lru") == {}
    assert mp.settings("lfu:x=1") == {}


def test_label_puts_a_setting_in_words():
    assert mp.label("ttdf") == "tau 5 s / 120 s, w 0.7 / 0.3"
    assert mp.label("ttdf:tau_short=1,tau_long=600,w_short=0.25") == \
        "tau 1 s / 600 s, w 0.25 / 0.3"
    assert mp.label(RR_SPEC) == "L 16, P 6"
    assert mp.label("lru") == "lru"


# --- Sweep rows ------------------------------------------------------------


def test_read_rows_keeps_rows_with_a_summary_and_no_error(tmp_path):
    good = {"policy": "lru", "summary": {"x": 1}}
    lines = [
        json.dumps(good),
        "not json {",
        "",
        json.dumps({
            "policy": "lfu",
            "error": "boom",
            "summary": {}
        }),
        json.dumps({"policy": "fifo"}),
        json.dumps({
            "policy": "ttdf",
            "summary": {}
        }),
    ]
    path = tmp_path / "sweep.jsonl"
    path.write_text("\n".join(lines))
    assert mp.read_rows(path) == [good, {"policy": "ttdf", "summary": {}}]
    assert mp.read_rows(tmp_path / "missing.jsonl") == []


# --- Seed-1 tuning ---------------------------------------------------------


def row(policy, cap, usable, plan=S1, window=(600.0, 1800.0)):
    """One windowed replay row of the tuning sweep."""
    return {
        "plan": plan,
        "policy": policy,
        "capacity": cap,
        "capacity_tokens": cap,
        "timing": "model",
        "window": list(window) if window else None,
        "summary": {
            "window": {
                "usable_reused_fraction": usable
            }
        },
    }


def both(policy, a, b, **kw):
    """Rows at the two test capacities (64, 128)."""
    return [row(policy, 64, a, **kw), row(policy, 128, b, **kw)]


CAPS = (64, 128)


def sweep_rows():
    # Usable reuse per setting at 64 and 128 tokens, and its mean:
    #   lru      0.30 0.50  0.40
    #   ttdf     0.30 0.50  0.40  (specified)
    #   TT_A     0.40 0.60  0.50  (best TTDF)
    #   TT_B     0.10 0.20  0.15
    #   RR_SPEC  0.20 0.40  0.30  (specified)
    #   RR_L64   0.20 0.40  0.30  (ties the spec: the spec is kept)
    #   RR_P1    0.10 0.10  0.10
    return [
        *both("lru", 0.3, 0.5),
        *both("ttdf", 0.3, 0.5),
        *both(TT_A, 0.4, 0.6),
        *both(TT_B, 0.1, 0.2),
        *both(RR_SPEC, 0.2, 0.4),
        *both(RR_L64, 0.2, 0.4),
        *both(RR_P1, 0.1, 0.1),
        # Left out: another seed's plan (it would make TT_B the best),
        *both(TT_B, 0.99, 0.99, plan=S2),
        # rows without a window (null, or no key at all),
        *both(TT_B, 0.98, 0.98, window=None),
        {k: v
         for k, v in row(TT_B, 64, 0.97).items() if k != "window"},
        # a setting missing one tuning capacity,
        row("ttdf:tau_short=2", 64, 0.99),
        # and capacities outside the tuning set (a ceiling row, 256).
        row("ttdf:tau_short=2", 256, 0.99),
        {
            **row("lru", 0, 0.99), "capacity": "ceiling"
        },
    ]


def test_tuning_picks_the_best_mean_per_policy_and_keeps_the_spec_on_a_tie():
    out = mp.tuning(sweep_rows(), capacities=CAPS)
    assert out["plan"] == "balanced-human-s1"
    assert out["capacities"] == [64, 128]
    assert out["lru"] == [0.3, 0.5]
    assert out["spec"] == mp.SPEC
    assert out["spec"] is not mp.SPEC
    # TTDF: TT_A beats the specified setting (0.50 against 0.40).
    tt = out["candidates"]["ttdf"]
    assert [c["policy"] for c in tt] == [TT_A, "ttdf", TT_B]
    assert [c["mean"] for c in tt
            ] == [pytest.approx(0.5),
                  pytest.approx(0.4),
                  pytest.approx(0.15)]
    assert tt[0]["usable"] == [0.4, 0.6]
    assert tt[0]["label"] == "tau 1 s / 60 s, w 0.7 / 0.3"
    assert tt[0]["settings"] == {
        "tau_short": 1.0,
        "tau_long": 60.0,
        "w_short": 0.7,
        "w_long": 0.3
    }
    sel = out["selected"]["ttdf"]
    assert sel["policy"] == TT_A and sel["is_spec"] is False
    assert sel["mean"] == pytest.approx(0.5)
    assert sel["spec_mean"] == pytest.approx(0.4)
    # RACE-RP: L 64 / P 3 only ties the spec, so the spec is kept.
    rr = out["candidates"]["race_rp"]
    # Equal means in policy-string order ("...-L16-P6" before "...-L64-").
    assert [c["policy"] for c in rr] == [RR_SPEC, RR_L64, RR_P1]
    assert [c["mean"] for c in rr] == sorted((c["mean"] for c in rr),
                                             reverse=True)
    sel = out["selected"]["race_rp"]
    assert sel["policy"] == RR_SPEC and sel["is_spec"] is True
    assert sel["mean"] == sel["spec_mean"] == pytest.approx(0.3)
    assert sel["label"] == "L 16, P 6"


def test_tuning_ties_keep_the_spec_whatever_the_row_order():
    # The tying setting comes first in the sweep; the spec still wins.
    rows = [*both(TT_A, 0.3, 0.5), *both("ttdf", 0.3, 0.5)]
    rows += [*both(RR_L64, 0.2, 0.4), *both(RR_SPEC, 0.2, 0.4)]
    out = mp.tuning(rows, capacities=CAPS)
    assert out["selected"]["ttdf"]["policy"] == "ttdf"
    assert out["selected"]["ttdf"]["is_spec"] is True
    assert out["selected"]["race_rp"]["policy"] == RR_SPEC
    # Without LRU rows there is no LRU curve.
    assert out["lru"] is None
    # A setting a hair above the spec is picked.
    rows = [*both("ttdf", 0.3, 0.5), *both(TT_A, 0.3, 0.5 + 1e-9)]
    rows += both(RR_SPEC, 0.2, 0.4)
    sel = mp.tuning(rows, capacities=CAPS)["selected"]["ttdf"]
    assert sel["policy"] == TT_A and sel["is_spec"] is False


def test_tuning_tie_above_the_spec_goes_to_the_smaller_policy_string():
    # TT_A ("...tau_short=1,...") and TT_B ("...tau_short=30") tie at 0.5,
    # above the spec's 0.4; RR_L64 ("...-L64-...") and RR_P1 ("...-L16-P1")
    # tie at 0.6, above the spec's 0.3. In either row order the smaller
    # string wins and the candidates come out in the same order.
    rows = [*both("ttdf", 0.3, 0.5), *both(TT_A, 0.4, 0.6)]
    rows += [*both(TT_B, 0.4, 0.6), *both(RR_SPEC, 0.2, 0.4)]
    rows += [*both(RR_L64, 0.5, 0.7), *both(RR_P1, 0.5, 0.7)]
    assert TT_A < TT_B and RR_P1 < RR_L64
    picks, orders = [], []
    for ordered in (rows, rows[::-1]):
        out = mp.tuning(ordered, capacities=CAPS)
        picks.append({n: s["policy"] for n, s in out["selected"].items()})
        orders.append({
            n: [c["policy"] for c in v]
            for n, v in out["candidates"].items()
        })
        assert out["selected"]["ttdf"]["is_spec"] is False
        assert out["selected"]["race_rp"]["is_spec"] is False
    assert picks[0] == picks[1] == {"ttdf": TT_A, "race_rp": RR_P1}
    assert orders[0] == orders[1] == {
        "ttdf": [TT_A, TT_B, "ttdf"],
        "race_rp": [RR_P1, RR_L64, RR_SPEC]
    }


def test_tuning_spec_wins_a_tie_with_a_smaller_policy_string():
    # RR_P1 ("...-L16-P1") sorts before the spec ("...-L16-P6") and ties it
    # at the best mean: the spec is still kept, in either row order.
    rows = [*both("ttdf", 0.3, 0.5), *both(RR_P1, 0.2, 0.4)]
    rows += both(RR_SPEC, 0.2, 0.4)
    assert RR_P1 < RR_SPEC
    for ordered in (rows, rows[::-1]):
        out = mp.tuning(ordered, capacities=CAPS)
        sel = out["selected"]["race_rp"]
        assert sel["policy"] == RR_SPEC and sel["is_spec"] is True
        assert [c["policy"]
                for c in out["candidates"]["race_rp"]] == [RR_P1, RR_SPEC]
    assert mp.policy_specs(out) == mp.SPEC


def test_tuning_is_none_without_complete_rows_for_both_specs():
    rows = sweep_rows()
    no_ttdf = [r for r in rows if r["policy"] != "ttdf"]
    assert mp.tuning(no_ttdf, capacities=CAPS) is None
    no_race = [r for r in rows if r["policy"] != RR_SPEC]
    assert mp.tuning(no_race, capacities=CAPS) is None
    # The specified TTDF setting lacks 128 tokens.
    part = [
        r for r in rows
        if not (r["policy"] == "ttdf" and r["capacity_tokens"] == 128)
    ]
    assert mp.tuning(part, capacities=CAPS) is None
    # The specified RACE-RP setting only on another seed's plan.
    other = [r for r in rows if r["policy"] != RR_SPEC]
    other += both(RR_SPEC, 0.2, 0.4, plan=S2)
    assert mp.tuning(other, capacities=CAPS) is None
    assert mp.tuning([], capacities=CAPS) is None


def test_tuning_defaults_to_the_seed1_plan_and_five_capacities():
    caps = mp.TUNING_CAPACITIES
    assert len(caps) == 5
    rows = []
    for policy, v in (("ttdf", 0.4), (TT_A, 0.2), (RR_SPEC, 0.3)):
        rows += [row(policy, c, v) for c in caps]
    # Four of the five capacities: TT_B does not count.
    rows += [row(TT_B, c, 0.9) for c in caps[:4]]
    out = mp.tuning(rows)
    assert out["plan"] == mp.TUNING_PLAN
    assert out["capacities"] == list(caps)
    assert [c["policy"] for c in out["candidates"]["ttdf"]] == ["ttdf", TT_A]
    assert out["selected"]["ttdf"]["is_spec"] is True
    assert out["candidates"]["ttdf"][0]["usable"] == [0.4] * 5


# --- Page names ------------------------------------------------------------


def test_policy_specs_add_a_tuned_entry_only_when_the_pick_differs():
    assert mp.policy_specs(None) == {
        "ttdf": "ttdf",
        "race_rp": "race_rp:sketch=results/m3-race/sketch-L16-P6.json"
    }
    specs = mp.policy_specs(None)
    specs["x"] = "y"
    assert "x" not in mp.SPEC
    assert mp.policy_specs({}) == mp.SPEC
    assert mp.policy_specs({"selected": {}}) == mp.SPEC
    tune = mp.tuning(sweep_rows(), capacities=CAPS)
    # TTDF's pick differs (TT_A); RACE-RP kept its spec.
    assert mp.policy_specs(tune) == {**mp.SPEC, "ttdf_tuned": TT_A}


# --- Evictions per workload ------------------------------------------------

NAMES = ["sharegpt", "bailian-chat", "bailian-api", "agentbank"]
GROUPS = {
    "sharegpt": ("sharegpt", ),
    "bailian": ("bailian-chat", "bailian-api"),
    "ccbench": ("ccbench", ),
}


def matrix(counts, regret):
    return {
        "rows_evictor": NAMES,
        "counts": counts,
        "regretted_share": dict(zip(NAMES, regret))
    }


# Rows: evictor; columns: owner (sharegpt, bailian-chat, bailian-api,
# agentbank). Seed A owners: 4, 4, 1, 4 blocks; seed B: 2, 0, 2, 0.
SEED_A = matrix([[0, 2, 1, 0], [3, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 4]],
                [0.5, 0.25, None, 1.0])
SEED_B = matrix([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0]],
                [0.0, None, 0.5, None])
SEED_NONE = matrix([[0] * 4 for _ in range(4)], [None] * 4)


def test_eviction_view_counts_blocks_by_owner_per_second():
    out = mp.eviction_view({"lru": [SEED_A, SEED_B]}, GROUPS, 2.0)
    lru = out["lru"]
    assert list(lru) == ["sharegpt", "bailian", "ccbench", "all"]
    # ShareGPT owns 4 and 2 evicted blocks over 2 s: 2.0 and 1.0 per s;
    # regretted 0.5 and 0.0.
    assert lru["sharegpt"]["seeds"] == [2.0, 1.0]
    assert lru["sharegpt"]["per_s"] == pytest.approx(1.5)
    assert lru["sharegpt"]["regretted"] == pytest.approx(0.25)
    # Bailian: 4 + 1 and 0 + 2 blocks; regretted (4 x 0.25 + 1 x 0) / 5 =
    # 0.2 (bailian-api's None share counts as 0) and (2 x 0.5) / 2 = 0.5.
    assert lru["bailian"]["seeds"] == [2.5, 1.0]
    assert lru["bailian"]["per_s"] == pytest.approx(1.75)
    assert lru["bailian"]["regretted"] == pytest.approx(0.35)
    # CC-Bench has no column: no evictions, nothing to regret.
    assert lru["ccbench"] == {
        "per_s": 0.0,
        "seeds": [0.0, 0.0],
        "regretted": None
    }
    # All owners: 13 and 4 blocks; regretted (2 + 1 + 0 + 4) / 13 and
    # (0 + 0 + 1 + 0) / 4.
    assert lru["all"]["seeds"] == [6.5, 2.0]
    assert lru["all"]["per_s"] == pytest.approx(4.25)
    assert lru["all"]["regretted"] == pytest.approx((7 / 13 + 0.25) / 2)


def test_eviction_view_regret_is_none_when_a_seed_has_no_evictions():
    out = mp.eviction_view({
        "lru": [SEED_A],
        "ttdf": [SEED_A, SEED_NONE]
    }, GROUPS, 2.0)
    assert set(out) == {"lru", "ttdf"}
    assert out["lru"]["sharegpt"]["regretted"] == 0.5
    ttdf = out["ttdf"]["sharegpt"]
    assert ttdf["seeds"] == [2.0, 0.0] and ttdf["per_s"] == 1.0
    assert ttdf["regretted"] is None
    assert out["ttdf"]["all"]["regretted"] is None
    assert isinstance(out["ttdf"]["all"]["per_s"], float)


# --- Overhead --------------------------------------------------------------


def write_stats(run, **stats):
    run.mkdir(parents=True)
    (run / mp.STATS_FILE).write_text(json.dumps(stats))


def test_overhead_view_engine_part_averages_each_policys_runs(tmp_path):
    write_stats(tmp_path / "t2",
                evict_us_mean=2.0,
                candidates_per_scan=10.0,
                max_candidates=40,
                metadata_bytes=2048,
                evictions=100)
    write_stats(tmp_path / "t3",
                evict_us_mean=4.0,
                candidates_per_scan=20.0,
                max_candidates=30,
                metadata_bytes=4096,
                evictions=300)
    # A run that never evicted has no time or scan size.
    write_stats(tmp_path / "t4",
                evict_us_mean=None,
                candidates_per_scan=None,
                max_candidates=0,
                metadata_bytes=6144,
                evictions=0)
    (tmp_path / "t5").mkdir()  # no stats file: skipped
    (tmp_path / "r2").mkdir()
    out = mp.overhead_view(tmp_path, {
        "ttdf": ["t2", "t3", "t4", "t5", "t6"],
        "race_rp": ["r2"],
    }, [], mp.SPEC)
    assert out["replay"] == {}
    assert list(out["engine"]) == ["ttdf"]  # race_rp has no stats
    ttdf = out["engine"]["ttdf"]
    assert ttdf["runs"] == 3
    assert ttdf["evict_us"] == pytest.approx(3.0)
    assert ttdf["candidates_per_scan"] == pytest.approx(15.0)
    assert ttdf["max_candidates"] == 40
    # (2 + 4 + 6) KiB / 3 and (100 + 300 + 0) / 3.
    assert ttdf["metadata_kib"] == pytest.approx(4.0)
    assert ttdf["evictions"] == pytest.approx(400 / 3)


def replay_row(policy, cap, **summary):
    return {"policy": policy, "capacity_tokens": cap, "summary": summary}


def test_overhead_view_replay_part_per_page_name_and_capacity(tmp_path):
    specs = {**mp.SPEC, "ttdf_tuned": TT_A}
    race = mp.SPEC["race_rp"]
    rows = [
        # RACE-RP's spec string is filed as race_rp: 2 and 4 us per
        # eviction, 1 and 3 KiB, 8 and 12 candidates per scan.
        replay_row(race,
                   65536,
                   evictions=1000,
                   evict_seconds=0.002,
                   evictor={
                       "metadata_bytes": 1024,
                       "candidates_per_scan": 8.0
                   }),
        replay_row(race,
                   65536,
                   evictions=500,
                   evict_seconds=0.002,
                   evictor={
                       "metadata_bytes": 3072,
                       "candidates_per_scan": 12.0
                   }),
        # LRU reports no evictor stats: no metadata or scan size.
        replay_row("lru", 65536, evictions=100, evict_seconds=0.0001),
        # No eviction, or no timing: left out.
        replay_row("lru", 131072, evictions=0, evict_seconds=0.0),
        replay_row("fifo", 65536, evictions=10),
        # The tuned TTDF string; its stats lack the metadata size.
        replay_row(TT_A,
                   65536,
                   evictions=10,
                   evict_seconds=0.00005,
                   evictor={"candidates_per_scan": 3.0}),
        replay_row("lfu",
                   65536,
                   evictions=100,
                   evict_seconds=0.0002,
                   evictor=None),
    ]
    out = mp.overhead_view(tmp_path, {}, rows, specs)
    assert out["engine"] == {}
    rep = out["replay"]
    assert set(rep) == {"race_rp", "lru", "ttdf_tuned", "lfu"}
    assert list(rep["race_rp"]) == ["65536"]
    cell = rep["race_rp"]["65536"]
    assert cell["evict_us"] == pytest.approx(3.0)
    assert cell["metadata_kib"] == pytest.approx(2.0)
    assert cell["candidates_per_scan"] == pytest.approx(10.0)
    assert list(rep["lru"]) == ["65536"]
    assert rep["lru"]["65536"]["evict_us"] == pytest.approx(1.0)
    assert rep["lru"]["65536"]["metadata_kib"] is None
    assert rep["lru"]["65536"]["candidates_per_scan"] is None
    tuned = rep["ttdf_tuned"]["65536"]
    assert tuned["evict_us"] == pytest.approx(5.0)
    assert tuned["metadata_kib"] is None
    assert tuned["candidates_per_scan"] == 3.0
    assert rep["lfu"]["65536"]["evict_us"] == pytest.approx(2.0)
    assert rep["lfu"]["65536"]["metadata_kib"] is None


# --- CSV -------------------------------------------------------------------


def full_summary():
    return {
        "config": {
            "reported_seeds": [2, 3]
        },
        "capacity_window": {
            "capacities": [64, 128],
            "seeds": [2, 3],
            # At 128 tokens a seed's row was missing: no per-seed values.
            "group_seeds": {
                "lru": {
                    "sharegpt": [[0.1, 0.2], None]
                }
            },
            "evictions": {
                "lru": {
                    "sharegpt": [1.5, None]
                }
            },
        },
        "mixed_groups": {
            "sharegpt": {
                "lru": {
                    "usable": {
                        "mean": 0.4,
                        "seeds": [0.4, None]
                    }
                },
                "ttdf": {
                    "usable": {
                        "seeds": [0.5, 0.6]
                    },
                    "verdict": "higher"
                },
                # A policy with no usable reuse in any seed (spread None).
                "fifo": {
                    "usable": None,
                    "verdict": "tie"
                },
                "prompt_tokens": 100.0,
                "odd": {
                    "no_usable": 1
                },
            }
        },
        "mixed_intervals": {
            "lru": {
                "curve": [{
                    "lo": 0,
                    "hi": 1,
                    "offered": 4,
                    "served": 2,
                    "share": 0.5
                }, {
                    "lo": 1,
                    "hi": None,
                    "offered": 0,
                    "served": 0,
                    "share": None
                }]
            }
        },
        "new_policies": {
            "evictions": {
                "ttdf": {
                    "sharegpt": {
                        "per_s": 1.5,
                        "seeds": [1.0, 2.0],
                        "regretted": 0.25
                    },
                    "ccbench": {
                        "per_s": 0.0,
                        "seeds": [0.0, 0.0],
                        "regretted": None
                    },
                }
            },
            "overhead": {
                "engine": {
                    "ttdf": {
                        "runs": 3,
                        "evict_us": 3.0,
                        "metadata_kib": 4.0,
                        "candidates_per_scan": 15.0
                    }
                },
                "replay": {
                    "lru": {
                        "65536": {
                            "evict_us": 1.0,
                            "metadata_kib": None,
                            "candidates_per_scan": None
                        }
                    }
                },
            },
            "tuning": {
                "capacities": [64, 128],
                "candidates": {
                    "ttdf": [{
                        "label": "tau 5 s / 120 s, w 0.7 / 0.3",
                        "policy": "ttdf",
                        "usable": [0.3, 0.5],
                        "mean": 0.4
                    }]
                },
            },
        },
    }


def test_write_csvs_writes_every_section_with_blank_none_cells(tmp_path):
    out = tmp_path / "data" / "task2"
    paths = mp.write_csvs(full_summary(), out)
    assert [p.name for p in paths] == [
        "capacity_by_workload.csv", "evictions_by_capacity.csv",
        "engine_128k_usable.csv", "engine_128k_served_by_wait.csv",
        "engine_128k_evictions.csv", "overhead.csv", "tuning_seed1.csv"
    ]
    assert all(p.parent == out for p in paths)
    assert read_csv(out / "capacity_by_workload.csv") == [
        [
            "policy", "workload", "capacity_tokens", "seed",
            "usable_prefix_reuse"
        ],
        ["lru", "sharegpt", "64", "2", "0.1"],
        ["lru", "sharegpt", "64", "3", "0.2"],
    ]
    assert read_csv(out / "evictions_by_capacity.csv") == [
        ["policy", "workload", "capacity_tokens", "evictions_per_s"],
        ["lru", "sharegpt", "64", "1.5"],
        ["lru", "sharegpt", "128", ""],
    ]
    # The prompt-token entry and a cell without "usable" are not policies.
    assert read_csv(out / "engine_128k_usable.csv") == [
        ["policy", "workload", "seed", "usable_prefix_reuse"],
        ["lru", "sharegpt", "2", "0.4"],
        ["lru", "sharegpt", "3", ""],
        ["ttdf", "sharegpt", "2", "0.5"],
        ["ttdf", "sharegpt", "3", "0.6"],
    ]
    assert read_csv(out / "engine_128k_served_by_wait.csv") == [
        [
            "policy", "wait_from_s", "wait_to_s", "reuses_offered",
            "reuses_served", "share_served"
        ],
        ["lru", "0", "1", "4", "2", "0.5"],
        ["lru", "1", "", "0", "0", ""],
    ]
    assert read_csv(out / "engine_128k_evictions.csv") == [
        ["policy", "workload", "evictions_per_s", "share_looked_up_again"],
        ["ttdf", "sharegpt", "1.5", "0.25"],
        ["ttdf", "ccbench", "0.0", ""],
    ]
    assert read_csv(out / "overhead.csv") == [
        [
            "source", "policy", "capacity_tokens", "evict_us_mean",
            "metadata_kib", "candidates_per_scan"
        ],
        ["engine", "ttdf", "131072", "3.0", "4.0", "15.0"],
        ["replay", "lru", "65536", "1.0", "", ""],
    ]
    assert read_csv(out / "tuning_seed1.csv") == [
        [
            "policy", "setting", "replay_policy", "usable_64", "usable_128",
            "mean"
        ],
        ["ttdf", "tau 5 s / 120 s, w 0.7 / 0.3", "ttdf", "0.3", "0.5", "0.4"],
    ]


def test_write_csvs_leaves_out_a_group_cell_without_usable_reuse(tmp_path):
    # A group with no prompt tokens in the window in any seed has a null
    # spread: its row is absent, the other policies' rows are kept.
    summary = {
        "config": {
            "reported_seeds": [2]
        },
        "mixed_groups": {
            "agentbank": {
                "lru": {
                    "usable": None
                },
                "ceiling": {
                    "usable": None
                }
            },
            "sharegpt": {
                "lru": {
                    "usable": {
                        "seeds": [0.4]
                    }
                },
                "ttdf": {
                    "usable": None,
                    "vs_lru": None,
                    "verdict": "tie"
                }
            },
        },
    }
    paths = mp.write_csvs(summary, tmp_path)
    assert [p.name for p in paths] == ["engine_128k_usable.csv"]
    assert read_csv(paths[0]) == [
        ["policy", "workload", "seed", "usable_prefix_reuse"],
        ["lru", "sharegpt", "2", "0.4"],
    ]
    # Only null cells: the header alone.
    del summary["mixed_groups"]["sharegpt"]
    assert read_csv(mp.write_csvs(summary, tmp_path / "none")[0]) == [[
        "policy", "workload", "seed", "usable_prefix_reuse"
    ]]


def test_write_csvs_skips_missing_sections(tmp_path):
    assert mp.write_csvs({}, tmp_path / "empty") == []
    assert list((tmp_path / "empty").iterdir()) == []
    # Empty or null sections write nothing either.
    summary = {
        "capacity_window": None,
        "mixed_groups": {},
        "mixed_intervals": None,
        "new_policies": {
            "evictions": {},
            "overhead": None,
            "tuning": None
        },
    }
    assert mp.write_csvs(summary, tmp_path / "none") == []
    # A capacity view without per-group values: headers only.
    paths = mp.write_csvs(
        {"capacity_window": {
            "capacities": [64],
            "seeds": [2]
        }}, tmp_path / "cw")
    assert [p.name for p in paths
            ] == ["capacity_by_workload.csv", "evictions_by_capacity.csv"]
    assert read_csv(paths[0]) == [[
        "policy", "workload", "capacity_tokens", "seed", "usable_prefix_reuse"
    ]]
    assert len(read_csv(paths[1])) == 1
    # Only the overhead section.
    paths = mp.write_csvs(
        {"new_policies": {
            "overhead": {
                "engine": {},
                "replay": {}
            }
        }}, tmp_path / "oh")
    assert [p.name for p in paths] == ["overhead.csv"]
    assert len(read_csv(paths[0])) == 1


# --- Command line ----------------------------------------------------------


def test_main_select_prints_the_held_out_policy_strings(tmp_path, capsys):
    path = tmp_path / "tune.jsonl"
    rows = []
    for policy, v in (("ttdf", 0.4), (TT_A, 0.5), (RR_SPEC, 0.3)):
        rows += [row(policy, c, v) for c in mp.TUNING_CAPACITIES]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    mp.main(["select", "--tuning", str(path)])
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "ttdf\tttdf",
        f"race_rp\t{mp.SPEC['race_rp']}",
        f"ttdf_tuned\t{TT_A}",
    ]


def test_main_select_stops_without_the_specified_rows(tmp_path, capsys):
    path = tmp_path / "tune.jsonl"
    path.write_text("\n".join(
        json.dumps(row("ttdf", c, 0.4)) for c in mp.TUNING_CAPACITIES))
    with pytest.raises(SystemExit) as stop:
        mp.main(["select", "--tuning", str(path)])
    assert str(path) in str(stop.value.code)
    assert capsys.readouterr().out == ""
    with pytest.raises(SystemExit):
        mp.main(["select", "--tuning", str(tmp_path / "missing.jsonl")])


def test_main_csv_prints_the_files_written(tmp_path, capsys):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(full_summary()))
    out = tmp_path / "csv"
    mp.main(["csv", "--summary", str(path), "--out", str(out)])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7
    assert lines[0] == str(out / "capacity_by_workload.csv")
    assert all(
        (out / name).exists() for name in ("overhead.csv", "tuning_seed1.csv"))
    # The CSV holds the summary's numbers after its JSON round trip.
    assert read_csv(out / "tuning_seed1.csv")[1][-1] == "0.4"
    assert read_csv(
        out / "overhead.csv")[2] == ["replay", "lru", "65536", "1.0", "", ""]


def test_main_needs_a_command():
    with pytest.raises(SystemExit):
        mp.main([])
