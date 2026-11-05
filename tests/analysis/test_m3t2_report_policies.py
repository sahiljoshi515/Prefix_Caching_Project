"""The results page's chart data with the new policies, TTDF, RACE-RP and
Sampled LRFU (``analysis/m3t2_report.py``): they join the mixed-workload
views of points 1, 2 and 4 once their runs are in the summary, and get a
point 7.

`summary` builds a small but complete synthetic ``summary.json`` that the
narrative and figure tests reuse."""
import copy

import pytest

from analysis import m3t2_policies as mp
from analysis import m3t2_report as rp

pytestmark = pytest.mark.m3

# The new policies the synthetic summary holds by default; the third,
# Sampled LRFU, only when a test asks for it.
NEW = ("ttdf", "race_rp")
ALL_NEW = ("ttdf", "race_rp", "sampled_lrfu")
SEEDS = [2, 3, 4]
CAPS = [32768, 65536, 131072, 262144]
FIXED = 131072
# Every seed is the mean plus one of these, so two policies' seeds differ
# by the same amount: the sign always agrees and only the mean decides.
OFFSETS = (-0.01, 0.0, 0.01)
# Replay usable reuse at 32K, 64K, 128K and 256K. Against LRU: LFU beats
# it at 32K to 64K, trails at 128K and ties at 256K; FIFO and RACE-RP
# trail at every size; TTDF beats LRU at 32K to 64K, trails it at 128K
# and ties at 256K (0.1 point); Sampled LRFU beats LRU at 128K to 256K,
# trails it at 32K and ties at 64K.
CURVES = {
    "lru": [0.40, 0.50, 0.60, 0.65],
    "lfu": [0.41, 0.51, 0.59, 0.65],
    "fifo": [0.35, 0.45, 0.55, 0.60],
    "belady": [0.55, 0.65, 0.72, 0.75],
    "ttdf": [0.42, 0.52, 0.58, 0.651],
    "race_rp": [0.38, 0.48, 0.58, 0.63],
    "sampled_lrfu": [0.39, 0.50, 0.62, 0.70],
    # Seed-1 pick: higher than TTDF as specified at 32K to 64K only.
    "ttdf_tuned": [0.43, 0.53, 0.58, 0.652],
}
GROUP_OFFSET = {
    "sharegpt": -0.3,
    "agentbank": 0.1,
    "ccbench": 0.2,
    "bailian": -0.2
}
GROUP_CEILING = {
    "sharegpt": 0.6,
    "agentbank": 0.9,
    "ccbench": 0.98,
    "bailian": 0.62
}
# Engine runs at 128K: usable reuse over the whole mix.
MIXED = {
    "lru": 0.56,
    "lfu": 0.54,
    "fifo": 0.50,
    "ttdf": 0.57,
    "race_rp": 0.555,
    # Sampled LRFU keeps the least on the engine, as in the real runs.
    "sampled_lrfu": 0.45,
    "ceiling": 0.78,
}
MIXED_OFFSET = {
    "sharegpt": -0.4,
    "agentbank": 0.1,
    "ccbench": 0.15,
    "bailian": -0.3
}
VERDICTS = {
    "lfu": {
        "all": "lower",
        "sharegpt": "lower",
        "agentbank": "tie",
        "ccbench": "higher",
        "bailian": "lower"
    },
    "fifo": {k: "lower"
             for k in ("all", *rp.GROUPS)},
    "ttdf": {
        "all": "higher",
        "sharegpt": "higher",
        "agentbank": "higher",
        "ccbench": "tie",
        "bailian": "lower"
    },
    "race_rp": {k: "tie"
                for k in ("all", *rp.GROUPS)},
    "sampled_lrfu": {k: "lower"
                     for k in ("all", *rp.GROUPS)},
}
PROMPT_TOKENS = {
    "all": 1000,
    "sharegpt": 100,
    "agentbank": 100,
    "ccbench": 500,
    "bailian": 300
}
# Interval bins (seconds) and the share each policy serves in each.
EDGES = (0, 0.1, 1, 10, 30, 300, None)
COARSE = ((0, 10), (10, 30), (30, 300), (300, None))
SHARES = {
    "lru": [1.0, 0.99, 0.95, 0.30, 0.0, 0.0],
    "lfu": [0.97, 0.95, 0.90, 0.40, 0.20, 0.10],
    "fifo": [0.95, 0.90, 0.85, 0.20, 0.0, 0.0],
    "ttdf": [0.99, 0.98, 0.96, 0.50, 0.10, 0.0],
    "race_rp": [0.98, 0.97, 0.90, 0.40, 0.30, 0.20],
    "sampled_lrfu": [0.99, 0.97, 0.93, 0.35, 0.0, 0.0],
    "ceiling": [1.0] * 6,
}
# Evicted blocks per second (engine) by workload, scaled per policy: every
# policy evicts Bailian's blocks most, and TTDF evicts the fewest.
EVICT = {"sharegpt": 30, "agentbank": 7, "ccbench": 80, "bailian": 400}
EVICT_SCALE = {
    "lru": 1.0,
    "lfu": 1.03,
    "fifo": 1.04,
    "ttdf": 0.9,
    "race_rp": 0.95,
    "sampled_lrfu": 0.98
}
WORKLOADS = ("sharegpt", "agentbank", "ccbench", "bailian-chat", "bailian-api",
             "bailian-thinking", "bailian-coder")
# Reuse lost against no eviction, per workload and seed: (evicted, cut off
# behind a missing block). Sampled LRFU evicts as much as LRU and strands
# blocks besides; the others evict a little less.
LOST = {"lru": (100, 0), "sampled_lrfu": (100, 50)}
# Requests per seed that hit a cached block behind a missing one.
STRANDED = {"fifo": [3, 0, 1], "sampled_lrfu": [120, 150, 130]}


def spread(v, offsets=OFFSETS):
    seeds = [v + d for d in offsets]
    return {"mean": v, "min": min(seeds), "max": max(seeds), "seeds": seeds}


def per_seed(values):
    return [[v + d for d in OFFSETS] for v in values]


def curve(shares, offered=5000):
    return [{
        "lo": lo,
        "hi": hi,
        "offered": offered,
        "served": round(x * offered),
        "share": x
    } for lo, hi, x in zip(EDGES[:-1], EDGES[1:], shares)]


def coarse(bins):
    out = []
    for lo, hi in COARSE:
        inside = [
            b for b in bins if b["lo"] >= lo and (
                hi is None or (b["hi"] is not None and b["hi"] <= hi))
        ]
        off = sum(b["offered"] for b in inside)
        srv = sum(b["served"] for b in inside)
        out.append({
            "lo": lo,
            "hi": hi,
            "offered": off,
            "served": srv,
            "share": srv / off
        })
    return out


def interval_view(policy):
    bins = curve(SHARES[policy])
    return {"curve": bins, "by_bin": coarse(bins)}


def capacity_window(replay, engine=(), tuned=()):
    """The replay curves: the baselines, Belady and the new policies in
    `replay` complete; the other new policies all None (not yet run), as
    ``m3t2_study`` writes them. Engine runs at 128K for the baselines and
    the new policies in `engine`, each with the replay of its seed."""

    def none():
        return [None] * len(CAPS)

    names = [*rp.POLICIES, "belady", *ALL_NEW, *tuned]
    cv, groups, gseeds = {}, {}, {}
    for p in names:
        if p in replay or p not in ALL_NEW:
            cv[p] = {"hit_rate": list(CURVES[p]), "seeds": per_seed(CURVES[p])}
            groups[p] = {
                g: [v + GROUP_OFFSET[g] for v in CURVES[p]]
                for g in rp.GROUPS
            }
            gseeds[p] = {g: per_seed(groups[p][g]) for g in rp.GROUPS}
        else:
            cv[p] = {"hit_rate": none(), "seeds": none()}
            groups[p] = {g: none() for g in rp.GROUPS}
            gseeds[p] = {g: none() for g in rp.GROUPS}
    runs = [{
        "seed":
        x,
        "policy":
        p,
        "capacity":
        FIXED,
        "engine":
        CURVES[p][2] + d + 0.002,
        "replay":
        CURVES[p][2] + d if p in replay or p not in ALL_NEW else None,
    } for p in (*rp.POLICIES, *engine) for x, d in zip(SEEDS, OFFSETS)]
    runs += [
        # A partial size: only seed 2 ran on the engine at 32K.
        {
            "seed": 2,
            "policy": "lru",
            "capacity": 32768,
            "engine": 0.395,
            "replay": 0.39
        },
        # An engine run without a replay value is left out of the markers.
        {
            "seed": 3,
            "policy": "lru",
            "capacity": 32768,
            "engine": 0.9,
            "replay": None
        },
    ]
    return {
        "capacities": list(CAPS),
        "fixed": FIXED,
        "ceiling": 0.8,
        "curve": cv,
        "groups": groups,
        "group_seeds": gseeds,
        "group_ceiling": dict(GROUP_CEILING),
        "engine": runs,
    }


def mixed_groups(engine):
    out = {}
    for k in ("all", *rp.GROUPS):
        off = 0.0 if k == "all" else MIXED_OFFSET[k]
        cell = {"prompt_tokens": PROMPT_TOKENS[k]}
        for p in (*rp.POLICIES, *engine, "ceiling"):
            cell[p] = {"usable": spread(MIXED[p] + off)}
            if p in VERDICTS:
                cell[p]["verdict"] = VERDICTS[p][k]
        out[k] = cell
    return out


def candidate(policy, mean):
    """A tuning candidate, labelled and parsed as ``m3t2_policies`` does."""
    return {
        "policy": policy,
        "label": mp.label(policy),
        "settings": mp.settings(policy),
        "usable": [mean] * 5,
        "mean": mean
    }


def tuning(names=NEW):
    """Seed-1 tuning of the policies in `names`: TTDF's pick differs from
    the specified setting (which ranks fourth); RACE-RP's and Sampled
    LRFU's specified settings are also the best."""
    ttdf = [
        candidate("ttdf:tau_short=1", 0.53),
        candidate("ttdf:tau_long=600", 0.52),
        candidate("ttdf:tau_short=30", 0.51),
        candidate("ttdf", 0.50),
        candidate("ttdf:w_short=0.3,w_long=0.7", 0.49),
    ]
    race = [
        candidate("race_rp:sketch=sketch-L16-P6.json", 0.60),
        candidate("race_rp:sketch=sketch-L64-P6.json", 0.59),
        candidate("race_rp:sketch=sketch-L16-P4.json", 0.58),
        candidate("race_rp:sketch=sketch-L16-P8.json", 0.57),
    ]
    lrfu = [
        candidate("sampled_lrfu", 0.62),
        candidate("sampled_lrfu:half_life=64", 0.61),
        candidate("sampled_lrfu:sample_size=16", 0.60),
    ]
    candidates = {"ttdf": ttdf, "race_rp": race, "sampled_lrfu": lrfu}
    selected = {
        "ttdf": {
            **ttdf[0], "is_spec": False,
            "spec_mean": 0.50
        },
        "race_rp": {
            **race[0], "is_spec": True,
            "spec_mean": 0.60
        },
        "sampled_lrfu": {
            **lrfu[0], "is_spec": True,
            "spec_mean": 0.62
        },
    }
    return {
        "capacities": [32768, 65536, 131072, 262144, 393216],
        "lru": [0.45] * 5,
        "candidates": {p: candidates[p]
                       for p in names},
        "selected": {p: selected[p]
                     for p in names},
    }


def lost(policies):
    """Reuse lost against no eviction per policy, seed and workload, with
    "all" (their sum) as ``m3t2_study`` writes it."""
    out = {}
    for p in policies:
        evicted, cut = LOST.get(p, (90, 0))
        seeds = {}
        for x in SEEDS:
            ws = {
                w: {
                    "kept": 400,
                    "evicted": evicted,
                    "cut_off": cut
                }
                for w in WORKLOADS
            }
            ws["all"] = {
                k: sum(v[k] for v in ws.values())
                for k in ("kept", "evicted", "cut_off")
            }
            seeds[str(x)] = ws
        out[p] = seeds
    return out


def policy_views(policies):
    return {
        p: {
            "hits_after_miss": list(STRANDED.get(p, [0, 0, 0])),
            "preempted": [0, 0, 0]
        }
        for p in policies
    }


def evictions(policies):
    out = {}
    for p in policies:
        view = {}
        for k in (*rp.GROUPS, "all"):
            base = (sum(EVICT.values())
                    if k == "all" else EVICT[k]) * EVICT_SCALE[p]
            view[k] = {
                "per_s": base,
                "seeds": [base - 1, base, base + 2],
                "regretted": 0.3
            }
        out[p] = view
    if "race_rp" in out:
        out["race_rp"]["agentbank"]["regretted"] = None
    return out


def overhead():
    return {
        "engine": {
            "ttdf": {
                "evict_us": 12.34,
                "candidates_per_scan": 250.0,
                "metadata_kib": 1234.4
            },
            # No count of blocks scored: the replay's is used instead.
            "race_rp": {
                "evict_us": 30.0,
                "metadata_kib": 2048.0
            },
            "sampled_lrfu": {
                "evict_us": 0.8,
                "candidates_per_scan": 4.0,
                "metadata_kib": 64.0
            },
        },
        "replay": {
            "lru": {
                str(FIXED): {
                    "evict_us": 1.5,
                    "metadata_kib": None,
                    "candidates_per_scan": None
                }
            },
            "ttdf": {
                str(FIXED): {
                    "evict_us": 10.0,
                    "metadata_kib": 1000.0,
                    "candidates_per_scan": 240.0
                }
            },
            "race_rp": {
                str(FIXED): {
                    "evict_us": 25.0,
                    "metadata_kib": 2000.0,
                    "candidates_per_scan": 300.0
                }
            },
            "sampled_lrfu": {
                str(FIXED): {
                    "evict_us": 0.5,
                    "metadata_kib": 60.0,
                    "candidates_per_scan": 4.0
                }
            },
        },
    }


def identify():
    acc = {
        "prompt_length": 0.5,
        "prefix_hashes": 0.8,
        "reuse_history": 0.7,
        "interarrival": 0.4,
        "all": 0.9,
        "all_but_timing": 0.85,
    }
    models = {
        k: {
            "accuracy": a,
            "balanced_accuracy": a - 0.1,
            "token_accuracy": a - 0.05
        }
        for k, a in acc.items()
    }
    models["all"].update({
        "confusion": {
            "sharegpt": {
                "sharegpt": 8,
                "agentbank": 2
            },
            "agentbank": {
                "agentbank": 9,
                "ccbench": 1
            },
            "ccbench": {
                "ccbench": 10
            },
            "bailian": {
                "bailian": 19,
                "sharegpt": 1
            },
        },
        "recall": {g: 0.9
                   for g in rp.GROUPS},
        "first_accuracy": 0.7,
        "later_accuracy": 0.95,
        "per_test": {f"balanced-human-s{x}": {
            "accuracy": 0.9
        }
                     for x in SEEDS},
    })
    return {
        "characteristics": {
            "models": models,
            "majority": {
                "label": "bailian",
                "accuracy": 0.57
            }
        }
    }


def isolation():
    rows = {}
    for g in rp.GROUPS:
        row = {
            "ceiling": spread(0.8),
            "requests": {
                "mean": 500
            },
            "mixed_requests": {
                "mean": 500
            },
            "prompt_tokens_per_s": {
                "mean": 1000
            },
        }
        for p in rp.POLICIES:
            row[p] = {
                "alone": spread(0.7),
                "mixed": spread(0.6),
                "verdict": "higher",
                "latency_alone_s": {
                    "mean": 1.0
                },
                "latency_mixed_s": {
                    "mean": 1.2
                },
            }
        rows[g] = row
    # One bar without a seed range.
    rows["sharegpt"]["lru"]["alone"] = spread(0.75, (0.0, 0.0, 0.0))
    return {
        "rows": rows,
        "intervals": {
            g: {p: {
                "curve": curve(SHARES[p])
            }
                for p in rp.POLICIES}
            for g in rp.GROUPS
        },
        "vs_lru": {
            g: {
                p: {
                    "verdict": "tie",
                    "diff": {
                        "mean": 0.0
                    }
                }
                for p in ("lfu", "fifo")
            }
            for g in rp.GROUPS
        },
    }


def ceiling_by_group():
    offered = {
        "sharegpt": [100, 50, 200, 50],
        "agentbank": [800, 100, 50, 50],
        "ccbench": [3000, 500, 300, 200],
        "bailian": [500, 300, 600, 600],
    }
    return {
        g: {
            "offered":
            sum(n),
            "by_bin": [{
                "lo": lo,
                "hi": hi,
                "offered": x,
                "served": x,
                "share": 1.0
            } for (lo, hi), x in zip(COARSE, n)],
            "served": {
                "quantiles": {
                    "p50": 120.0
                }
            },
        }
        for g, n in offered.items()
    }


def summary(engine=NEW, replay=NEW, tune=True, sketch=True):
    """A synthetic summary.json: the baselines everywhere; the new policies
    in `engine` with engine runs at 128K (mixed views, evictions), those in
    `replay` with replay curves; seed-1 tuning of the new policies present
    (with TTDF's tuned curve when TTDF has replay curves) and RACE-RP's
    sketch when asked."""
    engine, replay = tuple(engine), tuple(replay)
    present = [p for p in ALL_NEW if p in engine or p in replay]
    tune = tune and bool(present)
    tuned = ("ttdf_tuned", ) if tune and "ttdf" in replay else ()
    mix = {
        str(x): {
            "workloads": {
                w: {
                    "requests":
                    100,
                    "sessions":
                    20,
                    "prompt_token_share":
                    1 / len(WORKLOADS),
                    **({
                        "kept_session_fraction": 0.5
                    } if w.startswith("bailian") else {}),
                }
                for w in WORKLOADS
            }
        }
        for x in SEEDS
    }
    return {
        "config": {
            "block_size": 16,
            "capacity_tokens": FIXED,
            "w0_s": 600.0,
            "stop_s": 1800.0,
            "reported_seeds": list(SEEDS),
            "train_seeds": [1],
            "policies": list(rp.POLICIES),
            "new_policies": list(NEW),
        },
        "mix": mix,
        "mixed_intervals":
        {p: interval_view(p)
         for p in (*rp.POLICIES, *engine, "ceiling")},
        "ceiling_intervals_by_group": ceiling_by_group(),
        "mixed_groups": mixed_groups(engine),
        "identify": identify(),
        "capacity_window": capacity_window(replay, engine, tuned),
        "isolation": isolation(),
        "q2": {
            "sources": {w: {
                "other_workload": 0.01
            }
                        for w in WORKLOADS}
        },
        "provenance": {
            "commit": "0123456789abcdef"
        },
        "lost": lost((*rp.POLICIES, *engine)),
        "policy_views": policy_views((*rp.POLICIES, *engine)),
        "new_policies": {
            "engine":
            list(engine),
            "specs": {
                "ttdf": "ttdf",
                "race_rp": "race_rp:sketch=sketch-L16-P6.json",
                "sampled_lrfu": "sampled_lrfu"
            },
            "sketch": ({
                "observations": 3639507,
                "reused_within_horizon": 0.385884
            } if sketch else None),
            "tuning":
            tuning(present) if tune else None,
            "evictions":
            evictions((*rp.POLICIES, *engine)),
            "overhead":
            overhead(),
        },
    }


# --- The new policies --------------------------------------------------------


def test_three_new_policies_with_labels():
    assert rp.NEW_POLICIES == ALL_NEW
    assert rp.POLICY_LABELS["sampled_lrfu"] == "Sampled LRFU"
    assert rp.POLICY_LABELS["sampled_lrfu_tuned"] == "Sampled LRFU, tuned"


# --- mixed_policies, curve_policies ---------------------------------------


def test_mixed_policies_add_the_new_policies_with_engine_runs_in_order():
    assert rp.mixed_policies(summary(engine=())) == ["lru", "lfu", "fifo"]
    assert rp.mixed_policies(summary(engine=("race_rp", ))) == \
        ["lru", "lfu", "fifo", "race_rp"]
    s = summary()
    # The order is fixed, whatever order the summary lists them in.
    s["mixed_groups"]["all"] = dict(reversed(s["mixed_groups"]["all"].items()))
    assert rp.mixed_policies(s) == ["lru", "lfu", "fifo", "ttdf", "race_rp"]


def test_curve_policies_keep_only_complete_replay_curves():
    c = capacity_window(replay=("ttdf", ))
    c["curve"]["lfu"]["hit_rate"][1] = None  # one size missing
    assert rp.curve_policies(c, ["lru", "lfu", "ttdf", "race_rp", "nope"]) == \
        ["lru", "ttdf"]
    assert rp.curve_policies(c, []) == []


# --- Point 2 ----------------------------------------------------------------


def test_q2_view_covers_the_given_policies():
    s = summary()
    view = rp.q2_view(s, rp.mixed_policies(s))
    assert view["rows"] == ["all", *rp.GROUPS]
    assert list(view["values"]) == ["lru", "lfu", "fifo", "ttdf", "race_rp"]
    assert view["values"]["ttdf"]["all"]["mean"] == pytest.approx(0.57)
    assert view["values"]["race_rp"]["bailian"]["mean"] == \
        pytest.approx(0.555 - 0.3)
    assert view["ceiling"]["all"]["mean"] == pytest.approx(0.78)
    # Without a list, the baselines only.
    assert list(rp.q2_view(s)["values"]) == list(rp.POLICIES)


# --- Point 4 ----------------------------------------------------------------


def test_q4_view_defaults_to_the_baselines_and_belady():
    c = capacity_window(replay=NEW, engine=NEW)
    view = rp.q4_view(c)
    assert view["policies"] == ["lru", "lfu", "fifo", "belady"]
    assert list(view["curve"]) == view["policies"]
    # Engine markers of the new policies are not drawn without their line.
    assert {m["policy"] for m in view["engine"]} == {"lru", "lfu", "fifo"}


def test_q4_view_draws_the_named_policies_with_complete_curves():
    c = capacity_window(replay=("ttdf", ), engine=("ttdf", ))
    # A RACE-RP engine run with a replay value, but no RACE-RP curve.
    c["engine"].append({
        "seed": 2,
        "policy": "race_rp",
        "capacity": FIXED,
        "engine": 0.5,
        "replay": 0.5
    })
    view = rp.q4_view(c, ["lru", "lfu", "fifo", "ttdf", "race_rp"])
    assert view["policies"] == ["lru", "lfu", "fifo", "ttdf", "belady"]
    assert view["curve"]["ttdf"] == CURVES["ttdf"]
    marks = {(m["policy"], m["capacity"]): m for m in view["engine"]}
    assert ("race_rp", FIXED) not in marks
    ttdf = marks[("ttdf", FIXED)]
    assert ttdf["seeds"] == SEEDS
    assert ttdf["engine"] == pytest.approx(0.582)
    assert ttdf["replay"] == pytest.approx(0.58)
    # The 32K run without a replay value is left out; seed 2's stays.
    assert marks[("lru", 32768)]["seeds"] == [2]
    assert marks[("lru", 32768)]["engine"] == pytest.approx(0.395)
    assert set(view["labelDy"]) == set(view["policies"])
    assert view["yDomain"] == rp.y_range(
        [v for p in view["policies"]
         for v in CURVES[p]] + [m["engine"] for m in view["engine"]] + [0.8])


# --- Point 7 ----------------------------------------------------------------


@pytest.mark.parametrize("engine, replay", [((), ()), (NEW, ())])
def test_q7_view_is_none_without_a_new_policys_replay_curve(engine, replay):
    s = summary(engine=engine, replay=replay)
    assert rp.q7_view(s, [*rp.POLICIES, *NEW]) is None


def test_q7_view_is_none_without_per_workload_curves_or_a_window():
    s = summary()
    del s["capacity_window"]["groups"]
    assert rp.q7_view(s, [*rp.POLICIES, *NEW]) is None
    s = summary()
    del s["capacity_window"]
    assert rp.q7_view(s, [*rp.POLICIES, *NEW]) is None
    # Asked only for the baselines: nothing new to show.
    assert rp.q7_view(summary(), list(rp.POLICIES)) is None


def test_q7_view_panels_per_workload_with_the_ceiling():
    s = summary(replay=("race_rp", ))
    view = rp.q7_view(s, [*rp.POLICIES, *NEW])
    assert view["policies"] == ["lru", "lfu", "fifo", "race_rp"]
    assert view["capacities"] == CAPS and view["fixed"] == FIXED
    assert list(view["panels"]) == list(rp.GROUPS)
    for g in rp.GROUPS:
        panel = view["panels"][g]
        assert list(panel["curve"]) == view["policies"]
        assert panel["curve"]["race_rp"] == pytest.approx(
            [v + GROUP_OFFSET[g] for v in CURVES["race_rp"]])
        assert panel["ceiling"] == GROUP_CEILING[g]
        values = [v for p in view["policies"] for v in panel["curve"][p]]
        assert panel["yDomain"] == rp.y_range([*values, GROUP_CEILING[g]])
    # ShareGPT: 0.05 to 0.35 and its 0.60 ceiling.
    assert view["panels"]["sharegpt"]["yDomain"] == pytest.approx([0.0, 0.65])


def test_q7_view_evictions_per_engine_policy_and_workload():
    s = summary(engine=("ttdf", ))
    view = rp.q7_view(s, [*rp.POLICIES, *NEW])
    ev = view["evictions"]
    # Policies without engine runs have no evictions.
    assert list(ev) == ["lru", "lfu", "fifo", "ttdf"]
    for p in ev:
        assert list(ev[p]) == ["all", *rp.GROUPS]
    bailian = ev["ttdf"]["bailian"]
    assert bailian["mean"] == pytest.approx(360.0)
    assert (bailian["min"], bailian["max"]) == pytest.approx((359.0, 362.0))
    assert bailian["seeds"] == pytest.approx([359.0, 360.0, 362.0])
    assert bailian["regretted"] == 0.3
    assert ev["lru"]["all"]["mean"] == pytest.approx(517.0)
    # No evictions at all in the summary: an empty view.
    s["new_policies"]["evictions"] = None
    assert rp.q7_view(s, [*rp.POLICIES, *NEW])["evictions"] == {}
    del s["new_policies"]
    assert rp.q7_view(s, [*rp.POLICIES, *NEW])["evictions"] == {}


def test_q7_view_tuned_curves_only_when_complete():
    view = rp.q7_view(summary(), [*rp.POLICIES, *NEW])
    assert view["tuned"] == {"ttdf_tuned": CURVES["ttdf_tuned"]}
    s = summary()
    s["capacity_window"]["curve"]["ttdf_tuned"]["hit_rate"][0] = None
    assert rp.q7_view(s, [*rp.POLICIES, *NEW])["tuned"] == {}
    assert rp.q7_view(summary(tune=False), [*rp.POLICIES, *NEW])["tuned"] == {}


# --- build_data -------------------------------------------------------------


def test_build_data_adds_the_new_policies_to_the_mixed_views():
    s = summary()
    data = rp.build_data(s)
    mixed = ["lru", "lfu", "fifo", "ttdf", "race_rp"]
    assert data["policies"] == list(rp.POLICIES)
    assert data["mixedPolicies"] == mixed
    assert list(data["q1"]["curves"]) == mixed
    assert data["q1"]["curves"]["ttdf"] == s["mixed_intervals"]["ttdf"][
        "curve"]
    assert list(data["q2"]["values"]) == mixed
    assert data["q4"]["policies"] == [*mixed, "belady"]
    assert data["q7"]["policies"] == mixed
    assert data["policyLabels"]["race_rp"] == "RACE-RP"
    assert data["policyLabels"]["ttdf_tuned"] == "TTDF, tuned"
    # The alone views stay on the baselines.
    assert list(data["q5"]["values"]) == list(rp.POLICIES)


def test_build_data_with_sampled_lrfu():
    data = rp.build_data(summary(engine=ALL_NEW, replay=ALL_NEW))
    mixed = ["lru", "lfu", "fifo", *ALL_NEW]
    assert data["mixedPolicies"] == mixed
    assert list(data["q1"]["curves"]) == mixed
    assert list(data["q2"]["values"]) == mixed
    assert data["q4"]["policies"] == [*mixed, "belady"]
    assert data["q7"]["policies"] == mixed
    assert list(data["q7"]["evictions"]) == mixed
    assert data["q7"]["panels"]["ccbench"]["curve"]["sampled_lrfu"] == \
        pytest.approx([v + 0.2 for v in CURVES["sampled_lrfu"]])
    # Engine runs of Sampled LRFU only: the mixed views, no point 7.
    data = rp.build_data(summary(engine=("sampled_lrfu", ), replay=()))
    assert data["mixedPolicies"] == ["lru", "lfu", "fifo", "sampled_lrfu"]
    assert data["q7"] is None
    assert data["q4"]["policies"] == [*rp.POLICIES, "belady"]


def test_build_data_draws_replay_curves_before_engine_runs():
    # Replay curves for both new policies, engine runs for neither yet.
    data = rp.build_data(summary(engine=()))
    assert data["mixedPolicies"] == list(rp.POLICIES)
    assert list(data["q1"]["curves"]) == list(rp.POLICIES)
    assert data["q4"]["policies"] == [*rp.POLICIES, *NEW, "belady"]
    assert data["q7"]["policies"] == [*rp.POLICIES, *NEW]
    assert data["q7"]["evictions"].keys() == set(rp.POLICIES)


def test_build_data_without_new_policies_is_the_baseline_page():
    s = summary(engine=(), replay=(), tune=False, sketch=False)
    data = rp.build_data(s)
    assert data["mixedPolicies"] == list(rp.POLICIES)
    assert data["q4"]["policies"] == [*rp.POLICIES, "belady"]
    assert data["q7"] is None
    # Without the isolated runs, points 5 and 6 have no data.
    s2 = copy.deepcopy(s)
    del s2["isolation"]
    data2 = rp.build_data(s2)
    assert data2["q5"] is None and data2["q6"] is None
