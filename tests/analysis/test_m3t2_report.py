"""The results page's chart data and number formats (spec 007 FR-011)."""
from typing import Any, Dict

import pytest

from analysis import m3t2_report as rp

pytestmark = pytest.mark.m3


def test_number_formats():
    assert rp.pct(0.5581, 1) == "55.8%"
    assert rp.pct(None) == "–"
    assert rp.pts(0.0123) == "1.2 points"
    assert rp.pts(-0.01) == "1.0 point"
    assert rp.secs(0.17) == "170 ms"
    assert rp.secs(5.2) == "5.2 s"
    assert rp.secs(30) == "30 s"
    assert rp.secs(60) == "1 min"
    assert rp.secs(68.5) == "1.1 min"
    assert rp.secs(1200) == "20 min"
    assert rp.tokens(131072) == "128K" and rp.tokens(4194304) == "4M"


def test_end_labels_are_pushed_apart():
    # On a 0.54-tall axis, 0.64 and 0.62 are 0.02 x 270 / 0.54 = 10 px
    # apart, under the 13 px gap, so the lower one moves 3 px down; 0.77 is
    # far above both, and a line with no end value gets no label.
    dy = rp.label_offsets({
        "lru": 0.64,
        "lfu": 0.62,
        "belady": 0.77,
        "x": None
    }, 0.54)
    assert dy == {"belady": 0, "lru": 0, "lfu": 3}


def test_y_range_rounds_out_to_five_points():
    assert rp.y_range([0.407, 0.781, None]) == [0.35, 0.85]
    assert rp.y_range([0.01, 0.99]) == [0.0, 1.0]


def test_identification_view_orders_the_confusion_by_workload():
    models: Dict[str, Dict[str, Any]] = {
        k: {
            "accuracy": 0.5,
            "balanced_accuracy": 0.4,
            "token_accuracy": 0.3
        }
        for k in rp.MODELS
    }
    models["all"]["confusion"] = {
        "sharegpt": {
            "sharegpt": 3,
            "agentbank": 1
        },
        "bailian": {
            "bailian": 2
        },
    }
    view = rp.q3_view({
        "models": models,
        "majority": {
            "label": "bailian",
            "accuracy": 0.57
        }
    })
    assert [m["id"] for m in view["models"]] == list(rp.MODELS)
    c = view["confusion"]
    assert c["classes"] == ["sharegpt", "agentbank", "ccbench", "bailian"]
    assert c["counts"][0] == [3, 1, 0, 0]
    assert c["share"][0] == [0.75, 0.25, 0.0, 0.0]
    assert c["share"][2] == [0.0, 0.0, 0.0, 0.0]  # no CC-Bench requests


def test_alone_against_sharing_view():

    def spread(v):
        return {"mean": v, "min": v, "max": v}

    rows = {
        g: {
            "ceiling": spread(0.9),
            **{
                p: {
                    "alone": spread(0.8),
                    "mixed": spread(0.5),
                    "verdict": "higher"
                }
                for p in rp.POLICIES
            },
        }
        for g in rp.GROUPS
    }
    view = rp.q6_view({"rows": rows})
    assert len(view["rows"]) == 12
    first = view["rows"][0]
    assert (first["group"], first["policy"]) == ("sharegpt", "lru")
    assert first["agree"] == "yes: lower when sharing"
    assert view["ceiling"]["bailian"] == 0.9
