"""Signals that predict an idle block's reuse (spec 007, US2).

Every expected value is worked out by hand from the small inputs below.
"""
import json
import math

import numpy as np
import pytest

from analysis import signals as sg

pytestmark = pytest.mark.m3


def test_auc_counts_ties_as_half():
    assert sg.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    # Pairs (positive, negative): (2, 1) 1, (2, 2) 1/2, (3, 1) 1, (3, 2) 1.
    assert sg.auc([1, 2, 2, 3], [0, 0, 1, 1]) == 3.5 / 4
    assert sg.auc([0.5] * 4, [0, 1, 0, 1]) == 0.5
    assert sg.auc([0.9, 0.1], [0, 1]) == 0.0
    assert sg.auc([0.1, 0.2], [1, 1]) is None


def test_brier():
    assert sg.brier([0.5, 1.0], [1, 1]) == 0.125
    assert sg.brier([0.2, 0.2], [0, 1]) == pytest.approx((0.04 + 0.64) / 2)
    assert sg.brier([], []) is None


def test_logistic_regression_matches_group_log_odds():
    # One 0/1 feature: the unpenalized fit is each group's log-odds, 3 of 4
    # positives at x = 0 and 1 of 4 at x = 1.
    x = np.array([[0.0]] * 4 + [[1.0]] * 4)
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    beta = sg.fit_logistic(x, y, l2=0.0)
    assert beta[0] == pytest.approx(math.log(3))
    assert beta[1] == pytest.approx(-2 * math.log(3))
    assert sg.predict_logistic(beta,
                               [[0.0], [1.0]]) == pytest.approx([0.75, 0.25])


def test_logistic_regression_recovers_a_separator():
    # Separable along x = 1: the L2 term keeps the weights finite.
    x = np.array([[-1.0], [0.0], [2.0], [3.0]])
    beta = sg.fit_logistic(x, [0, 0, 1, 1], l2=1e-2)
    assert beta[1] > 0
    p = sg.predict_logistic(beta, [[0.5], [1.5]])
    assert p[0] < 0.5 < p[1]
    assert -beta[0] / beta[1] == pytest.approx(1.0)
    # One class only: the clipped rate, no weights.
    beta = sg.fit_logistic(x, [1, 1, 1, 1])
    assert beta[1] == 0.0
    assert sg.predict_logistic(beta, [[0.0]])[0] == pytest.approx(1 - 1e-6)


def frame(t_free, idle_s, next_lookup_t, **columns):
    n = len(t_free)
    out = {
        "t_free": np.array(t_free, dtype=float),
        "idle_s": np.array(idle_s, dtype=float),
        "next_lookup_t": np.array(next_lookup_t, dtype=float),
    }
    for name in sg.MODEL_COLUMNS:
        out[name] = np.asarray(columns.get(name, [0] * n))
    return out


# r0 revived at 10; r1 revived at 3; r2 never looked up again (censored at
# 100); r3 freed at 90, censored; r4 revived at 50; r5 revived at exactly 20.
TOY = frame([0, 0, 0, 90, 0, 0], [10, 3, 100, 10, 50, 20],
            [10, 3, math.nan, math.nan, 50, 20],
            owner_workload=[0, 0, 1, 1, 0, 1])


def test_landmark_targets():
    mask, y = sg.landmark(TOY, 5.0, 15.0, censor_t=100.0)
    # r1 left the pool before age 5; r3's window (95, 110] passes the end.
    assert mask.tolist() == [True, False, True, False, True, True]
    # Lookups in (5, 20]: r0 at 10 and r5 at 20 (the end is included).
    assert y.tolist() == [True, False, False, True]
    mask, y = sg.landmark(TOY, 0.0, 15.0, censor_t=100.0)
    assert mask.tolist() == [True, True, True, False, True, True]
    assert y.tolist() == [True, True, False, False, False]
    # Still idle at age a means idle_s > a: r0 (idle 10) is gone at 10.
    mask, _ = sg.landmark(TOY, 10.0, 15.0, censor_t=100.0)
    assert mask.tolist() == [False, False, True, False, True, True]


def test_stack_and_base_rates():
    rows = sg.stack([(TOY, 100.0)], (0.0, 5.0), 15.0)
    assert rows["age"].tolist() == [0.0] * 5 + [5.0] * 4
    assert rows["y"].tolist() == [True, True, False, False, False
                                  ] + [True, False, False, True]
    rates = sg.base_rates(rows, ["chat", "api"])
    assert rates["all"] == {"rows": 9, "reused": 4 / 9}
    # chat: r0, r1, r4 at age 0 and r0, r4 at 5.
    assert rates["chat"] == {"rows": 5, "reused": 3 / 5}
    assert rates["api"] == {"rows": 4, "reused": 1 / 4}


def test_encoder_drops_constant_columns():
    rows = {
        "age": np.array([5.0, 5.0, 5.0]),
        "owner_workload": np.array([0, 1, 1]),
        "hits_before": np.array([0.0, 1.0, 2.0]),
        "index": np.array([1.0, 1.0, 1.0]),
        "y": np.array([0, 1, 1]),
    }
    encoder = sg.fit_encoder(rows,
                             ("age", "owner_workload", "hits_before", "index"))
    x = sg.encode(encoder, rows)
    # No column for the single age or the constant index; two one-hot
    # columns and one standardized numeric column.
    assert x.shape == (3, 3)
    assert x[:, :2].tolist() == [[1, 0], [0, 1], [0, 1]]
    assert x[:, 2] == pytest.approx([-math.sqrt(1.5), 0, math.sqrt(1.5)])
    unseen = dict(rows, owner_workload=np.array([7, 7, 7]))
    assert sg.encode(encoder, unseen)[:, :2].tolist() == [[0, 0]] * 3


def run_frame(n_per_class):
    """Workload 0 always comes back 2 s after its free, workload 1 never;
    `index` is noise. Blocks are freed 1 s apart; the log ends at 1,000."""
    n = 2 * n_per_class
    workload = np.repeat([0, 1], n_per_class)
    t_free = np.arange(n, dtype=float)
    next_lookup = np.where(workload == 0, t_free + 2.0, np.nan)
    idle = np.where(workload == 0, 2.0, 1000.0 - t_free)
    return frame(t_free,
                 idle,
                 next_lookup,
                 owner_workload=workload,
                 inferred_class=workload,
                 index=np.log1p(np.arange(n) % 3).astype(float))


def test_score_models_on_a_known_signal():
    train = [(run_frame(20), 1000.0)]
    tests = {"s2": (run_frame(10), 1000.0)}
    out = sg.score_models(train,
                          tests, ["wa", "wb"],
                          landmarks=(0.0, 5.0),
                          horizons=(15.0, ),
                          feature_sets={
                              "idle_age": ("age", ),
                              "+oracle_workload": ("age", "owner_workload"),
                          })
    cells = out["models"]["+oracle_workload"]["H=15"]
    # At age 0 the workload separates reused from not perfectly.
    assert cells["a=0"]["auc"] == 1.0
    assert cells["a=0"]["by_run"]["s2"]["auc"] == 1.0
    assert cells["a=0"]["brier"] < 0.01
    # At one age the idle age is the same for every row: no signal.
    assert out["models"]["idle_age"]["H=15"]["a=0"]["auc"] == 0.5
    # At age 5 only workload 1 is still idle, never reused: one class.
    assert cells["a=5"]["auc"] is None
    assert out["rows"]["H=15"]["a=5"] == {"train": 20, "test": 10}
    # Pooled: age alone ranks age 0 (half reused) above age 5 (none).
    pooled = out["models"]["idle_age"]["H=15"]["pooled"]
    assert pooled["auc"] == pytest.approx(0.75)
    rates = out["base_rates"]["H=15"]["a=0"]["test"]
    assert rates["wa"] == {"rows": 10, "reused": 1.0}
    assert rates["wb"] == {"rows": 10, "reused": 0.0}


def test_load_residencies_and_owner_frame(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "t_free": [700.0, 650.0, 100.0, 800.0],
        "idle_s": [5.0, 10.0, 20.0, 1.0],
        "next_lookup_t": [705.0, math.nan, 120.0, 801.0],
        "owner_planned_s": [690.0, 640.0, 90.0, 1900.0],
        "hits_before": pa.array([0, 3, 1, 0], pa.int32()),
        "index": pa.array([0, 7, -1, 2], pa.int32()),
        "owner_req": ["wa:s1:0", "wb:s2:1", "wa:s1:0", None],
        "owner_workload": ["wa", "wb", "wa", None],
    })
    meta = {"w0_s": 600.0, "stop_s": 1800.0, "censor_t": 2000.0}
    table = table.replace_schema_metadata({"run": json.dumps(meta)})
    (tmp_path / "tables").mkdir()
    pq.write_table(table, tmp_path / "tables" / "residencies.parquet")
    residencies = sg.load_residencies(tmp_path)
    assert residencies.meta == meta
    assert residencies.strings["owner_req"] == ["wa:s1:0", "wb:s2:1"]
    assert residencies.columns["owner_req"].tolist() == [0, 1, 0, -1]
    features = {
        "wa:s1:0": {
            "inferred_turn": 0,
            "family_popularity": 3
        },
        "wb:s2:1": {
            "inferred_turn": 1,
            "family_popularity": 0
        },
    }
    labels = {"wa:s1:0": "wb", "wb:s2:1": "wb"}
    out = sg.owner_frame(residencies, features, labels, {"wa": 0, "wb": 1})
    # Row 2's owner was planned before W0 = 600; row 3 has no owner.
    assert out["t_free"].tolist() == [700.0, 650.0]
    assert out["owner_workload"].tolist() == [0, 1]
    assert out["inferred_class"].tolist() == [1, 1]
    assert out["inferred_turn"] == pytest.approx([0.0, math.log(2)])
    assert out["family_popularity"] == pytest.approx([math.log(4), 0.0])
    assert out["hits_before"] == pytest.approx([0.0, math.log(4)])
    assert out["index"] == pytest.approx([0.0, math.log(8)])


def write_run(run_dir, plan_dir, owners):
    """A ceiling run's residency table: 5 blocks per owner, freed at 0 to 4
    s; a `wa` owner's blocks come back 1 s after their free, a `wb`
    owner's never (censored at 100 s)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    for owner in owners:
        reused = owner.startswith("wa:")
        for i in range(5):
            rows.append({
                "t_free": float(i),
                "idle_s": 1.0 if reused else 100.0 - i,
                "next_lookup_t": i + 1.0 if reused else math.nan,
                "owner_planned_s": 0.0,
                "hits_before": 0,
                "index": i,
                "owner_req": owner,
                "owner_workload": owner.split(":", 1)[0],
            })
    meta = {
        "plan_dir": str(plan_dir),
        "block_size": 2,
        "censor_t": 100.0,
        "w0_s": 0.0,
        "stop_s": 1000.0,
    }
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {"run": json.dumps(meta)})
    (run_dir / "tables").mkdir(parents=True)
    pq.write_table(table, run_dir / "tables" / "residencies.parquet")
    return run_dir


def test_signals_report_end_to_end(tmp_path):
    from client_simulator.schema import TokenRequest
    from client_simulator.token_plan import write_plan

    def plan(path, prompts):
        write_plan([
            TokenRequest(rid, rid.rsplit(":", 1)[0], 0, p, [], arrival_s=t)
            for t, (rid, p) in enumerate(prompts.items())
        ], path / "plan.jsonl.gz", path / "responses.jsonl")
        return path

    # Two sessions per workload, alternating, so each time half of the
    # training sessions has both and the cross-fitted labels are right:
    # every `wa` prompt starts with block [1, 2], every `wb` with [5, 6].
    train = plan(
        tmp_path / "train", {
            "wa:s1:0": [1, 2, 3, 4],
            "wb:s2:0": [5, 6, 7, 8],
            "wa:s3:0": [1, 2, 9, 10],
            "wb:s4:0": [5, 6, 11, 12],
        })
    test = plan(tmp_path / "test", {
        "wa:s9:0": [1, 2, 13, 14],
        "wb:s8:0": [5, 6, 15, 16]
    })
    report = sg.signals_report(
        [write_run(tmp_path / "run1", train, sorted(train_ids(train)))],
        [write_run(tmp_path / "run2", test, ["wa:s9:0", "wb:s8:0"])],
        landmarks=(0.0, ),
        horizons=(10.0, ))
    assert report["residencies"]["run1"] == {
        "rows": 20,
        "inferred_class_accuracy": 1.0
    }
    assert report["residencies"]["run2"]["inferred_class_accuracy"] == 1.0
    cell = report["models"]["+inferred_class"]["H=10"]["a=0"]
    assert cell["auc"] == 1.0 and cell["by_run"]["run2"]["auc"] == 1.0
    assert report["models"]["idle_age"]["H=10"]["a=0"]["auc"] == 0.5
    assert report["rows"]["H=10"]["a=0"] == {"train": 20, "test": 10}
    assert report["base_rates"]["H=10"]["a=0"]["test"]["wa"] == {
        "rows": 5,
        "reused": 1.0
    }


def train_ids(plan_dir):
    from client_simulator.token_plan import read_plan
    return [r.request_id for r in read_plan(plan_dir / "plan.jsonl.gz")]
