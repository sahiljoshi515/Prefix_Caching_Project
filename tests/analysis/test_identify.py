"""Workload identification from label-free signals (spec 007, US2).

Toy plans use 2-token blocks, so every chain below can be followed by hand:
``[1, 2]`` is one block, ``[1, 2, 3]`` one full block and one partial.
"""
import math

import numpy as np
import pytest

from analysis import identify as idf
from client_simulator.schema import TokenRequest
from client_simulator.token_plan import write_plan

pytestmark = pytest.mark.m3

BS = 2


def R(rid, prompt, output, t, **meta):  # noqa: N802
    session = rid.rsplit(":", 1)[0]
    return TokenRequest(rid,
                        session,
                        int(rid.rsplit(":", 1)[1]),
                        list(prompt),
                        list(output),
                        arrival_s=t,
                        meta=meta)


# A three-turn session of `wa`, each turn's prompt extending the previous
# prompt and output, and an unrelated `wb` request that shares only block 1.
A0 = R("wa:s1:0", [1, 2, 3, 4], [5, 6], 0.0)
U0 = R("wb:s2:0", [1, 2, 20, 21], [22, 23], 5.0)
A1 = R("wa:s1:1", [1, 2, 3, 4, 5, 6, 7, 8], [9], 10.0)
A2 = R("wa:s1:2", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12], 30.0)


def by_id(rows):
    return {r["request_id"]: r for r in rows}


def test_families_are_vllm_chained_hashes():
    from vllm.core.block.prefix_caching_block import PrefixCachingBlock
    row = idf.request_features([R("wa:s1:0", [1, 2, 3, 4, 5], [6], 0.0)],
                               BS)[0]
    first = PrefixCachingBlock.hash_block_tokens(True, None, [1, 2])
    assert row["family_1"] == first
    assert row["family_2"] == PrefixCachingBlock.hash_block_tokens(
        False, first, [3, 4])
    # Tokens 5 and 6 fill a block only with the output: not a prompt block.
    assert row["prompt_blocks"] == 2 and row["family_8"] == row["family_2"]


def test_family_is_kth_block_or_last():
    assert idf.family([11, 12, 13], 2) == 12
    assert idf.family([11, 12, 13], 4) == 13
    assert idf.family([], 1) is None


def test_parent_matching_follows_the_session_not_block_one():
    rows = by_id(idf.request_features([A0, U0, A1, A2], BS))
    # U0 shares A0's block 1 but A0's chain (3 blocks) is not its prefix.
    assert rows["wb:s2:0"]["matched_depth"] == 1
    assert rows["wb:s2:0"]["inferred_parent"] is None
    assert rows["wb:s2:0"]["inferred_turn"] == 0
    # A1's prompt holds A0's prompt and output: 3 blocks seen, A0 parent.
    assert rows["wa:s1:1"]["matched_depth"] == 3
    assert rows["wa:s1:1"]["inferred_parent"] == "wa:s1:0"
    # A2's prompt holds A1's chain (4 full blocks; A1's token 9 is in a
    # partial block) and A0's: the longest, A1's, is the parent.
    assert rows["wa:s1:2"]["matched_depth"] == 4
    assert rows["wa:s1:2"]["inferred_parent"] == "wa:s1:1"


def test_turn_growth_age_and_family_popularity():
    rows = by_id(idf.request_features([A0, U0, A1, A2], BS))
    a0, a1, a2 = rows["wa:s1:0"], rows["wa:s1:1"], rows["wa:s1:2"]
    assert (a0["inferred_turn"], a1["inferred_turn"],
            a2["inferred_turn"]) == (0, 1, 2)
    assert a0["growth_tokens"] is None and a0["parent_age_s"] is None
    assert a1["growth_tokens"] == 8 - (4 + 2)
    assert a2["growth_tokens"] == 10 - (8 + 1)
    assert a1["parent_age_s"] == 10.0 and a2["parent_age_s"] == 20.0
    # Depth-2 family: block [3, 4] for the session, [20, 21] for U0.
    assert [
        a0["family_popularity"], a1["family_popularity"],
        a2["family_popularity"]
    ] == [0, 1, 2]
    assert rows["wb:s2:0"]["family_popularity"] == 0
    assert a0["time_since_family_s"] is None
    assert a2["time_since_family_s"] == 20.0
    assert a2["family_1"] == rows["wb:s2:0"]["family_1"]
    assert a2["family_2"] == a0["family_2"] != rows["wb:s2:0"]["family_2"]


def test_features_are_causal_in_planned_order():
    # Listed out of order: rows come back sorted by planned time, and a
    # request never sees one planned after it.
    rows = idf.request_features([A2, A1, U0, A0], BS)
    assert [r["request_id"]
            for r in rows] == ["wa:s1:0", "wb:s2:0", "wa:s1:1", "wa:s1:2"]
    alone = idf.request_features([A0], BS)[0]
    assert rows[0] == alone
    assert alone["matched_depth"] == 0 and alone["family_popularity"] == 0
    # meta.planned_s takes precedence over arrival_s: A0 planned after A1
    # is unseen by A1 and is not A1's parent.
    late = R("wa:s1:0", [1, 2, 3, 4], [5, 6], 0.0, planned_s=50.0)
    rows = idf.request_features([late, A1], BS)
    assert [r["request_id"] for r in rows] == ["wa:s1:1", "wa:s1:0"]
    assert rows[0]["matched_depth"] == 0
    assert rows[0]["inferred_parent"] is None
    assert rows[1]["matched_depth"] == 2  # A1's chain holds A0's prompt
    assert rows[1]["inferred_parent"] is None


def test_a_chain_of_seen_blocks_is_nobodys_parent():
    # V's whole chain is block [1, 2], already seen when it arrives: it is a
    # prefix of W's prompt but not W's parent.
    v = R("wb:s3:0", [1, 2], [], 6.0)
    w = R("wb:s4:0", [1, 2, 30, 31], [], 7.0)
    rows = by_id(idf.request_features([A0, v, w], BS))
    assert rows["wb:s4:0"]["matched_depth"] == 1
    assert rows["wb:s4:0"]["inferred_parent"] is None
    # The first request with a chain still counts.
    rows = by_id(idf.request_features([v, w], BS))
    assert rows["wb:s4:0"]["inferred_parent"] == "wb:s3:0"


def test_lookup_table_keeps_decisive_families():
    rows = ([{
        "family_1": 1,
        "workload": "wa"
    }] * 3 + [{
        "family_1": 2,
        "workload": w
    } for w in ("wa", "wb")] + [{
        "family_1": 3,
        "workload": "wa"
    }] * 9 + [{
        "family_1": 3,
        "workload": "wb"
    }, {
        "family_1": None,
        "workload": "wb"
    }])
    assert idf.lookup_table(rows, 1, 0.9) == {1: "wa", 3: "wa"}
    assert idf.lookup_table(rows, 1, 0.5) == {1: "wa", 2: "wa", 3: "wa"}


def test_cart_splits_a_separable_set():
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1, 2, 2])
    tree = idf.fit_tree(x, y, 3, max_depth=2, min_leaf=1)
    # Splits at 1.5 and 3.5 tie at the root; the lower threshold wins.
    assert tree["feature"] == 0 and tree["threshold"] == 1.5
    assert tree["right"]["threshold"] == 3.5
    assert idf.predict_tree(tree, [[0.2], [2.2], [4.9]]).tolist() == [0, 1, 2]
    # One level: the right leaf ties 2-2 and takes the lower class.
    stump = idf.fit_tree(x, y, 3, max_depth=1, min_leaf=1)
    assert idf.predict_tree(stump, [[0.2], [4.9]]).tolist() == [0, 1]
    assert stump["right"]["counts"] == [0, 2, 2]


def test_cart_needs_a_gain_and_leaf_size():
    xor = idf.fit_tree([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0], 2, 4, 1)
    assert "feature" not in xor  # no single split lowers the impurity
    small = idf.fit_tree([[0], [1], [2], [3]], [0, 0, 1, 1], 2, 4, 3)
    assert "feature" not in small  # each side would hold fewer than 3
    rules = idf.tree_rules(idf.fit_tree([[0], [1]], [0, 1], 2, 1, 1), ["x"],
                           ["wa", "wb"])
    assert rules == [
        "x <= 0.5 -> wa (1 training requests, 1.00 pure)",
        "x > 0.5 -> wb (1 training requests, 1.00 pure)"
    ]


def row(rid, f1, f2, prompt, t, parent=None, turn=0):
    """A `request_features` row with only what the ladder reads."""
    return {
        "request_id": rid,
        "session_id": rid.rsplit(":", 1)[0],
        "turn_index": turn,
        "workload": rid.split(":", 1)[0],
        "planned_s": t,
        "prompt_tokens": prompt,
        "family_1": f1,
        "family_2": f2,
        "family_4": f2,
        "family_8": f2,
        "matched_depth": 1,
        "inferred_parent": parent,
        "inferred_turn": turn,
        "parent_age_s": None,
        "growth_tokens": None,
        "family_popularity": 0,
    }


# Training: `wa` owns block-1 family 100; `wb` and `wc` share family 200,
# `wb` has short prompts and `wc` long ones, and each session has its own
# depth-2 family (one `wb` session's is 201). Sessions start in the order
# a1, b0, c0, b1, c1 | b2, c2, b3, c3, a2, so each time half (fold) has
# both `wb` and `wc`, and the tree learns from their 8 requests.
TRAIN = [row("wa:a1:0", 100, 1001, 50, 0), row("wa:a2:0", 100, 1002, 50, 9)]
for i in range(4):
    TRAIN.append(
        row(f"wb:b{i}:0", 200, 201 if i == 0 else 2010 + i, 10, 2 * i + 1))
    TRAIN.append(row(f"wc:c{i}:0", 200, 2020 + i, 1000, 2 * i + 2))
TEST = [
    row("wa:t1:0", 100, 9001, 50, 0),  # block 1 -> wa
    row("wb:t2:0", 200, 201, 10, 1),  # depth 2 -> wb
    row("wc:t3:0", 200, 9003, 1000, 2),  # tree -> wc
    # A later turn of t3 with a short prompt: the tree says wb, its parent
    # says wc.
    row("wc:t3:1", 200, 9003, 10, 3, parent="wc:t3:0", turn=1),
    row("wa:t5:0", 300, 9005, 1000, 4),  # unseen block 1; tree -> wc
]


def test_ladder_on_a_toy_plan():
    ladder = idf.fit_ladder(TRAIN, min_leaf=1)
    assert ladder.tables[1] == {100: "wa"}
    assert ladder.tables[2][201] == "wb"
    # Each `wa` request is resolved by the other half's block 1; the tree
    # learns from the 8 `wb` and `wc` requests the other half has not seen.
    assert ladder.tree["counts"] == [0, 4, 4]
    out = idf.apply_ladder(ladder, TEST)
    assert [(r["step"], r["label"]) for r in out] == [
        ("block1", "wa"),
        ("family2", "wb"),
        ("tree", "wc"),
        ("session", "wc"),
        ("tree", "wc"),
    ]
    assert out[3]["tree"] == "wb"
    report = idf.evaluate(out)
    steps = {s["step"]: s for s in report["steps"]}
    assert steps["block1"]["coverage"] == steps["block1"]["accuracy"] == 0.2
    assert steps["block1"]["precision"] == 1.0
    assert steps["family4"]["accuracy"] == 0.4
    assert steps["tree"]["coverage"] == 1.0
    assert steps["tree"]["accuracy"] == pytest.approx(0.6)
    assert steps["session"]["accuracy"] == pytest.approx(0.8)
    assert report["propagation"] == {
        "changed": 1,
        "made_right": 1,
        "made_wrong": 0
    }
    final = report["final"]
    assert final["recall"] == {"wa": 0.5, "wb": 1.0, "wc": 1.0}
    assert final["confusion"]["wa"] == {"wa": 1, "wc": 1}
    assert final["first_accuracy"] == 0.75 and final["later_accuracy"] == 1.0
    assert report["tree"]["later_accuracy"] == 0.0
    assert report["step_shares"]["wc"] == {"tree": 0.5, "session": 0.5}


def test_crossfit_and_ladder_labels():
    ladder = idf.fit_ladder(TRAIN, min_leaf=1)
    labels = idf.ladder_labels(ladder, TEST, step="tree")
    assert labels["wc:t3:1"] == "wb"
    assert idf.ladder_labels(ladder, TEST)["wc:t3:1"] == "wc"
    # Each half of TRAIN's sessions labelled by a ladder of the other half.
    assert set(idf.crossfit_labels(
        TRAIN, min_leaf=1)) == {r["request_id"]
                                for r in TRAIN}
    folds = idf.session_folds(TRAIN)
    assert sorted(folds.values()) == [0] * 5 + [1] * 5
    assert folds["wa:a1"] == 0 and folds["wa:a2"] == 1


def test_tree_features_mark_missing_parents():
    x = idf.tree_matrix([TEST[0]], idf.TREE_FEATURES)
    values = dict(zip(idf.TREE_FEATURES, x[0]))
    assert values["log_prompt_tokens"] == pytest.approx(math.log(51))
    assert values["log_growth"] == values["log_parent_age"] == -1.0
    without = idf.fit_ladder(TRAIN, timing=False, min_leaf=1)
    assert "log_parent_age" not in without.features


def test_each_characteristic_alone_and_together():
    train = [{**r, "time_since_family_s": None} for r in TRAIN]
    test = [{**r, "time_since_family_s": None} for r in TEST]
    report = idf.characteristics_report(train, {"t": test},
                                        depths=(1, 2, 3),
                                        min_leaf=1)
    assert report["classes"] == ["wa", "wb", "wc"]
    # wb and wc tie with 4 training requests each: the first name wins.
    assert report["majority"] == {"label": "wb", "accuracy": 0.2}
    models = report["models"]
    assert list(models) == [
        "prompt_length", "prefix_hashes", "reuse_history", "interarrival",
        "all", "all_but_timing"
    ]
    # Length alone: 50 -> wa, 10 -> wb, 1000 -> wc, so the short later turn
    # of t3 and the long wa request are wrong. One split cannot tell three
    # lengths apart, so cross-validation on TRAIN picks depth 2.
    length = models["prompt_length"]
    assert length["features"] == ["log_prompt_tokens"]
    assert length["depth"] == 2
    assert length["accuracy"] == pytest.approx(0.6)
    assert length["recall"] == {"wa": 0.5, "wb": 1.0, "wc": 0.5}
    assert length["balanced_accuracy"] == pytest.approx(2 / 3)
    assert length["token_accuracy"] == pytest.approx(1060 / 2070)
    assert length["confusion"]["wc"] == {"wc": 1, "wb": 1}
    assert length["first_accuracy"] == 0.75
    assert length["later_accuracy"] == 0.0
    # Hashes alone: in training, only wa's block 1 is known to the other
    # half of the sessions, so the tree learns "wa lookup -> wa, else wb"
    # (wb and wc tie); t1 and t2 are right.
    hashes = models["prefix_hashes"]
    assert hashes["features"] == [] and hashes["lookups"]
    assert hashes["accuracy"] == pytest.approx(0.4)
    assert models["all"]["accuracy"] == pytest.approx(0.6)
    assert models["all"]["per_test"]["t"]["accuracy"] == pytest.approx(0.6)
    assert "log_parent_age" not in models["all_but_timing"]["features"]
    assert "log_parent_age" in models["all"]["features"]


def test_lookup_inputs_are_one_hot_with_a_column_for_unresolved():
    tables = {1: {100: "wa"}, 2: {201: "wb"}, 4: {}}
    x = idf.lookup_inputs(TEST[:3], tables, ["wa", "wb", "wc"])
    assert x.tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    # Cross-fitted: a training session is looked up in the other half's
    # tables, so the a1 request (fold 0) is known from a2 (fold 1), and no
    # wb or wc session's own family can resolve it.
    x = idf.crossfit_lookup_inputs(TRAIN, ["wa", "wb", "wc"], 0.9)
    assert x[:, 0].tolist() == [1, 1] + [0] * 8
    assert x[2:, 3].tolist() == [1] * 8


def test_depth_is_picked_on_the_training_halves():
    x = np.array([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2, 0, 1, 2])
    folds = np.array([0, 0, 0, 1, 1, 1])
    assert idf.pick_depth(x, y, folds, 3, (1, 2, 3), 1) == 2


def test_grouped_rows_take_their_group_as_workload():
    rows = idf.grouped([{
        "workload": "bailian-api",
        "x": 1
    }], lambda w: w.split("-")[0])
    assert rows == [{"workload": "bailian", "x": 1}]


def write(path, requests):
    write_plan(requests, path / "plan.jsonl.gz", path / "responses.jsonl")
    return path


def test_identify_report_on_toy_plans(tmp_path):
    train = write(tmp_path / "train", [
        R("wa:s1:0", [1, 2, 3, 4], [5, 6], 0.0),
        R("wb:s2:0", [7, 8, 9, 10], [11], 1.0),
        R("wb:s3:0", [7, 8, 12, 13], [14], 2.0),
    ])
    test = write(tmp_path / "test", [
        R("wa:s9:0", [1, 2, 40, 41], [42], 0.0),
        R("wb:s8:0", [7, 8, 9, 10], [43], 1.0),
    ])
    report = idf.identify_report(train, [test],
                                 block_size=BS,
                                 min_leaf=1,
                                 group_of=lambda w: w)
    pooled = report["with_timing"]["pooled"]
    assert [s["accuracy"] for s in pooled["steps"]] == [1.0] * 5
    assert report["unseen_family_share"]["test"] == {
        "1": 0.0,
        "2": 0.5,
        "4": 0.5,
        "8": 0.5
    }
    # Depth 2: three singleton families, so no purity without them.
    families = report["families"]["train"]["2"]
    assert families["families"] == 3 and families["singleton_families"] == 3
    assert families["purity_without_singletons"] is None
    # Depth 1: wb's two requests share block [7, 8].
    families = report["families"]["train"]["1"]
    assert families["purity_without_singletons"] == 1.0
    assert report["parents"]["test"]["later_turns"] == 0
    chars = report["characteristics"]
    assert chars["classes"] == ["wa", "wb"]
    assert chars["models"]["all"]["requests"] == 2


def test_ccbench_is_split_by_whether_training_ran_the_task():
    from analysis.mix import ccbench_task
    rows = [{
        "workload": "ccbench",
        "session_id": s,
        "prompt_tokens": 1,
        "inferred_turn": 0
    } for s in ("ccbench:t1@m", "ccbench:t2@m")]
    known = {ccbench_task("ccbench:t1@m")}
    out = idf._model_scores(rows, ["ccbench", "bailian"], known)
    assert out["ccbench_known_requests"] == 1
    assert out["ccbench_known_accuracy"] == 1.0
    assert out["ccbench_new_accuracy"] == 0.0
    assert "ccbench_known_requests" not in idf._model_scores(
        rows, ["ccbench"] * 2)
