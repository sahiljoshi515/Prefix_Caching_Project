"""Which workload sent a request, from what a server can see (spec 007).

The course asks: "How do I know a request belongs to which workload
type?" For an eviction policy a label matters only if it helps predict
reuse, which ``analysis/signals.py`` measures. This module measures how
accurately the workload can be told at all, from signals a server can
compute without any label::

    python -m analysis.identify --train data/processed/m3/balanced-human-s1 \\
        --test data/processed/m3/balanced-human-s{2,3,4} --out identify.json

**Features** (`request_features`) are computed causally, in planned-send
order, from a *ghost table*: every block hash the earlier requests produced
(each one's prompt and output tokens, full blocks only, chained with vLLM's
own block hash by `cachesim.block_hashes`). A request sees only requests
planned before it, each as if it had finished at once (the plan's
zero-latency send times). The table
holds no cache state: it remembers every block ever produced, whether or not
a real cache would still hold it. So the features are the same under every
eviction policy and capacity, one computation serves every run of a plan,
and a label inferred from them cannot feed back into the policy it informs.

Raw token ids and token statistics are never used: Bailian prompts are
synthetic random ids keyed by trace, which would make the task trivially
and falsely easy. Only block hashes, lengths, counts and times are used.
The times of ShareGPT and AgentBank requests come from the mix's pacing
inputs (bailian-chat's pauses, CC-Bench's tool latencies), so results are
given with and without the timing feature.

**The ladder** (`fit_ladder`, `apply_ladder`) is learned on one plan and
applied to others. Each step labels only the requests the earlier steps
leave unresolved:

1. block 1: the workload of the training requests with the same first block
   hash, if at least `min_share` of them agree (ShareGPT, AgentBank and
   CC-Bench all start with Qwen's default system prompt, so their first
   block is ambiguous and passes down);
2. families: the same rule at depth 2, then 4. Hashes are chained, so a
   family unseen at depth 2 is unseen at depth 4 too: depth 4 helps only
   where depth 2 is seen but mixed;
3. a CART tree (depth at most 4) on numeric features, for every request the
   lookups leave. It is fitted on the training requests that lookup tables
   learned on the other half of the training sessions leave, the kind of
   request it will see;
4. session propagation: a request the lookups leave takes its inferred
   parent's label (the parent is earlier, so already labelled) instead of
   the tree's. This is online and label-free, and it carries a session's
   first label through all of its turns.

`identify_report` scores the ladder after each step against the oracle
label (the request id's prefix, never an input), both whole and stopped
after the tree.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Set, Tuple)

import numpy as np

from analysis import characterize as ch
from analysis.cachesim import block_hashes
from client_simulator.schema import TokenRequest
from client_simulator.token_plan import read_plan

Row = Dict[str, Any]

FAMILY_DEPTHS = (1, 2, 4, 8)
# The ladder's lookup steps: depth -> step name.
LOOKUPS = {1: "block1", 2: "family2", 4: "family4"}
STEPS = tuple(LOOKUPS.values()) + ("tree", "session")
POPULARITY_DEPTH = 2


def _log(value: Optional[float]) -> float:
    """log(1 + value) of a measured value (negatives count as 0), or -1
    when there is none, so a tree can split "no parent" off."""
    if value is None:
        return -1.0
    return math.log1p(max(value, 0.0))


# The tree's inputs, computed from a `request_features` row.
TREE_FEATURES: Dict[str, Callable[[Row], float]] = {
    "log_prompt_tokens": lambda r: math.log1p(r["prompt_tokens"]),
    "inferred_turn": lambda r: float(r["inferred_turn"]),
    "log_growth": lambda r: _log(r["growth_tokens"]),
    "log_parent_age": lambda r: _log(r["parent_age_s"]),
    "matched_depth": lambda r: float(r["matched_depth"]),
    "log_family_popularity": lambda r: math.log1p(r["family_popularity"]),
}
TIMING_TREE_FEATURES = ("log_parent_age", )

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def send_time(request: TokenRequest) -> float:
    """When the plan sends `request` if requests took no time:
    ``meta["planned_s"]``, else its arrival time."""
    planned = request.meta.get("planned_s")
    return float(planned if planned is not None else request.arrival_s)


def family(hashes: Sequence[int], k: int) -> Optional[int]:
    """The depth-k prefix family: the hash of the k-th block, or of the
    last full block if there are fewer; None with no full block."""
    return hashes[min(k, len(hashes)) - 1] if hashes else None


def request_features(requests: Iterable[TokenRequest],
                     block_size: int = 16) -> List[Row]:
    """One row per request, in planned-send order (ties in plan order).

    Features, all label-free and causal:

    - ``prompt_tokens``;
    - ``family_1`` (the block-1 hash), ``family_2``, ``family_4`` and
      ``family_8`` (`family`);
    - ``matched_depth``: leading prompt blocks already in the ghost table;
    - ``inferred_parent``: the earlier request whose whole chain (prompt
      and output, full blocks) is the longest prefix of this prompt; in a
      conversation, the latest earlier turn. Only a request whose chain
      added a block to the ghost table counts: a chain repeated exactly
      names its first request, and a request made only of blocks seen
      before (say, a one-block chat that is all system prompt, a prefix of
      every later request with that system prompt) is nobody's parent.
      None if there is none;
    - ``inferred_turn``: the parent's plus 1, or 0;
    - ``parent_age_s`` (timing): this send time minus the parent's;
    - ``growth_tokens``: prompt tokens minus the parent's prompt and output
      tokens (at least -15 with 16-token blocks: the parent's partial last
      block is not in its chain);
    - ``family_popularity``: earlier requests with the same depth-2 family;
    - ``time_since_family_s`` (timing): since the latest of them.

    Parent features are None without a parent. ``request_id``,
    ``session_id``, ``turn_index`` and ``workload`` are the oracle labels,
    kept for scoring only.
    """
    requests = list(requests)
    order = sorted(range(len(requests)),
                   key=lambda i: (send_time(requests[i]), i))
    seen: Set[int] = set()  # the ghost table
    chain_end: Dict[int, int] = {}  # last hash of a chain -> latest row
    popularity: Counter = Counter()
    last_seen: Dict[int, float] = {}
    rows: List[Row] = []
    for i in order:
        r = requests[i]
        t = send_time(r)
        chain = block_hashes(r.prompt_token_ids + r.output_token_ids,
                             block_size)
        prompt = chain[:len(r.prompt_token_ids) // block_size]
        matched = 0
        while matched < len(prompt) and prompt[matched] in seen:
            matched += 1
        parent = None
        for j in range(matched - 1, -1, -1):
            if prompt[j] in chain_end:
                parent = rows[chain_end[prompt[j]]]
                break
        families = {k: family(prompt, k) for k in FAMILY_DEPTHS}
        key = families[POPULARITY_DEPTH]
        row = {
            "request_id":
            r.request_id,
            "session_id":
            r.session_id,
            "turn_index":
            r.turn_index,
            "workload":
            r.workload,
            "planned_s":
            t,
            "prompt_tokens":
            len(r.prompt_token_ids),
            "output_tokens":
            len(r.output_token_ids),
            "prompt_blocks":
            len(prompt),
            **{f"family_{k}": f
               for k, f in families.items()},
            "matched_depth":
            matched,
            "inferred_parent":
            parent["request_id"] if parent else None,
            "inferred_turn":
            parent["inferred_turn"] + 1 if parent else 0,
            "parent_age_s":
            t - parent["planned_s"] if parent else None,
            "growth_tokens":
            (len(r.prompt_token_ids) - parent["prompt_tokens"] -
             parent["output_tokens"]) if parent else None,
            "family_popularity":
            popularity[key] if key is not None else 0,
            "time_since_family_s":
            t - last_seen[key] if key in last_seen else None,
        }
        rows.append(row)
        if chain and chain[-1] not in seen:  # chained: the last is new
            chain_end[chain[-1]] = len(rows) - 1
        seen.update(chain)
        if key is not None:
            popularity[key] += 1
            last_seen[key] = t
    return rows


def plan_features(plan_dir: Path, block_size: int = 16) -> List[Row]:
    return request_features(read_plan(Path(plan_dir) / "plan.jsonl.gz"),
                            block_size)


# --------------------------------------------------------------------------
# CART tree (numpy only)
# --------------------------------------------------------------------------


def _gini(counts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    return 1.0 - (counts**2).sum(axis=-1) / np.maximum(sizes, 1)**2


def _best_split(x: np.ndarray, y: np.ndarray, n_classes: int,
                min_leaf: int) -> Optional[Tuple[int, float]]:
    """The split `x[:, f] <= threshold` with the lowest weighted Gini
    impurity; None if no split lowers it. Ties go to the first feature
    and the lowest threshold."""
    n = len(y)
    onehot = np.eye(n_classes)[y]
    parent = _gini(onehot.sum(axis=0), np.array(n))
    best, best_impurity = None, parent - 1e-12
    left_n = np.arange(1, n)
    for f in range(x.shape[1]):
        order = np.argsort(x[:, f], kind="stable")
        xs = x[order, f]
        below = np.cumsum(onehot[order], axis=0)
        left, right = below[:-1], below[-1] - below[:-1]
        impurity = (left_n * _gini(left, left_n) +
                    (n - left_n) * _gini(right, n - left_n)) / n
        valid = ((xs[:-1] < xs[1:]) & (left_n >= min_leaf) &
                 (n - left_n >= min_leaf))
        if not valid.any():
            continue
        impurity = np.where(valid, impurity, np.inf)
        i = int(np.argmin(impurity))
        if impurity[i] < best_impurity:
            best_impurity = impurity[i]
            best = (f, float((xs[i] + xs[i + 1]) / 2))
    return best


def fit_tree(x: np.ndarray,
             y: np.ndarray,
             n_classes: int,
             max_depth: int = 4,
             min_leaf: int = 5) -> Dict[str, Any]:
    """A CART classification tree: binary splits ``x[:, feature] <=
    threshold`` that most lower the Gini impurity, to `max_depth`, with
    at least `min_leaf` training rows per leaf.

    Nodes are JSON-ready dicts. Every node has ``counts`` (training rows per
    class) and ``label`` (the majority, the lowest class on a tie); an inner
    node also has ``feature``, ``threshold``, ``left`` and ``right``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=n_classes)
    node: Dict[str, Any] = {
        "counts": counts.tolist(),
        "label": int(counts.argmax())
    }
    if max_depth == 0 or counts.max() == len(y) or len(y) < 2 * min_leaf:
        return node
    split = _best_split(x, y, n_classes, min_leaf)
    if split is None:
        return node
    feature, threshold = split
    go_left = x[:, feature] <= threshold
    node.update(feature=feature,
                threshold=threshold,
                left=fit_tree(x[go_left], y[go_left], n_classes, max_depth - 1,
                              min_leaf),
                right=fit_tree(x[~go_left], y[~go_left], n_classes,
                               max_depth - 1, min_leaf))
    return node


def predict_tree(tree: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Each row's leaf label."""
    x = np.asarray(x, dtype=float)
    out = np.empty(len(x), dtype=int)

    def walk(node: Dict[str, Any], rows: np.ndarray) -> None:
        if "feature" not in node:
            out[rows] = node["label"]
            return
        go_left = x[rows, node["feature"]] <= node["threshold"]
        walk(node["left"], rows[go_left])
        walk(node["right"], rows[~go_left])

    walk(tree, np.arange(len(x)))
    return out


def tree_rules(tree: Dict[str, Any], features: Sequence[str],
               classes: Sequence[str]) -> List[str]:
    """The tree as one readable rule per leaf."""
    rules: List[str] = []

    def walk(node: Dict[str, Any], path: List[str]) -> None:
        if "feature" not in node:
            n = sum(node["counts"])
            share = node["counts"][node["label"]] / n if n else 0.0
            rules.append(f"{' and '.join(path) or 'always'} -> "
                         f"{classes[node['label']]} "
                         f"({n} training requests, {share:.2f} pure)")
            return
        name, threshold = features[node["feature"]], node["threshold"]
        walk(node["left"], path + [f"{name} <= {threshold:.3g}"])
        walk(node["right"], path + [f"{name} > {threshold:.3g}"])

    walk(tree, [])
    return rules


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def lookup_table(rows: Sequence[Row], k: int,
                 min_share: float) -> Dict[int, str]:
    """Depth-k family -> workload, for the families of `rows` whose most
    common workload holds at least `min_share` of their requests."""
    by_family: Dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        if r[f"family_{k}"] is not None:
            by_family[r[f"family_{k}"]][r["workload"]] += 1
    table = {}
    for f, counts in by_family.items():
        workload, n = counts.most_common(1)[0]
        if n >= min_share * sum(counts.values()):
            table[f] = workload
    return table


def _lookup(tables: Dict[int, Dict[int, str]],
            row: Row) -> Tuple[Optional[str], Optional[str]]:
    """The first lookup step that resolves `row`: (step, label)."""
    for k, step in LOOKUPS.items():
        label = tables[k].get(row[f"family_{k}"])
        if label is not None:
            return step, label
    return None, None


def tree_matrix(rows: Sequence[Row], features: Sequence[str]) -> np.ndarray:
    return np.array([[TREE_FEATURES[f](r) for f in features] for r in rows],
                    dtype=float).reshape(len(rows), len(features))


def session_folds(rows: Sequence[Row]) -> Dict[str, int]:
    """Session -> 0 for the earlier-starting half of the sessions, else 1:
    a split in time, as a later seed is a later trace window."""
    starts: Dict[str, float] = {}
    for r in rows:
        starts.setdefault(r["session_id"], r["planned_s"])
    ordered = sorted(starts, key=lambda s: (starts[s], s))
    half = len(ordered) // 2
    return {s: int(i >= half) for i, s in enumerate(ordered)}


@dataclass
class Ladder:
    tables: Dict[int, Dict[int, str]]
    tree: Dict[str, Any]
    features: Tuple[str, ...]
    classes: Tuple[str, ...]


def fit_ladder(train: Sequence[Row],
               timing: bool = True,
               min_share: float = 0.9,
               max_depth: int = 4,
               min_leaf: int = 5) -> Ladder:
    """Learn the ladder (module docstring) on `train`'s rows.

    The tree's training rows are the requests the lookups would leave if
    they had not seen the request's own sessions: each half of the sessions
    (`session_folds`) is looked up in tables learned on the other half.
    Fitted on every training request instead, the tree would learn from a
    mix the lookups never pass it.
    """
    features = tuple(f for f in TREE_FEATURES
                     if timing or f not in TIMING_TREE_FEATURES)
    classes = tuple(sorted({r["workload"] for r in train}))
    folds = session_folds(train)
    residual: List[Row] = []
    for fold in (0, 1):
        other = [r for r in train if folds[r["session_id"]] != fold]
        tables = {k: lookup_table(other, k, min_share) for k in LOOKUPS}
        residual += [
            r for r in train
            if folds[r["session_id"]] == fold and _lookup(tables, r)[1] is None
        ]
    if not residual:  # every request resolved: fit on all of them
        residual = list(train)
    index = {c: i for i, c in enumerate(classes)}
    tree = fit_tree(tree_matrix(residual, features),
                    np.array([index[r["workload"]] for r in residual]),
                    len(classes), max_depth, min_leaf)
    return Ladder({k: lookup_table(train, k, min_share)
                   for k in LOOKUPS}, tree, features, classes)


def apply_ladder(ladder: Ladder, rows: Sequence[Row]) -> List[Row]:
    """Label `rows` (in planned-send order) online.

    Each output row has ``lookup_step`` and ``lookup`` (the lookup that
    resolved it, else None), ``tree`` (the tree's label for a row no lookup
    resolved), ``label`` (the final label) and ``step`` (the step that gave
    it), plus the row's ``request_id``, ``session_id``, ``workload`` and
    ``inferred_turn`` for scoring.
    """
    trees = predict_tree(ladder.tree, tree_matrix(rows, ladder.features))
    final: Dict[str, str] = {}
    out: List[Row] = []
    for r, leaf in zip(rows, trees):
        lookup_step, lookup = _lookup(ladder.tables, r)
        tree = None if lookup else ladder.classes[leaf]
        parent = final.get(r["inferred_parent"]) if tree else None
        if lookup:
            step, label = lookup_step, lookup
        elif parent:
            step, label = "session", parent
        else:
            step, label = "tree", tree
        final[r["request_id"]] = label
        out.append({
            "request_id": r["request_id"],
            "session_id": r["session_id"],
            "workload": r["workload"],
            "inferred_turn": r["inferred_turn"],
            "lookup_step": lookup_step,
            "lookup": lookup,
            "tree": tree,
            "label": label,
            "step": step,
        })
    return out


def ladder_labels(ladder: Ladder,
                  rows: Sequence[Row],
                  step: str = "session") -> Dict[str, str]:
    """Request id -> label, running the ladder up to `step` ("tree" stops
    before session propagation)."""
    return {
        r["request_id"]: _label_after(r, step)
        for r in apply_ladder(ladder, rows)
    }


def crossfit_labels(rows: Sequence[Row],
                    step: str = "session",
                    **kwargs: Any) -> Dict[str, str]:
    """Out-of-sample ladder labels for a training plan's own requests: each
    half of the sessions (`session_folds`) is labelled by a ladder fitted on
    the other half. In-sample, the lookups would know every family."""
    folds = session_folds(rows)
    fold_of = {r["request_id"]: folds[r["session_id"]] for r in rows}
    labels: Dict[str, str] = {}
    for fold in (0, 1):
        ladder = fit_ladder(
            [r for r in rows if folds[r["session_id"]] != fold], **kwargs)
        for rid, label in ladder_labels(ladder, rows, step).items():
            if fold_of[rid] == fold:
                labels[rid] = label
    return labels


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _label_after(row: Row, step: str) -> Optional[str]:
    """The row's label once the ladder has run up to `step`."""
    if step == "session":
        return row["label"]
    if step == "tree":
        return row["lookup"] or row["tree"]
    position = STEPS.index(step)
    if row["lookup_step"] and STEPS.index(row["lookup_step"]) <= position:
        return row["lookup"]
    return None


def _share(part: int, whole: int) -> Optional[float]:
    return part / whole if whole else None


def _accuracy(rows: Sequence[Row], step: str = "session") -> Optional[float]:
    return _share(sum(_label_after(r, step) == r["workload"] for r in rows),
                  len(rows))


def _scores(rows: Sequence[Row], step: str,
            known_tasks: Set[str]) -> Dict[str, Any]:
    """Accuracy, recall, confusion (true -> predicted -> requests), the
    first-request split and CC-Bench, for the labels after `step`."""
    from analysis.mix import ccbench_task
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        confusion[r["workload"]][_label_after(r, step)] += 1
    ccbench = [r for r in rows if r["workload"] == "ccbench"]
    repeated = [ccbench_task(r["session_id"]) in known_tasks for r in ccbench]
    return {
        "accuracy":
        _accuracy(rows, step),
        "recall": {
            w: _share(c[w], sum(c.values()))
            for w, c in sorted(confusion.items())
        },
        "confusion": {w: dict(c)
                      for w, c in sorted(confusion.items())},
        "first_accuracy":
        _accuracy([r for r in rows if r["inferred_turn"] == 0], step),
        "later_accuracy":
        _accuracy([r for r in rows if r["inferred_turn"] > 0], step),
        "ccbench_accuracy":
        _accuracy(ccbench, step),
        "ccbench_known_task_requests":
        sum(repeated),
        "ccbench_known_task_accuracy":
        _accuracy([r for r, k in zip(ccbench, repeated) if k], step),
        "ccbench_new_task_accuracy":
        _accuracy([r for r, k in zip(ccbench, repeated) if not k], step),
        "accuracy_without_ccbench":
        _accuracy([r for r in rows if r["workload"] != "ccbench"], step),
    }


def evaluate(labelled: Sequence[Row],
             known_tasks: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Score `apply_ladder`'s output against the oracle workload.

    - ``steps``: after each step, the share of requests labelled
      (``coverage``), the share labelled right (``accuracy``: an unlabelled
      request counts as wrong) and the share right among those labelled
      (``precision``);
    - ``step_shares``: per workload, the share of its requests whose final
      label came from each step;
    - ``final`` (the whole ladder) and ``tree`` (stopping after the tree):
      accuracy, recall per workload, confusion matrix, accuracy of first
      requests (inferred turn 0) and later ones, and CC-Bench alone (74
      tasks, repeated across seeds), split by whether the training plan ran
      the same task (`known_tasks`), with every other workload's accuracy;
    - ``propagation``: requests whose label propagation changed, and how
      many of those it made right or wrong.
    """
    n = len(labelled)
    steps = []
    for step in STEPS:
        labels = [_label_after(r, step) for r in labelled]
        covered = sum(label is not None for label in labels)
        right = sum(label == r["workload"]
                    for label, r in zip(labels, labelled))
        steps.append({
            "step": step,
            "coverage": _share(covered, n),
            "accuracy": _share(right, n),
            "precision": _share(right, covered),
        })
    by_step: Dict[str, Counter] = defaultdict(Counter)
    for r in labelled:
        by_step[r["workload"]][r["step"]] += 1
    changed = [
        r for r in labelled
        if r["step"] == "session" and r["label"] != r["tree"]
    ]
    return {
        "requests": n,
        "first_requests": sum(r["inferred_turn"] == 0 for r in labelled),
        "steps": steps,
        "step_shares": {
            w: {s: c[s] / sum(c.values())
                for s in STEPS if c[s]}
            for w, c in sorted(by_step.items())
        },
        "final": _scores(labelled, "session", known_tasks or set()),
        "tree": _scores(labelled, "tree", known_tasks or set()),
        "propagation": {
            "changed": len(changed),
            "made_right": sum(r["label"] == r["workload"] for r in changed),
            "made_wrong": sum(r["tree"] == r["workload"] for r in changed),
        },
    }


def family_report(rows: Sequence[Row]) -> Dict[str, Any]:
    """Within one plan, `characterize.family_purity` at each depth (with
    and without singleton families, which are pure by construction).
    Requests with no full block have no family and are left out."""
    return {
        str(k): ch.family_purity({
            r["request_id"]: r[f"family_{k}"]
            for r in rows if r[f"family_{k}"] is not None
        })
        for k in FAMILY_DEPTHS
    }


def unseen_share(train: Sequence[Row], test: Sequence[Row]) -> Dict[str, Any]:
    """Per depth, the share of `test` requests whose family no `train`
    request has (no full block counts as unseen)."""
    out = {}
    for k in FAMILY_DEPTHS:
        known = {r[f"family_{k}"] for r in train} - {None}
        out[str(k)] = _share(sum(r[f"family_{k}"] not in known for r in test),
                             len(test))
    return out


def parent_check(rows: Sequence[Row]) -> Dict[str, Any]:
    """How often the inferred parent is the session's previous turn (the
    oracle), over requests after a session's first."""
    previous = {(r["session_id"], r["turn_index"]): r["request_id"]
                for r in rows}
    later = [r for r in rows if r["turn_index"] > 0]
    found = [r for r in later if r["inferred_parent"] is not None]
    return {
        "later_turns":
        len(later),
        "parent_found":
        _share(len(found), len(later)),
        "parent_is_previous_turn":
        _share(
            sum(r["inferred_parent"] == previous.get((r["session_id"],
                                                      r["turn_index"] - 1))
                for r in later), len(later)),
        "inferred_turn_is_turn":
        _share(sum(r["inferred_turn"] == r["turn_index"] for r in rows),
               len(rows)),
        "first_turns_with_parent":
        sum(r["turn_index"] == 0 and r["inferred_parent"] is not None
            for r in rows),
    }


def _pool(per_test: Dict[str, List[Row]]) -> List[Row]:
    return [r for rows in per_test.values() for r in rows]


def identify_report(train_plan_dir: Path,
                    test_plan_dirs: Sequence[Path],
                    block_size: int = 16,
                    min_share: float = 0.9,
                    max_depth: int = 4,
                    min_leaf: int = 5) -> Dict[str, Any]:
    """The ladder learned on one plan and scored on others, with and
    without the timing feature; family purity, unseen families and the
    parent check per plan."""
    from analysis.mix import ccbench_task
    train = plan_features(train_plan_dir, block_size)
    tests = {
        Path(d).name: plan_features(d, block_size)
        for d in test_plan_dirs
    }
    known_tasks = {
        ccbench_task(r["session_id"])
        for r in train if r["workload"] == "ccbench"
    }
    report: Dict[str, Any] = {
        "train_plan": str(train_plan_dir),
        "test_plans": [str(d) for d in test_plan_dirs],
        "settings": {
            "block_size": block_size,
            "min_share": min_share,
            "max_depth": max_depth,
            "min_leaf": min_leaf,
            "lookup_depths": list(LOOKUPS),
            "tree_features": list(TREE_FEATURES),
            "timing_features": list(TIMING_TREE_FEATURES),
        },
        "families": {
            Path(train_plan_dir).name: family_report(train),
            **{name: family_report(rows)
               for name, rows in tests.items()},
        },
        "unseen_family_share":
        {name: unseen_share(train, rows)
         for name, rows in tests.items()},
        "parents": {
            Path(train_plan_dir).name: parent_check(train),
            **{name: parent_check(rows)
               for name, rows in tests.items()},
        },
    }
    for timing in (True, False):
        ladder = fit_ladder(train, timing, min_share, max_depth, min_leaf)
        labelled = {
            name: apply_ladder(ladder, rows)
            for name, rows in tests.items()
        }
        report["with_timing" if timing else "without_timing"] = {
            "tree_features": list(ladder.features),
            "lookup_families":
            {LOOKUPS[k]: len(t)
             for k, t in ladder.tables.items()},
            "tree": tree_rules(ladder.tree, ladder.features, ladder.classes),
            "tests": {
                name: evaluate(rows, known_tasks)
                for name, rows in labelled.items()
            },
            "pooled": evaluate(_pool(labelled), known_tasks),
        }
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    for variant in ("with_timing", "without_timing"):
        pooled = report[variant]["pooled"]
        steps = ", ".join(f"{s['step']} {s['accuracy']:.3f}"
                          for s in pooled["steps"])
        print(f"{variant}: accuracy after each step: {steps}")
        for name in ("tree", "final"):
            scores = pooled[name]
            recall = ", ".join(f"{w} {r:.2f}"
                               for w, r in scores["recall"].items())
            print(f"  {name}: recall {recall}; without CC-Bench "
                  f"{scores['accuracy_without_ccbench']:.3f}, CC-Bench "
                  f"{scores['ccbench_accuracy']:.3f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--min-share", type=float, default=0.9)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--out", type=Path, help="JSON file to write")
    args = parser.parse_args(argv)
    report = identify_report(args.train, args.test, args.block_size,
                             args.min_share, args.max_depth, args.min_leaf)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    _print_summary(report)


if __name__ == "__main__":
    main()
