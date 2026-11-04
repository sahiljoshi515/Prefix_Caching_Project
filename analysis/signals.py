"""Which signals predict that an idle cached block is read again soon
(spec 007; input to the eviction policy's design).

An evictor chooses among the blocks in the free pool, and the best victim
is a block that will not be looked up again soon. So a signal is worth
having if it ranks idle blocks by how soon they come back. This module
measures that on runs that never evict (the "ceiling"), where a block's
next lookup is its own demand, the same under any policy::

    python -m analysis.signals --train results/dev/sim-s1-ceiling \\
        --test <ceiling runs of seeds 2 to 4> --out signals.json

**Landmarks.** A policy only ever looks at blocks that have already
waited. At idle age ``a`` (0, 5, 30 and 120 s), the rows are the
residencies (``analysis/tables.py``) still in the free pool at ``t_free +
a`` (``idle_s > a``: not revived, evicted or censored by then) whose next
``H`` seconds the log covers (``t_free + a + H <= censor_t``). The target is
1 if the block's hash is looked up in ``(t_free + a, t_free + a + H]``, for
H of 15, 60 and 300 s. A row counts if its owner (the request whose free put
the block in the pool) was planned in the analysis window ``[w0_s,
stop_s)``.

**Signals.**

- ``age``, the idle age, pooled across landmarks: what LRU sees;
- ``owner_workload``: the owner's workload (oracle);
- ``inferred_class``: the owner's label from ``analysis/identify.py``'s
  ladder learned on the training plan, stopping after the tree (which beats
  session propagation on the training seed). A training run gets
  cross-fitted labels (`identify.crossfit_labels`), so the model does not
  learn from labels that already know their own sessions;
- ``inferred_turn``: the owner's inferred turn;
- ``hits_before``: hits since the block was computed, what LFU sees;
- ``index``: the block's position in the owner's chain;
- ``family_popularity``: earlier requests in the owner's depth-2 family.

Numeric signals enter as ``log(1 + x)``, standardized on the training rows;
the idle age and the labels as one-hot columns.

**Models.** A logistic regression per feature set (`FEATURE_SETS`), fitted
on the training runs (seed 1) and scored on the test runs (seeds 2 to 4)
by AUC and Brier score, at each landmark and with every landmark pooled.
AUC is the chance that a block reused within H ranks above one that is not
(0.5: no signal). The Brier score is the mean squared error of the
predicted probability (lower is better). Within one landmark every row has
the same idle age, so "idle age only" scores 0.5 there: LRU's order carries
no information among blocks of one age. Base rates per workload and
landmark make the effect readable: "of chat blocks idle for 30 s, X% come
back within 60 s".
"""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis import identify

Frame = Dict[str, np.ndarray]

LANDMARKS_S = (0.0, 5.0, 30.0, 120.0)
HORIZONS_S = (15.0, 60.0, 300.0)
# One-hot signals (integer codes; -1 is unknown) and log(1 + x) signals.
CATEGORICAL = ("age", "owner_workload", "inferred_class")
NUMERIC = ("inferred_turn", "hits_before", "index", "family_popularity")
MODEL_COLUMNS = CATEGORICAL[1:] + NUMERIC
LABEL_FREE = ("age", "inferred_class", "inferred_turn", "hits_before", "index",
              "family_popularity")
FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "idle_age": ("age", ),
    "+oracle_workload": ("age", "owner_workload"),
    "+inferred_class": ("age", "inferred_class"),
    "+inferred_turn": ("age", "inferred_turn"),
    "+hits_before": ("age", "hits_before"),
    "+index": ("age", "index"),
    "+family_popularity": ("age", "family_popularity"),
    "label_free": LABEL_FREE,
    "label_free+oracle": LABEL_FREE + ("owner_workload", ),
}
STRING_COLUMNS = ("owner_req", "owner_workload")
COLUMNS = ("t_free", "idle_s", "next_lookup_t", "owner_planned_s",
           "hits_before", "index") + STRING_COLUMNS
W0_S = 600.0
CHUNK_ROWS = 1 << 20

# --------------------------------------------------------------------------
# Scores and the model
# --------------------------------------------------------------------------


def auc(scores: Sequence[float], y: Sequence[bool]) -> Optional[float]:
    """Rank-based AUC (Mann-Whitney): the chance that a random positive
    scores above a random negative, ties counting half. None without both
    classes."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=bool)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    _, inverse, counts = np.unique(scores,
                                   return_inverse=True,
                                   return_counts=True)
    # The average 1-based rank of each distinct score.
    below = np.cumsum(counts) - counts
    ranks = (below + (counts + 1) / 2.0)[inverse]
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) /
                 (positives * negatives))


def brier(p: Sequence[float], y: Sequence[bool]) -> Optional[float]:
    """Mean squared error of predicted probabilities; None for no rows."""
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return None
    return float(np.mean((p - np.asarray(y, dtype=float))**2))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def fit_logistic(x: np.ndarray,
                 y: Sequence[bool],
                 l2: float = 1e-4,
                 max_iter: int = 50,
                 tol: float = 1e-8) -> np.ndarray:
    """Logistic regression by Newton's method (IRLS) on the mean log-loss
    plus ``l2 / 2 * |w|^2`` (the intercept is not penalized).

    Returns ``[intercept, *weights]``. With rows of one class only, the
    weights are 0 and the intercept is the log-odds of that class's rate,
    clipped to [1e-6, 1 - 1e-6].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = x.shape
    rate = min(max(float(y.mean()) if n else 0.5, 1e-6), 1 - 1e-6)
    beta = np.zeros(p + 1)
    beta[0] = math.log(rate / (1 - rate))
    if n == 0 or y.min() == y.max():
        return beta
    design = np.hstack([np.ones((n, 1)), x])
    penalty = np.full(p + 1, l2)
    penalty[0] = 0.0
    for _ in range(max_iter):
        prob = _sigmoid(design @ beta)
        gradient = design.T @ (prob - y) / n + penalty * beta
        hessian = ((design.T * (prob * (1 - prob))) @ design / n +
                   np.diag(penalty))
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def predict_logistic(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _sigmoid(beta[0] + np.asarray(x, dtype=float) @ beta[1:])


@dataclass
class Encoder:
    """The one-hot levels of each categorical feature and the mean and
    standard deviation of each numeric one, from training rows."""
    features: Tuple[str, ...]
    levels: Dict[str, np.ndarray]
    mean: Dict[str, float]
    std: Dict[str, float]


def fit_encoder(frame: Frame, features: Sequence[str]) -> Encoder:
    """Levels and scales of `features`. A categorical feature with one level
    (the idle age within one landmark) and a numeric one with no spread get
    no column."""
    levels, mean, std = {}, {}, {}
    for f in features:
        values = frame[f]
        if f in CATEGORICAL:
            found = np.unique(values)
            levels[f] = found if len(found) > 1 else found[:0]
        else:
            mean[f] = float(values.mean()) if len(values) else 0.0
            std[f] = float(values.std()) if len(values) else 0.0
    return Encoder(tuple(features), levels, mean, std)


def encode(encoder: Encoder, frame: Frame) -> np.ndarray:
    """The design matrix: one 0/1 column per training level (a level unseen
    in training gets none), then each standardized numeric feature."""
    n = len(frame["y"])
    columns: List[np.ndarray] = []
    for f in encoder.features:
        if f in CATEGORICAL:
            columns += [(frame[f] == level).astype(float)
                        for level in encoder.levels[f]]
        elif encoder.std[f] > 0:
            columns.append((frame[f] - encoder.mean[f]) / encoder.std[f])
    return np.column_stack(columns) if columns else np.zeros((n, 0))


def predict(encoder: Encoder, beta: np.ndarray, frame: Frame) -> np.ndarray:
    """Predicted probabilities, `CHUNK_ROWS` rows at a time."""
    n = len(frame["y"])
    out = np.empty(n)
    for lo in range(0, n, CHUNK_ROWS):
        part = {k: v[lo:lo + CHUNK_ROWS] for k, v in frame.items()}
        out[lo:lo + CHUNK_ROWS] = predict_logistic(beta, encode(encoder, part))
    return out


# --------------------------------------------------------------------------
# Landmark rows
# --------------------------------------------------------------------------


def landmark(frame: Frame, age: float, horizon: float,
             censor_t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Which residencies are at risk at idle age `age` with their whole
    horizon in the log, and for those, whether the hash is looked up within
    `horizon` seconds after that age (module docstring)."""
    start = frame["t_free"] + age
    at_risk = (frame["idle_s"] > age) & (start + horizon <= censor_t)
    with np.errstate(invalid="ignore"):  # NaN: never looked up again
        reused = ((frame["next_lookup_t"] > start) &
                  (frame["next_lookup_t"] <= start + horizon))
    return at_risk, reused[at_risk]


def stack(frames: Sequence[Tuple[Frame, float]], ages: Sequence[float],
          horizon: float) -> Frame:
    """The landmark rows of every (frame, censor time) at each of `ages`:
    the model columns, the age, and the target ``y``."""
    parts: List[Frame] = []
    for frame, censor_t in frames:
        for age in ages:
            mask, y = landmark(frame, age, horizon, censor_t)
            part = {k: frame[k][mask] for k in MODEL_COLUMNS}
            part["age"] = np.full(len(y), age, dtype=np.float32)
            part["y"] = y
            parts.append(part)
    return {
        k: np.concatenate([p[k] for p in parts])
        for k in (*MODEL_COLUMNS, "age", "y")
    }


def _subsample(frame: Frame, limit: Optional[int], seed: int = 0) -> Frame:
    n = len(frame["y"])
    if limit is None or n <= limit:
        return frame
    keep = np.sort(np.random.default_rng(seed).choice(n, limit, False))
    return {k: v[keep] for k, v in frame.items()}


def _mean(values: np.ndarray) -> Optional[float]:
    return float(np.mean(values)) if len(values) else None


def base_rates(rows: Frame, names: Sequence[str]) -> Dict[str, Any]:
    """Overall and per owner workload (`names` by code): landmark rows and
    the share reused within the horizon."""
    out = {"all": {"rows": len(rows["y"]), "reused": _mean(rows["y"])}}
    for code in np.unique(rows["owner_workload"]):
        mask = rows["owner_workload"] == code
        name = names[code] if 0 <= code < len(names) else "unknown"
        out[name] = {"rows": int(mask.sum()), "reused": _mean(rows["y"][mask])}
    return out


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


@dataclass
class Residencies:
    """A run's residency table: numeric columns, the codes of its string
    columns (-1 for null), each string column's values by code, and the
    table's ``run`` metadata."""
    columns: Frame
    strings: Dict[str, List[Optional[str]]]
    meta: Dict[str, Any]


def _dictionary(column) -> Tuple[np.ndarray, List[Optional[str]]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    array = column.combine_chunks()
    if not pa.types.is_dictionary(array.type):
        array = pc.dictionary_encode(array)
    codes = pc.fill_null(array.indices, -1).to_numpy(zero_copy_only=False)
    return codes.astype(np.int64), array.dictionary.to_pylist()


def load_residencies(run_dir: Path) -> Residencies:
    """A run's ``tables/residencies.parquet`` (``analysis/tables.py``)."""
    import pyarrow.parquet as pq
    table = pq.read_table(Path(run_dir) / "tables" / "residencies.parquet",
                          columns=list(COLUMNS))
    meta = json.loads((table.schema.metadata or {}).get(b"run", b"{}"))
    columns: Frame = {}
    strings: Dict[str, List[Optional[str]]] = {}
    for name in COLUMNS:
        if name in STRING_COLUMNS:
            columns[name], strings[name] = _dictionary(table.column(name))
        else:
            columns[name] = np.asarray(
                table.column(name).to_numpy(zero_copy_only=False), dtype=float)
    return Residencies(columns, strings, meta)


def resolve_plan(plan_dir: str) -> Path:
    """A plan folder from run metadata: as given, else from the repo root."""
    path = Path(plan_dir)
    if path.exists() or path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _plan_key(plan: Path) -> Any:
    """One key per plan folder however its path is spelled (macOS paths
    ignore case: run metadata has both ``Projects`` and ``projects``)."""
    try:
        stat = plan.stat()
        return (stat.st_dev, stat.st_ino)
    except OSError:
        return str(plan)


def _log1p(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


def owner_frame(residencies: Residencies, features: Dict[str, identify.Row],
                labels: Dict[str, str], vocabulary: Dict[str, int]) -> Frame:
    """The residencies owned by a request planned in the analysis window and
    known to the plan, with the owner's signals. Workload names become codes
    of `vocabulary` (-1 if absent)."""
    c, meta = residencies.columns, residencies.meta
    owners = residencies.strings["owner_req"] + [None]  # code -1: null

    def by_owner(value, dtype) -> np.ndarray:
        return np.array([value(o) for o in owners], dtype=dtype)

    known = by_owner(lambda o: o in features, bool)
    turn = by_owner(
        lambda o: features[o]["inferred_turn"] if o in features else 0, float)
    popularity = by_owner(
        lambda o: features[o]["family_popularity"]
        if o in features else 0, float)
    inferred = by_owner(lambda o: vocabulary.get(labels.get(o), -1), np.int16)
    workloads = np.array([
        vocabulary.get(w, -1)
        for w in residencies.strings["owner_workload"] + [None]
    ],
                         dtype=np.int16)
    planned = c["owner_planned_s"]
    keep = (known[c["owner_req"]] & (planned >= meta.get("w0_s", W0_S)) &
            (planned < meta.get("stop_s", math.inf)))
    owner = c["owner_req"][keep]
    return {
        "t_free": c["t_free"][keep],
        "idle_s": c["idle_s"][keep],
        "next_lookup_t": c["next_lookup_t"][keep],
        "owner_workload": workloads[c["owner_workload"][keep]],
        "inferred_class": inferred[owner],
        "inferred_turn": _log1p(turn)[owner],
        "family_popularity": _log1p(popularity)[owner],
        "hits_before": _log1p(c["hits_before"][keep]),
        "index": _log1p(c["index"][keep]),
    }


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def score_models(train: Sequence[Tuple[Frame, float]],
                 tests: Dict[str, Tuple[Frame, float]],
                 names: Sequence[str],
                 landmarks: Sequence[float] = LANDMARKS_S,
                 horizons: Sequence[float] = HORIZONS_S,
                 feature_sets: Optional[Dict[str, Tuple[str, ...]]] = None,
                 l2: float = 1e-4,
                 max_train_rows: Optional[int] = 200_000) -> Dict[str, Any]:
    """Fit every feature set on `train` and score it on `tests`, per
    horizon (``"H=<s>"``), at each landmark (``"a=<s>"``) and with all
    landmarks pooled (``"pooled"``).

    Frames are `owner_frame` outputs paired with their censor time; `names`
    are the workload codes' names. Each fit uses at most `max_train_rows`
    rows, drawn at random with a fixed seed; scoring uses every test row.
    """
    feature_sets = feature_sets or FEATURE_SETS
    cells = [(f"a={a:g}", (a, )) for a in landmarks]
    cells.append(("pooled", tuple(landmarks)))
    out: Dict[str, Any] = {
        "rows": {},
        "base_rates": {},
        "models": {name: {}
                   for name in feature_sets},
    }
    for horizon in horizons:
        h = f"H={horizon:g}"
        out["rows"][h], out["base_rates"][h] = {}, {}
        for name in feature_sets:
            out["models"][name][h] = {}
        for cell, ages in cells:
            fit_rows = stack(train, ages, horizon)
            test_rows = {
                n: stack([t], ages, horizon)
                for n, t in tests.items()
            }
            all_test = {
                k: np.concatenate([r[k] for r in test_rows.values()])
                for k in ("owner_workload", "y")
            }
            y = all_test["y"]
            out["rows"][h][cell] = {
                "train": len(fit_rows["y"]),
                "test": len(y)
            }
            if cell != "pooled":
                out["base_rates"][h][cell] = {
                    "train": base_rates(fit_rows, names),
                    "test": base_rates(all_test, names),
                }
            sample = _subsample(fit_rows, max_train_rows)
            for name, features in feature_sets.items():
                encoder = fit_encoder(sample, features)
                beta = fit_logistic(encode(encoder, sample), sample["y"], l2)
                p = {
                    n: predict(encoder, beta, r)
                    for n, r in test_rows.items()
                }
                pooled = np.concatenate(list(p.values()))
                out["models"][name][h][cell] = {
                    "auc": auc(pooled, y),
                    "brier": brier(pooled, y),
                    "by_run": {
                        n: {
                            "auc": auc(p[n], test_rows[n]["y"]),
                            "brier": brier(p[n], test_rows[n]["y"]),
                        }
                        for n in test_rows
                    },
                }
    return out


def _plan_signals(
    rows: List[identify.Row], ladder: Optional[identify.Ladder]
) -> Tuple[Dict[str, identify.Row], Dict[str, str]]:
    """A plan's request features by id, and each request's inferred class
    (the ladder up to the tree): from `ladder`, or cross-fitted when
    `ladder` is None (the training plan)."""
    if ladder is None:
        labels = identify.crossfit_labels(rows, step="tree")
    else:
        labels = identify.ladder_labels(ladder, rows, step="tree")
    return {r["request_id"]: r for r in rows}, labels


def signals_report(train_run_dirs: Sequence[Path],
                   test_run_dirs: Sequence[Path],
                   landmarks: Sequence[float] = LANDMARKS_S,
                   horizons: Sequence[float] = HORIZONS_S,
                   l2: float = 1e-4,
                   max_train_rows: Optional[int] = 200_000) -> Dict[str, Any]:
    """The landmark models (module docstring) fitted on `train_run_dirs`
    and scored on `test_run_dirs`: ceiling runs with ``tables/``. The ladder
    is learned on the first training run's plan."""
    loaded = {
        Path(d): load_residencies(d)
        for d in [*train_run_dirs, *test_run_dirs]
    }
    first = loaded[Path(train_run_dirs[0])].meta
    block_size = int(first.get("block_size", 16))
    train_plan = resolve_plan(first["plan_dir"])
    train_rows = identify.plan_features(train_plan, block_size)
    ladder = identify.fit_ladder(train_rows)
    names = sorted(
        set(ladder.classes)
        | {
            w
            for r in loaded.values()
            for w in r.strings["owner_workload"] if w is not None
        })
    vocabulary = {w: i for i, w in enumerate(names)}
    plans: Dict[Any, Tuple[Dict[str, identify.Row], Dict[str, str]]] = {}

    def frames(dirs: Sequence[Path]) -> Dict[str, Tuple[Frame, float]]:
        out = {}
        for d in dirs:
            residencies = loaded[Path(d)]
            plan = resolve_plan(residencies.meta["plan_dir"])
            key = _plan_key(plan)
            if key not in plans:
                is_train = key == _plan_key(train_plan)
                plans[key] = _plan_signals(
                    train_rows if is_train else identify.plan_features(
                        plan, block_size), None if is_train else ladder)
            frame = owner_frame(residencies, *plans[key], vocabulary)
            out[Path(d).name] = (frame, float(residencies.meta["censor_t"]))
        return out

    train = frames(train_run_dirs)
    tests = frames(test_run_dirs)
    report: Dict[str, Any] = {
        "train_runs": [str(d) for d in train_run_dirs],
        "test_runs": [str(d) for d in test_run_dirs],
        "ladder_plan": str(train_plan),
        "settings": {
            "landmarks_s": list(landmarks),
            "horizons_s": list(horizons),
            "l2": l2,
            "max_train_rows": max_train_rows,
            "feature_sets": {k: list(v)
                             for k, v in FEATURE_SETS.items()},
        },
        "residencies": {
            name: {
                "rows":
                len(frame["t_free"]),
                "inferred_class_accuracy":
                _mean(frame["inferred_class"] == frame["owner_workload"]),
            }
            for name, (frame, _) in {
                **train,
                **tests
            }.items()
        },
    }
    report.update(
        score_models(list(train.values()), tests, names, landmarks, horizons,
                     None, l2, max_train_rows))
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    for h, cells in report["rows"].items():
        names = list(cells)
        print(f"AUC on the test runs at {h} s, by idle age "
              f"({', '.join(names)}):")
        for model, by_h in report["models"].items():
            values = " ".join("  n/a" if by_h[h][c]["auc"] is None else
                              f"{by_h[h][c]['auc']:.3f}" for c in names)
            print(f"  {model:20s} {values}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--out", type=Path, help="JSON file to write")
    args = parser.parse_args(argv)
    report = signals_report(args.train,
                            args.test,
                            l2=args.l2,
                            max_train_rows=args.max_train_rows)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    _print_summary(report)


if __name__ == "__main__":
    main()
