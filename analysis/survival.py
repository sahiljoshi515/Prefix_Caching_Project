"""Survival analysis of block reuse times (spec 007, User Story 2).

A cached block either comes back (its next access: an *event*) or the log
ends first. A block the log stops watching is *right-censored*: we know only
that it waited at least that long. Counting censored blocks as reused at the
end would shorten reuse times, and dropping them would hide how many blocks
are never reused. These estimators keep both:

- `kaplan_meier`: the share of blocks still waiting at each time, with no
  model assumed.
- `hazard`: the reuse rate within time bins (events over time at risk), to
  see whether the rate falls as a block ages.
- `fit_exponential`: the constant-rate model ``S(t) = exp(-rate t)``; its
  maximum-likelihood rate is events over total time at risk.
- `fit_cure`: the exponential *cure* model
  ``S(t) = 1 - p + p exp(-rate t)``: only a share ``p`` of blocks is ever
  reused, at ``rate``; the rest never are. Most blocks are never reused, so
  a plain exponential fits them poorly.
- `ks_distance`: the largest gap between the Kaplan-Meier curve and a
  model's survival curve (0 is a perfect fit).

The ATC'25 paper (Wang et al., Fig. 15) fits an exponential to the reuse
times of each request category; these functions test that on our data.
Inputs are equal-length arrays: durations in seconds, and flags that are
True when the block was reused at that duration (False: censored then).
"""
import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np

ArrayLike = Sequence[float]
# Times at which `fit_summary` reports the reuse probability, 1 - S(t).
REUSE_WITHIN_S = (1.0, 10.0, 60.0, 600.0)


def _arrays(durations: ArrayLike,
            observed: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(durations, dtype=float)
    e = np.asarray(observed, dtype=bool)
    if d.ndim != 1 or d.shape != e.shape:
        raise ValueError("durations and observed must be 1-D and equal "
                         "in length")
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("durations must be finite and not negative")
    return d, e


def kaplan_meier(durations: ArrayLike,
                 observed: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray]:
    """The Kaplan-Meier curve: each distinct event time, and the share of
    blocks still waiting just after it.

    At a time with both events and censorings, the events come first: a
    block censored at ``t`` still counts as at risk at ``t``.
    """
    d, e = _arrays(durations, observed)
    if not e.any():
        return np.empty(0), np.empty(0)
    order = np.argsort(d, kind="stable")
    times, first = np.unique(d[order], return_index=True)
    at_risk = len(d) - first
    events = np.add.reduceat(e[order].astype(np.int64), first)
    keep = events > 0
    return times[keep], np.cumprod(1.0 - events[keep] / at_risk[keep])


def survival_at(times: np.ndarray, survival: np.ndarray,
                at: ArrayLike) -> np.ndarray:
    """A Kaplan-Meier curve's value at each time in `at` (1 before the
    first event)."""
    index = np.searchsorted(times, np.asarray(at, dtype=float), side="right")
    return np.concatenate(([1.0], survival))[index]


def hazard(durations: ArrayLike, observed: Sequence[bool],
           edges: ArrayLike) -> np.ndarray:
    """The reuse rate in each bin ``[edges[k], edges[k + 1])``: events in
    the bin over the time at risk spent in it (NaN if none)."""
    d, e = _arrays(durations, observed)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be at least two increasing values")
    lo, hi = edges[:-1], edges[1:]
    bins = np.searchsorted(edges, d[e], side="right") - 1
    inside = (bins >= 0) & (bins < len(lo))
    events = np.bincount(bins[inside], minlength=len(lo))
    # Time at risk in [a, b): b - a for each block waiting past b, plus
    # d - a for each block whose wait ends inside the bin.
    ordered = np.sort(d)
    sums = np.concatenate(([0.0], np.cumsum(ordered)))
    i = np.searchsorted(ordered, lo, side="left")
    j = np.searchsorted(ordered, hi, side="left")
    exposure = (len(d) - j) * (hi - lo) + (sums[j] - sums[i]) - lo * (j - i)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(exposure > 0, events / exposure, np.nan)


def fit_exponential(durations: ArrayLike, observed: Sequence[bool]) -> float:
    """Maximum-likelihood rate of ``S(t) = exp(-rate t)`` under right
    censoring: events over total time at risk (0 without events)."""
    d, e = _arrays(durations, observed)
    events = int(e.sum())
    if events == 0:
        return 0.0
    exposure = float(d.sum())
    return events / exposure if exposure > 0 else math.inf


def fit_cure(durations: ArrayLike,
             observed: Sequence[bool],
             tol: float = 1e-10,
             max_iter: int = 20000) -> Tuple[float, float]:
    """Maximum-likelihood ``(p, rate)`` of the cure model
    ``S(t) = 1 - p + p exp(-rate t)``, by expectation-maximization.

    Each step weighs a censored block by the chance that it is one of the
    reused ones, given how long it has waited; then ``p`` is the expected
    share of reused blocks and ``rate`` their events over their time at
    risk. Without censoring, ``p`` is 1 and the fit is `fit_exponential`.
    """
    d, e = _arrays(durations, observed)
    events = int(e.sum())
    if events == 0:
        return 0.0, 0.0
    if events == len(d):
        return 1.0, fit_exponential(d, e)
    event_time = float(d[e].sum())
    if event_time == 0:  # Every reuse at once: an infinite rate.
        return events / len(d), math.inf
    # Censored blocks share few distinct waits (one per request), so the
    # steps run over the distinct values.
    waits, counts = np.unique(d[~e], return_counts=True)
    p, rate = events / len(d), events / event_time
    for _ in range(max_iter):
        stay = np.exp(-rate * waits)
        with np.errstate(invalid="ignore"):
            weight = p * stay / (1.0 - p + p * stay)
        weight = np.nan_to_num(weight, nan=1.0)  # p = 1, stay = 0
        new_p = (events + float(counts @ weight)) / len(d)
        new_rate = events / (event_time + float(counts @ (weight * waits)))
        done = (abs(new_p - p) < tol and abs(new_rate - rate) < tol * rate)
        p, rate = new_p, new_rate
        if done:
            break
    return p, rate


def _model(rate: float, p: float, t: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        stay = np.where(t > 0, np.exp(-rate * t), 1.0)
    return 1.0 - p + p * stay


def ks_distance(durations: ArrayLike,
                observed: Sequence[bool],
                rate: float,
                p: float = 1.0) -> float:
    """Largest gap between the Kaplan-Meier curve and the model
    ``S(t) = 1 - p + p exp(-rate t)`` over ``[0, longest duration]``.
    ``p = 1`` (the default) is the plain exponential."""
    d, e = _arrays(durations, observed)
    if len(d) == 0:
        return 0.0
    end = float(_model(rate, p, np.array([d.max()]))[0])
    times, survival = kaplan_meier(d, e)
    if len(times) == 0:  # The curve stays at 1; the model only falls.
        return abs(1.0 - end)
    model = _model(rate, p, times)
    before = np.concatenate(([1.0], survival[:-1]))
    return float(
        max(
            np.abs(survival - model).max(),
            np.abs(before - model).max(), abs(survival[-1] - end)))


def fit_summary(durations: ArrayLike,
                observed: Sequence[bool]) -> Dict[str, Any]:
    """Both models' fits, their KS distances, and the measured chance of a
    reuse within each of `REUSE_WITHIN_S` (``1 - S(t)``, Kaplan-Meier)."""
    d, e = _arrays(durations, observed)
    rate = fit_exponential(d, e)
    p, cure_rate = fit_cure(d, e)
    times, survival = kaplan_meier(d, e)
    within = 1.0 - survival_at(times, survival, REUSE_WITHIN_S)
    return {
        "n": int(len(d)),
        "events": int(e.sum()),
        "rate": rate,
        "ks": ks_distance(d, e, rate),
        "cure_p": p,
        "cure_rate": cure_rate,
        "cure_ks": ks_distance(d, e, cure_rate, p),
        "reused_within":
        {f"{s:g}s": float(w)
         for s, w in zip(REUSE_WITHIN_S, within)},
    }
