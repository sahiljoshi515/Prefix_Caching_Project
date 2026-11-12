"""Survival estimators for reuse times (analysis/survival.py).

Small cases are worked out by hand; the model fits are checked on samples
drawn from the model itself, with a fixed seed.
"""
import math

import numpy as np
import pytest

from analysis import survival as sv

pytestmark = pytest.mark.m3


def test_kaplan_meier_by_hand():
    # At t=1: 5 at risk, 1 reused -> 0.8. At t=2: 4 at risk (the block
    # censored at 2 still counts), 1 reused -> 0.8 * 3/4 = 0.6. At t=3:
    # 2 at risk, 1 reused -> 0.3. The censoring at 5 adds no step.
    times, s = sv.kaplan_meier([1, 2, 2, 3, 5], [1, 1, 0, 1, 0])
    assert times.tolist() == [1, 2, 3]
    assert s == pytest.approx([0.8, 0.6, 0.3])


def test_kaplan_meier_ties_and_no_events():
    times, s = sv.kaplan_meier([2, 2, 2, 4], [1, 1, 0, 1])
    # t=2: 4 at risk, 2 reused -> 0.5; t=4: 1 at risk, 1 reused -> 0.
    assert times.tolist() == [2, 4]
    assert s == pytest.approx([0.5, 0.0])
    times, s = sv.kaplan_meier([1, 3], [0, 0])
    assert len(times) == 0 and len(s) == 0


def test_survival_at_is_a_right_continuous_step():
    times, s = sv.kaplan_meier([1, 2, 2, 3, 5], [1, 1, 0, 1, 0])
    at = sv.survival_at(times, s, [0, 0.5, 1, 1.5, 2, 3, 10])
    assert at == pytest.approx([1, 1, 0.8, 0.8, 0.6, 0.3, 0.3])


def test_hazard_bins_by_hand():
    # Bin [0, 1): one reuse (0.5); time at risk 0.5 + 1 + 1 + 1 = 3.5.
    # Bin [1, 2): no reuse (1.5 is censored); 0.5 + 1 + 1 = 2.5.
    # Bin [2, 4): reuses at 2.5 and 3; 0.5 + 1 = 1.5.
    rates = sv.hazard([0.5, 1.5, 2.5, 3], [1, 0, 1, 1], [0, 1, 2, 4])
    assert rates == pytest.approx([1 / 3.5, 0.0, 2 / 1.5])
    # Nobody is at risk past 3: NaN.
    assert np.isnan(sv.hazard([1], [1], [0, 2, 5])[1])


def test_exponential_mle_by_hand_and_edge_cases():
    assert sv.fit_exponential([1, 2, 3], [1, 0, 1]) == pytest.approx(2 / 6)
    assert sv.fit_exponential([1, 2], [0, 0]) == 0.0
    assert sv.fit_exponential([0, 0], [1, 1]) == math.inf


def test_exponential_mle_recovers_the_rate():
    rng = np.random.default_rng(7)
    rate = 0.2
    reuse = rng.exponential(1 / rate, 50_000)
    censor = rng.uniform(0, 10, 50_000)
    durations = np.minimum(reuse, censor)
    rate_hat = sv.fit_exponential(durations, reuse <= censor)
    assert rate_hat == pytest.approx(rate, rel=0.05)


def test_ks_distance_by_hand():
    # One reuse at t=1, model exp(-ln2 t): the curve drops from 1 to 0 at
    # t=1 where the model is 0.5, so the gap is 0.5 on both sides.
    assert sv.ks_distance([1], [1], math.log(2)) == pytest.approx(0.5)
    # No reuse until 2: the curve stays at 1, the model falls to e^-2.
    assert sv.ks_distance([2], [0], 1.0) == pytest.approx(1 - math.exp(-2))


def test_ks_of_an_exponential_sample_is_small():
    rng = np.random.default_rng(11)
    reuse = rng.exponential(5.0, 20_000)
    censor = rng.uniform(0, 30, 20_000)
    durations = np.minimum(reuse, censor)
    observed = reuse <= censor
    rate = sv.fit_exponential(durations, observed)
    assert sv.ks_distance(durations, observed, rate) < 0.02
    # A wrong rate is far off.
    assert sv.ks_distance(durations, observed, 3 * rate) > 0.2


def test_cure_model_recovers_share_and_rate():
    rng = np.random.default_rng(3)
    n, p, rate = 40_000, 0.3, 0.5
    reused = rng.random(n) < p
    reuse = np.where(reused, rng.exponential(1 / rate, n), np.inf)
    censor = rng.uniform(0, 30, n)
    durations = np.minimum(reuse, censor)
    observed = reuse <= censor
    p_hat, rate_hat = sv.fit_cure(durations, observed)
    assert p_hat == pytest.approx(p, abs=0.01)
    assert rate_hat == pytest.approx(rate, rel=0.05)
    # The cure model fits; a plain exponential cannot (most never return).
    assert sv.ks_distance(durations, observed, rate_hat, p_hat) < 0.02
    plain = sv.fit_exponential(durations, observed)
    assert sv.ks_distance(durations, observed, plain) > 0.2


def test_cure_model_edge_cases():
    assert sv.fit_cure([1, 2], [0, 0]) == (0.0, 0.0)
    p, rate = sv.fit_cure([1, 3], [1, 1])  # no censoring: plain fit
    assert p == 1.0 and rate == pytest.approx(0.5)
    p, rate = sv.fit_cure([0, 0, 5], [1, 1, 0])
    assert p == pytest.approx(2 / 3) and rate == math.inf


def test_fit_summary_fields():
    out = sv.fit_summary([1, 2, 2, 3, 5], [1, 1, 0, 1, 0])
    assert out["n"] == 5 and out["events"] == 3
    assert out["rate"] == pytest.approx(3 / 13)
    # 1 - S: by 1 s 0.2 reused; by 10 s 0.7.
    assert out["reused_within"]["1s"] == pytest.approx(0.2)
    assert out["reused_within"]["10s"] == pytest.approx(0.7)
    assert 0 < out["cure_p"] <= 1


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        sv.kaplan_meier([1, -1], [1, 1])
    with pytest.raises(ValueError):
        sv.kaplan_meier([1, 2], [1])
    with pytest.raises(ValueError):
        sv.hazard([1], [1], [0])
