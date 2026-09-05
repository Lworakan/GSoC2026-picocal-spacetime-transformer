"""Smoothing and derivatives: what makes wrist speed mean anything."""

import math

import pytest

from badminton_coach.filters import LandmarkFilter, OneEuro, RunningMedian, derivative, mean, median


def test_first_sample_passes_through():
    assert OneEuro().filter(4.2, 0.0) == 4.2


def test_jitter_is_suppressed_around_a_constant():
    f = OneEuro(min_cutoff=0.5, beta=0.0)
    out = 0.0
    for i in range(200):
        out = f.filter(10 + (1 if i % 2 else -1), i / 60)
    assert abs(out - 10) < 0.25


def test_beta_reduces_lag_on_a_fast_ramp():
    fast = OneEuro(min_cutoff=1.0, beta=1.0)
    slow = OneEuro(min_cutoff=1.0, beta=0.0)
    last_fast = last_slow = 0.0
    for i in range(60):
        t = i / 60
        last_fast = fast.filter(t * 20, t)
        last_slow = slow.filter(t * 20, t)
    truth = (59 / 60) * 20
    assert abs(truth - last_fast) < abs(truth - last_slow)


def test_repeated_or_rewound_timestamps_are_tolerated():
    f = OneEuro()
    f.filter(1, 1.0)
    assert f.filter(2, 1.0) == 2
    assert f.filter(3, 0.5) == 3


def test_landmark_filter_leaves_visibility_alone():
    out = LandmarkFilter(2, 3).filter([[1, 2, 3, 0.9], [4, 5, 6, 0.1]], 0.0)
    assert out[0][3] == 0.9
    assert out[1][3] == 0.1


def test_running_median_ignores_a_single_outlier():
    m = RunningMedian(5)
    for v in (0.48, 0.47, 0.49, 0.48, 0.52):
        m.push(v)
    before = m.value
    m.push(9.9)
    assert abs(m.value - before) < 0.05


def test_running_median_skips_non_finite():
    m = RunningMedian(5)
    m.push(1)
    m.push(float("nan"))
    m.push(3)
    assert m.value == 2


def test_median_and_mean_ignore_non_finite():
    assert median([1, float("nan"), 3]) == 2
    assert mean([1, float("nan"), 3]) == 2
    assert math.isnan(median([]))


def test_derivative_of_t_squared():
    times = [0, 1, 2, 3, 4]
    d = derivative([t * t for t in times], times)
    assert d[1] == pytest.approx(2)
    assert d[2] == pytest.approx(4)
    assert d[3] == pytest.approx(6)
