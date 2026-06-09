"""Unit tests for evaluation metrics (run in CI)."""
import math

import pytest

from picocal.evaluation import bias, fractional_error, p68, rmse


def test_fractional_error_perfect_prediction():
    fe = fractional_error([10.0, 20.0], [10.0, 20.0])
    assert fe == [0.0, 0.0]


def test_bias_is_mean_fractional_error():
    # pred 10% high then 10% low -> zero mean bias
    assert bias([11.0, 9.0], [10.0, 10.0]) == pytest.approx(0.0)


def test_rmse_known_value():
    assert rmse([1.0, 2.0], [0.0, 0.0]) == pytest.approx(math.sqrt(2.5))


def test_p68_in_unit_interval():
    pred = [10.5, 9.5, 11.0, 9.0, 10.0]
    true = [10.0] * 5
    val = p68(pred, true)
    assert 0.0 <= val <= 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        fractional_error([1.0], [1.0, 2.0])


def test_zero_true_energy_raises():
    with pytest.raises(ValueError):
        fractional_error([1.0], [0.0])
