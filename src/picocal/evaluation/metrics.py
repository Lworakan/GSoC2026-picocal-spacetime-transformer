"""Evaluation metrics, consistent across dataset stages and models.

These mirror the metrics used in the evaluation task so the 2D baseline and the
new 3D PicoCal results are directly comparable. Implemented with plain Python so
they are dependency-light and unit-testable in CI.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def fractional_error(pred: Sequence[float], true: Sequence[float]) -> list[float]:
    """Return per-event (E_pred - E_true) / E_true."""
    if len(pred) != len(true):
        raise ValueError("pred and true must have the same length")
    out = []
    for p, t in zip(pred, true):
        if t == 0:
            raise ValueError("true energy must be non-zero")
        out.append((p - t) / t)
    return out


def bias(pred: Sequence[float], true: Sequence[float]) -> float:
    """Mean of the fractional error (the resolution bias)."""
    fe = fractional_error(pred, true)
    return sum(fe) / len(fe)


def rmse(pred: Sequence[float], true: Sequence[float]) -> float:
    """Root-mean-square error in absolute energy units (GeV)."""
    if len(pred) != len(true):
        raise ValueError("pred and true must have the same length")
    sq = [(p - t) ** 2 for p, t in zip(pred, true)]
    return math.sqrt(sum(sq) / len(sq))


def p68(pred: Sequence[float], true: Sequence[float]) -> float:
    """68th percentile of |fractional error| — robust to non-Gaussian tails."""
    fe = sorted(abs(v) for v in fractional_error(pred, true))
    idx = max(0, math.ceil(0.68 * len(fe)) - 1)
    return fe[idx]
