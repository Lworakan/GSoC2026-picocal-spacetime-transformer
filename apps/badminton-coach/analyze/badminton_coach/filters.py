"""Signal conditioning for pose landmarks.

Raw MediaPipe landmarks on a player filmed from across a court jitter badly: on
the reference clip the raw right-wrist speed peaked at 41 m/s, about three times
the fastest wrist speed ever measured in a badminton smash, so it is noise rather
than movement. Everything downstream reads filtered values.

Mirrors ``web/js/core/filters.js``.
"""

from __future__ import annotations

import math
from statistics import median as _median


class OneEuro:
    """A single-channel 1-Euro filter.

    A fixed low-pass filter forces a choice between jitter at rest and lag during
    a swing, and a smash is over in about 100 ms, so lag is not affordable. The
    1-Euro filter (Casiez, Roussel & Vogel, CHI 2012) widens its own cutoff as the
    signal speeds up: still at rest, sharp through the swing.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.25, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value: float, t: float) -> float:
        if self.x is None or self.t is None or not t > self.t:
            self.x, self.t, self.dx = value, t, 0.0
            return value
        dt = t - self.t
        dx_raw = (value - self.x) / dt
        self.dx += OneEuro.alpha(dt, self.d_cutoff) * (dx_raw - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += OneEuro.alpha(dt, cutoff) * (value - self.x)
        self.t = t
        return self.x


class LandmarkFilter:
    """A 1-Euro filter applied independently to every coordinate."""

    def __init__(self, count: int, dims: int = 3, **options):
        self.dims = dims
        self.channels = [OneEuro(**options) for _ in range(count * dims)]

    def reset(self) -> None:
        for c in self.channels:
            c.reset()

    def filter(self, points, t: float) -> list[list[float]]:
        out = []
        for i, p in enumerate(points):
            row = list(p)
            for d in range(self.dims):
                row[d] = self.channels[i * self.dims + d].filter(p[d], t)
            out.append(row)
        return out


class RunningMedian:
    """Median over a sliding window.

    Used for body-scale estimates: one bad frame would otherwise rescale every
    normalised measurement on that frame. On the reference clip the per-frame
    trunk length wanders between 0.36 m and 0.53 m for a trunk that is, of
    course, one fixed length.
    """

    def __init__(self, size: int = 31):
        self.size = size
        self.buffer: list[float] = []

    def reset(self) -> None:
        self.buffer.clear()

    def push(self, value: float) -> float:
        if value is not None and math.isfinite(value):
            self.buffer.append(value)
            if len(self.buffer) > self.size:
                self.buffer.pop(0)
        return self.value

    @property
    def value(self) -> float:
        return _median(self.buffer) if self.buffer else float("nan")


def median(values) -> float:
    ok = [v for v in values if v is not None and math.isfinite(v)]
    return _median(ok) if ok else float("nan")


def mean(values) -> float:
    ok = [v for v in values if v is not None and math.isfinite(v)]
    return sum(ok) / len(ok) if ok else float("nan")


def derivative(values, times) -> list[float]:
    """Central difference of a scalar series, robust to uneven frame spacing.

    Half the noise gain of a forward difference and no half-frame phase shift,
    which matters because the peak of this derivative is what we call contact.
    """
    out = [float("nan")] * len(values)
    for i in range(1, len(values) - 1):
        dt = times[i + 1] - times[i - 1]
        if dt > 0:
            out[i] = (values[i + 1] - values[i - 1]) / dt
    if len(values) > 1:
        dt0 = times[1] - times[0]
        if dt0 > 0:
            out[0] = (values[1] - values[0]) / dt0
        n = len(values) - 1
        dtn = times[n] - times[n - 1]
        if dtn > 0:
            out[n] = (values[n] - values[n - 1]) / dtn
    return out
