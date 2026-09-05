"""Small vector helpers.

Points are plain ``(x, y, z)`` tuples or lists so landmark data passes straight
through from MediaPipe and from the JSON the browser app exports, with no
per-frame object wrapping.

Mirrors ``web/js/core/vec3.js``; the parity test checks the two agree.
"""

from __future__ import annotations

import math

Vec = "tuple[float, float, float]"


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def distance(a, b):
    return math.dist(a[:3], b[:3])


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def normalize(a):
    n = length(a)
    return (0.0, 0.0, 0.0) if n < 1e-9 else (a[0] / n, a[1] / n, a[2] / n)


def reject(a, axis):
    """Component of ``a`` perpendicular to the unit vector ``axis``."""
    k = dot(a, axis)
    return (a[0] - axis[0] * k, a[1] - axis[1] * k, a[2] - axis[2] * k)


def angle_between(a, b) -> float:
    """Unsigned angle between two vectors, in degrees (0..180)."""
    na, nb = length(a), length(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    c = max(-1.0, min(1.0, dot(a, b) / (na * nb)))
    return math.degrees(math.acos(c))


def joint_angle(a, b, c) -> float:
    """Interior angle at ``b`` in the chain a-b-c, in degrees.

    180 is a straight limb, 0 fully folded -- the convention used throughout the
    sports-biomechanics literature, so values here compare directly with
    published ones.
    """
    return angle_between(sub(a, b), sub(c, b))


def signed_angle(a, b, axis) -> float:
    """Signed angle from ``a`` to ``b`` about ``axis``, in degrees (-180..180).

    Positive follows the right-hand rule around ``axis``.
    """
    n = normalize(axis)
    pa, pb = reject(a, n), reject(b, n)
    if length(pa) < 1e-9 or length(pb) < 1e-9:
        return float("nan")
    angle = angle_between(pa, pb)
    return -angle if dot(cross(pa, pb), n) < 0 else angle
