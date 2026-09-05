"""Test helpers, including the synthetic skeleton the biomechanics are checked against.

Mirrors ``tests/skeleton.js`` on the JavaScript side: a figure built from chosen
joint angles, so the maths is tested against geometry rather than against
whatever the pose model happened to output on one video.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from badminton_coach import landmarks as L  # noqa: E402


def _unit(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def _perpendicular_to(axis, candidates):
    """The first candidate with a usable component perpendicular to ``axis``."""
    for c in candidates:
        k = sum(a * b for a, b in zip(c, axis))
        r = tuple(ci - ai * k for ci, ai in zip(c, axis))
        if math.sqrt(sum(x * x for x in r)) > 1e-6:
            return _unit(r)
    return (0.0, 0.0, 1.0)


def make_skeleton(
    trunk=0.5, shoulder_width=0.36, hip_width=0.28,
    upper_arm=0.30, forearm=0.27,
    elbow_right=180.0, arm_elevation=0.0, arm_azimuth=0.0,
    lean_forward=0.0, lean_right=0.0, shoulder_twist=0.0,
    knee_angle=180.0, thigh=0.42, shin=0.42, stance=0.4,
):
    """A figure facing +z, y up, built from the angles given."""
    p = [[0.0, 0.0, 0.0] for _ in range(L.COUNT)]

    up = (
        -math.sin(math.radians(lean_right)),
        math.cos(math.radians(lean_forward)) * math.cos(math.radians(lean_right)),
        math.sin(math.radians(lean_forward)),
    )
    u = _unit(up)
    mid_shoulder = tuple(c * trunk for c in u)

    p[L.LEFT_HIP] = [hip_width / 2, 0.0, 0.0]
    p[L.RIGHT_HIP] = [-hip_width / 2, 0.0, 0.0]

    tw = math.radians(shoulder_twist)
    sx = math.cos(tw) * shoulder_width / 2
    sz = math.sin(tw) * shoulder_width / 2
    p[L.LEFT_SHOULDER] = [mid_shoulder[0] + sx, mid_shoulder[1], mid_shoulder[2] + sz]
    p[L.RIGHT_SHOULDER] = [mid_shoulder[0] - sx, mid_shoulder[1], mid_shoulder[2] - sz]

    e = math.radians(arm_elevation)
    a = math.radians(arm_azimuth)
    direction = (-math.sin(e) * math.sin(a), -math.cos(e), math.sin(e) * math.cos(a))
    shoulder = p[L.RIGHT_SHOULDER]
    elbow = [shoulder[i] + direction[i] * upper_arm for i in range(3)]
    p[L.RIGHT_ELBOW] = elbow

    bend = math.pi - math.radians(elbow_right)
    perp = _perpendicular_to(direction, [u, (0, 0, 1), (1, 0, 0)])
    fore = [direction[i] * math.cos(bend) + perp[i] * math.sin(bend) for i in range(3)]
    p[L.RIGHT_WRIST] = [elbow[i] + fore[i] * forearm for i in range(3)]

    p[L.LEFT_ELBOW] = [p[L.LEFT_SHOULDER][0], p[L.LEFT_SHOULDER][1] - upper_arm,
                       p[L.LEFT_SHOULDER][2]]
    p[L.LEFT_WRIST] = [p[L.LEFT_ELBOW][0], p[L.LEFT_ELBOW][1] - forearm, p[L.LEFT_ELBOW][2]]

    bend_knee = math.radians(180 - knee_angle)
    for hip_i, knee_i, ankle_i, sign in (
        (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, 1),
        (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, -1),
    ):
        hip = p[hip_i]
        lateral = sign * stance / 2 - hip[0]
        thigh_dir = _unit((lateral, -thigh, 0.0))
        knee_point = [hip[i] + thigh_dir[i] * thigh for i in range(3)]
        knee_perp = _perpendicular_to(thigh_dir, [(0, 0, 1), (1, 0, 0)])
        shin_dir = [thigh_dir[i] * math.cos(bend_knee) + knee_perp[i] * math.sin(bend_knee)
                    for i in range(3)]
        p[knee_i] = knee_point
        p[ankle_i] = [knee_point[i] + shin_dir[i] * shin for i in range(3)]

    p[L.LEFT_FOOT_INDEX] = [p[L.LEFT_ANKLE][0], p[L.LEFT_ANKLE][1], p[L.LEFT_ANKLE][2] + 0.1]
    p[L.RIGHT_FOOT_INDEX] = [p[L.RIGHT_ANKLE][0], p[L.RIGHT_ANKLE][1], p[L.RIGHT_ANKLE][2] + 0.1]
    p[L.NOSE] = [mid_shoulder[0], mid_shoulder[1] + 0.2, mid_shoulder[2] + 0.1]
    return p


def to_mediapipe(points):
    """The same skeleton in MediaPipe's raw convention (y down, z away)."""
    return [[x, -y, -z] for x, y, z in points]


def fake_image(points, visibility=1.0):
    return [[0.5 + p[0] * 0.2, 0.5 - p[1] * 0.2, 0.0, visibility] for p in points]


@pytest.fixture
def skeleton():
    return make_skeleton
