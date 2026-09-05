"""Geometry primitives: the base every angle is built on."""

import math

import pytest

from badminton_coach.geometry import (
    angle_between, cross, distance, dot, joint_angle, midpoint, normalize, reject, signed_angle,
)


def test_joint_angle_straight_and_right_angle():
    assert joint_angle((0, 1, 0), (0, 0, 0), (0, -1, 0)) == pytest.approx(180)
    assert joint_angle((1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(90)
    assert joint_angle((1, 0, 0), (0, 0, 0), (1, 0, 0)) == pytest.approx(0)


def test_joint_angle_ignores_limb_length():
    assert joint_angle((5, 0, 0), (0, 0, 0), (0, 0.01, 0)) == pytest.approx(90)


def test_signed_angle_follows_the_right_hand_rule():
    assert signed_angle((1, 0, 0), (0, 1, 0), (0, 0, 1)) == pytest.approx(90)
    assert signed_angle((0, 1, 0), (1, 0, 0), (0, 0, 1)) == pytest.approx(-90)


def test_degenerate_inputs_give_nan():
    assert math.isnan(angle_between((0, 0, 0), (1, 0, 0)))
    assert math.isnan(signed_angle((0, 0, 1), (0, 0, 1), (0, 0, 1)))
    assert normalize((0, 0, 0)) == (0, 0, 0)


def test_reject_removes_the_component_along_the_axis():
    r = reject((1, 2, 3), (0, 1, 0))
    assert dot(r, (0, 1, 0)) == pytest.approx(0)
    assert r == pytest.approx((1, 0, 3))


def test_cross_is_right_handed():
    assert cross((1, 0, 0), (0, 1, 0)) == (0, 0, 1)


def test_midpoint_and_distance():
    assert midpoint((0, 0, 0), (2, 4, 6)) == (1, 2, 3)
    assert distance((0, 0, 0), (3, 4, 0)) == pytest.approx(5)
