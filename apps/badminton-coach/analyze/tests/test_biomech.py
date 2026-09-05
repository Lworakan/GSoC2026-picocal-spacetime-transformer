"""Biomechanics against a synthetic skeleton with known angles."""

import math

import pytest

from badminton_coach.biomech import (
    body_frame, body_scale, core_visibility, elbow_angle, frame_metrics,
    hip_shoulder_separation, shoulder_azimuth, shoulder_elevation, to_body,
    trunk_lean, world_to_up_frame,
)
from badminton_coach.landmarks import CORE, COUNT, LEFT_SHOULDER, RIGHT_SHOULDER
from conftest import make_skeleton, to_mediapipe


def test_world_to_up_frame_undoes_mediapipe_convention():
    world = make_skeleton()
    restored = world_to_up_frame(to_mediapipe(world))
    for a, b in zip(restored, world):
        assert a == pytest.approx(tuple(b))


def test_body_frame_is_orthonormal_and_faces_forward():
    f = body_frame(make_skeleton())
    from badminton_coach.geometry import dot, length
    assert dot(f.up, f.right) == pytest.approx(0, abs=1e-9)
    assert dot(f.up, f.forward) == pytest.approx(0, abs=1e-9)
    assert dot(f.right, f.forward) == pytest.approx(0, abs=1e-9)
    assert length(f.up) == pytest.approx(1)
    # The figure faces +z, and its right shoulder is at negative x.
    assert f.forward[2] > 0.99
    assert f.right[0] < -0.99


@pytest.mark.parametrize("want", [180, 150, 120, 90, 60])
def test_elbow_angle_reproduces_what_it_was_built_with(want):
    assert elbow_angle(make_skeleton(elbow_right=want), "right") == pytest.approx(want, abs=0.01)


def test_elbow_angle_is_independent_of_how_the_arm_is_held():
    a = elbow_angle(make_skeleton(elbow_right=110, arm_elevation=0), "right")
    b = elbow_angle(make_skeleton(elbow_right=110, arm_elevation=160, arm_azimuth=-40), "right")
    assert a == pytest.approx(b, abs=0.01)


@pytest.mark.parametrize("want", [0, 45, 90, 135, 180])
def test_shoulder_elevation(want):
    w = make_skeleton(arm_elevation=want)
    assert shoulder_elevation(body_frame(w), w, "right") == pytest.approx(want, abs=0.01)


@pytest.mark.parametrize("want", [0, 90, -70])
def test_shoulder_azimuth_is_signed_towards_the_racket_side(want):
    w = make_skeleton(arm_elevation=90, arm_azimuth=want)
    assert shoulder_azimuth(body_frame(w), w, "right", "right") == pytest.approx(want, abs=0.01)


def test_shoulder_azimuth_is_undefined_for_an_arm_hanging_down():
    w = make_skeleton(arm_elevation=0)
    assert math.isnan(shoulder_azimuth(body_frame(w), w, "right", "right"))


def test_left_handed_players_mirror_the_azimuth():
    w = make_skeleton(arm_elevation=90, arm_azimuth=90)
    right = shoulder_azimuth(body_frame(w), w, "right", "right")
    as_left = shoulder_azimuth(body_frame(w), w, "right", "left")
    assert right == pytest.approx(-as_left, abs=0.01)


def test_trunk_lean_separates_forwards_from_sideways():
    m = trunk_lean(body_frame(make_skeleton(lean_forward=30)), "right")
    assert m["total"] == pytest.approx(30, abs=0.05)
    assert m["forward"] == pytest.approx(30, abs=0.05)
    assert m["lateral"] == pytest.approx(0, abs=0.05)

    m = trunk_lean(body_frame(make_skeleton(lean_right=25)), "right")
    assert m["total"] == pytest.approx(25, abs=0.05)
    assert m["forward"] == pytest.approx(0, abs=0.05)
    assert m["lateral"] == pytest.approx(25, abs=0.05)


def test_trunk_lean_components_are_not_both_zero():
    # Pins the bug where the decomposition was taken against the torso's own
    # axes, which are orthogonal to its up-vector, making every component zero.
    m = trunk_lean(body_frame(make_skeleton(lean_forward=20, lean_right=20)), "right")
    assert abs(m["forward"]) > 5 and abs(m["lateral"]) > 5
    assert 20 < m["total"] < 40


@pytest.mark.parametrize("twist", [0, 20, 40, -30])
def test_hip_shoulder_separation(twist):
    w = make_skeleton(shoulder_twist=twist)
    got = hip_shoulder_separation(body_frame(w), w, "right")
    assert abs(got) == pytest.approx(abs(twist), abs=0.05)


def test_hand_position_reads_height_lateral_and_forward():
    overhead = frame_metrics(make_skeleton(arm_elevation=180), "right")
    assert overhead["hand"]["height"] > 0.8

    across = frame_metrics(make_skeleton(arm_elevation=90, arm_azimuth=-110), "right")
    assert across["hand"]["lateral"] < 0

    in_front = frame_metrics(make_skeleton(arm_elevation=90, arm_azimuth=0), "right")
    assert in_front["hand"]["forward"] > 0.5


def test_metrics_are_scale_free():
    def at(k):
        return frame_metrics(make_skeleton(
            trunk=0.5 * k, shoulder_width=0.36 * k, hip_width=0.28 * k,
            upper_arm=0.30 * k, forearm=0.27 * k, thigh=0.42 * k, shin=0.42 * k,
            stance=0.4 * k, arm_elevation=120, arm_azimuth=30, elbow_right=140,
        ), "right")

    small, large = at(0.8), at(1.25)
    for key in ("height", "lateral", "forward"):
        assert small["hand"][key] == pytest.approx(large["hand"][key], abs=0.005)
    assert small["stance_width"] == pytest.approx(large["stance_width"], abs=0.005)
    assert small["elbow"] == pytest.approx(large["elbow"], abs=0.01)


def test_body_scale_reports_the_lengths_it_was_built_from():
    s = body_scale(make_skeleton(trunk=0.5, shoulder_width=0.36, upper_arm=0.3, forearm=0.27))
    assert s.trunk == pytest.approx(0.5)
    assert s.shoulder_width == pytest.approx(0.36)
    assert s.arm_right == pytest.approx(0.57)


def test_supplied_trunk_length_overrides_the_per_frame_one():
    w = make_skeleton(trunk=0.5, arm_elevation=180)
    noisy = frame_metrics(w, "right")
    stable = frame_metrics(w, "right", trunk=1.0)
    assert stable["hand"]["height"] == pytest.approx(noisy["hand"]["height"] / 2, abs=0.01)


def test_to_body_puts_the_shoulders_at_the_origin():
    w = make_skeleton()
    f = body_frame(w)
    mid = to_body(f, [(w[LEFT_SHOULDER][i] + w[RIGHT_SHOULDER][i]) / 2 for i in range(3)])
    assert mid == pytest.approx((0, 0, 0), abs=1e-9)


def test_core_visibility():
    image = [[0, 0, 0, 1.0] for _ in range(COUNT)]
    for i in CORE:
        image[i][3] = 0.5
    assert core_visibility(image, CORE) == pytest.approx(0.5)
    assert core_visibility([[0, 0, 0]] * COUNT, CORE) == 0.0
