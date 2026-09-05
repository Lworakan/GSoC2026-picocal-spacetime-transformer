"""The court homography, its guards, and the movement measures built on it."""

import json

import pytest

from badminton_coach import landmarks as L
from badminton_coach.court import (
    BASE_POSITION, CALIBRATION_ORDER, Court, CourtCalibration, HALF_COURT_CORNERS,
    apply_homography, court_track, distance_covered, distance_from_base,
    homography_from_quad, invert3x3, is_convex_quad, on_court, recovery_times,
    solve_linear, zone_occupancy, zone_of,
)

VIEW = [(0.20, 0.60), (0.80, 0.60), (0.95, 0.95), (0.05, 0.95)]


def test_solve_linear():
    x = solve_linear([[2, 1], [1, 3]], [5, 10])
    assert x == pytest.approx([1, 3])
    assert solve_linear([[1, 2], [2, 4]], [1, 2]) is None


def test_corners_map_exactly_onto_the_court():
    cal = CourtCalibration(VIEW)
    assert cal.valid
    for name, point in zip(CALIBRATION_ORDER, cal.points):
        assert cal.court_point(point) == pytest.approx(HALF_COURT_CORNERS[name], abs=1e-9)


@pytest.mark.parametrize("point", [(0, 0), (1.5, 4), (-2.8, 6.5), (3.05, 6.7)])
def test_court_and_image_coordinates_round_trip(point):
    cal = CourtCalibration(VIEW)
    assert cal.court_point(cal.image_point(point)) == pytest.approx(point, abs=1e-6)


def test_a_bow_tie_tap_order_is_rejected():
    bad = CourtCalibration([(0.20, 0.60), (0.95, 0.95), (0.80, 0.60), (0.05, 0.95)])
    assert not bad.valid
    assert not is_convex_quad(bad.points)


def test_duplicated_corners_are_rejected():
    assert not CourtCalibration([(0.2, 0.6), (0.2, 0.6), (0.9, 0.9), (0.1, 0.9)]).valid


def test_swapping_sides_reverses_a_mirrored_view():
    cal = CourtCalibration(VIEW)
    swapped = cal.swap_sides()
    assert swapped.valid
    assert cal.mirrored != swapped.mirrored
    assert cal.court_point(VIEW[1])[0] == pytest.approx(-swapped.court_point(VIEW[1])[0])


def test_calibration_survives_a_save_and_reload():
    cal = CourtCalibration(VIEW)
    back = CourtCalibration.from_json(json.loads(json.dumps(cal.to_json())))
    assert back.valid
    assert back.court_point((0.5, 0.8)) == pytest.approx(cal.court_point((0.5, 0.8)))
    assert CourtCalibration.from_json(None) is None
    assert CourtCalibration.from_json({"points": [[0, 0]]}) is None


def test_homography_and_its_inverse_agree():
    H = homography_from_quad(VIEW, [HALF_COURT_CORNERS[k] for k in CALIBRATION_ORDER])
    Hi = invert3x3(H)
    there = apply_homography(H, (0.5, 0.8))
    assert apply_homography(Hi, there) == pytest.approx((0.5, 0.8), abs=1e-9)
    assert invert3x3([[1, 0, 0], [2, 0, 0], [3, 0, 0]]) is None


def test_zones_tile_the_half_court():
    assert zone_of((0, 1.0))["name"] == "front-centre"
    assert zone_of((-2.5, 1.0))["name"] == "front-left"
    assert zone_of((2.5, 5.5))["name"] == "rear-right"
    assert zone_of((0, 3.5))["name"] == "mid-centre"
    assert zone_of((0, 9))["depth"] == "rear"


def test_base_distance_and_boundary():
    assert distance_from_base(BASE_POSITION) == pytest.approx(0)
    assert distance_from_base((0, 6.0)) == pytest.approx(3.0)
    assert on_court((0, 0))
    assert on_court((Court.width / 2, Court.half_length))
    assert not on_court((0, 20))


def _frame_at(t, x, y):
    image = [[0.5, 0.5, 0.0, 1.0] for _ in range(L.COUNT)]
    image[L.LEFT_ANKLE] = [x, y, 0.0, 1.0]
    image[L.RIGHT_ANKLE] = [x, y, 0.0, 1.0]
    return {"t": t, "image": image}


def test_court_track_maps_feet_and_drops_invisible_frames():
    cal = CourtCalibration(VIEW)
    centre = cal.image_point((0, 3.0))
    frames = [
        _frame_at(0.0, centre[0], centre[1]),
        {"t": 0.1, "image": None},
        {"t": 0.2, "image": [[0.5, 0.5, 0.0, 0.05] for _ in range(L.COUNT)]},
    ]
    positions = court_track(frames, cal)
    assert positions[0]["x"] == pytest.approx(0, abs=1e-6)
    assert positions[0]["y"] == pytest.approx(3.0, abs=1e-6)
    assert positions[0]["zone"]["name"] == "mid-centre"
    assert positions[1] is None
    assert positions[2] is None


def test_distance_covered_ignores_teleports():
    walk = [{"t": 0, "x": 0, "y": 0}, {"t": 0.1, "x": 0, "y": 0.5}, {"t": 0.2, "x": 0, "y": 1.0}]
    assert distance_covered(walk) == pytest.approx(1.0)
    assert distance_covered(walk + [{"t": 0.3, "x": 0, "y": 9.0}]) == pytest.approx(1.0)


def test_zone_occupancy_ignores_long_gaps():
    positions = [
        {"t": 0.0, "zone": {"name": "mid-centre"}},
        {"t": 0.1, "zone": {"name": "mid-centre"}},
        {"t": 0.2, "zone": {"name": "mid-centre"}},
        {"t": 9.0, "zone": {"name": "mid-centre"}},
    ]
    assert zone_occupancy(positions)["mid-centre"] == pytest.approx(0.2)


def test_recovery_times_reports_success_and_failure():
    positions = [
        {"t": 1.0, "x": 0, "y": 6.0, "zone": {"name": "rear-centre"}, "base": 3.0},
        {"t": 1.8, "x": 0, "y": 3.4, "zone": {"name": "mid-centre"}, "base": 0.4},
        {"t": 3.0, "x": 0, "y": 6.5, "zone": {"name": "rear-centre"}, "base": 3.5},
        {"t": 5.9, "x": 0, "y": 6.5, "zone": {"name": "rear-centre"}, "base": 3.5},
    ]
    first, second = recovery_times([{"index": 0, "t": 1.0}, {"index": 1, "t": 3.0}], positions)
    assert first["recovered"]
    assert first["recovery_seconds"] == pytest.approx(0.8)
    assert not second["recovered"]
    assert second["recovery_seconds"] is None
