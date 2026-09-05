"""The Python analysis must agree with the JavaScript analysis.

Two implementations of the same maths drift apart silently: a threshold changed
on one side, a sign flipped on the other, and the report stops matching what the
player saw on court. This test pins them together on real landmark data.

``tests/fixtures/expected-metrics.json`` is written by the JavaScript core
(``node tools/dump-metrics.mjs``). If this test fails, one of the two moved --
find out which before regenerating the file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from badminton_coach.coach import coach_session, coach_stroke
from badminton_coach.session import PoseSession, SessionOptions, detect_racket_arm

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"

#: Angles in degrees, lengths in trunk lengths. The two implementations do the
#: same operations in the same order, so they agree to far better than this; the
#: tolerance is for float64 formatting differences, not for real disagreement.
TOL = 1e-6


@pytest.fixture(scope="module")
def landmarks():
    return json.loads((FIXTURES / "landmarks.json").read_text())


@pytest.fixture(scope="module")
def expected():
    path = FIXTURES / "expected-metrics.json"
    if not path.exists():  # pragma: no cover - developer error
        pytest.skip("run `node tools/dump-metrics.mjs` to generate the reference")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def session(landmarks):
    s = PoseSession(SessionOptions(racket_arm=landmarks["racket_arm"]))
    for f in landmarks["frames"]:
        s.push(f["t"], f["world"], f["image"], f["frame"])
    return s


def approx(name, got, want):
    if want is None:
        assert got is None or not math.isfinite(got), f"{name}: expected undefined, got {got}"
        return
    assert got is not None and math.isfinite(got), f"{name}: expected {want}, got {got}"
    assert abs(got - want) <= TOL, f"{name}: expected {want}, got {got} (diff {abs(got - want):.3g})"


def test_same_number_of_usable_frames(session, expected):
    assert len(session.frames) == expected["frameCount"]


def test_frame_metrics_match(session, expected):
    by_frame = {f["frame"]: f for f in session.frames}
    keys = [
        ("elbow", "elbow"), ("elbowOff", "elbow_off"),
        ("shoulderElevation", "shoulder_elevation"),
        ("shoulderElevationOff", "shoulder_elevation_off"),
        ("shoulderAzimuth", "shoulder_azimuth"),
        ("separation", "separation"),
        ("kneeLeft", "knee_left"), ("kneeRight", "knee_right"),
        ("stanceWidth", "stance_width"),
        ("trunkLength", "trunk_length"), ("armLength", "arm_length"),
    ]
    for row in expected["metrics"]:
        frame = by_frame[row["frame"]]
        m = frame["metrics"]
        for js_key, py_key in keys:
            approx(f"frame {row['frame']} {js_key}", m[py_key], row[js_key])
        approx(f"frame {row['frame']} trunkLeanTotal", m["trunk_lean"]["total"], row["trunkLeanTotal"])
        approx(f"frame {row['frame']} trunkLeanForward", m["trunk_lean"]["forward"], row["trunkLeanForward"])
        approx(f"frame {row['frame']} trunkLeanLateral", m["trunk_lean"]["lateral"], row["trunkLeanLateral"])
        for js_key, py_key in [("handHeight", "height"), ("handLateral", "lateral"),
                               ("handForward", "forward"), ("handExtension", "extension")]:
            approx(f"frame {row['frame']} {js_key}", m["hand"][py_key], row[js_key])


def test_racket_arm_guess_matches(session, expected):
    guess = detect_racket_arm(session.frames)
    assert guess["arm"] == expected["racketArmGuess"]["arm"]
    approx("racket arm margin", guess["margin"], expected["racketArmGuess"]["margin"])


def test_same_strokes_detected(session, expected):
    strokes = session.strokes()
    assert len(strokes) == len(expected["strokes"]), (
        f"found {len(strokes)} strokes, JavaScript found {len(expected['strokes'])}"
    )
    for got, want in zip(strokes, expected["strokes"]):
        assert got["frame"] == want["frame"]
        assert got["shot"] == want["shot"]
        assert got["side"] == want["side"]
        assert got["height"] == want["height"]
        approx("stroke t", got["t"], want["t"])
        approx("confidence", got["confidence"], want["confidence"])
        approx("peak speed", got["peak_speed"], want["peakSpeed"])
        approx("peak speed m/s", got["peak_speed_ms"], want["peakSpeedMs"])
        approx("start t", got["start_t"], want["startT"])
        approx("end t", got["end_t"], want["endT"])
        approx("contact elbow", got["contact"]["elbow"], want["contactElbow"])
        approx("contact elevation", got["contact"]["shoulder_elevation"], want["contactElevation"])
        approx("hand height", got["contact"]["hand"]["height"], want["contactHandHeight"])
        approx("hand lateral", got["contact"]["hand"]["lateral"], want["contactHandLateral"])
        approx("hand forward", got["contact"]["hand"]["forward"], want["contactHandForward"])
        approx("trunk lean forward", got["contact"]["trunk_lean"]["forward"],
               want["contactTrunkLeanForward"])
        approx("backswing min elbow", got["backswing"]["min_elbow"], want["backswingMinElbow"])
        approx("backswing max separation", got["backswing"]["max_separation"],
               want["backswingMaxSeparation"])
        approx("backswing off arm", got["backswing"]["max_off_arm_elevation"],
               want["backswingMaxOffArmElevation"])
        approx("window min knee", got["window"]["min_knee"], want["windowMinKnee"])


def test_same_coaching_cues(session, expected):
    strokes = session.strokes()
    cues = [c for s in strokes for c in coach_stroke(s)]
    assert len(cues) == len(expected["strokeCues"])
    for got, want in zip(cues, expected["strokeCues"]):
        assert got["id"] == want["id"]
        assert got["stroke_index"] == want["strokeIndex"]
        assert got["status"] == want["status"], f"{got['id']}: {got['status']} vs {want['status']}"
        approx(got["id"], got["value"], want["value"])


def test_same_session_cues(session, expected):
    cues = coach_session(session.frames, session.strokes())
    assert [c["id"] for c in cues] == [c["id"] for c in expected["sessionCues"]]
    for got, want in zip(cues, expected["sessionCues"]):
        assert got["status"] == want["status"]
        approx(got["id"], got["value"], want["value"])
