"""The report: it must be assembled and rendered without a video anywhere near it."""

import json
from pathlib import Path

import pytest

from badminton_coach.court import CourtCalibration
from badminton_coach.report import build_report, render_markdown, write_report
from badminton_coach.session import PoseSession, SessionOptions

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
VIEW = [(0.20, 0.60), (0.80, 0.60), (0.95, 0.95), (0.05, 0.95)]


@pytest.fixture(scope="module")
def track():
    data = json.loads((FIXTURES / "landmarks.json").read_text())
    return {
        "video": {"path": "fixture", "fps": data["fps"], "width": 1920, "height": 1080,
                  "frames_analysed": len(data["frames"])},
        "model": "fixture",
        "coverage": 1.0,
        "detector_stats": {"full": 1, "crop": len(data["frames"]) - 1},
        "frames": data["frames"],
    }


@pytest.fixture(scope="module")
def session(track):
    s = PoseSession(SessionOptions(racket_arm="right"))
    return s.push_track(track)


def test_report_is_json_serialisable(track, session):
    report = build_report(track, session, session.analyse())
    json.dumps(report)  # must not raise: NaN would produce invalid JSON
    assert "NaN" not in json.dumps(report)


def test_report_contains_the_analysis(track, session):
    analysis = session.analyse()
    report = build_report(track, session, analysis)
    assert report["summary"]["count"] == len(analysis["strokes"])
    assert report["racket_arm"] == "right"
    assert report["court"] is None


def test_markdown_renders_without_a_court(track, session):
    md = render_markdown(build_report(track, session, session.analyse()))
    assert md.startswith("# Badminton session report")
    assert "The court was not set up" in md
    assert "How to read these numbers" in md


def test_markdown_renders_with_a_court(track, session):
    report = build_report(track, session, session.analyse(), CourtCalibration(VIEW))
    md = render_markdown(report)
    assert "distance covered" in md
    assert report["court"]["frames_on_court"] >= 0


def test_session_cues_are_not_duplicated_into_the_fault_ranking(track, session):
    report = build_report(track, session, session.analyse(), CourtCalibration(VIEW))
    ranked = {g["id"] for g in report["ranked_faults"]}
    session_ids = {c["id"] for c in report["session_cues"]}
    assert not (ranked & session_ids)


def test_write_report_produces_both_files(track, session, tmp_path):
    paths = write_report(build_report(track, session, session.analyse()), tmp_path)
    assert paths["json"].exists() and paths["markdown"].exists()
    assert json.loads(paths["json"].read_text())["tool"] == "badminton-coach analyze"


def test_a_report_with_no_strokes_still_renders(track):
    empty = PoseSession(SessionOptions(racket_arm="right"))
    analysis = empty.analyse()
    md = render_markdown(build_report(track, empty, analysis))
    assert "No swings were detected" in md
