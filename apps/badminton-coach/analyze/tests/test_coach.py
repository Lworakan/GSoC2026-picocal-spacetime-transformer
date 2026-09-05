"""The coaching rules: grading, applicability, and how faults are ranked."""

import math

import pytest

from badminton_coach.coach import RULES, coach_session, coach_stroke, grade, rank_cues


def stroke(**overrides):
    base = {
        "index": 0, "t": 1.0,
        "shot": "forehand-overhead", "side": "forehand", "height": "overhead",
        "confidence": 0.9,
        "contact": {
            "elbow": 165, "elbow_off": 170, "shoulder_elevation": 160,
            "shoulder_elevation_off": 60, "shoulder_azimuth": 20, "separation": 25,
            "trunk_lean": {"total": 15, "forward": 10, "lateral": 8},
            "knee_left": 160, "knee_right": 155, "stance_width": 1.4,
            "hand": {"height": 0.5, "lateral": 0.6, "forward": 0.3, "extension": 0.95},
        },
        "backswing": {
            "min_elbow": 95, "max_separation": 30,
            "max_shoulder_elevation": 165, "max_off_arm_elevation": 140,
        },
        "window": {"max_elbow": 172, "min_knee": 140},
    }
    for key in ("contact", "backswing", "window"):
        if key in overrides:
            base[key] = {**base[key], **overrides.pop(key)}
    base.update(overrides)
    return base


def test_grade_bands():
    band = {"min": 150, "warn_min": 135}
    assert grade(160, band) == "good"
    assert grade(142, band) == "warn"
    assert grade(120, band) == "bad"
    assert grade(float("nan"), band) == "unknown"


def test_grade_two_sided_band():
    band = {"min": 95, "max": 145, "warn_min": 80, "warn_max": 160}
    assert grade(120, band) == "good"
    assert grade(150, band) == "warn"
    assert grade(170, band) == "bad"
    assert grade(70, band) == "bad"


def test_a_well_played_overhead_raises_no_complaints():
    cues = coach_stroke(stroke())
    assert cues
    assert [c for c in cues if c["status"] != "good"] == []


def test_a_bent_arm_at_contact_is_called_out():
    cues = coach_stroke(stroke(contact={"elbow": 110}))
    cue = next(c for c in cues if c["id"] == "overhead-elbow-extension")
    assert cue["status"] == "bad"
    assert len(cue["why"]) > 40


def test_a_dropped_free_arm_and_missing_turn_are_caught():
    cues = coach_stroke(stroke(backswing={"max_off_arm_elevation": 40, "max_separation": 5}))
    assert next(c for c in cues if c["id"] == "overhead-free-arm")["status"] == "bad"
    assert next(c for c in cues if c["id"] == "overhead-body-rotation")["status"] == "bad"


def test_rules_only_fire_for_their_own_shot_family():
    overhead = [c["id"] for c in coach_stroke(stroke())]
    net = [c["id"] for c in coach_stroke(stroke(
        shot="forehand-underarm", height="underarm",
        contact={"hand": {"height": -0.8, "lateral": 0.5, "forward": 0.3, "extension": 0.8}},
    ))]
    assert "overhead-free-arm" in overhead
    assert "overhead-free-arm" not in net
    assert "net-lunge-knee" in net


def test_a_coin_flip_shot_is_not_coached():
    assert coach_stroke(stroke(confidence=0.05)) == []


def test_a_missing_measurement_is_skipped():
    cues = coach_stroke(stroke(backswing={"max_off_arm_elevation": float("nan")}))
    assert not any(c["id"] == "overhead-free-arm" for c in cues)


def test_every_rule_states_a_target_and_a_reason():
    for rule in RULES:
        assert rule.target_text()
        assert len(rule.why) > 40, rule.id
        assert rule.label


def test_rule_ids_are_unique():
    ids = [r.id for r in RULES]
    assert len(set(ids)) == len(ids)


def test_session_cues_read_posture_away_from_contact():
    frames = [{"t": i / 30, "metrics": {"knee_left": 178, "knee_right": 176, "stance_width": 1.5}}
              for i in range(60)]
    cues = coach_session(frames, [])
    assert next(c for c in cues if c["id"] == "ready-knee-bend")["status"] == "bad"
    assert next(c for c in cues if c["id"] == "ready-stance-width")["status"] == "good"


def test_frames_near_a_contact_are_excluded():
    frames = [{"t": 1.0 + (i - 10) / 100,
               "metrics": {"knee_left": 178, "knee_right": 176, "stance_width": 1.5}}
              for i in range(20)]
    assert coach_session(frames, [{"t": 1.0}]) == []


def test_recovery_cues_appear_with_court_data():
    recovery = [
        {"recovery_seconds": 2.4, "recovered": True},
        {"recovery_seconds": None, "recovered": False},
        {"recovery_seconds": 2.6, "recovered": True},
    ]
    cues = coach_session([], [], recovery=recovery)
    assert next(c for c in cues if c["id"] == "recovery-time")["status"] == "bad"
    rate = next(c for c in cues if c["id"] == "recovery-rate")
    assert rate["value"] == pytest.approx(2 / 3)


def test_rank_cues_groups_repeats_and_puts_the_worst_first():
    cues = [
        {"id": "a", "label": "A", "why": "x", "unit": "d", "target": "t", "value": 1, "status": "warn"},
        {"id": "b", "label": "B", "why": "y", "unit": "d", "target": "t", "value": 2, "status": "bad"},
        {"id": "b", "label": "B", "why": "y", "unit": "d", "target": "t", "value": 4, "status": "bad"},
        {"id": "c", "label": "C", "why": "z", "unit": "d", "target": "t", "value": 9, "status": "good"},
    ]
    ranked = rank_cues(cues)
    assert len(ranked) == 2
    assert ranked[0]["id"] == "b"
    assert ranked[0]["count"] == 2
    assert ranked[0]["mean"] == pytest.approx(3)
