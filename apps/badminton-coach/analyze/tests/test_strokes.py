"""Stroke detection and shot naming, against synthesised swings."""

import math

import pytest

from badminton_coach.biomech import frame_metrics
from badminton_coach.strokes import (
    ShotThresholds, StrokeOptions, classify_shot, detect_strokes, summarise_strokes,
    wrist_speed_series,
)
from conftest import make_skeleton


def make_swing_sequence(contact_times, seconds=4.0, fps=60, **pose):
    """A sequence where the racket arm sweeps through a swing at each contact time."""
    frames = []
    for i in range(int(seconds * fps)):
        t = i / fps
        elevation = 20.0
        for contact in contact_times:
            phase = (t - contact) / 0.25
            if -1 < phase < 1:
                elevation += 150 * math.exp(-(phase * phase) * 3)
        world = make_skeleton(arm_elevation=elevation, **pose)
        frames.append({
            "t": t, "frame": i, "world": world,
            "image": [[0.5, 0.5, 0.0, 1.0] for _ in world],
            "metrics": frame_metrics(world, "right", trunk=0.5, arm_length=0.57),
        })
    return frames


def test_wrist_speed_is_zero_for_a_still_figure():
    frames = make_swing_sequence([])
    speed = [v for v in wrist_speed_series(frames, "right", [0.5] * len(frames))
             if math.isfinite(v)]
    assert max(speed) < 0.01


def test_one_swing_is_one_stroke_at_the_right_moment():
    strokes = detect_strokes(make_swing_sequence([1.5]))
    assert len(strokes) == 1
    assert abs(strokes[0]["t"] - 1.5) < 0.12


def test_separate_swings_are_counted_separately():
    assert len(detect_strokes(make_swing_sequence([0.8, 2.0, 3.2]))) == 3


def test_refractory_collapses_peaks_that_are_too_close():
    frames = make_swing_sequence([1.5, 1.62])
    assert len(detect_strokes(frames, StrokeOptions(refractory=0.2))) == 2
    assert len(detect_strokes(frames, StrokeOptions(refractory=0.5))) == 1


def test_the_strongest_peak_survives_a_collapse():
    frames = make_swing_sequence([1.5, 1.62])
    both = detect_strokes(frames, StrokeOptions(refractory=0.2))
    strongest = max(both, key=lambda s: s["peak_speed"])
    kept = detect_strokes(frames, StrokeOptions(refractory=0.5))[0]
    assert kept["t"] == pytest.approx(strongest["t"])


def test_a_stroke_carries_its_phases():
    stroke = detect_strokes(make_swing_sequence([1.5]))[0]
    assert stroke["start_t"] < stroke["t"] < stroke["end_t"]
    assert stroke["backswing_duration"] > 0
    assert math.isfinite(stroke["backswing"]["max_shoulder_elevation"])
    assert math.isfinite(stroke["window"]["min_knee"])


def test_the_threshold_controls_what_counts_as_a_swing():
    frames = make_swing_sequence([1.5])
    assert detect_strokes(frames, StrokeOptions(peak_speed=100)) == []
    assert len(detect_strokes(frames, StrokeOptions(peak_speed=1))) == 1


def test_a_barely_visible_swing_is_skipped():
    frames = make_swing_sequence([1.5])
    for f in frames:
        f["image"] = [[0.5, 0.5, 0.0, 0.1] for _ in f["image"]]
    assert detect_strokes(frames) == []


def test_peak_speed_is_reported_in_both_units():
    stroke = detect_strokes(make_swing_sequence([1.5]))[0]
    assert stroke["peak_speed"] > 0
    assert stroke["peak_speed_ms"] == pytest.approx(stroke["peak_speed"] * 0.5)


def test_height_separates_the_shot_families():
    assert classify_shot({"height": 0.6, "lateral": 0.5})["height"] == "overhead"
    assert classify_shot({"height": -0.2, "lateral": 0.5})["height"] == "drive"
    assert classify_shot({"height": -0.9, "lateral": 0.5})["height"] == "underarm"


def test_crossing_the_midline_makes_a_backhand():
    assert classify_shot({"height": 0.5, "lateral": 0.8})["side"] == "forehand"
    assert classify_shot({"height": 0.5, "lateral": -0.4})["side"] == "backhand"
    assert classify_shot({"height": 0.5, "lateral": 0.15})["side"] == "roundhead"


def test_round_the_head_only_exists_above_the_shoulders():
    assert classify_shot({"height": -0.2, "lateral": 0.15})["side"] == "forehand"


def test_confidence_falls_near_a_boundary():
    th = ShotThresholds()
    assert classify_shot({"height": 0.7, "lateral": 0.9})["confidence"] > 0.9
    borderline = classify_shot({"height": th.overhead_height + 0.001, "lateral": 0.9})
    assert borderline["confidence"] < 0.05


def test_summarise_strokes():
    strokes = [
        {"shot": "forehand-overhead", "side": "forehand", "height": "overhead", "peak_speed_ms": 10},
        {"shot": "backhand-drive", "side": "backhand", "height": "drive", "peak_speed_ms": 6},
        {"shot": "forehand-overhead", "side": "forehand", "height": "overhead", "peak_speed_ms": 8},
    ]
    s = summarise_strokes(strokes)
    assert s["count"] == 3
    assert s["forehand"] == 2
    assert s["backhand"] == 1
    assert s["overhead"] == 2
    assert s["by_shot"]["forehand-overhead"] == 2
    assert s["max_peak_speed_ms"] == 10
    assert s["mean_peak_speed_ms"] == pytest.approx(8)


def test_a_short_sequence_yields_nothing():
    assert detect_strokes([]) == []
    assert detect_strokes(make_swing_sequence([])[:3]) == []
