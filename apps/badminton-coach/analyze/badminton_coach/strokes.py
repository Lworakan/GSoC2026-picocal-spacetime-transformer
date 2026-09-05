"""Find swings in a pose sequence and say what kind of shot each one was.

Mirrors ``web/js/core/strokes.js``. The reasoning, in short:

* The signal is racket-wrist speed *relative to the hips*, in trunk lengths per
  second. Hip-relative so that running to the back of the court is not read as a
  swing; in trunk lengths so one threshold covers a junior and an adult, and a
  phone 5 m away and one 15 m away.
* A stroke is a local maximum of that speed. Peak wrist speed is the standard
  stand-in for contact in racket-sports analysis: true contact is within a frame
  or two of it, and unlike the shuttle, the wrist is something a pose model can
  actually see.
* At contact the hand is expressed in the torso frame. Its height relative to the
  shoulders separates overhead from drive from underarm; its lateral position,
  signed so positive is always the racket side, separates forehand from backhand
  -- a backhand being exactly the shot where the racket hand has crossed the
  body's midline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import landmarks as L
from .biomech import core_visibility
from .geometry import distance, midpoint, sub

NAN = float("nan")


@dataclass
class StrokeOptions:
    """Tunables for :func:`detect_strokes`. Speeds are in trunk lengths per second."""

    peak_speed: float = 6.0
    """A peak must reach this to count as a swing rather than a fidget."""

    quiet_speed: float = 2.5
    """Speed below which the arm counts as quiet, used to find a swing's edges."""

    refractory: float = 0.30
    """Minimum gap between two contacts; badminton exchanges rarely beat this."""

    max_backswing: float = 0.80
    max_follow_through: float = 0.60
    min_visibility: float = 0.5
    peak_window: int = 3
    """Half-width, in frames, of the neighbourhood a peak must dominate."""


@dataclass
class ShotThresholds:
    """Family boundaries, in trunk lengths from the shoulder line."""

    overhead_height: float = 0.10
    underarm_height: float = -0.45
    forehand_lateral: float = 0.28
    backhand_lateral: float = 0.02


def wrist_speed_series(frames, racket_arm: str, trunk_lengths) -> list[float]:
    """Racket-wrist speed relative to the hips, in trunk lengths per second."""
    wrist_index = L.side("WRIST", racket_arm)
    relative = [
        sub(f["world"][wrist_index], midpoint(f["world"][L.LEFT_HIP], f["world"][L.RIGHT_HIP]))
        for f in frames
    ]
    speed = [NAN] * len(frames)
    for i in range(1, len(frames) - 1):
        dt = frames[i + 1]["t"] - frames[i - 1]["t"]
        if dt <= 0:
            continue
        trunk = trunk_lengths[i]
        if not (trunk and math.isfinite(trunk) and trunk > 1e-6):
            continue
        speed[i] = distance(relative[i + 1], relative[i - 1]) / dt / trunk
    if len(speed) > 2:
        speed[0] = speed[1]
        speed[-1] = speed[-2]
    return speed


def classify_shot(hand: dict, thresholds: ShotThresholds | None = None) -> dict:
    """Name the shot from the hand's position in the torso frame at contact."""
    th = thresholds or ShotThresholds()
    height, lateral = hand["height"], hand["lateral"]

    if height >= th.overhead_height:
        height_class = "overhead"
    elif height >= th.underarm_height:
        height_class = "drive"
    else:
        height_class = "underarm"

    if lateral >= th.forehand_lateral:
        side_class = "forehand"
    elif lateral <= th.backhand_lateral:
        side_class = "backhand"
    else:
        side_class = "roundhead"

    # A shot played over the head is only its own category when it is actually
    # overhead; lower down, the same lateral band is a straight-on forehand.
    if side_class == "roundhead" and height_class != "overhead":
        side_class = "forehand"

    # Distance from the nearest decision boundary, as a rough confidence: shots
    # near a boundary are the ones a human would hesitate over too.
    side_margin = min(abs(lateral - th.forehand_lateral), abs(lateral - th.backhand_lateral))
    height_margin = min(abs(height - th.overhead_height), abs(height - th.underarm_height))
    confidence = max(0.0, min(1.0, min(side_margin, height_margin) / 0.25))

    return {
        "shot": f"{side_class}-{height_class}",
        "side": side_class,
        "height": height_class,
        "confidence": confidence,
    }


def _local_peaks(speed, options: StrokeOptions) -> list[int]:
    w = options.peak_window
    peaks = []
    for i in range(w, len(speed) - w):
        v = speed[i]
        if not (v is not None and math.isfinite(v) and v >= options.peak_speed):
            continue
        dominant = True
        for j in range(i - w, i + w + 1):
            if j == i:
                continue
            other = speed[j]
            if other is None or not math.isfinite(other):
                continue
            # `>=` on the left and `>` on the right keeps exactly one index on a plateau.
            if (other >= v) if j < i else (other > v):
                dominant = False
                break
        if dominant:
            peaks.append(i)
    return peaks


def _apply_refractory(peaks, speed, times, refractory: float) -> list[int]:
    # Strongest first, so that when two candidates are too close we keep the real
    # contact rather than whichever came first in time.
    kept: list[int] = []
    for p in sorted(peaks, key=lambda i: speed[i], reverse=True):
        if all(abs(times[p] - times[k]) >= refractory for k in kept):
            kept.append(p)
    return sorted(kept)


def _walk_to(speed, times, start: int, direction: int, limit: float, quiet: float) -> int:
    i = start
    while True:
        nxt = i + direction
        if nxt < 0 or nxt >= len(speed):
            break
        if abs(times[nxt] - times[start]) > limit:
            break
        i = nxt
        v = speed[i]
        if v is not None and math.isfinite(v) and v <= quiet:
            break
    return i


def _min_by(items, fn) -> float:
    values = [fn(x) for x in items]
    values = [v for v in values if v is not None and math.isfinite(v)]
    return min(values) if values else NAN


def _max_by(items, fn) -> float:
    values = [fn(x) for x in items]
    values = [v for v in values if v is not None and math.isfinite(v)]
    return max(values) if values else NAN


def detect_strokes(frames, options: StrokeOptions | None = None,
                   thresholds: ShotThresholds | None = None) -> list[dict]:
    """Detect every stroke in a sequence of analysed frames.

    Each frame is ``{"t", "world", "metrics", "image"?}`` with ``world`` already
    in the y-up frame and ``metrics`` from :func:`biomech.frame_metrics`.
    """
    opts = options or StrokeOptions()
    th = thresholds or ShotThresholds()
    if len(frames) < 5:
        return []

    times = [f["t"] for f in frames]
    trunks = [f["metrics"]["trunk_length"] for f in frames]
    racket_arm = frames[0]["metrics"]["racket_arm"]
    speed = wrist_speed_series(frames, racket_arm, trunks)
    peaks = _apply_refractory(_local_peaks(speed, opts), speed, times, opts.refractory)

    strokes = []
    for peak in peaks:
        image = frames[peak].get("image")
        visibility = core_visibility(image) if image else 1.0
        if visibility < opts.min_visibility:
            continue

        start = _walk_to(speed, times, peak, -1, opts.max_backswing, opts.quiet_speed)
        end = _walk_to(speed, times, peak, +1, opts.max_follow_through, opts.quiet_speed)
        contact = frames[peak]["metrics"]
        label = classify_shot(contact["hand"], th)
        window = frames[start:end + 1]
        backswing = frames[start:peak + 1]

        strokes.append({
            "index": len(strokes),
            "frame": frames[peak].get("frame"),
            "t": times[peak],
            "start_t": times[start],
            "end_t": times[end],
            "start_frame": frames[start].get("frame"),
            "end_frame": frames[end].get("frame"),
            **label,
            "visibility": visibility,
            "peak_speed": speed[peak],
            "peak_speed_ms": speed[peak] * contact["trunk_length"],
            "backswing_duration": times[peak] - times[start],
            "follow_through_duration": times[end] - times[peak],
            "contact": {
                "elbow": contact["elbow"],
                "elbow_off": contact["elbow_off"],
                "shoulder_elevation": contact["shoulder_elevation"],
                "shoulder_elevation_off": contact["shoulder_elevation_off"],
                "shoulder_azimuth": contact["shoulder_azimuth"],
                "separation": contact["separation"],
                "trunk_lean": {k: v for k, v in contact["trunk_lean"].items() if k != "heading"},
                "knee_left": contact["knee_left"],
                "knee_right": contact["knee_right"],
                "stance_width": contact["stance_width"],
                "hand": {k: contact["hand"][k] for k in ("height", "lateral", "forward", "extension")},
            },
            # A coach reads the backswing as much as the contact: the deepest
            # elbow bend and the largest twist before contact are what the arm
            # had available to release.
            "backswing": {
                "min_elbow": _min_by(backswing, lambda f: f["metrics"]["elbow"]),
                "max_separation": _max_by(backswing, lambda f: f["metrics"]["separation"]),
                "max_shoulder_elevation": _max_by(backswing, lambda f: f["metrics"]["shoulder_elevation"]),
                "max_off_arm_elevation": _max_by(backswing, lambda f: f["metrics"]["shoulder_elevation_off"]),
            },
            "window": {
                "max_elbow": _max_by(window, lambda f: f["metrics"]["elbow"]),
                "min_knee": _min_by(window, lambda f: min(f["metrics"]["knee_left"], f["metrics"]["knee_right"])),
            },
        })
    return strokes


def summarise_strokes(strokes) -> dict:
    """Tally shots by name, for the session summary."""
    by_shot: dict[str, int] = {}
    for s in strokes:
        by_shot[s["shot"]] = by_shot.get(s["shot"], 0) + 1
    speeds = [s["peak_speed_ms"] for s in strokes if math.isfinite(s["peak_speed_ms"])]
    return {
        "count": len(strokes),
        "by_shot": by_shot,
        "forehand": sum(1 for s in strokes if s["side"] == "forehand"),
        "backhand": sum(1 for s in strokes if s["side"] == "backhand"),
        "roundhead": sum(1 for s in strokes if s["side"] == "roundhead"),
        "overhead": sum(1 for s in strokes if s["height"] == "overhead"),
        "mean_peak_speed_ms": sum(speeds) / len(speeds) if speeds else NAN,
        "max_peak_speed_ms": max(speeds) if speeds else NAN,
    }
