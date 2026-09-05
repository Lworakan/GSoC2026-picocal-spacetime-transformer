"""Glue: raw pose frames in, conditioned metrics and detected strokes out.

The same class runs over a whole clip here and, in its JavaScript twin
(``web/js/core/session.js``), live in the browser -- so the numbers shown on
court during practice are the numbers in the report afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import landmarks as L
from .biomech import body_scale, core_visibility, frame_metrics, world_to_up_frame
from .filters import LandmarkFilter, RunningMedian
from .strokes import ShotThresholds, StrokeOptions, detect_strokes, summarise_strokes, wrist_speed_series


@dataclass
class SessionOptions:
    racket_arm: str = "right"
    world_filter: dict = field(default_factory=lambda: {"min_cutoff": 1.5, "beta": 0.35})
    """1-Euro settings for the metric world landmarks."""

    image_filter: dict = field(default_factory=lambda: {"min_cutoff": 1.0, "beta": 0.15})
    """Smoother settings for the on-screen image landmarks: the overlay is judged
    by eye, and a jittery skeleton looks broken."""

    scale_window: int = 45
    """Window, in frames, of the running median used for body scale."""

    min_visibility: float = 0.35


def detect_racket_arm(frames) -> dict | None:
    """Guess which hand holds the racket from which wrist swings harder."""
    if len(frames) < 10:
        return None
    trunks = [f["metrics"]["trunk_length"] for f in frames]

    def score(arm: str) -> float:
        speeds = [v for v in wrist_speed_series(frames, arm, trunks)
                  if v is not None and math.isfinite(v)]
        if len(speeds) < 5:
            return -math.inf
        # The 95th percentile, not the mean: the non-racket arm also moves a lot
        # in badminton (it balances and points), but only the racket arm is whipped.
        speeds.sort()
        return speeds[int(0.95 * (len(speeds) - 1))]

    right, left = score("right"), score("left")
    if not math.isfinite(right) or not math.isfinite(left):
        return None
    margin = abs(right - left) / max(right, left, 1e-6)
    return {"arm": "right" if right >= left else "left",
            "margin": margin, "right": right, "left": left}


class PoseSession:
    def __init__(self, options: SessionOptions | None = None):
        self.options = options or SessionOptions()
        self.world_filter = LandmarkFilter(L.COUNT, 3, **self.options.world_filter)
        self.image_filter = LandmarkFilter(L.COUNT, 2, **self.options.image_filter)
        self.trunk_median = RunningMedian(self.options.scale_window)
        self.arm_median = RunningMedian(self.options.scale_window)
        self.frames: list[dict] = []
        self.dropped = 0

    @property
    def racket_arm(self) -> str:
        return self.options.racket_arm

    def set_racket_arm(self, arm: str) -> None:
        """Change the racket hand, re-measuring every stored frame.

        A mid-session correction fixes the history too, rather than leaving a seam
        in the data.
        """
        if arm not in ("left", "right"):
            raise ValueError(f"bad racket arm: {arm}")
        if arm == self.options.racket_arm:
            return
        self.options.racket_arm = arm
        for f in self.frames:
            f["metrics"] = frame_metrics(
                f["world"], arm,
                trunk=f["metrics"]["trunk_length"],
                arm_length=f["metrics"]["arm_length"],
            )

    def push(self, t: float, world, image=None, frame=None) -> dict | None:
        """Add one detected pose. ``world`` is raw MediaPipe (y-down)."""
        if not world or len(world) < L.COUNT:
            return None
        if image and core_visibility(image) < self.options.min_visibility:
            self.dropped += 1
            return None

        filtered = self.world_filter.filter(world_to_up_frame(world), t)
        filtered_image = self.image_filter.filter(image, t) if image else None

        raw = body_scale(filtered)
        trunk = self.trunk_median.push(raw.trunk)
        arm_raw = raw.arm_right if self.racket_arm == "right" else raw.arm_left
        arm_length = self.arm_median.push(arm_raw)

        analysed = {
            "t": t,
            "frame": frame,
            "world": filtered,
            "image": filtered_image,
            "metrics": frame_metrics(filtered, self.racket_arm, trunk=trunk, arm_length=arm_length),
        }
        self.frames.append(analysed)
        return analysed

    def push_track(self, track: dict) -> "PoseSession":
        """Load a whole track written by :mod:`badminton_coach.extract`."""
        for f in track["frames"]:
            self.push(f["t"], f["world"], f.get("image"), f.get("frame"))
        return self

    def strokes(self, options: StrokeOptions | None = None,
                thresholds: ShotThresholds | None = None) -> list[dict]:
        return detect_strokes(self.frames, options, thresholds)

    def analyse(self, options: StrokeOptions | None = None,
                thresholds: ShotThresholds | None = None) -> dict:
        strokes = self.strokes(options, thresholds)
        return {
            "racket_arm": self.racket_arm,
            "frames": len(self.frames),
            "dropped": self.dropped,
            "duration": (self.frames[-1]["t"] - self.frames[0]["t"]) if self.frames else 0.0,
            "strokes": strokes,
            "summary": summarise_strokes(strokes),
        }
