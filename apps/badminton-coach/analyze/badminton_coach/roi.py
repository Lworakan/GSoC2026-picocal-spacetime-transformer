"""Two-stage pose tracking: find the player once, then follow a cropped ROI.

Filming a badminton court with a phone puts the player at 15-25% of the frame
height, which is where a single full-frame pass starts dropping detections --
measured at 58% of frames on the reference clip.  Cropping a padded square around
the last known pose and upscaling it before inference took the same clip to 96%,
and it is also *cheaper*, because a 384x384 crop is a fraction of a 1080p frame.

The state machine is kept free of any particular pose library: callers hand in
two callbacks, so the browser app and the Python pipeline can share this design
and be tested with a fake detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in normalised image coordinates (0..1)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def distance_to(self, x: float, y: float) -> float:
        return ((self.cx - x) ** 2 + (self.cy - y) ** 2) ** 0.5


def bounds(points: Sequence[Sequence[float]]) -> Box:
    """Bounding box of a landmark list of ``[x, y, ...]``."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return Box(min(xs), min(ys), max(xs), max(ys))


def crop_window(box: Box, pad: float = 0.85, aspect: float = 1.0) -> Box:
    """A square-ish window around ``box``, padded for the arm swing.

    The pad is generous on purpose: a badminton player's racket arm leaves the
    torso box entirely at the top of a smash, and a crop that clips the wrist is
    worse than no crop at all.  ``aspect`` is width/height of the source frame,
    used to keep the crop square *in pixels* rather than in normalised units.
    """
    half = max(box.width * aspect, box.height) * pad
    half_x = half / aspect
    return Box(
        max(0.0, box.cx - half_x),
        max(0.0, box.cy - half),
        min(1.0, box.cx + half_x),
        min(1.0, box.cy + half),
    )


def unproject(points: Sequence[Sequence[float]], window: Box) -> list[list[float]]:
    """Map landmarks measured inside ``window`` back to full-frame coordinates."""
    out = []
    for p in points:
        mapped = [window.x0 + p[0] * window.width, window.y0 + p[1] * window.height]
        mapped.extend(p[2:])
        out.append(mapped)
    return out


@dataclass
class RoiTrackerOptions:
    pad: float = 0.85
    """Crop padding as a multiple of the pose's larger dimension."""

    max_misses: int = 8
    """Consecutive empty crops before falling back to a full-frame search."""

    max_size_ratio: float = 2.2
    """Reject a crop result whose size jumps by more than this -- it is a different
    person who wandered into the crop, not our player."""

    max_jump: float = 0.25
    """Reject a crop result whose centre moves further than this in one frame."""

    reacquire_radius: float = 0.35
    """When re-acquiring, only accept a full-frame detection this close to where
    the player was last seen."""


@dataclass
class RoiTracker:
    """Follow one person across frames.

    ``detect_full`` takes a frame and returns every pose it can find, as a list of
    landmark lists in normalised full-frame coordinates.  ``detect_crop`` takes a
    frame and a :class:`Box` and returns at most one pose, in coordinates local to
    that crop.
    """

    detect_full: Callable[[object], list]
    detect_crop: Callable[[object, Box], object | None]
    options: RoiTrackerOptions = field(default_factory=RoiTrackerOptions)
    aspect: float = 1.0

    box: Box | None = None
    misses: int = 0
    #: Set by the caller (a tap in the app, ``--subject-at`` on the CLI) to say
    #: which person to lock on to at the next acquisition.
    target_hint: tuple[float, float] | None = None
    stats: dict = field(default_factory=lambda: {"full": 0, "crop": 0, "hits": 0, "reacquisitions": 0})

    @property
    def tracking(self) -> bool:
        return self.box is not None

    def reset(self, hint: tuple[float, float] | None = None) -> None:
        self.box = None
        self.misses = 0
        self.target_hint = hint

    def step(self, frame) -> object | None:
        """Return this frame's pose for the tracked player, or ``None``."""
        if self.box is not None:
            pose = self._step_crop(frame)
            if pose is not None:
                return pose
            # A miss keeps the lock and reports nothing for this frame. Falling
            # straight through to a full-frame search would spend a full-
            # resolution inference on every dropped frame and make ``max_misses``
            # mean nothing; ``_step_crop`` clears ``box`` once it has missed too
            # often, and only then do we search again.
            if self.box is not None:
                return None
        return self._step_full(frame)

    def _step_crop(self, frame):
        window = crop_window(self.box, self.options.pad, self.aspect)
        self.stats["crop"] += 1
        result = self.detect_crop(frame, window)
        if result is None:
            self.misses += 1
            if self.misses > self.options.max_misses:
                self.box = None
                self.misses = 0
            return None

        points = unproject(result["image"], window)
        new_box = bounds(points)
        if not self._plausible(new_box):
            self.misses += 1
            if self.misses > self.options.max_misses:
                self.box = None
                self.misses = 0
            return None

        self.misses = 0
        self.box = new_box
        self._last_box = new_box
        self.stats["hits"] += 1
        return dict(result, image=points, source="crop", box=new_box)

    def _plausible(self, new_box: Box) -> bool:
        """Guard against the crop latching on to a different body."""
        old = self.box
        if old is None:
            return True
        if old.height > 1e-6 and new_box.height > 1e-6:
            ratio = max(old.height / new_box.height, new_box.height / old.height)
            if ratio > self.options.max_size_ratio:
                return False
        if new_box.distance_to(old.cx, old.cy) > self.options.max_jump:
            return False
        return True

    def _step_full(self, frame):
        self.stats["full"] += 1
        poses = self.detect_full(frame) or []
        if not poses:
            return None
        pose = self._choose(poses)
        if pose is None:
            return None
        self.box = bounds(pose["image"])
        self._last_box = self.box
        self.misses = 0
        self.stats["hits"] += 1
        self.stats["reacquisitions"] += 1
        return dict(pose, source="full", box=self.box)

    def _choose(self, poses: list):
        """Pick which full-frame detection is our player.

        A tap from the user wins.  Otherwise, if we were tracking someone a moment
        ago, prefer whoever is nearest to where they were -- never simply the
        biggest pose, because the biggest pose is often a bystander walking
        between the phone and the court.
        """
        hint = self.target_hint
        if hint is not None:
            inside = [p for p in poses if bounds(p["image"]).contains(*hint)]
            pool = inside or poses
            chosen = min(pool, key=lambda p: bounds(p["image"]).distance_to(*hint))
            self.target_hint = None
            return chosen

        last = self._last_box
        if last is not None:
            near = [
                p
                for p in poses
                if bounds(p["image"]).distance_to(last.cx, last.cy)
                <= self.options.reacquire_radius
            ]
            if near:
                return min(near, key=lambda p: bounds(p["image"]).distance_to(last.cx, last.cy))
        return max(poses, key=lambda p: bounds(p["image"]).height)

    def __post_init__(self):
        #: Where the player was last seen, kept across a loss so that
        #: re-acquisition can prefer that spot over the largest pose in frame.
        self._last_box: Box | None = None
