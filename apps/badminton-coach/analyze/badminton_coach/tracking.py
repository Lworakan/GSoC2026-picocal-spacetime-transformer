"""Associate per-frame pose detections into stable person tracks.

MediaPipe's multi-pose output is not identity-stable across frames: the order in
which poses come back can change, and a pose can drop out for a few frames when
the player is occluded or motion-blurred.  On a badminton court that matters,
because the neighbouring court is usually in shot and we must not let the metrics
jump to a stranger mid-rally.

The association here is deliberately simple -- greedy nearest-centroid with a
size gate -- because the players we care about are far apart in the image and
move slowly relative to the frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


def _centroid(landmarks: Iterable[tuple[float, float]]) -> tuple[float, float]:
    xs, ys = zip(*landmarks)
    return sum(xs) / len(xs), sum(ys) / len(ys)


@dataclass
class Track:
    """One person followed through the clip."""

    track_id: int
    frames: list[dict] = field(default_factory=list)
    last_frame: int = -1
    last_centroid: tuple[float, float] = (0.0, 0.0)
    last_height: float = 0.0

    @property
    def first_frame(self) -> int:
        return self.frames[0]["frame"] if self.frames else -1

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def span(self) -> int:
        """Frames between first and last detection, gaps included."""
        return 0 if not self.frames else self.last_frame - self.first_frame + 1

    def mean_height(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f["bbox_height"] for f in self.frames) / len(self.frames)

    def mean_x(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f["centroid"][0] for f in self.frames) / len(self.frames)


class Tracker:
    """Greedy nearest-centroid tracker over normalised image coordinates.

    ``max_distance`` and ``max_size_ratio`` are in normalised-image units and are
    tuned for a phone filming a court from the side, where a player crosses at
    most a few percent of the frame per 1/30 s.
    """

    def __init__(
        self,
        max_distance: float = 0.12,
        max_size_ratio: float = 2.0,
        max_age: int = 15,
    ) -> None:
        self.max_distance = max_distance
        self.max_size_ratio = max_size_ratio
        self.max_age = max_age
        self.tracks: list[Track] = []
        self._next_id = 0

    def _new_track(self) -> Track:
        track = Track(track_id=self._next_id)
        self._next_id += 1
        self.tracks.append(track)
        return track

    def update(self, frame_index: int, detections: list[dict]) -> None:
        """Assign this frame's detections to tracks, creating tracks as needed.

        Each detection must carry ``centroid`` and ``bbox_height``; everything
        else in the dict is stored verbatim on the track.
        """
        live = [t for t in self.tracks if frame_index - t.last_frame <= self.max_age]

        # Score every (track, detection) pair, then take them cheapest-first.
        pairs = []
        for ti, track in enumerate(live):
            for di, det in enumerate(detections):
                dx = track.last_centroid[0] - det["centroid"][0]
                dy = track.last_centroid[1] - det["centroid"][1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self.max_distance:
                    continue
                if track.last_height > 0 and det["bbox_height"] > 0:
                    ratio = max(
                        track.last_height / det["bbox_height"],
                        det["bbox_height"] / track.last_height,
                    )
                    if ratio > self.max_size_ratio:
                        continue
                pairs.append((dist, ti, di))
        pairs.sort()

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _dist, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            self._attach(live[ti], frame_index, detections[di])

        for di, det in enumerate(detections):
            if di not in used_dets:
                self._attach(self._new_track(), frame_index, det)

    @staticmethod
    def _attach(track: Track, frame_index: int, det: dict) -> None:
        record = dict(det)
        record["frame"] = frame_index
        track.frames.append(record)
        track.last_frame = frame_index
        track.last_centroid = det["centroid"]
        track.last_height = det["bbox_height"]

    def finished(self, min_frames: int = 5) -> list[Track]:
        """Tracks worth keeping, longest first."""
        keep = [t for t in self.tracks if t.n_frames >= min_frames]
        keep.sort(key=lambda t: t.n_frames, reverse=True)
        return keep


def pick_subject(tracks: list[Track]) -> Track | None:
    """Choose which track is 'the player' when the user has not said.

    The subject of a self-filmed practice clip is the person who is both large in
    frame (near court, so the pose is actually measurable) and present for most of
    it.  Scoring on ``mean_height * n_frames`` picks that person over a bystander
    on the next court, who is smaller, and over a passer-by, who is brief.
    """
    if not tracks:
        return None
    return max(tracks, key=lambda t: t.mean_height() * t.n_frames)
