"""Run MediaPipe Pose over a video and write one identity-stable landmark track.

This is the only part of the pipeline that needs MediaPipe.  Everything
downstream -- angles, stroke detection, court mapping, coaching -- consumes the
JSON this writes, so the analysis can be re-run and re-tuned in seconds without
touching the video again.

Detection runs in two stages (see :mod:`badminton_coach.roi`): a full-frame
search to find the player, then a padded crop around the previous pose for every
following frame.  On the reference clip that is the difference between usable
data and unusable data.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .roi import Box, RoiTracker, RoiTrackerOptions
from .tracking import Tracker, pick_subject

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

DEFAULT_MODEL_DIR = Path(
    os.environ.get("BADMINTON_MODEL_DIR", Path.home() / ".cache" / "badminton-coach")
)

TRACK_FORMAT = 1


def ensure_model(variant: str = "full", model_dir: Path | None = None) -> Path:
    """Return a local path to the requested pose model, downloading on first use."""
    if variant not in MODEL_URLS:
        raise ValueError(f"unknown model variant {variant!r}; pick one of {sorted(MODEL_URLS)}")
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"pose_landmarker_{variant}.task"
    if not path.exists() or path.stat().st_size < 1_000_000:
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
        tmp.replace(path)
    return path


@dataclass
class ExtractOptions:
    model: str = "full"
    """Pose model variant. ``lite`` is fastest, ``heavy`` is the most accurate on a
    small, distant subject -- which is the usual case when a phone films a court."""

    num_poses: int = 4
    """People to look for during a full-frame search. The next court over usually
    contributes two or three."""

    crop_size: int = 384
    """Pixel size the ROI crop is resampled to before inference."""

    min_detection_confidence: float = 0.35
    min_presence_confidence: float = 0.35
    min_tracking_confidence: float = 0.35
    stride: int = 1
    """Analyse every Nth frame. Stroke detection wants 1; raise it only to preview."""

    max_frames: int | None = None
    subject_at: tuple[float, float] | None = None
    """Normalised (x, y) naming who to track, e.g. from a tap in the app."""

    model_dir: Path | None = None
    roi: RoiTrackerOptions = field(default_factory=RoiTrackerOptions)


def extract(video_path: str | Path, options: ExtractOptions | None = None,
            progress=None) -> dict:
    """Extract the subject's pose track from ``video_path``.

    Returns a JSON-serialisable dict; see ``docs/track-format.md`` for the schema.
    """
    import cv2  # imported lazily so the pure-maths modules stay dependency-free
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = options or ExtractOptions()
    model_path = ensure_model(options.model, options.model_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = width / height if height else 1.0

    def make(num_poses: int):
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=num_poses,
                min_pose_detection_confidence=options.min_detection_confidence,
                min_pose_presence_confidence=options.min_presence_confidence,
                min_tracking_confidence=options.min_tracking_confidence,
            )
        )

    full_detector = make(options.num_poses)
    crop_detector = make(1)

    def to_image(bgr):
        return mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        )

    def detect_full(frame):
        result = full_detector.detect(to_image(frame))
        return _poses_from(result)

    def detect_crop(frame, window: Box):
        x0 = int(window.x0 * width)
        x1 = int(window.x1 * width)
        y0 = int(window.y0 * height)
        y1 = int(window.y1 * height)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        patch = cv2.resize(
            frame[y0:y1, x0:x1],
            (options.crop_size, options.crop_size),
            interpolation=cv2.INTER_LINEAR,
        )
        poses = _poses_from(crop_detector.detect(to_image(patch)))
        return poses[0] if poses else None

    tracker = RoiTracker(
        detect_full=detect_full,
        detect_crop=detect_crop,
        options=options.roi,
        aspect=aspect,
    )
    if options.subject_at is not None:
        tracker.target_hint = options.subject_at

    frames: list[dict] = []
    frame_index = 0
    analysed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if options.max_frames is not None and analysed >= options.max_frames:
                break
            if frame_index % options.stride:
                frame_index += 1
                continue

            pose = tracker.step(frame)
            if pose is not None:
                box = pose["box"]
                frames.append(
                    {
                        "frame": frame_index,
                        "t": round(frame_index / fps, 5),
                        "source": pose["source"],
                        "box": [round(box.x0, 5), round(box.y0, 5),
                                round(box.x1, 5), round(box.y1, 5)],
                        "image": [[round(v, 5) for v in p] for p in pose["image"]],
                        "world": [[round(v, 5) for v in p] for p in pose["world"]]
                        if pose.get("world")
                        else None,
                    }
                )
            frame_index += 1
            analysed += 1
            if progress and analysed % 60 == 0:
                progress(analysed, frame_index)
    finally:
        # MediaPipe 1.x raises from __del__ during interpreter teardown if a
        # landmarker is left open; closing here keeps that out of the log.
        for detector in (full_detector, crop_detector):
            try:
                detector.close()
            except Exception:  # pragma: no cover - teardown only
                pass
        cap.release()

    return {
        "format": TRACK_FORMAT,
        "video": {
            "path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "frames_analysed": analysed,
            "stride": options.stride,
        },
        "model": options.model,
        "detector_stats": tracker.stats,
        "coverage": round(len(frames) / analysed, 4) if analysed else 0.0,
        "frames": frames,
    }


def _poses_from(result) -> list[dict]:
    """Normalise a MediaPipe result into plain lists."""
    poses = []
    image_poses = result.pose_landmarks or []
    world_poses = result.pose_world_landmarks or []
    for i, pose in enumerate(image_poses):
        world = world_poses[i] if i < len(world_poses) else None
        poses.append(
            {
                "image": [
                    [p.x, p.y, p.z, float(getattr(p, "visibility", 0.0) or 0.0)]
                    for p in pose
                ],
                "world": [[p.x, p.y, p.z] for p in world] if world is not None else None,
            }
        )
    return poses


def scan_people(video_path: str | Path, options: ExtractOptions | None = None,
                scan_stride: int = 15) -> list[dict]:
    """Sample the clip full-frame and report who is in it.

    Used by ``--list-people`` so you can see which person to pass to
    ``--subject-at`` before committing to a full analysis run.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = options or ExtractOptions()
    model_path = ensure_model(options.model, options.model_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=options.num_poses,
            min_pose_detection_confidence=options.min_detection_confidence,
        )
    )
    tracker = Tracker(max_age=scan_stride * 3)
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % scan_stride == 0:
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                detections = []
                for pose in _poses_from(detector.detect(image)):
                    xs = [p[0] for p in pose["image"]]
                    ys = [p[1] for p in pose["image"]]
                    detections.append(
                        {
                            "t": round(index / fps, 3),
                            "centroid": (sum(xs) / len(xs), sum(ys) / len(ys)),
                            "bbox_height": max(ys) - min(ys),
                            "image": pose["image"],
                        }
                    )
                tracker.update(index, detections)
            index += 1
    finally:
        try:
            detector.close()
        except Exception:  # pragma: no cover - teardown only
            pass
        cap.release()

    tracks = tracker.finished(min_frames=2)
    subject = pick_subject(tracks)
    return [
        {
            "id": t.track_id,
            "seen_in": t.n_frames,
            "first_t": round(t.frames[0]["t"], 2),
            "last_t": round(t.frames[-1]["t"], 2),
            "mean_height": round(t.mean_height(), 3),
            "at": [round(t.last_centroid[0], 3), round(t.last_centroid[1], 3)],
            "suggested": subject is not None and t.track_id == subject.track_id,
        }
        for t in tracks
    ]


def write_tracks(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


def load_tracks(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
