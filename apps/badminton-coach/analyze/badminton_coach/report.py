"""Assemble the analysis into a report, and draw it back onto the video.

Two outputs, for two different moments: a JSON file for anything downstream, and
a Markdown file a player or coach actually reads. The annotated video is the
third: numbers are persuasive only when you can see the frame they came from.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import landmarks as L
from .coach import coach_session, coach_stroke, rank_cues
from .court import (
    Court, CourtCalibration, court_track, distance_covered, recovery_times, zone_occupancy,
)


def _round(value, digits=3):
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    return value


def build_report(track: dict, session, analysis: dict,
                 calibration: CourtCalibration | None = None) -> dict:
    """Everything the pipeline knows, in one JSON-serialisable dict."""
    strokes = analysis["strokes"]
    stroke_cues = [c for s in strokes for c in coach_stroke(s)]

    court_block = None
    session_recovery = None
    if calibration is not None and calibration.valid:
        positions = court_track(session.frames, calibration)
        session_recovery = recovery_times(strokes, positions)
        seen = [p for p in positions if p]
        court_block = {
            "calibration": calibration.to_json(),
            "mirrored": calibration.mirrored,
            "frames_on_court": len(seen),
            "distance_covered_m": distance_covered(positions),
            "zone_seconds": zone_occupancy(positions),
            "recovery": session_recovery,
            "positions": [
                {"t": p["t"], "x": p["x"], "y": p["y"], "zone": p["zone"]["name"]} if p else None
                for p in positions
            ],
        }

    session_cues = coach_session(session.frames, strokes, recovery=session_recovery)

    return _round({
        "format": 1,
        "tool": "badminton-coach analyze",
        "video": track.get("video", {}),
        "model": track.get("model"),
        "detector": {
            "stats": track.get("detector_stats"),
            "coverage": track.get("coverage"),
        },
        "racket_arm": analysis["racket_arm"],
        "frames": analysis["frames"],
        "dropped_frames": analysis["dropped"],
        "duration_s": analysis["duration"],
        "summary": analysis["summary"],
        "strokes": strokes,
        "stroke_cues": stroke_cues,
        "session_cues": session_cues,
        # Stroke cues only: session cues get their own section, and ranking them
        # together made every session cue appear twice in the report.
        "ranked_faults": rank_cues(stroke_cues),
        "court": court_block,
    }, 4)


SHOT_LABELS = {
    "forehand-overhead": "Forehand overhead",
    "backhand-overhead": "Backhand overhead",
    "roundhead-overhead": "Round-the-head",
    "forehand-drive": "Forehand drive",
    "backhand-drive": "Backhand drive",
    "forehand-underarm": "Forehand underarm / net",
    "backhand-underarm": "Backhand underarm / net",
}

STATUS_MARK = {"good": "OK", "warn": "!", "bad": "!!", "unknown": "?"}


def _fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict) -> str:
    """A report a player can read without knowing what a homography is."""
    out: list[str] = []
    video = report.get("video") or {}
    out.append("# Badminton session report\n")
    out.append(f"*Source*: `{video.get('path', 'unknown')}` — "
               f"{_fmt(report.get('duration_s'), 1)} s, "
               f"{video.get('width')}x{video.get('height')} at {_fmt(video.get('fps'), 1)} fps  ")
    out.append(f"*Racket hand*: {report.get('racket_arm')}  ")
    detector = report.get("detector") or {}
    if detector.get("coverage") is not None:
        out.append(f"*Player found in*: {detector['coverage'] * 100:.0f}% of frames "
                   f"({report.get('frames')} usable)\n")

    summary = report.get("summary") or {}
    out.append("\n## Shots\n")
    if not summary.get("count"):
        out.append("No swings were detected. If the player was hitting, lower the swing "
                   "threshold (`--sensitivity`) or check that the whole body is in frame.\n")
    else:
        out.append(f"- **{summary['count']}** swings detected  ")
        out.append(f"- forehand **{summary.get('forehand', 0)}**, "
                   f"backhand **{summary.get('backhand', 0)}**, "
                   f"round-the-head **{summary.get('roundhead', 0)}**  ")
        out.append(f"- fastest wrist speed **{_fmt(summary.get('max_peak_speed_ms'), 1)} m/s**, "
                   f"mean **{_fmt(summary.get('mean_peak_speed_ms'), 1)} m/s**\n")

        out.append("\n| # | time | shot | confidence | elbow at contact | arm elevation | "
                   "wrist speed | contact height |")
        out.append("|---|------|------|-----------|------------------|---------------|"
                   "-------------|----------------|")
        for s in report["strokes"]:
            out.append(
                f"| {s['index'] + 1} | {_fmt(s['t'], 2)} s | "
                f"{SHOT_LABELS.get(s['shot'], s['shot'])} | {_fmt(s['confidence'], 2)} | "
                f"{_fmt(s['contact']['elbow'], 0)}° | "
                f"{_fmt(s['contact']['shoulder_elevation'], 0)}° | "
                f"{_fmt(s['peak_speed_ms'], 1)} m/s | "
                f"{_fmt(s['contact']['hand']['height'], 2)} |"
            )

    ranked = report.get("ranked_faults") or []
    if ranked:
        out.append("\n## What to work on\n")
        out.append("Ordered by how often it went wrong. These are coaching heuristics, not "
                   "clinical measurements — read them as \"worth watching the video for\".\n")
        for g in ranked:
            out.append(f"\n### {g['label']} — {g['count']} time(s), "
                       f"{g['bad']} clearly outside target")
            out.append(f"\n- measured: **{_fmt(g['mean'])}{g['unit']}** on average")
            out.append(f"- target: {g['target']}")
            out.append(f"- why it matters: {g['why']}")
    elif summary.get("count"):
        out.append("\n## What to work on\n\nNothing fell outside its target range.\n")

    session_cues = report.get("session_cues") or []
    if session_cues:
        out.append("\n## Between shots\n")
        for c in session_cues:
            out.append(f"- [{STATUS_MARK.get(c['status'], '?')}] **{c['label']}**: "
                       f"{_fmt(c['value'])}{c['unit']} (target {c['target']}) — {c['why']}")

    court = report.get("court")
    out.append("\n## Court movement\n")
    if not court:
        out.append("The court was not set up, so movement was not measured. Pass "
                   "`--court` with the four corners of the near half to get court "
                   "positions, zone times and recovery.\n")
    else:
        out.append(f"- distance covered: **{_fmt(court['distance_covered_m'], 1)} m**  ")
        recovered = [r for r in court["recovery"] if r["recovered"]]
        if court["recovery"]:
            out.append(f"- returned to base after **{len(recovered)}/{len(court['recovery'])}** "
                       f"shots  ")
        zones = sorted(court["zone_seconds"].items(), key=lambda kv: -kv[1])
        if zones:
            out.append("\n| zone | time |")
            out.append("|------|------|")
            for name, seconds in zones:
                out.append(f"| {name} | {_fmt(seconds, 1)} s |")

    out.append("\n## How to read these numbers\n")
    out.append("- **Angles between body parts** (elbow, knee, arm-to-trunk, trunk twist) do "
               "not depend on where the camera was.")
    out.append("- **Trunk lean** is measured against the camera's vertical, so it is only "
               "as upright as the phone was.")
    out.append("- **Contact** is taken as the moment of peak wrist speed, which is within a "
               "frame or two of the real thing — the shuttle itself is not tracked.")
    out.append("- **Heights and distances of the hand** are in trunk lengths, so they compare "
               "across players and camera distances.\n")
    return "\n".join(out) + "\n"


def write_report(report: dict, out_dir: Path, stem: str = "report") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=1))
    md_path.write_text(render_markdown(report))
    return {"json": json_path, "markdown": md_path}


# -- annotated video -------------------------------------------------------

_BONE = (226, 232, 240)
_RACKET = (22, 115, 249)      # BGR
_FREE = (248, 189, 56)
_GOOD = (94, 197, 34)
_WARN = (76, 204, 250)
_BAD = (68, 68, 239)


def render_annotated_video(track: dict, session, strokes, out_path: Path,
                           calibration: CourtCalibration | None = None,
                           source: str | None = None) -> Path | None:
    """Draw the skeleton, the measured angles and the detected shots onto the clip.

    Returns ``None`` if the source video cannot be opened, rather than failing the
    whole run: the report is the primary output and the video is a bonus.
    """
    import cv2

    source = source or track["video"]["path"]
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return None
    fps = track["video"].get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    by_frame = {f["frame"]: f for f in session.frames if f.get("frame") is not None}
    # A shot is called out for a moment around its contact, so it is readable at
    # normal playback speed rather than a single-frame flash.
    banners: dict[int, dict] = {}
    for s in strokes:
        if s.get("frame") is None:
            continue
        for offset in range(-int(0.2 * fps), int(0.8 * fps)):
            banners.setdefault(s["frame"] + offset, s)

    arm = session.racket_arm
    racket_bones = {(L.side("SHOULDER", arm), L.side("ELBOW", arm)),
                    (L.side("ELBOW", arm), L.side("WRIST", arm))}
    free = L.other_arm(arm)
    free_bones = {(L.side("SHOULDER", free), L.side("ELBOW", free)),
                  (L.side("ELBOW", free), L.side("WRIST", free))}

    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            analysed = by_frame.get(index)
            if analysed and analysed.get("image"):
                image = analysed["image"]
                pt = lambda i: (int(image[i][0] * width), int(image[i][1] * height))
                for a, b in L.CONNECTIONS:
                    colour = (_RACKET if (a, b) in racket_bones
                              else _FREE if (a, b) in free_bones else _BONE)
                    cv2.line(frame, pt(a), pt(b), colour, 3, cv2.LINE_AA)
                cv2.circle(frame, pt(L.side("WRIST", arm)), 7, _RACKET, -1, cv2.LINE_AA)

                m = analysed["metrics"]
                elbow = m["elbow"]
                colour = _GOOD if elbow >= 150 else _WARN if elbow >= 120 else _BAD
                ex, ey = pt(L.side("ELBOW", arm))
                cv2.putText(frame, f"{elbow:.0f}deg", (ex + 12, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

                lines = [
                    f"elbow {m['elbow']:.0f}  elevation {m['shoulder_elevation']:.0f}",
                    f"trunk lean {m['trunk_lean']['total']:.0f}  twist {m['separation']:.0f}",
                    f"knee {min(m['knee_left'], m['knee_right']):.0f}  stance {m['stance_width']:.2f}x",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (16, 34 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if calibration is not None and calibration.valid:
                _draw_court(frame, calibration, width, height)

            shot = banners.get(index)
            if shot:
                label = f"{SHOT_LABELS.get(shot['shot'], shot['shot'])}  " \
                        f"{shot['peak_speed_ms']:.1f} m/s  elbow {shot['contact']['elbow']:.0f}deg"
                cv2.rectangle(frame, (0, height - 52), (width, height), (20, 20, 20), -1)
                cv2.putText(frame, label, (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
            index += 1
    finally:
        writer.release()
        cap.release()
    return out_path


def _draw_court(frame, calibration: CourtCalibration, width: int, height: int) -> None:
    import cv2

    half = Court.width / 2
    singles = Court.singles_width / 2
    polylines = [
        [(-half, 0), (half, 0), (half, Court.half_length), (-half, Court.half_length), (-half, 0)],
        [(-singles, 0), (-singles, Court.half_length)],
        [(singles, 0), (singles, Court.half_length)],
        [(-half, Court.short_service_line), (half, Court.short_service_line)],
        [(0, Court.short_service_line), (0, Court.half_length)],
    ]
    for line in polylines:
        points = []
        for court_point in line:
            image = calibration.image_point(court_point)
            if image is None:
                continue
            points.append((int(image[0] * width), int(image[1] * height)))
        for a, b in zip(points, points[1:]):
            cv2.line(frame, a, b, (21, 204, 250), 2, cv2.LINE_AA)
