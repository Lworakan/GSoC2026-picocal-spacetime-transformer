"""Command-line entry point.

    python -m badminton_coach match.mov -o out/
    python -m badminton_coach match.mov --list-people
    python -m badminton_coach match.mov --court 0.18,0.72 0.86,0.66 0.98,0.95 0.02,0.99
    python -m badminton_coach --from-track out/track.json -o out/

Pose extraction is the slow part and is cached as ``track.json``; re-running the
analysis with different thresholds costs a second, not a minute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .coach import coach_stroke
from .court import CALIBRATION_ORDER, CourtCalibration
from .extract import ExtractOptions, extract, load_tracks, scan_people, write_tracks
from .report import build_report, render_annotated_video, write_report
from .session import PoseSession, SessionOptions, detect_racket_arm
from .strokes import ShotThresholds, StrokeOptions


def _point(text: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a normalised x,y point like 0.18,0.72 — got {text!r}") from exc
    return x, y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="badminton_coach",
        description="Badminton posture and stroke analysis from a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", nargs="?", help="video file to analyse")
    parser.add_argument("-o", "--out", default="badminton-report",
                        help="output directory (default: %(default)s)")
    parser.add_argument("--from-track", metavar="TRACK.json",
                        help="skip pose extraction and re-analyse a saved track "
                             "(also reads the JSON exported by the web app)")
    parser.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                        help="pose model: heavy is the most accurate on a distant "
                             "player, lite the fastest (default: %(default)s)")
    parser.add_argument("--racket-hand", choices=["right", "left", "auto"], default="right",
                        help="which hand holds the racket (default: %(default)s)")
    parser.add_argument("--sensitivity", type=float, default=6.0, metavar="TL/s",
                        help="wrist speed, in trunk lengths per second, that counts as a "
                             "swing. Lower finds more shots and more false ones "
                             "(default: %(default)s)")
    parser.add_argument("--subject-at", type=_point, metavar="X,Y",
                        help="normalised point on the player to track, when more than one "
                             "person is in shot")
    parser.add_argument("--court", type=_point, nargs=4, metavar=("NET_L", "NET_R", "BACK_R", "BACK_L"),
                        help="four normalised image points: the corners of the near half, "
                             "in the order " + ", ".join(CALIBRATION_ORDER))
    parser.add_argument("--list-people", action="store_true",
                        help="scan the clip and report who is in it, then exit")
    parser.add_argument("--annotate", action="store_true",
                        help="also write an annotated .mp4 with the skeleton and angles drawn on")
    parser.add_argument("--stride", type=int, default=1,
                        help="analyse every Nth frame (default: %(default)s)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many analysed frames")
    parser.add_argument("--quiet", action="store_true", help="only print the output paths")
    return parser


def _load_track(args, log) -> dict:
    """Get a landmark track, either from a saved file or by running MediaPipe."""
    if args.from_track:
        track = load_tracks(args.from_track)
        # The web app's export uses the same frame shape but a different envelope.
        if "frames" not in track:
            raise SystemExit(f"{args.from_track} does not look like a track file")
        track.setdefault("video", {"path": args.video or args.from_track, "fps": 30.0})
        return track

    if not args.video:
        raise SystemExit("give a video file, or --from-track to re-analyse a saved one")

    log(f"extracting pose from {args.video} (model: {args.model})")
    options = ExtractOptions(
        model=args.model,
        stride=args.stride,
        max_frames=args.max_frames,
        subject_at=args.subject_at,
    )
    track = extract(args.video, options,
                    progress=lambda n, _f: log(f"  {n} frames", end="\r"))
    log(f"  player found in {track['coverage'] * 100:.0f}% of "
        f"{track['video']['frames_analysed']} frames "
        f"({track['detector_stats']['full']} full-frame searches)")
    return track


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else (
        lambda *a, **k: print(*a, file=sys.stderr, **k))

    if args.list_people:
        if not args.video:
            raise SystemExit("--list-people needs a video")
        for person in scan_people(args.video, ExtractOptions(model=args.model)):
            mark = " <- suggested" if person["suggested"] else ""
            print(f"person {person['id']:2d}  seen {person['seen_in']:3d}x  "
                  f"{person['first_t']:6.2f}-{person['last_t']:6.2f}s  "
                  f"height {person['mean_height']:.3f}  "
                  f"--subject-at {person['at'][0]:.3f},{person['at'][1]:.3f}{mark}")
        return 0

    out_dir = Path(args.out)
    track = _load_track(args, log)

    if not track["frames"]:
        raise SystemExit("no pose was detected in this clip — check that a whole person "
                         "is visible, or try --model heavy")

    racket_arm = "right" if args.racket_hand == "auto" else args.racket_hand
    session = PoseSession(SessionOptions(racket_arm=racket_arm)).push_track(track)

    if args.racket_hand == "auto":
        guess = detect_racket_arm(session.frames)
        if guess and guess["margin"] >= 0.15:
            session.set_racket_arm(guess["arm"])
            log(f"  racket hand looks like: {guess['arm']} "
                f"(margin {guess['margin'] * 100:.0f}%)")
        else:
            log("  could not tell which hand holds the racket; assuming right. "
                "Pass --racket-hand to be sure.")

    analysis = session.analyse(StrokeOptions(peak_speed=args.sensitivity), ShotThresholds())
    log(f"  {analysis['summary']['count']} swings detected")

    calibration = CourtCalibration(args.court) if args.court else None
    if calibration is not None and not calibration.valid:
        log("  the four court corners do not form a court — ignoring them. "
            "Give them in order: " + ", ".join(CALIBRATION_ORDER))
        calibration = None

    report = build_report(track, session, analysis, calibration)
    paths = write_report(report, out_dir)
    track_path = write_tracks(track, out_dir / "track.json")
    written = [track_path, paths["json"], paths["markdown"]]

    if args.annotate:
        source = args.video or track["video"].get("path")
        log("  rendering annotated video")
        video_path = render_annotated_video(
            track, session, analysis["strokes"], out_dir / "annotated.mp4",
            calibration=calibration, source=source,
        )
        if video_path:
            written.append(video_path)
        else:
            log("  could not open the source video; skipped the annotated render")

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
