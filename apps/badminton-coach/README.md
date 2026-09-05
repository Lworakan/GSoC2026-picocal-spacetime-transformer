# Badminton posture coach

Arm angles, forehand/backhand detection and court movement from a phone camera.
Two pieces that share one analysis core:

- **`web/`** — a mobile web app. Open it on a phone, point the camera at the
  court or load a clip from the camera roll, and it measures posture live.
  No install, no account, no upload: MediaPipe runs in the browser and the video
  never leaves the device.
- **`analyze/`** — a Python pipeline for a recorded clip. Runs MediaPipe Pose over
  the whole video and writes a report, plus an annotated copy of the video with
  the skeleton and the measured angles drawn on.

> **These are coaching heuristics, not clinical measurements.** The target ranges
> are the things badminton coaches say about technique, written down as numbers so
> a phone can check them. A good player will break several of them on purpose.
> Read a flagged cue as "worth watching the video for", not as a verdict.

---

## The web app

```bash
npm run serve            # http://localhost:8080
```

It has to be served over http rather than opened as a file: ES modules, the
service worker and camera access all need a real origin. Any static host works —
there is no build step.

On a phone, open the page and add it to the home screen; it then runs full-screen
and works offline after the first load.

### Using it

1. **Racket hand.** Right, left, or auto (guessed from which wrist swings harder,
   correctable with one tap).
2. **Camera** for live feedback, or **open a video** to analyse a clip. A recorded
   clip is stepped frame by frame rather than played, so every frame is measured
   rather than whichever ones the phone kept up with.
3. **Tap the player** to lock on to them. Useful when the next court is in shot.
4. **Live** shows joint angles as they happen, drawn on the player as well as
   listed. **Shots** is a feed of detected strokes with the coaching cues for each
   one. **Court** and **Summary** are described below.
5. **Export data (JSON)** saves the whole session, landmark track included, in the
   format `analyze/` reads — so a session captured on a phone can be re-analysed
   on a laptop without recording anything again.

### The court

Tapping the four corners of your half — net-left, net-right, back-right,
back-left — gives court positions in metres, zone occupancy, distance covered and
how long you take to get back to base after each shot. The court model is drawn
back onto the video so a bad calibration is visible rather than silently wrong,
and a bow-tie tap order is refused rather than accepted.

All four corners must be in frame. On a camera placed at the side of a hall they
often are not, in which case the angle measurements still work and the court
section simply stays empty. See [docs/filming.md](docs/filming.md).

### Offline

```bash
npm run vendor           # or: npm run vendor lite | heavy
```

Downloads the MediaPipe runtime and a pose model into `web/vendor/`, after which
the app needs no network at all — which is the normal state of affairs inside a
sports hall. Without it, both are fetched from a CDN on first use and cached.
`web/vendor/` is git-ignored: those are large third-party binaries, fetched on
demand rather than committed.

---

## The Python pipeline

```bash
pip install -r analyze/requirements.txt
cd analyze
python -m badminton_coach match.mov -o out/ --annotate
```

Writes `out/track.json` (the landmarks), `out/report.json`, `out/report.md`, and
with `--annotate` an `out/annotated.mp4`.

Useful flags:

| flag | what it does |
|---|---|
| `--racket-hand right\|left\|auto` | which hand holds the racket |
| `--model lite\|full\|heavy` | `heavy` is the most accurate on a distant player |
| `--list-people` | scan the clip and print who is in it, with the `--subject-at` value for each |
| `--subject-at X,Y` | normalised point on the player to track |
| `--court NET_L NET_R BACK_R BACK_L` | four normalised corner points, e.g. `--court 0.18,0.72 0.86,0.66 0.98,0.95 0.02,0.99` |
| `--sensitivity N` | wrist speed, in trunk lengths per second, that counts as a swing |
| `--from-track out/track.json` | re-analyse without re-running pose |

Pose extraction is the slow part and is cached in `track.json`, so re-running the
analysis with different thresholds costs a second rather than a minute.

---

## How it finds the player

A phone filming a badminton court puts the player at 15–25% of the frame height,
which is where a single full-frame pass starts dropping detections. Measured on
the reference clip:

| approach | frames with a usable pose |
|---|---|
| full-frame detection every frame | **58%** |
| full-frame once, then a padded crop around the previous pose | **96%** |

So both implementations acquire the player once, then crop a padded square around
the last known pose and upscale it before inference. It is also *cheaper* — a
384×384 crop is a fraction of a 1080p frame — which is what makes it usable on a
phone. When the crop comes up empty several frames running, it falls back to a
full-frame search and prefers whoever is nearest to where the player was, rather
than the largest person in frame: the largest person is often somebody walking
between the phone and the court.

## What is measured

Elbow and knee angles, arm elevation and direction, trunk lean and trunk twist
(the "X-factor"), stance width, contact height and position, and wrist speed.
Full definitions, conventions and limits: [docs/measurements.md](docs/measurements.md).

Strokes are found as peaks of racket-wrist speed relative to the hips, and named
from where the hand is in the torso frame at contact — height above the shoulder
line separates overhead from drive from underarm, and whether the hand has crossed
the body's midline separates forehand from backhand.

---

## Layout

```
web/
  index.html, styles.css      mobile-first UI, Thai and English
  js/core/                    the analysis: no DOM, no MediaPipe, unit-tested in Node
    vec3, filters             geometry primitives; 1-Euro smoothing, running median
    landmarks, biomech        landmark names; joint angles and the torso frame
    strokes, coach            swing detection and naming; the coaching rules
    court, roitracker         homography and zones; two-stage person tracking
    session                   ties it together, live and batch
  js/pose.js                  the only file that knows about MediaPipe
  js/overlay.js, js/app.js    canvas drawing; the controller
analyze/badminton_coach/      the same analysis in Python, plus extraction and reports
tests/                        Node tests, the shared landmark fixture, the reference metrics
analyze/tests/                pytest, including the JavaScript/Python parity check
tools/                        dev server, offline vendoring, reference-metrics dump
```

## Tests

```bash
npm test                            # 109 tests
cd analyze && python -m pytest      # 117 tests
```

The two implementations are kept in step by a parity test. `tests/fixtures/
landmarks.json` holds real MediaPipe landmarks from a practice clip;
`tools/dump-metrics.mjs` writes what the JavaScript core computes from them, and
`analyze/tests/test_parity.py` asserts the Python core produces the same numbers
to within 1e-6 — every frame metric, every detected stroke, every coaching cue.
A Node test regenerates the reference file and fails if it has drifted, so
neither side can move without the other noticing.

Joint angles are tested against a synthetic skeleton built from chosen angles
(`tests/skeleton.js`, mirrored in `analyze/tests/conftest.py`), so the maths is
checked against geometry rather than against whatever the pose model happened to
output on one video.

## Licence

Same as the repository root.
