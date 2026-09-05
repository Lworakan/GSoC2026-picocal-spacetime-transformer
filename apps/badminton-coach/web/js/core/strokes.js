/**
 * Find swings in a pose sequence and say what kind of shot each one was.
 *
 * ## Detecting the swing
 *
 * The signal is the speed of the racket wrist *relative to the hips*, in trunk
 * lengths per second. Two choices matter:
 *
 * - Relative to the hips, so that running to the back of the court does not read
 *   as a swing. MediaPipe's world landmarks are already hip-centred, so this
 *   falls out for free.
 * - In trunk lengths rather than metres, so one threshold works for a junior and
 *   an adult, and for a phone 5 m away and one 15 m away.
 *
 * A stroke is a local maximum of that speed. Peak wrist speed is a standard
 * stand-in for the moment of contact in racket-sports analysis: the true contact
 * is within a frame or two of it, and unlike the shuttle, the wrist is something
 * a pose model can actually see.
 *
 * ## Naming the shot
 *
 * At contact the hand is expressed in the torso frame (see `biomech.js`). Its
 * height relative to the shoulder line separates overhead shots from drives from
 * underarm lifts; its lateral position -- signed so that positive is always the
 * racket side -- separates forehand from backhand, because a backhand is
 * precisely the shot where the racket hand has crossed the body's midline.
 */

import { distance, sub } from './vec3.js';
import { LM, CORE, side } from './landmarks.js';
import { midpoint } from './vec3.js';
import { coreVisibility } from './biomech.js';

/** Tunables for {@link detectStrokes}. Speeds are in trunk lengths per second. */
export const DEFAULT_STROKE_OPTIONS = {
  /** A peak must reach this to count as a swing rather than a fidget. */
  peakSpeed: 6.0,
  /** Speed below which the arm counts as quiet, used to find the swing's edges. */
  quietSpeed: 2.5,
  /** Minimum gap between two contacts; badminton exchanges rarely beat this. */
  refractory: 0.30,
  /** How far back from contact to look for the start of the backswing. */
  maxBackswing: 0.80,
  /** How far forward from contact to look for the end of the follow-through. */
  maxFollowThrough: 0.60,
  /** Skip swings where the model could barely see the joints we are measuring. */
  minVisibility: 0.5,
  /** Half-width, in frames, of the neighbourhood a peak must dominate. */
  peakWindow: 3,
};

/** Thresholds separating shot families, in trunk lengths from the shoulder line. */
export const SHOT_THRESHOLDS = {
  overheadHeight: 0.10,
  underarmHeight: -0.45,
  forehandLateral: 0.28,
  backhandLateral: 0.02,
};

/**
 * Racket-wrist speed relative to the hips, per frame, in trunk lengths per second.
 *
 * @param {Array} samples frames of `{ t, world }` with world already y-up
 * @param {'left'|'right'} racketArm
 * @param {number[]} trunkLengths stable trunk length per frame, in metres
 */
export function wristSpeedSeries(samples, racketArm, trunkLengths) {
  const wristIndex = side('WRIST', racketArm);
  const relative = samples.map((s) => {
    const hip = midpoint(s.world[LM.LEFT_HIP], s.world[LM.RIGHT_HIP]);
    return sub(s.world[wristIndex], hip);
  });
  const speed = new Array(samples.length).fill(NaN);
  for (let i = 1; i < samples.length - 1; i += 1) {
    const dt = samples[i + 1].t - samples[i - 1].t;
    if (!(dt > 0)) continue;
    const trunk = trunkLengths[i];
    if (!(trunk > 1e-6)) continue;
    speed[i] = distance(relative[i + 1], relative[i - 1]) / dt / trunk;
  }
  if (speed.length > 2) {
    speed[0] = speed[1];
    speed[speed.length - 1] = speed[speed.length - 2];
  }
  return speed;
}

/**
 * Name the shot from the hand's position in the torso frame at contact.
 *
 * @returns {{shot: string, side: string, height: string, confidence: number}}
 */
export function classifyShot(hand, thresholds = SHOT_THRESHOLDS) {
  const { height, lateral } = hand;
  let heightClass;
  if (height >= thresholds.overheadHeight) heightClass = 'overhead';
  else if (height >= thresholds.underarmHeight) heightClass = 'drive';
  else heightClass = 'underarm';

  let sideClass;
  if (lateral >= thresholds.forehandLateral) sideClass = 'forehand';
  else if (lateral <= thresholds.backhandLateral) sideClass = 'backhand';
  else sideClass = 'roundhead';

  // A shot played over the head only makes sense as its own category when it is
  // actually overhead; lower down, the same lateral band is just a straight-on
  // forehand played close to the body.
  if (sideClass === 'roundhead' && heightClass !== 'overhead') sideClass = 'forehand';

  // How far the hand sits from the nearest decision boundary, as a rough
  // confidence. Shots near a boundary are the ones a human would also hesitate on.
  const sideMargin = Math.min(
    Math.abs(lateral - thresholds.forehandLateral),
    Math.abs(lateral - thresholds.backhandLateral),
  );
  const heightMargin = Math.min(
    Math.abs(height - thresholds.overheadHeight),
    Math.abs(height - thresholds.underarmHeight),
  );
  const confidence = Math.max(0, Math.min(1, (Math.min(sideMargin, heightMargin) / 0.25)));

  return {
    shot: `${sideClass}-${heightClass}`,
    side: sideClass,
    height: heightClass,
    confidence,
  };
}

function localPeaks(speed, options) {
  const peaks = [];
  const w = options.peakWindow;
  for (let i = w; i < speed.length - w; i += 1) {
    const v = speed[i];
    if (!(v >= options.peakSpeed)) continue;
    let dominant = true;
    for (let j = i - w; j <= i + w; j += 1) {
      if (j === i) continue;
      // `>` on the left and `>=` on the right keeps exactly one index on a plateau.
      if (j < i ? speed[j] >= v : speed[j] > v) {
        dominant = false;
        break;
      }
    }
    if (dominant) peaks.push(i);
  }
  return peaks;
}

function applyRefractory(peaks, speed, times, refractory) {
  // Strongest first, so that when two candidates are too close together we keep
  // the real contact rather than whichever came first in time.
  const byStrength = peaks.slice().sort((a, b) => speed[b] - speed[a]);
  const kept = [];
  for (const p of byStrength) {
    if (kept.every((k) => Math.abs(times[p] - times[k]) >= refractory)) kept.push(p);
  }
  return kept.sort((a, b) => a - b);
}

function walkTo(speed, times, from, direction, limitSeconds, quiet) {
  let i = from;
  while (true) {
    const next = i + direction;
    if (next < 0 || next >= speed.length) break;
    if (Math.abs(times[next] - times[from]) > limitSeconds) break;
    i = next;
    if (speed[i] <= quiet) break;
  }
  return i;
}

/**
 * Detect every stroke in a sequence of analysed frames.
 *
 * @param {Array} frames each `{ t, world, metrics, image? }`, world in the y-up
 *   frame and `metrics` from `biomech.frameMetrics`
 * @param {object} [options] see {@link DEFAULT_STROKE_OPTIONS}
 * @returns {Array} one record per stroke
 */
export function detectStrokes(frames, options = {}) {
  const opts = { ...DEFAULT_STROKE_OPTIONS, ...options };
  const thresholds = { ...SHOT_THRESHOLDS, ...(options.thresholds || {}) };
  if (frames.length < 5) return [];

  const times = frames.map((f) => f.t);
  const trunks = frames.map((f) => f.metrics.trunkLength);
  const racketArm = frames[0].metrics.racketArm;
  const speed = wristSpeedSeries(frames, racketArm, trunks);

  const peaks = applyRefractory(localPeaks(speed, opts), speed, times, opts.refractory);

  const strokes = [];
  for (const peak of peaks) {
    const visibility = frames[peak].image
      ? coreVisibility(frames[peak].image, CORE)
      : 1;
    if (visibility < opts.minVisibility) continue;

    const start = walkTo(speed, times, peak, -1, opts.maxBackswing, opts.quietSpeed);
    const end = walkTo(speed, times, peak, +1, opts.maxFollowThrough, opts.quietSpeed);
    const contact = frames[peak].metrics;
    const label = classifyShot(contact.hand, thresholds);

    const window = frames.slice(start, end + 1);
    const backswing = frames.slice(start, peak + 1);

    strokes.push({
      index: strokes.length,
      frame: frames[peak].frame,
      t: times[peak],
      startT: times[start],
      endT: times[end],
      startFrame: frames[start].frame,
      endFrame: frames[end].frame,
      ...label,
      visibility,
      peakSpeed: speed[peak],
      peakSpeedMs: speed[peak] * contact.trunkLength,
      backswingDuration: times[peak] - times[start],
      followThroughDuration: times[end] - times[peak],
      contact: {
        elbow: contact.elbow,
        elbowOff: contact.elbowOff,
        shoulderElevation: contact.shoulderElevation,
        shoulderElevationOff: contact.shoulderElevationOff,
        shoulderAzimuth: contact.shoulderAzimuth,
        separation: contact.separation,
        trunkLean: contact.trunkLean,
        kneeLeft: contact.kneeLeft,
        kneeRight: contact.kneeRight,
        stanceWidth: contact.stanceWidth,
        hand: {
          height: contact.hand.height,
          lateral: contact.hand.lateral,
          forward: contact.hand.forward,
          extension: contact.hand.extension,
        },
      },
      // A coach reads the backswing as much as the contact: the deepest elbow
      // bend and the largest hip-shoulder twist before contact are what the
      // arm had available to release.
      backswing: {
        minElbow: minBy(backswing, (f) => f.metrics.elbow),
        maxSeparation: maxBy(backswing, (f) => f.metrics.separation),
        maxShoulderElevation: maxBy(backswing, (f) => f.metrics.shoulderElevation),
        maxOffArmElevation: maxBy(backswing, (f) => f.metrics.shoulderElevationOff),
      },
      window: {
        maxElbow: maxBy(window, (f) => f.metrics.elbow),
        minKnee: minBy(window, (f) => Math.min(f.metrics.kneeLeft, f.metrics.kneeRight)),
      },
    });
  }
  return strokes;
}

function minBy(items, fn) {
  let best = Infinity;
  for (const item of items) {
    const v = fn(item);
    if (Number.isFinite(v) && v < best) best = v;
  }
  return Number.isFinite(best) ? best : NaN;
}

function maxBy(items, fn) {
  let best = -Infinity;
  for (const item of items) {
    const v = fn(item);
    if (Number.isFinite(v) && v > best) best = v;
  }
  return Number.isFinite(best) ? best : NaN;
}

/** Tally shots by name, for the session summary. */
export function summariseStrokes(strokes) {
  const byShot = {};
  for (const s of strokes) byShot[s.shot] = (byShot[s.shot] || 0) + 1;
  const speeds = strokes.map((s) => s.peakSpeedMs).filter(Number.isFinite);
  return {
    count: strokes.length,
    byShot,
    forehand: strokes.filter((s) => s.side === 'forehand').length,
    backhand: strokes.filter((s) => s.side === 'backhand').length,
    roundhead: strokes.filter((s) => s.side === 'roundhead').length,
    overhead: strokes.filter((s) => s.height === 'overhead').length,
    meanPeakSpeedMs: speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : NaN,
    maxPeakSpeedMs: speeds.length ? Math.max(...speeds) : NaN,
  };
}
