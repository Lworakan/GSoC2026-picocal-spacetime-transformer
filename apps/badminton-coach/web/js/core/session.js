/**
 * Glue: raw pose frames in, conditioned metrics and detected strokes out.
 *
 * The same class runs live in the browser and offline over a whole clip, so the
 * numbers shown on court during practice are the numbers in the report
 * afterwards. Live use calls {@link PoseSession#push} per frame and
 * {@link PoseSession#recentStrokes}; batch use pushes everything and calls
 * {@link PoseSession#analyse}.
 */

import { LandmarkFilter, RunningMedian } from './filters.js';
import { worldToUpFrame, frameMetrics, bodyScale } from './biomech.js';
import { LANDMARK_COUNT, CORE, side, LM } from './landmarks.js';
import { detectStrokes, summariseStrokes, wristSpeedSeries } from './strokes.js';
import { distance, midpoint, sub } from './vec3.js';
import { coreVisibility } from './biomech.js';

export const DEFAULT_SESSION_OPTIONS = {
  racketArm: 'right',
  /** 1-Euro settings for the metric world landmarks. */
  worldFilter: { minCutoff: 1.5, beta: 0.35 },
  /** 1-Euro settings for the on-screen image landmarks; smoother, since the
   *  overlay is judged by eye and a jittery skeleton looks broken. */
  imageFilter: { minCutoff: 1.0, beta: 0.15 },
  /** Window, in frames, of the running median used for body scale. */
  scaleWindow: 45,
  /** Frames kept in memory. At 30 fps this is two minutes; live sessions do not
   *  need more, and it bounds memory on a phone. */
  maxFrames: 3600,
  /** Drop frames the pose model could barely see. */
  minVisibility: 0.35,
};

/** Guess which hand holds the racket from which wrist swings harder. */
export function detectRacketArm(frames) {
  if (frames.length < 10) return null;
  const trunks = frames.map((f) => f.metrics?.trunkLength ?? NaN);
  const score = (arm) => {
    const speeds = wristSpeedSeries(frames, arm, trunks).filter(Number.isFinite);
    if (speeds.length < 5) return -Infinity;
    // The 95th percentile, not the mean: the non-racket arm also moves a lot in
    // badminton (it balances and points), but only the racket arm gets whipped.
    const sorted = speeds.slice().sort((a, b) => a - b);
    return sorted[Math.floor(0.95 * (sorted.length - 1))];
  };
  const right = score('right');
  const left = score('left');
  if (!Number.isFinite(right) || !Number.isFinite(left)) return null;
  const margin = Math.abs(right - left) / Math.max(right, left, 1e-6);
  return { arm: right >= left ? 'right' : 'left', margin, right, left };
}

export class PoseSession {
  constructor(options = {}) {
    this.options = { ...DEFAULT_SESSION_OPTIONS, ...options };
    this.worldFilter = new LandmarkFilter(LANDMARK_COUNT, 3, this.options.worldFilter);
    this.imageFilter = new LandmarkFilter(LANDMARK_COUNT, 2, this.options.imageFilter);
    this.trunkMedian = new RunningMedian(this.options.scaleWindow);
    this.armMedian = new RunningMedian(this.options.scaleWindow);
    this.frames = [];
    this.dropped = 0;
    this._strokeCount = 0;
  }

  get racketArm() {
    return this.options.racketArm;
  }

  /**
   * Change which hand is treated as the racket hand.
   *
   * Every stored frame is re-measured, so a mid-session correction fixes the
   * history too rather than leaving a seam in the data.
   */
  setRacketArm(arm) {
    if (arm !== 'left' && arm !== 'right') throw new Error(`bad racket arm: ${arm}`);
    if (arm === this.options.racketArm) return;
    this.options.racketArm = arm;
    for (const f of this.frames) {
      f.metrics = frameMetrics(f.world, {
        racketArm: arm,
        trunk: f.metrics.trunkLength,
        armLength: f.metrics.armLength,
      });
    }
  }

  reset() {
    this.worldFilter.reset();
    this.imageFilter.reset();
    this.trunkMedian.reset();
    this.armMedian.reset();
    this.frames = [];
    this.dropped = 0;
    this._strokeCount = 0;
  }

  /**
   * Add one detected pose.
   *
   * @param {object} sample
   * @param {number} sample.t seconds
   * @param {number[][]} sample.world raw MediaPipe world landmarks (y-down)
   * @param {number[][]} [sample.image] normalised image landmarks with visibility
   * @param {number} [sample.frame] source frame number, for reporting
   * @returns {object|null} the analysed frame, or null if it was too poor to use
   */
  push({ t, world, image = null, frame = null }) {
    if (!world || world.length < LANDMARK_COUNT) return null;
    if (image && coreVisibility(image, CORE) < this.options.minVisibility) {
      this.dropped += 1;
      return null;
    }

    const up = worldToUpFrame(world);
    const filtered = this.worldFilter.filter(up, t);
    const filteredImage = image ? this.imageFilter.filter(image, t) : null;

    const raw = bodyScale(filtered);
    const trunk = this.trunkMedian.push(raw.trunk);
    const armRaw = this.options.racketArm === 'right' ? raw.armRight : raw.armLeft;
    const armLength = this.armMedian.push(armRaw);

    const metrics = frameMetrics(filtered, {
      racketArm: this.options.racketArm,
      trunk,
      armLength,
    });

    const analysed = { t, frame, world: filtered, image: filteredImage, metrics };
    this.frames.push(analysed);
    if (this.frames.length > this.options.maxFrames) this.frames.shift();
    return analysed;
  }

  /** Every stroke in the buffer. */
  strokes(options = {}) {
    return detectStrokes(this.frames, { ...options, racketArm: this.racketArm });
  }

  /**
   * Strokes found since the last call -- the live path.
   *
   * The most recent frames are held back by `settle` seconds, because a peak
   * cannot be confirmed as a peak until the frames after it have arrived.
   */
  recentStrokes({ settle = 0.25, ...options } = {}) {
    if (!this.frames.length) return [];
    const cutoff = this.frames[this.frames.length - 1].t - settle;
    const settled = this.frames.filter((f) => f.t <= cutoff);
    const all = detectStrokes(settled, options);
    const fresh = all.slice(this._strokeCount);
    this._strokeCount = all.length;
    return fresh;
  }

  /** Full analysis of everything buffered. */
  analyse(options = {}) {
    const strokes = this.strokes(options);
    return {
      racketArm: this.racketArm,
      frames: this.frames.length,
      dropped: this.dropped,
      duration: this.frames.length
        ? this.frames[this.frames.length - 1].t - this.frames[0].t
        : 0,
      strokes,
      summary: summariseStrokes(strokes),
    };
  }

  /** Latest frame's metrics, for the live readout. */
  get latest() {
    return this.frames.length ? this.frames[this.frames.length - 1] : null;
  }

  /**
   * Current racket-wrist speed, in trunk lengths per second.
   *
   * Computed over the last few frames only, so the live readout costs nothing
   * even when the session buffer holds two minutes of play.
   */
  speedNow(window = 5) {
    const tail = this.frames.slice(-window);
    if (tail.length < 3) return NaN;
    const speeds = wristSpeedSeries(
      tail,
      this.racketArm,
      tail.map((f) => f.metrics.trunkLength),
    ).filter(Number.isFinite);
    return speeds.length ? speeds[speeds.length - 1] : NaN;
  }
}
