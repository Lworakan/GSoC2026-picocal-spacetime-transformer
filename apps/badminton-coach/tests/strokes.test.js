/**
 * Stroke detection and shot naming.
 *
 * Swings are synthesised as a wrist that accelerates and decelerates, so the
 * detector is tested against a signal whose peaks we placed ourselves.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeSkeleton } from './skeleton.js';
import { frameMetrics } from '../web/js/core/biomech.js';
import {
  detectStrokes, classifyShot, wristSpeedSeries, summariseStrokes, SHOT_THRESHOLDS,
} from '../web/js/core/strokes.js';

/**
 * A sequence at 60 fps where the racket arm sweeps through a swing at each of
 * `contactTimes`, and rests otherwise.
 */
function makeSwingSequence(contactTimes, { seconds = 4, fps = 60, pose = {} } = {}) {
  const frames = [];
  for (let i = 0; i < seconds * fps; i += 1) {
    const t = i / fps;
    // Each swing is a smooth sweep of the arm through 150 degrees of elevation
    // over 250 ms, which is roughly the duration of a real overhead action.
    let elevation = 20;
    for (const contact of contactTimes) {
      const phase = (t - contact) / 0.25;
      if (phase > -1 && phase < 1) elevation += 150 * Math.exp(-(phase * phase) * 3);
    }
    const world = makeSkeleton({ ...pose, armElevation: elevation });
    frames.push({
      t,
      frame: i,
      world,
      image: world.map(() => [0.5, 0.5, 0, 1]),
      metrics: frameMetrics(world, { racketArm: 'right', trunk: 0.5, armLength: 0.57 }),
    });
  }
  return frames;
}

test('wrist speed is zero for a still figure', () => {
  const frames = makeSwingSequence([]);
  const speed = wristSpeedSeries(frames, 'right', frames.map(() => 0.5)).filter(Number.isFinite);
  assert.ok(Math.max(...speed) < 0.01, `still figure moved at ${Math.max(...speed)}`);
});

test('one swing produces exactly one stroke, at the right moment', () => {
  const strokes = detectStrokes(makeSwingSequence([1.5]));
  assert.equal(strokes.length, 1);
  assert.ok(Math.abs(strokes[0].t - 1.5) < 0.12, `contact at ${strokes[0].t}`);
});

test('separate swings are counted separately', () => {
  const strokes = detectStrokes(makeSwingSequence([0.8, 2.0, 3.2]));
  assert.equal(strokes.length, 3);
});

test('the refractory period collapses peaks that are too close together', () => {
  // Two overlapping swings put speed peaks about 350 ms apart. A refractory
  // window wider than that has to keep exactly one of them.
  const frames = makeSwingSequence([1.5, 1.62]);
  assert.equal(detectStrokes(frames, { refractory: 0.2 }).length, 2);
  const collapsed = detectStrokes(frames, { refractory: 0.5 });
  assert.equal(collapsed.length, 1);
});

test('when peaks collapse, the strongest survives', () => {
  // Not simply the earliest: on a real exchange the first peak is often the
  // backswing turning over, and the second is the contact.
  const frames = makeSwingSequence([1.5, 1.62]);
  const both = detectStrokes(frames, { refractory: 0.2 });
  const strongest = both.reduce((a, b) => (b.peakSpeed > a.peakSpeed ? b : a));
  const [kept] = detectStrokes(frames, { refractory: 0.5 });
  assert.ok(Math.abs(kept.t - strongest.t) < 1e-9, 'kept the weaker peak');
});

test('a stroke carries its backswing and follow-through', () => {
  const [stroke] = detectStrokes(makeSwingSequence([1.5]));
  assert.ok(stroke.startT < stroke.t && stroke.t < stroke.endT);
  assert.ok(stroke.backswingDuration > 0);
  assert.ok(Number.isFinite(stroke.backswing.maxShoulderElevation));
  assert.ok(Number.isFinite(stroke.window.minKnee));
});

test('raising the threshold rejects a gentle swing', () => {
  const frames = makeSwingSequence([1.5]);
  assert.equal(detectStrokes(frames, { peakSpeed: 100 }).length, 0);
  assert.equal(detectStrokes(frames, { peakSpeed: 1 }).length, 1);
});

test('a swing the model could barely see is skipped', () => {
  const frames = makeSwingSequence([1.5]).map((f) => ({
    ...f, image: f.image.map(() => [0.5, 0.5, 0, 0.1]),
  }));
  assert.equal(detectStrokes(frames).length, 0);
});

test('peak speed is reported in metres per second as well as trunk lengths', () => {
  const [stroke] = detectStrokes(makeSwingSequence([1.5]));
  assert.ok(stroke.peakSpeed > 0);
  assert.ok(Math.abs(stroke.peakSpeedMs - stroke.peakSpeed * 0.5) < 1e-6);
});

test('classifyShot: height separates overhead, drive and underarm', () => {
  assert.equal(classifyShot({ height: 0.6, lateral: 0.5 }).height, 'overhead');
  assert.equal(classifyShot({ height: -0.2, lateral: 0.5 }).height, 'drive');
  assert.equal(classifyShot({ height: -0.9, lateral: 0.5 }).height, 'underarm');
});

test('classifyShot: crossing the midline is what makes a backhand', () => {
  assert.equal(classifyShot({ height: 0.5, lateral: 0.8 }).side, 'forehand');
  assert.equal(classifyShot({ height: 0.5, lateral: -0.4 }).side, 'backhand');
  assert.equal(classifyShot({ height: 0.5, lateral: 0.15 }).side, 'roundhead');
});

test('classifyShot: round-the-head only exists above the shoulders', () => {
  assert.equal(classifyShot({ height: -0.2, lateral: 0.15 }).side, 'forehand');
});

test('classifyShot: confidence falls near a boundary', () => {
  const clear = classifyShot({ height: 0.7, lateral: 0.9 });
  const borderline = classifyShot({ height: SHOT_THRESHOLDS.overheadHeight + 0.001, lateral: 0.9 });
  assert.ok(clear.confidence > 0.9);
  assert.ok(borderline.confidence < 0.05);
});

test('summariseStrokes tallies sides and speeds', () => {
  const strokes = [
    { shot: 'forehand-overhead', side: 'forehand', height: 'overhead', peakSpeedMs: 10 },
    { shot: 'backhand-drive', side: 'backhand', height: 'drive', peakSpeedMs: 6 },
    { shot: 'forehand-overhead', side: 'forehand', height: 'overhead', peakSpeedMs: 8 },
  ];
  const s = summariseStrokes(strokes);
  assert.equal(s.count, 3);
  assert.equal(s.forehand, 2);
  assert.equal(s.backhand, 1);
  assert.equal(s.overhead, 2);
  assert.equal(s.byShot['forehand-overhead'], 2);
  assert.equal(s.maxPeakSpeedMs, 10);
  assert.equal(s.meanPeakSpeedMs, 8);
});

test('too short a sequence yields nothing rather than throwing', () => {
  assert.deepEqual(detectStrokes([]), []);
  assert.deepEqual(detectStrokes(makeSwingSequence([]).slice(0, 3)), []);
});
