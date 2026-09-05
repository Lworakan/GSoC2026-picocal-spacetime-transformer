/** The coaching rules: grading, applicability, and how faults are ranked. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { grade, RULES, coachStroke, coachSession, rankCues, targetText } from '../web/js/core/coach.js';

/** A stroke record shaped like the ones detectStrokes emits. */
function stroke(overrides = {}) {
  const base = {
    index: 0,
    t: 1.0,
    shot: 'forehand-overhead',
    side: 'forehand',
    height: 'overhead',
    confidence: 0.9,
    contact: {
      elbow: 165, elbowOff: 170, shoulderElevation: 160, shoulderElevationOff: 60,
      shoulderAzimuth: 20, separation: 25,
      trunkLean: { total: 15, forward: 10, lateral: 8 },
      kneeLeft: 160, kneeRight: 155, stanceWidth: 1.4,
      hand: { height: 0.5, lateral: 0.6, forward: 0.3, extension: 0.95 },
    },
    backswing: {
      minElbow: 95, maxSeparation: 30, maxShoulderElevation: 165, maxOffArmElevation: 140,
    },
    window: { maxElbow: 172, minKnee: 140 },
  };
  return {
    ...base, ...overrides,
    contact: { ...base.contact, ...(overrides.contact || {}) },
    backswing: { ...base.backswing, ...(overrides.backswing || {}) },
    window: { ...base.window, ...(overrides.window || {}) },
  };
}

test('grade places values in good, warn and bad bands', () => {
  const band = { min: 150, warnMin: 135 };
  assert.equal(grade(160, band), 'good');
  assert.equal(grade(142, band), 'warn');
  assert.equal(grade(120, band), 'bad');
  assert.equal(grade(NaN, band), 'unknown');
});

test('grade handles a two-sided band', () => {
  const band = { min: 95, max: 145, warnMin: 80, warnMax: 160 };
  assert.equal(grade(120, band), 'good');
  assert.equal(grade(150, band), 'warn');
  assert.equal(grade(170, band), 'bad');
  assert.equal(grade(70, band), 'bad');
});

test('a well-played overhead raises no complaints', () => {
  const cues = coachStroke(stroke());
  assert.ok(cues.length > 0, 'the overhead rules should have applied');
  assert.deepEqual(cues.filter((c) => c.status !== 'good'), []);
});

test('a bent arm at contact is called out on an overhead', () => {
  const cues = coachStroke(stroke({ contact: { elbow: 110 } }));
  const cue = cues.find((c) => c.id === 'overhead-elbow-extension');
  assert.equal(cue.status, 'bad');
  assert.ok(cue.why.length > 40, 'a cue must explain the mechanism');
});

test('a dropped free arm and a missing shoulder turn are both caught', () => {
  const cues = coachStroke(stroke({
    backswing: { maxOffArmElevation: 40, maxSeparation: 5 },
  }));
  assert.equal(cues.find((c) => c.id === 'overhead-free-arm').status, 'bad');
  assert.equal(cues.find((c) => c.id === 'overhead-body-rotation').status, 'bad');
});

test('rules only fire for the shot family they belong to', () => {
  const overhead = coachStroke(stroke()).map((c) => c.id);
  const net = coachStroke(stroke({
    shot: 'forehand-underarm', height: 'underarm',
    contact: { hand: { height: -0.8, lateral: 0.5, forward: 0.3, extension: 0.8 } },
  })).map((c) => c.id);
  assert.ok(overhead.includes('overhead-free-arm'));
  assert.ok(!net.includes('overhead-free-arm'));
  assert.ok(net.includes('net-lunge-knee'));
});

test('a shot whose type is a coin-flip is not coached', () => {
  assert.deepEqual(coachStroke(stroke({ confidence: 0.05 })), []);
});

test('a missing measurement is skipped rather than graded', () => {
  const cues = coachStroke(stroke({ backswing: { maxOffArmElevation: NaN } }));
  assert.ok(!cues.some((c) => c.id === 'overhead-free-arm'));
});

test('every rule states a target and a reason', () => {
  for (const rule of RULES) {
    assert.ok(targetText(rule).length > 0, `${rule.id} has no readable target`);
    assert.ok(rule.why.length > 40, `${rule.id} does not explain itself`);
    assert.ok(rule.label.length > 0);
  }
});

test('rule ids are unique', () => {
  const ids = RULES.map((r) => r.id);
  assert.equal(new Set(ids).size, ids.length);
});

test('session cues read posture from the frames away from any contact', () => {
  const frames = Array.from({ length: 60 }, (_, i) => ({
    t: i / 30,
    metrics: { kneeLeft: 178, kneeRight: 176, stanceWidth: 1.5 },
  }));
  const cues = coachSession(frames, []);
  const knee = cues.find((c) => c.id === 'ready-knee-bend');
  assert.equal(knee.status, 'bad', 'standing bolt upright should be flagged');
  assert.equal(cues.find((c) => c.id === 'ready-stance-width').status, 'good');
});

test('frames near a contact are excluded from the between-shots posture', () => {
  // Every frame is within the quiet window of the one stroke, so there is
  // nothing left to judge and no cue should be invented.
  const frames = Array.from({ length: 20 }, (_, i) => ({
    t: 1.0 + (i - 10) / 100,
    metrics: { kneeLeft: 178, kneeRight: 176, stanceWidth: 1.5 },
  }));
  assert.deepEqual(coachSession(frames, [{ t: 1.0 }]), []);
});

test('recovery cues appear once court data is available', () => {
  const recovery = [
    { recoverySeconds: 2.4, recovered: true },
    { recoverySeconds: null, recovered: false },
    { recoverySeconds: 2.6, recovered: true },
  ];
  const cues = coachSession([], [], { recovery });
  assert.equal(cues.find((c) => c.id === 'recovery-time').status, 'bad');
  const rate = cues.find((c) => c.id === 'recovery-rate');
  assert.ok(Math.abs(rate.value - 2 / 3) < 1e-9);
});

test('rankCues groups repeats and puts the worst first', () => {
  const cues = [
    { id: 'a', label: 'A', why: 'x', unit: '°', target: 't', value: 1, status: 'warn' },
    { id: 'b', label: 'B', why: 'y', unit: '°', target: 't', value: 2, status: 'bad' },
    { id: 'b', label: 'B', why: 'y', unit: '°', target: 't', value: 4, status: 'bad' },
    { id: 'c', label: 'C', why: 'z', unit: '°', target: 't', value: 9, status: 'good' },
  ];
  const ranked = rankCues(cues);
  assert.equal(ranked.length, 2, 'good cues are not faults');
  assert.equal(ranked[0].id, 'b');
  assert.equal(ranked[0].count, 2);
  assert.equal(ranked[0].mean, 3);
});
