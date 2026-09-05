/**
 * Biomechanics against a synthetic skeleton with known angles.
 *
 * Testing against the reference video would only prove the code agrees with
 * itself; a figure built from chosen angles proves it agrees with geometry.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeSkeleton, toMediaPipe } from './skeleton.js';
import {
  frameMetrics, bodyFrame, bodyScale, worldToUpFrame, toBody, elbowAngle,
  shoulderElevation, shoulderAzimuth, trunkLean, hipShoulderSeparation, coreVisibility,
} from '../web/js/core/biomech.js';
import { LM, CORE } from '../web/js/core/landmarks.js';

const close = (a, b, tol = 0.05) =>
  assert.ok(Math.abs(a - b) <= tol, `expected ${a} to be within ${tol} of ${b}`);

test('worldToUpFrame undoes MediaPipe y-down, z-away', () => {
  const world = makeSkeleton();
  const restored = worldToUpFrame(toMediaPipe(world));
  for (let i = 0; i < world.length; i += 1) {
    for (let d = 0; d < 3; d += 1) close(restored[i][d], world[i][d], 1e-12);
  }
});

test('the body frame is orthonormal and faces the way the figure does', () => {
  const f = bodyFrame(makeSkeleton());
  const d = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  close(d(f.up, f.right), 0, 1e-9);
  close(d(f.up, f.forward), 0, 1e-9);
  close(d(f.right, f.forward), 0, 1e-9);
  close(Math.hypot(...f.up), 1, 1e-9);
  // The figure faces +z, and its right shoulder is at negative x.
  assert.ok(f.forward[2] > 0.99);
  assert.ok(f.right[0] < -0.99);
});

test('elbow angle reproduces the angle it was built with', () => {
  for (const want of [180, 150, 120, 90, 60]) {
    close(elbowAngle(makeSkeleton({ elbowRight: want }), 'right'), want, 0.01);
  }
});

test('elbow angle does not depend on how the arm is held', () => {
  const a = elbowAngle(makeSkeleton({ elbowRight: 110, armElevation: 0 }), 'right');
  const b = elbowAngle(makeSkeleton({ elbowRight: 110, armElevation: 160, armAzimuth: -40 }), 'right');
  close(a, b, 0.01);
});

test('shoulder elevation is 0 at the flank, 90 horizontal, 180 overhead', () => {
  for (const want of [0, 45, 90, 135, 180]) {
    const w = makeSkeleton({ armElevation: want });
    close(shoulderElevation(bodyFrame(w), w, 'right'), want, 0.01);
  }
});

test('shoulder azimuth is positive to the racket side and negative across the body', () => {
  const at = (azimuth) => {
    const w = makeSkeleton({ armElevation: 90, armAzimuth: azimuth });
    return shoulderAzimuth(bodyFrame(w), w, 'right', 'right');
  };
  close(at(0), 0, 0.01);
  close(at(90), 90, 0.01);
  close(at(-70), -70, 0.01);
});

test('shoulder azimuth is undefined for an arm with no horizontal direction', () => {
  const w = makeSkeleton({ armElevation: 0 });
  assert.ok(Number.isNaN(shoulderAzimuth(bodyFrame(w), w, 'right', 'right')));
});

test('a left-handed player mirrors the azimuth sign', () => {
  const w = makeSkeleton({ armElevation: 90, armAzimuth: 90 });
  const right = shoulderAzimuth(bodyFrame(w), w, 'right', 'right');
  const asLeft = shoulderAzimuth(bodyFrame(w), w, 'right', 'left');
  close(right, -asLeft, 0.01);
});

test('trunk lean separates forwards from sideways', () => {
  let m = trunkLean(bodyFrame(makeSkeleton({ leanForward: 30 })), 'right');
  close(m.total, 30, 0.05);
  close(m.forward, 30, 0.05);
  close(m.lateral, 0, 0.05);

  m = trunkLean(bodyFrame(makeSkeleton({ leanRight: 25 })), 'right');
  close(m.total, 25, 0.05);
  close(m.forward, 0, 0.05);
  close(m.lateral, 25, 0.05);
});

test('trunk lean components are not both zero for a combined lean', () => {
  // The bug this pins: decomposing against the torso's own axes, which are
  // orthogonal to its up-vector, made every component identically zero.
  const m = trunkLean(bodyFrame(makeSkeleton({ leanForward: 20, leanRight: 20 })), 'right');
  assert.ok(Math.abs(m.forward) > 5 && Math.abs(m.lateral) > 5);
  assert.ok(m.total > 20 && m.total < 40);
});

test('hip-shoulder separation reads the twist between the two lines', () => {
  for (const twist of [0, 20, 40, -30]) {
    const w = makeSkeleton({ shoulderTwist: twist });
    close(Math.abs(hipShoulderSeparation(bodyFrame(w), w, 'right')), Math.abs(twist), 0.05);
  }
});

test('hand position: height, lateral and forward are in trunk lengths', () => {
  const overhead = frameMetrics(makeSkeleton({ armElevation: 180 }), { racketArm: 'right' });
  assert.ok(overhead.hand.height > 0.8, 'an overhead arm puts the hand well above the shoulders');

  const across = frameMetrics(
    makeSkeleton({ armElevation: 90, armAzimuth: -110 }), { racketArm: 'right' },
  );
  assert.ok(across.hand.lateral < 0, 'a hand across the midline reads negative');

  const inFront = frameMetrics(
    makeSkeleton({ armElevation: 90, armAzimuth: 0 }), { racketArm: 'right' },
  );
  assert.ok(inFront.hand.forward > 0.5, 'an arm held out in front reads forward');
});

test('normalising by trunk length makes the metrics scale-free', () => {
  // A junior and an adult in the same position must produce the same numbers,
  // which is what lets one set of thresholds serve both. The figure has to be
  // scaled uniformly for this to be the claim being tested.
  const at = (k) => frameMetrics(
    makeSkeleton({
      trunk: 0.5 * k, shoulderWidth: 0.36 * k, hipWidth: 0.28 * k,
      upperArm: 0.30 * k, forearm: 0.27 * k, thigh: 0.42 * k, shin: 0.42 * k,
      stance: 0.4 * k, armElevation: 120, armAzimuth: 30, elbowRight: 140,
    }),
    { racketArm: 'right' },
  );
  const small = at(0.8);
  const large = at(1.25);
  close(small.hand.height, large.hand.height, 0.005);
  close(small.hand.lateral, large.hand.lateral, 0.005);
  close(small.hand.forward, large.hand.forward, 0.005);
  close(small.stanceWidth, large.stanceWidth, 0.005);
  close(small.elbow, large.elbow, 0.01);
});

test('body scale reports the lengths it was built from', () => {
  const s = bodyScale(makeSkeleton({ trunk: 0.5, shoulderWidth: 0.36, upperArm: 0.3, forearm: 0.27 }));
  close(s.trunk, 0.5, 1e-9);
  close(s.shoulderWidth, 0.36, 1e-9);
  close(s.armRight, 0.57, 1e-9);
});

test('a supplied trunk length overrides the noisy per-frame one', () => {
  const w = makeSkeleton({ trunk: 0.5, armElevation: 180 });
  const noisy = frameMetrics(w, { racketArm: 'right' });
  const stable = frameMetrics(w, { racketArm: 'right', trunk: 1.0 });
  close(stable.hand.height, noisy.hand.height / 2, 0.01);
});

test('toBody puts the shoulders at the origin', () => {
  const w = makeSkeleton();
  const f = bodyFrame(w);
  const mid = toBody(f, [
    (w[LM.LEFT_SHOULDER][0] + w[LM.RIGHT_SHOULDER][0]) / 2,
    (w[LM.LEFT_SHOULDER][1] + w[LM.RIGHT_SHOULDER][1]) / 2,
    (w[LM.LEFT_SHOULDER][2] + w[LM.RIGHT_SHOULDER][2]) / 2,
  ]);
  for (const c of mid) close(c, 0, 1e-9);
});

test('core visibility averages only the landmarks it is asked about', () => {
  const image = Array.from({ length: 33 }, () => [0, 0, 0, 1]);
  for (const i of CORE) image[i][3] = 0.5;
  close(coreVisibility(image, CORE), 0.5, 1e-9);
  assert.equal(coreVisibility([], CORE), 0);
});
