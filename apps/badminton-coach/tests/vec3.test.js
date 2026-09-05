/** Geometry primitives: the base every angle in the app is built on. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  jointAngle, signedAngle, angleBetween, normalize, reject, cross, dot, midpoint, distance,
} from '../web/js/core/vec3.js';

const close = (a, b, tol = 1e-9) =>
  assert.ok(Math.abs(a - b) <= tol, `expected ${a} to be within ${tol} of ${b}`);

test('jointAngle is 180 for a straight limb and 90 for a right angle', () => {
  close(jointAngle([0, 1, 0], [0, 0, 0], [0, -1, 0]), 180);
  close(jointAngle([1, 0, 0], [0, 0, 0], [0, 1, 0]), 90);
  close(jointAngle([1, 0, 0], [0, 0, 0], [1, 0, 0]), 0);
});

test('jointAngle ignores limb length, only direction', () => {
  close(jointAngle([5, 0, 0], [0, 0, 0], [0, 0.01, 0]), 90);
});

test('signedAngle follows the right-hand rule about its axis', () => {
  close(signedAngle([1, 0, 0], [0, 1, 0], [0, 0, 1]), 90);
  close(signedAngle([0, 1, 0], [1, 0, 0], [0, 0, 1]), -90);
});

test('degenerate inputs give NaN rather than a wrong number', () => {
  assert.ok(Number.isNaN(angleBetween([0, 0, 0], [1, 0, 0])));
  assert.ok(Number.isNaN(signedAngle([0, 0, 1], [0, 0, 1], [0, 0, 1])));
  assert.deepEqual(normalize([0, 0, 0]), [0, 0, 0]);
});

test('reject removes the component along the axis', () => {
  const r = reject([1, 2, 3], [0, 1, 0]);
  close(dot(r, [0, 1, 0]), 0);
  assert.deepEqual(r, [1, 0, 3]);
});

test('cross of the basis vectors is right-handed', () => {
  assert.deepEqual(cross([1, 0, 0], [0, 1, 0]), [0, 0, 1]);
});

test('midpoint and distance', () => {
  assert.deepEqual(midpoint([0, 0, 0], [2, 4, 6]), [1, 2, 3]);
  close(distance([0, 0, 0], [3, 4, 0]), 5);
});
