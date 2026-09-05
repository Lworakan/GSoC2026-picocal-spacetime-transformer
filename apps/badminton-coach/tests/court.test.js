/** The court homography, its guards, and the movement measures built on it. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CourtCalibration, CALIBRATION_ORDER, HALF_COURT_CORNERS, COURT, BASE_POSITION,
  homographyFromQuad, applyHomography, invert3x3, solveLinear, isConvexQuad,
  zoneOf, distanceFromBase, onCourt, courtTrack, distanceCovered, zoneOccupancy,
  recoveryTimes,
} from '../web/js/core/court.js';
import { LM } from '../web/js/core/landmarks.js';

const close = (a, b, tol = 1e-6) =>
  assert.ok(Math.abs(a - b) <= tol, `expected ${a} to be within ${tol} of ${b}`);

// A plausible oblique view of one half court.
const VIEW = [[0.20, 0.60], [0.80, 0.60], [0.95, 0.95], [0.05, 0.95]];

test('solveLinear solves a small system and rejects a singular one', () => {
  const x = solveLinear([[2, 1], [1, 3]], [5, 10]);
  close(x[0], 1);
  close(x[1], 3);
  assert.equal(solveLinear([[1, 2], [2, 4]], [1, 2]), null);
});

test('the homography maps the tapped corners exactly onto the court corners', () => {
  const cal = new CourtCalibration(VIEW);
  assert.ok(cal.valid);
  CALIBRATION_ORDER.forEach((name, i) => {
    const got = cal.courtPoint(cal.points[i]);
    close(got[0], HALF_COURT_CORNERS[name][0], 1e-9);
    close(got[1], HALF_COURT_CORNERS[name][1], 1e-9);
  });
});

test('court and image coordinates round-trip', () => {
  const cal = new CourtCalibration(VIEW);
  for (const point of [[0, 0], [1.5, 4], [-2.8, 6.5], [3.05, 6.7]]) {
    const back = cal.courtPoint(cal.imagePoint(point));
    close(back[0], point[0], 1e-6);
    close(back[1], point[1], 1e-6);
  }
});

test('a bow-tie tap order is rejected instead of silently mapping nonsense', () => {
  // Net-left, back-right, net-right, back-left: a plausible mis-tap that yields
  // an invertible matrix and completely wrong positions.
  const bad = new CourtCalibration([[0.20, 0.60], [0.95, 0.95], [0.80, 0.60], [0.05, 0.95]]);
  assert.equal(bad.valid, false);
  assert.equal(isConvexQuad(bad.points), false);
});

test('duplicated corners are rejected', () => {
  assert.equal(new CourtCalibration([[0.2, 0.6], [0.2, 0.6], [0.9, 0.9], [0.1, 0.9]]).valid, false);
});

test('a mirrored view is legitimate, and swapping sides reverses it', () => {
  const cal = new CourtCalibration(VIEW);
  assert.ok(cal.valid);
  const swapped = cal.swapSides();
  assert.ok(swapped.valid);
  assert.notEqual(cal.mirrored, swapped.mirrored);
  const wasRight = cal.courtPoint(VIEW[1]);
  const nowLeft = swapped.courtPoint(VIEW[1]);
  close(wasRight[0], -nowLeft[0], 1e-9);
});

test('calibration survives a save and reload', () => {
  const cal = new CourtCalibration(VIEW);
  const back = CourtCalibration.fromJSON(JSON.parse(JSON.stringify(cal.toJSON())));
  assert.ok(back.valid);
  close(back.courtPoint([0.5, 0.8])[1], cal.courtPoint([0.5, 0.8])[1], 1e-9);
  assert.equal(CourtCalibration.fromJSON(null), null);
  assert.equal(CourtCalibration.fromJSON({ points: [[0, 0]] }), null);
});

test('applyHomography and invert3x3 agree', () => {
  const H = homographyFromQuad(VIEW, CALIBRATION_ORDER.map((k) => HALF_COURT_CORNERS[k]));
  const Hi = invert3x3(H);
  const there = applyHomography(H, [0.5, 0.8]);
  const back = applyHomography(Hi, there);
  close(back[0], 0.5, 1e-9);
  close(back[1], 0.8, 1e-9);
  assert.equal(invert3x3([[1, 0, 0], [2, 0, 0], [3, 0, 0]]), null);
});

test('zones tile the half court', () => {
  assert.equal(zoneOf([0, 1.0]).name, 'front-centre');
  assert.equal(zoneOf([-2.5, 1.0]).name, 'front-left');
  assert.equal(zoneOf([2.5, 5.5]).name, 'rear-right');
  assert.equal(zoneOf([0, 3.5]).name, 'mid-centre');
  // Even a position past the baseline gets a name rather than crashing.
  assert.equal(zoneOf([0, 9]).depth, 'rear');
});

test('base distance and the court boundary', () => {
  close(distanceFromBase(BASE_POSITION), 0);
  close(distanceFromBase([0, 6.0]), 3.0);
  assert.ok(onCourt([0, 0]));
  assert.ok(onCourt([COURT.width / 2, COURT.halfLength]));
  assert.ok(!onCourt([0, 20]));
});

const frameAt = (t, x, y) => {
  const image = Array.from({ length: 33 }, () => [0.5, 0.5, 0, 1]);
  image[LM.LEFT_ANKLE] = [x, y, 0, 1];
  image[LM.RIGHT_ANKLE] = [x, y, 0, 1];
  return { t, image };
};

test('courtTrack maps feet to court metres and drops invisible frames', () => {
  const cal = new CourtCalibration(VIEW);
  const centreImage = cal.imagePoint([0, 3.0]);
  const frames = [
    frameAt(0, centreImage[0], centreImage[1]),
    { t: 0.1, image: null },
    { t: 0.2, image: Array.from({ length: 33 }, () => [0.5, 0.5, 0, 0.05]) },
  ];
  const positions = courtTrack(frames, cal, LM);
  close(positions[0].x, 0, 1e-6);
  close(positions[0].y, 3.0, 1e-6);
  assert.equal(positions[0].zone.name, 'mid-centre');
  assert.equal(positions[1], null);
  assert.equal(positions[2], null);
});

test('distanceCovered adds up steps and ignores teleports', () => {
  const walk = [
    { t: 0, x: 0, y: 0 }, { t: 0.1, x: 0, y: 0.5 }, { t: 0.2, x: 0, y: 1.0 },
  ];
  close(distanceCovered(walk), 1.0, 1e-9);
  const glitch = [...walk, { t: 0.3, x: 0, y: 9.0 }];
  close(distanceCovered(glitch), 1.0, 1e-9);
});

test('zoneOccupancy accumulates time and ignores long gaps', () => {
  const positions = [
    { t: 0.0, zone: { name: 'mid-centre' } },
    { t: 0.1, zone: { name: 'mid-centre' } },
    { t: 0.2, zone: { name: 'mid-centre' } },
    { t: 9.0, zone: { name: 'mid-centre' } },
  ];
  close(zoneOccupancy(positions)['mid-centre'], 0.2, 1e-9);
});

test('recoveryTimes reports when the player got home, and when they did not', () => {
  const positions = [
    { t: 1.0, x: 0, y: 6.0, zone: { name: 'rear-centre' }, base: 3.0 },
    { t: 1.8, x: 0, y: 3.4, zone: { name: 'mid-centre' }, base: 0.4 },
    { t: 3.0, x: 0, y: 6.5, zone: { name: 'rear-centre' }, base: 3.5 },
    { t: 5.9, x: 0, y: 6.5, zone: { name: 'rear-centre' }, base: 3.5 },
  ];
  const [first, second] = recoveryTimes(
    [{ index: 0, t: 1.0 }, { index: 1, t: 3.0 }], positions,
  );
  assert.equal(first.recovered, true);
  close(first.recoverySeconds, 0.8, 1e-9);
  assert.equal(second.recovered, false);
  assert.equal(second.recoverySeconds, null);
});
