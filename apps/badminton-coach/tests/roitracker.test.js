/**
 * The two-stage tracker, driven by a fake detector.
 *
 * The behaviour that matters is who it locks on to and when it lets go, which is
 * testable without a pose model at all.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RoiTracker, Box, bounds, cropWindow, unproject } from '../web/js/core/roitracker.js';

/** A pose that is a box of landmarks centred on (cx, cy). */
const poseAt = (cx, cy, h = 0.2, w = 0.08) => ({
  image: [
    [cx - w / 2, cy - h / 2, 0, 1],
    [cx + w / 2, cy - h / 2, 0, 1],
    [cx + w / 2, cy + h / 2, 0, 1],
    [cx - w / 2, cy + h / 2, 0, 1],
  ],
  world: [[0, 0, 0]],
});

test('bounds and cropWindow', () => {
  const b = bounds(poseAt(0.5, 0.5, 0.2, 0.1).image);
  assert.ok(Math.abs(b.cx - 0.5) < 1e-9);
  assert.ok(Math.abs(b.height - 0.2) < 1e-9);

  // In a 16:9 frame the window must be wider than it is tall in normalised
  // units, so that it is square in pixels.
  const w = cropWindow(b, 0.85, 16 / 9);
  assert.ok(w.width * (16 / 9) > w.height * 0.99);
});

test('cropWindow stays inside the frame', () => {
  const w = cropWindow(new Box(0.0, 0.0, 0.1, 0.1), 2.0, 1);
  assert.ok(w.x0 >= 0 && w.y0 >= 0 && w.x1 <= 1 && w.y1 <= 1);
});

test('unproject maps crop-local landmarks back to the full frame', () => {
  const window = new Box(0.2, 0.4, 0.6, 0.8);
  const [p] = unproject([[0.5, 0.5, 0, 1]], window);
  assert.ok(Math.abs(p[0] - 0.4) < 1e-9);
  assert.ok(Math.abs(p[1] - 0.6) < 1e-9);
  assert.equal(p[3], 1, 'visibility is carried through');
});

function tracker({ full = [], crop = null, ...rest } = {}) {
  const calls = { full: 0, crop: 0 };
  const t = new RoiTracker({
    detectFull: async () => { calls.full += 1; return typeof full === 'function' ? full() : full; },
    detectCrop: async () => { calls.crop += 1; return typeof crop === 'function' ? crop() : crop; },
    ...rest,
  });
  return { t, calls };
}

test('the first frame is a full-frame search, later frames use the crop', async () => {
  const { t, calls } = tracker({ full: [poseAt(0.3, 0.5)], crop: poseAt(0.5, 0.5, 0.5, 0.4) });
  const first = await t.step(null);
  assert.equal(first.source, 'full');
  const second = await t.step(null);
  assert.equal(second.source, 'crop');
  assert.equal(calls.full, 1);
  assert.equal(calls.crop, 1);
});

test('with no history the biggest person is chosen', async () => {
  const { t } = tracker({ full: [poseAt(0.3, 0.5, 0.15), poseAt(0.8, 0.5, 0.6)] });
  const pose = await t.step(null);
  assert.ok(Math.abs(bounds(pose.image).cx - 0.8) < 1e-6);
});

test('a tap overrides the size preference', async () => {
  const { t } = tracker({ full: [poseAt(0.3, 0.5, 0.15), poseAt(0.8, 0.5, 0.6)] });
  t.lockOn(0.3, 0.5);
  const pose = await t.step(null);
  assert.ok(Math.abs(bounds(pose.image).cx - 0.3) < 1e-6);
});

test('after a loss, the nearest person wins over the biggest one', async () => {
  // This is the bystander case: someone walks between the phone and the court
  // and is far larger in frame than the player.
  const { t } = tracker({
    full: [poseAt(0.30, 0.5, 0.20), poseAt(0.34, 0.5, 0.90)],
    crop: null,
    options: { maxMisses: 0 },
  });
  t.lockOn(0.30, 0.5);
  await t.step(null);
  await t.step(null);           // crop fails, tracker drops the lock
  const pose = await t.step(null);
  assert.ok(Math.abs(bounds(pose.image).cx - 0.30) < 1e-6, 'jumped to the bystander');
});

test('a crop result that jumps too far is rejected', async () => {
  // A pose in the far corner of the crop unprojects well away from the last
  // known position: that is a different body the crop happened to contain.
  const { t } = tracker({
    full: [poseAt(0.3, 0.5, 0.2)],
    crop: poseAt(0.95, 0.95, 0.08, 0.04),
    options: { maxJump: 0.1 },
  });
  await t.step(null);
  const box = t.box;
  assert.equal(await t.step(null), null);
  assert.equal(t.box, box, 'the last good box is kept while missing');
});

test('a crop result of a wildly different size is rejected', async () => {
  const { t } = tracker({ full: [poseAt(0.5, 0.5, 0.2)], crop: poseAt(0.5, 0.5, 0.99) });
  await t.step(null);
  t.options.maxSizeRatio = 1.2;
  assert.equal(await t.step(null), null);
});

test('the lock is dropped only after several consecutive misses', async () => {
  const { t, calls } = tracker({
    full: [poseAt(0.5, 0.5, 0.2)], crop: null, options: { maxMisses: 3 },
  });
  await t.step(null);
  assert.equal(calls.full, 1);
  for (let i = 0; i < 3; i += 1) {
    assert.equal(await t.step(null), null);
    assert.equal(calls.full, 1, 'gave up too early');
  }
  await t.step(null);
  assert.equal(calls.full, 2, 'never went back to a full-frame search');
});

test('an empty frame yields null without throwing', async () => {
  const { t } = tracker({ full: [] });
  assert.equal(await t.step(null), null);
});

test('reset clears the lock', async () => {
  const { t, calls } = tracker({ full: [poseAt(0.5, 0.5)], crop: poseAt(0.5, 0.5, 0.5) });
  await t.step(null);
  assert.ok(t.tracking);
  t.reset();
  assert.ok(!t.tracking);
  await t.step(null);
  assert.equal(calls.full, 2);
});
