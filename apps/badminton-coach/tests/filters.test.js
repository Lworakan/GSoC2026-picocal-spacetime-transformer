/** Smoothing and derivatives: the things that make wrist speed mean anything. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { OneEuro, LandmarkFilter, RunningMedian, derivative, median, mean } from '../web/js/core/filters.js';

test('OneEuro passes the first sample through unchanged', () => {
  const f = new OneEuro();
  assert.equal(f.filter(4.2, 0), 4.2);
});

test('OneEuro suppresses jitter around a constant', () => {
  const f = new OneEuro({ minCutoff: 0.5, beta: 0.0 });
  let out = 0;
  for (let i = 0; i < 200; i += 1) out = f.filter(10 + (i % 2 ? 1 : -1), i / 60);
  assert.ok(Math.abs(out - 10) < 0.25, `residual jitter ${Math.abs(out - 10)}`);
});

test('OneEuro tracks a fast ramp without falling far behind', () => {
  // beta is what buys this: with beta 0 the same ramp lags several times as much.
  const fast = new OneEuro({ minCutoff: 1.0, beta: 1.0 });
  const slow = new OneEuro({ minCutoff: 1.0, beta: 0.0 });
  let lastFast = 0;
  let lastSlow = 0;
  for (let i = 0; i < 60; i += 1) {
    const t = i / 60;
    lastFast = fast.filter(t * 20, t);
    lastSlow = slow.filter(t * 20, t);
  }
  const truth = (59 / 60) * 20;
  assert.ok(Math.abs(truth - lastFast) < Math.abs(truth - lastSlow));
});

test('OneEuro tolerates a repeated or rewound timestamp', () => {
  const f = new OneEuro();
  f.filter(1, 1.0);
  assert.equal(f.filter(2, 1.0), 2);
  assert.equal(f.filter(3, 0.5), 3);
});

test('LandmarkFilter leaves extra columns such as visibility untouched', () => {
  const f = new LandmarkFilter(2, 3);
  const out = f.filter([[1, 2, 3, 0.9], [4, 5, 6, 0.1]], 0);
  assert.equal(out[0][3], 0.9);
  assert.equal(out[1][3], 0.1);
});

test('RunningMedian is unmoved by a single outlier', () => {
  const m = new RunningMedian(5);
  for (const v of [0.48, 0.47, 0.49, 0.48, 0.52]) m.push(v);
  const before = m.value;
  m.push(9.9);
  assert.ok(Math.abs(m.value - before) < 0.05, 'one bad frame moved the body scale');
});

test('RunningMedian skips non-finite input', () => {
  const m = new RunningMedian(5);
  m.push(1);
  m.push(NaN);
  m.push(3);
  assert.equal(m.value, 2);
});

test('median and mean ignore non-finite values', () => {
  assert.equal(median([1, NaN, 3]), 2);
  assert.equal(mean([1, NaN, 3]), 2);
  assert.ok(Number.isNaN(median([])));
});

test('derivative of t squared is 2t at the interior points', () => {
  const times = [0, 1, 2, 3, 4];
  const values = times.map((t) => t * t);
  const d = derivative(values, times);
  assert.equal(d[1], 2);
  assert.equal(d[2], 4);
  assert.equal(d[3], 6);
});
