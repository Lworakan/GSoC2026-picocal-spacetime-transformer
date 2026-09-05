/**
 * The session pipeline, and the guard that keeps the two implementations in step.
 *
 * The last test re-derives `tests/fixtures/expected-metrics.json` from the shared
 * landmark fixture. That file is what the Python parity test compares itself
 * against, so if this fails the JavaScript side moved and the Python side has not
 * been told.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { PoseSession, detectRacketArm } from '../web/js/core/session.js';
import { makeSkeleton, toMediaPipe, fakeImage } from './skeleton.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = () => JSON.parse(readFileSync(join(here, 'fixtures', 'landmarks.json'), 'utf8'));

function loadedSession(options = {}) {
  const input = fixture();
  const session = new PoseSession({ racketArm: input.racket_arm, ...options });
  for (const f of input.frames) {
    session.push({ t: f.t, world: f.world, image: f.image, frame: f.frame });
  }
  return session;
}

test('push accepts MediaPipe world landmarks and stores them y-up', () => {
  const session = new PoseSession();
  const world = makeSkeleton({ armElevation: 180 });
  const out = session.push({ t: 0, world: toMediaPipe(world), image: fakeImage(world) });
  assert.ok(out.metrics.hand.height > 0.5, 'an overhead arm should read above the shoulders');
});

test('a frame the model could barely see is dropped', () => {
  const session = new PoseSession();
  const world = makeSkeleton();
  assert.equal(session.push({ t: 0, world: toMediaPipe(world), image: fakeImage(world, 0.05) }), null);
  assert.equal(session.dropped, 1);
  assert.equal(session.frames.length, 0);
});

test('malformed input is refused rather than half-analysed', () => {
  const session = new PoseSession();
  assert.equal(session.push({ t: 0, world: null }), null);
  assert.equal(session.push({ t: 0, world: [[0, 0, 0]] }), null);
});

test('switching the racket hand re-measures the frames already stored', () => {
  const session = loadedSession();
  const before = session.frames[50].metrics.hand.lateral;
  session.setRacketArm('left');
  const after = session.frames[50].metrics.hand.lateral;
  assert.notEqual(before, after, 'history was left measured against the old hand');
  assert.equal(session.racketArm, 'left');
  assert.throws(() => session.setRacketArm('either'));
});

test('the buffer is bounded so a long session cannot grow without limit', () => {
  const session = new PoseSession({ maxFrames: 10 });
  const world = toMediaPipe(makeSkeleton());
  for (let i = 0; i < 40; i += 1) session.push({ t: i / 30, world, frame: i });
  assert.equal(session.frames.length, 10);
  assert.equal(session.latest.frame, 39);
});

test('reset clears everything', () => {
  const session = loadedSession();
  assert.ok(session.frames.length > 0);
  session.reset();
  assert.equal(session.frames.length, 0);
  assert.equal(session.latest, null);
});

test('the racket hand is guessed correctly on the real clip', () => {
  const guess = detectRacketArm(loadedSession().frames);
  assert.equal(guess.arm, 'right', 'the player in the fixture is right-handed');
  assert.ok(guess.margin > 0.05);
});

test('detectRacketArm declines to guess from too little data', () => {
  assert.equal(detectRacketArm([]), null);
});

test('speedNow reads the current wrist speed without scanning the buffer', () => {
  const session = loadedSession();
  assert.ok(Number.isFinite(session.speedNow()));
  assert.ok(Number.isNaN(new PoseSession().speedNow()));
});

test('analyse summarises the whole buffer', () => {
  const analysis = loadedSession().analyse();
  assert.ok(analysis.frames > 200);
  assert.ok(analysis.duration > 3);
  assert.equal(analysis.summary.count, analysis.strokes.length);
});

test('recentStrokes reports each stroke once', () => {
  const input = fixture();
  const session = new PoseSession({ racketArm: input.racket_arm });
  let emitted = 0;
  for (const f of input.frames) {
    session.push({ t: f.t, world: f.world, image: f.image, frame: f.frame });
    emitted += session.recentStrokes().length;
  }
  assert.equal(emitted, session.strokes().length, 'live and batch disagree on the count');
});

test('the checked-in reference metrics still match what the code produces', () => {
  // Regenerating writes the same file; a difference here means the analysis
  // changed, and the Python parity test is now comparing against a stale
  // reference. Re-read the diff before running `node tools/dump-metrics.mjs`.
  const path = join(here, 'fixtures', 'expected-metrics.json');
  const before = readFileSync(path, 'utf8');
  execFileSync(process.execPath, [join(here, '..', 'tools', 'dump-metrics.mjs')], {
    stdio: 'ignore',
  });
  const after = readFileSync(path, 'utf8');
  assert.equal(after, before, 'tests/fixtures/expected-metrics.json is out of date');
});
