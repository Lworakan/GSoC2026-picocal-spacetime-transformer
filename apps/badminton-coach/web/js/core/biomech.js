/**
 * Turn a frame of pose landmarks into badminton-relevant joint angles.
 *
 * ## Which landmarks
 *
 * Angles are computed from MediaPipe's *world* landmarks, which are metric
 * (metres), origin at the mid-hip, and independent of where the player is in the
 * frame. Image landmarks would make the same joint read differently depending on
 * whether the player is near or far from the phone, which is exactly the
 * situation on a badminton court.
 *
 * ## Coordinate convention
 *
 * MediaPipe world landmarks use y-down and z-away-from-camera. {@link toUpFrame}
 * flips both, giving a right-handed frame with **y up** and **z towards the
 * camera**, which is what the rest of this file assumes.
 *
 * ## The body frame
 *
 * Angles that describe *technique* rather than *anatomy* -- is the arm across the
 * body? is the chest turned? -- only mean something relative to the player. So we
 * build an orthonormal frame on the torso (`right`, `up`, `forward`) and express
 * the racket arm in it. That makes every measurement invariant to which way the
 * player happens to be facing the phone.
 *
 * One caveat worth stating plainly: MediaPipe's world frame is aligned to the
 * *camera*, not to gravity. Angles between body parts (elbow, knee, arm-to-trunk,
 * shoulder-to-hip twist) are unaffected. Angles quoted against the vertical --
 * trunk lean -- assume the phone is held roughly upright and level, and are
 * flagged as such in the output.
 */

import {
  sub, add, cross, dot, normalize, length, distance, midpoint,
  reject, angleBetween, jointAngle, signedAngle,
} from './vec3.js';
import { LM, side, otherArm } from './landmarks.js';

/** Convert one MediaPipe world landmark to the y-up, z-toward-camera frame. */
export const toUpFrame = (p) => [p[0], -p[1], -p[2]];

/** Convert a whole landmark array. */
export const worldToUpFrame = (points) => points.map(toUpFrame);

/**
 * The torso frame: an orthonormal basis attached to the player.
 *
 * - `right` points out of the player's right shoulder,
 * - `up` runs mid-hip to mid-shoulder,
 * - `forward` comes out of the chest.
 *
 * `up` is taken as primary and `right` is orthogonalised against it, because the
 * shoulder line is the noisier of the two when an arm is swinging.
 */
export function bodyFrame(world) {
  const midShoulder = midpoint(world[LM.LEFT_SHOULDER], world[LM.RIGHT_SHOULDER]);
  const midHip = midpoint(world[LM.LEFT_HIP], world[LM.RIGHT_HIP]);
  const up = normalize(sub(midShoulder, midHip));
  const shoulderAxis = sub(world[LM.RIGHT_SHOULDER], world[LM.LEFT_SHOULDER]);
  const right = normalize(reject(shoulderAxis, up));
  // cross(up, right) points out of the chest in a right-handed y-up frame.
  const forward = normalize(cross(up, right));
  return { origin: midShoulder, midShoulder, midHip, right, up, forward };
}

/** Express a world point in the body frame, in metres. */
export function toBody(frame, point) {
  const d = sub(point, frame.origin);
  return [dot(d, frame.right), dot(d, frame.up), dot(d, frame.forward)];
}

/**
 * Body-scale estimates used to normalise lengths.
 *
 * Normalising by the player's own trunk makes thresholds transferable between a
 * tall adult and a junior, and between a clip shot from 5 m and one from 15 m.
 */
export function bodyScale(world) {
  const midShoulder = midpoint(world[LM.LEFT_SHOULDER], world[LM.RIGHT_SHOULDER]);
  const midHip = midpoint(world[LM.LEFT_HIP], world[LM.RIGHT_HIP]);
  return {
    trunk: distance(midShoulder, midHip),
    shoulderWidth: distance(world[LM.LEFT_SHOULDER], world[LM.RIGHT_SHOULDER]),
    hipWidth: distance(world[LM.LEFT_HIP], world[LM.RIGHT_HIP]),
    armLeft:
      distance(world[LM.LEFT_SHOULDER], world[LM.LEFT_ELBOW]) +
      distance(world[LM.LEFT_ELBOW], world[LM.LEFT_WRIST]),
    armRight:
      distance(world[LM.RIGHT_SHOULDER], world[LM.RIGHT_ELBOW]) +
      distance(world[LM.RIGHT_ELBOW], world[LM.RIGHT_WRIST]),
  };
}

/** Elbow flexion: 180 is a straight arm, 90 a right angle. */
export function elbowAngle(world, arm) {
  return jointAngle(
    world[side('SHOULDER', arm)],
    world[side('ELBOW', arm)],
    world[side('WRIST', arm)],
  );
}

/** Knee flexion: 180 is a straight leg. */
export function kneeAngle(world, leg) {
  return jointAngle(
    world[side('HIP', leg)],
    world[side('KNEE', leg)],
    world[side('ANKLE', leg)],
  );
}

/**
 * How far the upper arm is raised away from the side of the body, in degrees.
 *
 * 0 is the arm hanging down the flank; 90 is horizontal; 180 is straight
 * overhead. This is the number a coach means by "get your elbow up".
 */
export function shoulderElevation(frame, world, arm) {
  const upperArm = sub(world[side('ELBOW', arm)], world[side('SHOULDER', arm)]);
  const down = [-frame.up[0], -frame.up[1], -frame.up[2]];
  return angleBetween(upperArm, down);
}

/**
 * Where the upper arm points in the horizontal plane, in degrees.
 *
 * Measured from straight ahead of the chest and signed towards the racket side:
 * near 0 the arm is out in front, +90 is straight out to the racket side, and a
 * negative value means the arm has crossed the body -- the geometry of a
 * backhand.
 */
export function shoulderAzimuth(frame, world, arm, racketArm = arm) {
  const upperArm = sub(world[side('ELBOW', arm)], world[side('SHOULDER', arm)]);
  const flat = reject(upperArm, frame.up);
  const raw = signedAngle(frame.forward, flat, frame.up);
  // signedAngle is positive towards the player's left about `up`; flip so that
  // positive always means "towards the racket side" for either handedness.
  const towardsRacketSide = racketArm === 'right' ? -1 : 1;
  return raw * towardsRacketSide;
}

/**
 * Trunk lean away from vertical, decomposed into forwards and sideways.
 *
 * `total` is the tilt of the mid-hip-to-mid-shoulder line away from vertical.
 * `forward` is positive when leaning the way the player is facing, `lateral`
 * positive when leaning towards the racket side.
 *
 * The decomposition is taken against *world-horizontal* axes derived from which
 * way the player faces -- not against the torso's own axes, which are orthogonal
 * to its up-vector by construction and would make every component zero.
 *
 * Vertical here is the camera's vertical, so this is the one measurement in the
 * file that assumes the phone is held roughly upright.
 */
export function trunkLean(frame, racketArm = 'right') {
  const up = frame.up;
  const vertical = [0, 1, 0];
  const total = angleBetween(up, vertical);

  // Where the player faces, flattened onto the floor plane.
  let heading = normalize(reject(frame.forward, vertical));
  if (length(heading) < 1e-6) {
    // Chest pointing straight up or down (a deep dive, or a bad frame): fall back
    // to the shoulder line so the axes stay defined.
    heading = normalize(reject(frame.right, vertical));
  }
  // With y up, cross(heading, vertical) points out of the player's right side.
  const rightAxis = normalize(cross(heading, vertical));
  const towardsRacketSide = racketArm === 'right' ? 1 : -1;

  const upright = dot(up, vertical);
  const forward = Math.atan2(dot(up, heading), upright) * (180 / Math.PI);
  const lateral =
    Math.atan2(dot(up, rightAxis), upright) * (180 / Math.PI) * towardsRacketSide;
  return { total, forward, lateral, heading };
}

/**
 * Shoulder-hip separation ("X-factor"), in degrees.
 *
 * The twist stored between the pelvis and the ribcage during the backswing is
 * where an overhead shot's power comes from; unwinding it is what a coach means
 * by "hit with your body, not your arm". Positive means the shoulders are turned
 * further away from the target than the hips -- the loaded direction for the
 * given racket hand.
 */
export function hipShoulderSeparation(frame, world, racketArm = 'right') {
  const shoulderLine = reject(
    sub(world[LM.RIGHT_SHOULDER], world[LM.LEFT_SHOULDER]),
    frame.up,
  );
  const hipLine = reject(sub(world[LM.RIGHT_HIP], world[LM.LEFT_HIP]), frame.up);
  const raw = signedAngle(hipLine, shoulderLine, frame.up);
  return raw * (racketArm === 'right' ? 1 : -1);
}

/**
 * Stance width as a multiple of shoulder width.
 *
 * Around 1 is a normal standing base; a badminton ready stance is wider, and a
 * lunge to the net is wider still.
 */
export function stanceWidth(world, scale) {
  const feet = distance(world[LM.LEFT_ANKLE], world[LM.RIGHT_ANKLE]);
  return scale.shoulderWidth > 1e-6 ? feet / scale.shoulderWidth : NaN;
}

/**
 * Everything about the racket hand's position, in body-frame trunk lengths.
 *
 * - `height`: 0 is level with the shoulders, positive is above them.
 * - `lateral`: positive on the racket side, negative once the hand has crossed
 *   the midline. This single number is what separates a forehand from a
 *   backhand.
 * - `forward`: positive in front of the chest.
 * - `extension`: 0..1, how much of the arm's length is actually being used.
 */
export function handPosition(frame, world, arm, trunk, armLength, racketArm = arm) {
  const wrist = toBody(frame, world[side('WRIST', arm)]);
  const unit = trunk > 1e-6 ? trunk : NaN;
  const towardsRacketSide = racketArm === 'right' ? 1 : -1;
  const reach = distance(world[side('WRIST', arm)], world[side('SHOULDER', arm)]);
  return {
    height: wrist[1] / unit,
    lateral: (wrist[0] * towardsRacketSide) / unit,
    forward: wrist[2] / unit,
    extension: armLength > 1e-6 ? Math.min(1.5, reach / armLength) : NaN,
    body: wrist,
  };
}

/** Mean visibility of the landmarks that the angles above depend on. */
export function coreVisibility(image, indices) {
  let sum = 0;
  let n = 0;
  for (const i of indices) {
    const v = image?.[i]?.[3];
    if (Number.isFinite(v)) {
      sum += v;
      n += 1;
    }
  }
  return n ? sum / n : 0;
}

/**
 * Compute the full metric set for one frame.
 *
 * @param {number[][]} world world landmarks, already in the y-up frame
 * @param {object} options
 * @param {'left'|'right'} options.racketArm which hand holds the racket
 * @param {number} options.trunk stable trunk length in metres (a running median,
 *   not this frame's value -- see `filters.RunningMedian`)
 * @param {number} options.armLength stable racket-arm length in metres
 */
export function frameMetrics(world, { racketArm = 'right', trunk, armLength } = {}) {
  const frame = bodyFrame(world);
  const scale = bodyScale(world);
  const trunkLen = Number.isFinite(trunk) && trunk > 1e-6 ? trunk : scale.trunk;
  const armLen =
    Number.isFinite(armLength) && armLength > 1e-6
      ? armLength
      : racketArm === 'right'
        ? scale.armRight
        : scale.armLeft;
  const off = otherArm(racketArm);

  return {
    frame,
    scale,
    racketArm,
    elbow: elbowAngle(world, racketArm),
    elbowOff: elbowAngle(world, off),
    shoulderElevation: shoulderElevation(frame, world, racketArm),
    shoulderElevationOff: shoulderElevation(frame, world, off),
    shoulderAzimuth: shoulderAzimuth(frame, world, racketArm, racketArm),
    trunkLean: trunkLean(frame, racketArm),
    separation: hipShoulderSeparation(frame, world, racketArm),
    kneeLeft: kneeAngle(world, 'left'),
    kneeRight: kneeAngle(world, 'right'),
    stanceWidth: stanceWidth(world, scale),
    hand: handPosition(frame, world, racketArm, trunkLen, armLen, racketArm),
    trunkLength: trunkLen,
    armLength: armLen,
  };
}
