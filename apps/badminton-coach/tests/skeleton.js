/**
 * A synthetic skeleton with angles we choose, so the biomechanics can be tested
 * against a known answer rather than against whatever the pose model happened to
 * output on one video.
 *
 * The figure faces +z, stands with y up, and is built from a handful of
 * parameters: joint angles in, joint angles out.
 */

import { LM, LANDMARK_COUNT } from '../web/js/core/landmarks.js';

const deg = (d) => (d * Math.PI) / 180;

const unit = (v) => {
  const n = Math.hypot(...v) || 1;
  return v.map((c) => c / n);
};

/** The first candidate with a usable component perpendicular to `axis`. */
function perpendicularTo(axis, candidates) {
  for (const c of candidates) {
    const k = c[0] * axis[0] + c[1] * axis[1] + c[2] * axis[2];
    const r = [c[0] - axis[0] * k, c[1] - axis[1] * k, c[2] - axis[2] * k];
    if (Math.hypot(...r) > 1e-6) return unit(r);
  }
  return [0, 0, 1];
}



/**
 * @param {object} pose
 * @param {number} [pose.trunk=0.5] mid-hip to mid-shoulder, metres
 * @param {number} [pose.shoulderWidth=0.36]
 * @param {number} [pose.upperArm=0.3]
 * @param {number} [pose.forearm=0.27]
 * @param {number} [pose.elbowRight=180] elbow angle of the right arm, degrees
 * @param {number} [pose.armElevation=0] right arm raised from the flank, degrees
 * @param {number} [pose.armAzimuth=0] right arm direction: 0 forwards, +90 out to the right
 * @param {number} [pose.leanForward=0] trunk lean towards +z, degrees
 * @param {number} [pose.leanRight=0] trunk lean towards the player's right, degrees
 * @param {number} [pose.shoulderTwist=0] shoulder line rotated about the vertical, degrees
 * @param {number} [pose.kneeAngle=180]
 * @param {number} [pose.stance=0.4] lateral spread of the knees, metres; the
 *   ankles end up a little wider still, which is what `stanceWidth` measures
 */
export function makeSkeleton(pose = {}) {
  const {
    trunk = 0.5, shoulderWidth = 0.36, hipWidth = 0.28,
    upperArm = 0.30, forearm = 0.27,
    elbowRight = 180, armElevation = 0, armAzimuth = 0,
    leanForward = 0, leanRight = 0, shoulderTwist = 0,
    kneeAngle = 180, thigh = 0.42, shin = 0.42, stance = 0.4,
  } = pose;

  const p = Array.from({ length: LANDMARK_COUNT }, () => [0, 0, 0]);

  // Trunk axis, tilted forwards (+z) and/or towards the player's right (-x,
  // because a figure facing +z has its right hand at negative x).
  const up = [
    -Math.sin(deg(leanRight)),
    Math.cos(deg(leanForward)) * Math.cos(deg(leanRight)),
    Math.sin(deg(leanForward)),
  ];
  const n = Math.hypot(...up);
  const u = up.map((v) => v / n);

  const midHip = [0, 0, 0];
  const midShoulder = [u[0] * trunk, u[1] * trunk, u[2] * trunk];

  // Hip line runs along the player's right, which is -x when facing +z.
  p[LM.LEFT_HIP] = [hipWidth / 2, 0, 0];
  p[LM.RIGHT_HIP] = [-hipWidth / 2, 0, 0];

  // Shoulder line, optionally twisted about the vertical relative to the hips.
  const tw = deg(shoulderTwist);
  const sx = (Math.cos(tw) * shoulderWidth) / 2;
  const sz = (Math.sin(tw) * shoulderWidth) / 2;
  p[LM.LEFT_SHOULDER] = [midShoulder[0] + sx, midShoulder[1], midShoulder[2] + sz];
  p[LM.RIGHT_SHOULDER] = [midShoulder[0] - sx, midShoulder[1], midShoulder[2] - sz];

  // Right arm. Elevation is measured from straight down the flank; azimuth turns
  // the arm in the horizontal plane, 0 forwards and +90 out to the player's right.
  const e = deg(armElevation);
  const a = deg(armAzimuth);
  const dir = [
    -Math.sin(e) * Math.sin(a),   // player's right is -x
    -Math.cos(e),
    Math.sin(e) * Math.cos(a),
  ];
  const shoulder = p[LM.RIGHT_SHOULDER];
  const elbow = [
    shoulder[0] + dir[0] * upperArm,
    shoulder[1] + dir[1] * upperArm,
    shoulder[2] + dir[2] * upperArm,
  ];
  p[LM.RIGHT_ELBOW] = elbow;

  // Bend the forearm in a plane containing the upper arm. The bend direction is
  // whichever of the trunk axis or +z is not parallel to the upper arm -- with an
  // arm hanging straight down the trunk axis is parallel to it, and using it
  // would collapse the forearm to zero length.
  const bend = Math.PI - deg(elbowRight);
  const perp = perpendicularTo(dir, [u, [0, 0, 1], [1, 0, 0]]);
  const fore = [
    dir[0] * Math.cos(bend) + perp[0] * Math.sin(bend),
    dir[1] * Math.cos(bend) + perp[1] * Math.sin(bend),
    dir[2] * Math.cos(bend) + perp[2] * Math.sin(bend),
  ];
  p[LM.RIGHT_WRIST] = [
    elbow[0] + fore[0] * forearm,
    elbow[1] + fore[1] * forearm,
    elbow[2] + fore[2] * forearm,
  ];

  // Left arm hangs straight down; enough for the tests that need it present.
  p[LM.LEFT_ELBOW] = [p[LM.LEFT_SHOULDER][0], p[LM.LEFT_SHOULDER][1] - upperArm, p[LM.LEFT_SHOULDER][2]];
  p[LM.LEFT_WRIST] = [p[LM.LEFT_ELBOW][0], p[LM.LEFT_ELBOW][1] - forearm, p[LM.LEFT_ELBOW][2]];

  // Legs. The thigh runs from the hip out to the stance width; the shin is the
  // thigh direction rotated forwards about x by the complement of the knee angle,
  // which makes hip-knee-ankle come out at exactly `kneeAngle` whatever the
  // stance.
  const bendKnee = deg(180 - kneeAngle);
  for (const [hipIdx, kneeIdx, ankleIdx, sign] of [
    [LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE, +1],
    [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, -1],
  ]) {
    const hip = p[hipIdx];
    const lateral = (sign * stance) / 2 - hip[0];
    const thighDir = unit([lateral, -thigh, 0]);
    const kneePoint = [
      hip[0] + thighDir[0] * thigh,
      hip[1] + thighDir[1] * thigh,
      hip[2] + thighDir[2] * thigh,
    ];
    // Same construction as the elbow: bend within the plane containing the thigh
    // and +z, so the knee angle is exact whatever the stance.
    const kneePerp = perpendicularTo(thighDir, [[0, 0, 1], [1, 0, 0]]);
    const shinDir = [
      thighDir[0] * Math.cos(bendKnee) + kneePerp[0] * Math.sin(bendKnee),
      thighDir[1] * Math.cos(bendKnee) + kneePerp[1] * Math.sin(bendKnee),
      thighDir[2] * Math.cos(bendKnee) + kneePerp[2] * Math.sin(bendKnee),
    ];
    p[kneeIdx] = kneePoint;
    p[ankleIdx] = [
      kneePoint[0] + shinDir[0] * shin,
      kneePoint[1] + shinDir[1] * shin,
      kneePoint[2] + shinDir[2] * shin,
    ];
  }
  p[LM.LEFT_FOOT_INDEX] = [p[LM.LEFT_ANKLE][0], p[LM.LEFT_ANKLE][1], p[LM.LEFT_ANKLE][2] + 0.1];
  p[LM.RIGHT_FOOT_INDEX] = [p[LM.RIGHT_ANKLE][0], p[LM.RIGHT_ANKLE][1], p[LM.RIGHT_ANKLE][2] + 0.1];
  p[LM.NOSE] = [midShoulder[0], midShoulder[1] + 0.2, midShoulder[2] + 0.1];

  return p;
}

/** The same skeleton in MediaPipe's raw convention (y down, z away). */
export const toMediaPipe = (points) => points.map(([x, y, z]) => [x, -y, -z]);

/** Fully-visible image landmarks, good enough for visibility gates. */
export const fakeImage = (points, visibility = 1) =>
  points.map(([x, y]) => [0.5 + x * 0.2, 0.5 - y * 0.2, 0, visibility]);
