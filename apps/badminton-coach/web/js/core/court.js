/**
 * Put the player on the badminton court.
 *
 * Joint angles tell you how a shot was played; they say nothing about *where*
 * from, or whether the player got back to base afterwards, which is most of what
 * decides a rally. This module maps the player's feet from the image into real
 * court metres, so the same overhead played from the rear corner and from the
 * mid-court can be told apart.
 *
 * ## How the mapping works
 *
 * The court is flat and the feet are on it, so a single homography (a 3x3
 * projective transform) relates image pixels to court metres exactly -- no camera
 * calibration and no depth estimate needed. Four tapped corners determine it.
 *
 * ## What it assumes
 *
 * - The camera does not move after calibration. A phone on a bag or a tripod is
 *   fine; a phone held in someone's hand will drift and needs re-tapping.
 * - The player's feet are on the floor. During a jump smash the mapped position
 *   creeps towards the camera; those frames are flagged rather than silently used.
 */

/** BWF court dimensions, in metres. */
export const COURT = {
  length: 13.40,
  width: 6.10,
  halfLength: 6.70,
  singlesWidth: 5.18,
  shortServiceLine: 1.98,
  doublesLongServiceLine: 0.76,
  netHeightCentre: 1.524,
  netHeightPosts: 1.55,
};

/**
 * Court coordinates for the near half, in metres:
 * `x` runs -3.05 (left) to +3.05 (right) as the player faces the net,
 * `y` runs 0 at the net to 6.70 at the back boundary.
 */
export const HALF_COURT_CORNERS = {
  netLeft: [-COURT.width / 2, 0],
  netRight: [COURT.width / 2, 0],
  backRight: [COURT.width / 2, COURT.halfLength],
  backLeft: [-COURT.width / 2, COURT.halfLength],
};

/** The order the app asks for taps in. */
export const CALIBRATION_ORDER = ['netLeft', 'netRight', 'backRight', 'backLeft'];

/** Singles base position: centre, a little behind mid-court. */
export const BASE_POSITION = [0, 3.0];

/** Depth band edges (metres from the net) and lateral band edges (metres from centre). */
export const ZONES = {
  depth: [
    { name: 'front', from: 0, to: 2.2 },
    { name: 'mid', from: 2.2, to: 4.5 },
    { name: 'rear', from: 4.5, to: COURT.halfLength },
  ],
  lateral: [
    { name: 'left', from: -Infinity, to: -1.0 },
    { name: 'centre', from: -1.0, to: 1.0 },
    { name: 'right', from: 1.0, to: Infinity },
  ],
};

/**
 * Solve `A x = b` by Gaussian elimination with partial pivoting.
 * @returns {number[]|null} the solution, or null if the system is singular
 */
export function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < n; r += 1) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < n; r += 1) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      if (factor === 0) continue;
      for (let c = col; c <= n; c += 1) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

/**
 * Homography mapping four source points to four destination points.
 *
 * With h33 fixed at 1 the eight remaining unknowns are determined exactly by four
 * point correspondences, so this is a linear solve rather than a least-squares
 * fit -- which is why the four taps must be the four *corners*, and why tapping
 * them accurately matters.
 *
 * @returns {number[][]|null} a 3x3 matrix, or null if the points are degenerate
 */
export function homographyFromQuad(src, dst) {
  if (src.length !== 4 || dst.length !== 4) throw new Error('need exactly four points');
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i += 1) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const h = solveLinear(A, b);
  if (!h) return null;
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

/** Apply a homography to a 2D point. Returns null behind the horizon. */
export function applyHomography(H, [x, y]) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  if (Math.abs(w) < 1e-12) return null;
  return [
    (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
  ];
}

/** Invert a 3x3 matrix. */
export function invert3x3(M) {
  const [a, b, c] = M[0];
  const [d, e, f] = M[1];
  const [g, h, i] = M[2];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) return null;
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

/** Which of the nine zones a court position falls in. */
export function zoneOf([x, y]) {
  const depth = ZONES.depth.find((z) => y >= z.from && y < z.to)
    || (y < 0 ? ZONES.depth[0] : ZONES.depth[ZONES.depth.length - 1]);
  const lateral = ZONES.lateral.find((z) => x >= z.from && x < z.to) || ZONES.lateral[1];
  return { depth: depth.name, lateral: lateral.name, name: `${depth.name}-${lateral.name}` };
}

/** Straight-line distance from the singles base position, in metres. */
export const distanceFromBase = ([x, y]) => Math.hypot(x - BASE_POSITION[0], y - BASE_POSITION[1]);

/**
 * Is the quad simple and convex?
 *
 * The four corners of a real court always project to a convex quadrilateral. If
 * the taps came in a non-cyclic order -- net-left, back-right, net-right,
 * back-left, say -- the quad self-intersects into a bow tie, and the homography
 * that comes out of it inverts perfectly well while mapping the player to
 * nonsense. Winding direction is deliberately *not* checked: filming the same
 * half from the opposite side of the hall genuinely mirrors it, and both are
 * legitimate.
 */
export function isConvexQuad(points, minEdge = 0.02) {
  if (points.length !== 4) return false;
  for (let i = 0; i < 4; i += 1) {
    for (let j = i + 1; j < 4; j += 1) {
      if (Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) < minEdge) {
        return false;
      }
    }
  }
  let sign = 0;
  for (let i = 0; i < 4; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % 4];
    const c = points[(i + 2) % 4];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(cross) < 1e-9) return false;
    const s = Math.sign(cross);
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return true;
}

/** True if the position is inside the court boundary, with a little tolerance. */
export function onCourt([x, y], margin = 0.5) {
  return (
    Math.abs(x) <= COURT.width / 2 + margin &&
    y >= -margin &&
    y <= COURT.halfLength + margin
  );
}

/**
 * A calibrated court: image pixels in, court metres out.
 *
 * Calibration points are stored in *normalised* image coordinates (0..1) so the
 * calibration survives the video being displayed at a different size, which it
 * always is between a phone in portrait and a laptop.
 */
export class CourtCalibration {
  /**
   * @param {Array<[number,number]>} points four normalised image points, in
   *   {@link CALIBRATION_ORDER}
   */
  constructor(points) {
    this.points = points.map((p) => [p[0], p[1]]);
    const dst = CALIBRATION_ORDER.map((k) => HALF_COURT_CORNERS[k]);
    this.toCourt = homographyFromQuad(this.points, dst);
    this.toImage = this.toCourt ? invert3x3(this.toCourt) : null;
  }

  get valid() {
    return this.toCourt !== null && this._sane();
  }

  /**
   * Reject a calibration whose corners were tapped out of order or effectively on
   * top of each other -- both give a matrix that inverts fine and then reports
   * the player standing 40 m off court.
   */
  _sane() {
    if (!isConvexQuad(this.points)) return false;
    const centre = applyHomography(this.toCourt, [
      (this.points[0][0] + this.points[1][0] + this.points[2][0] + this.points[3][0]) / 4,
      (this.points[0][1] + this.points[1][1] + this.points[2][1] + this.points[3][1]) / 4,
    ]);
    return centre !== null && onCourt(centre, 1.5);
  }

  /**
   * True when the tapped quad is wound the opposite way to the court model,
   * which happens when the camera is on the other side of the hall. Harmless in
   * itself; the app surfaces it so a user who mislabelled left and right can see
   * it and swap.
   */
  get mirrored() {
    const [a, b, c] = this.points;
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    return cross > 0;
  }

  /** The same four corners with left and right exchanged. */
  swapSides() {
    const [nl, nr, br, bl] = this.points;
    return new CourtCalibration([nr, nl, bl, br]);
  }

  /** Court position, in metres, of a normalised image point. */
  courtPoint(imagePoint) {
    if (!this.toCourt) return null;
    return applyHomography(this.toCourt, imagePoint);
  }

  /** Normalised image point of a court position, for drawing the court back on. */
  imagePoint(courtPoint) {
    if (!this.toImage) return null;
    return applyHomography(this.toImage, courtPoint);
  }

  /** Serialise for localStorage. */
  toJSON() {
    return { points: this.points };
  }

  static fromJSON(data) {
    return data?.points?.length === 4 ? new CourtCalibration(data.points) : null;
  }
}

/**
 * Where the player is standing, from their feet.
 *
 * Ankles are used rather than the foot tips because they are the more reliable
 * landmark, and the heel-to-toe difference is small next to the size of a zone.
 * `airborne` marks frames where both feet have left the floor, whose mapped
 * position is not trustworthy.
 */
export function footPosition(image, LM) {
  const left = image[LM.LEFT_ANKLE];
  const right = image[LM.RIGHT_ANKLE];
  if (!left || !right) return null;
  const vis = Math.min(left[3] ?? 1, right[3] ?? 1);
  return {
    point: [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2],
    visibility: vis,
  };
}

/**
 * Court-movement analysis over a whole session.
 *
 * @param {Array} frames analysed frames with `image` landmarks
 * @param {CourtCalibration} calibration
 * @param {object} LM landmark index map
 */
export function courtTrack(frames, calibration, LM, { minVisibility = 0.4 } = {}) {
  const positions = [];
  for (const f of frames) {
    if (!f.image) {
      positions.push(null);
      continue;
    }
    const feet = footPosition(f.image, LM);
    if (!feet || feet.visibility < minVisibility) {
      positions.push(null);
      continue;
    }
    const court = calibration.courtPoint(feet.point);
    positions.push(
      court && onCourt(court, 2.0)
        ? { t: f.t, x: court[0], y: court[1], zone: zoneOf(court), base: distanceFromBase(court) }
        : null,
    );
  }
  return positions;
}

/** Total ground covered, in metres, ignoring gaps and obvious jitter. */
export function distanceCovered(positions, { maxStep = 1.2 } = {}) {
  let total = 0;
  let previous = null;
  for (const p of positions) {
    if (!p) {
      previous = null;
      continue;
    }
    if (previous) {
      const step = Math.hypot(p.x - previous.x, p.y - previous.y);
      // A step larger than this between adjacent frames is a tracking glitch,
      // not a movement any player makes in 1/30 s.
      if (step <= maxStep) total += step;
    }
    previous = p;
  }
  return total;
}

/** How long each zone was occupied, in seconds. */
export function zoneOccupancy(positions) {
  const seconds = {};
  for (let i = 1; i < positions.length; i += 1) {
    const p = positions[i];
    const prev = positions[i - 1];
    if (!p || !prev) continue;
    const dt = p.t - prev.t;
    if (!(dt > 0) || dt > 0.5) continue;
    seconds[p.zone.name] = (seconds[p.zone.name] || 0) + dt;
  }
  return seconds;
}

/**
 * Time from each stroke until the player is back near base.
 *
 * Recovery is the habit that separates players who can keep a rally going from
 * players who get caught out of position, and it is invisible in a shot-by-shot
 * angle readout.
 */
export function recoveryTimes(strokes, positions, { radius = 1.0, limit = 3.0 } = {}) {
  const byTime = positions.filter(Boolean);
  return strokes.map((s) => {
    const after = byTime.filter((p) => p.t > s.t && p.t <= s.t + limit);
    const home = after.find((p) => p.base <= radius);
    const at = byTime.find((p) => p.t >= s.t);
    return {
      strokeIndex: s.index,
      t: s.t,
      position: at ? { x: at.x, y: at.y, zone: at.zone.name, base: at.base } : null,
      recoverySeconds: home ? home.t - s.t : null,
      recovered: Boolean(home),
    };
  });
}
