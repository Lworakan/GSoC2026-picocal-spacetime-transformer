/**
 * Minimal 3-vector helpers.
 *
 * Vectors are plain `[x, y, z]` arrays so that landmark data can be passed
 * straight through from MediaPipe and from the Python pipeline's JSON without
 * an object-wrapping step on every frame.
 */

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

export const length = (a) => Math.hypot(a[0], a[1], a[2]);
export const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
export const midpoint = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];

/** Unit vector, or `[0,0,0]` for a degenerate input. */
export function normalize(a) {
  const n = length(a);
  return n < 1e-9 ? [0, 0, 0] : [a[0] / n, a[1] / n, a[2] / n];
}

/** Component of `a` perpendicular to the unit vector `axis`. */
export function reject(a, axis) {
  const k = dot(a, axis);
  return [a[0] - axis[0] * k, a[1] - axis[1] * k, a[2] - axis[2] * k];
}

export const RAD_TO_DEG = 180 / Math.PI;

/** Unsigned angle between two vectors, in degrees (0..180). */
export function angleBetween(a, b) {
  const na = length(a);
  const nb = length(b);
  if (na < 1e-9 || nb < 1e-9) return NaN;
  const c = Math.min(1, Math.max(-1, dot(a, b) / (na * nb)));
  return Math.acos(c) * RAD_TO_DEG;
}

/**
 * Interior angle at `b` in the chain a-b-c, in degrees.
 *
 * 180 is a straight limb, 0 is fully folded. This is the standard way joint
 * angles are reported in the sports-biomechanics literature, so an elbow angle
 * from here can be compared with published values directly.
 */
export const jointAngle = (a, b, c) => angleBetween(sub(a, b), sub(c, b));

/**
 * Signed angle from `a` to `b` about `axis`, in degrees (-180..180).
 *
 * Positive follows the right-hand rule around `axis`.
 */
export function signedAngle(a, b, axis) {
  const n = normalize(axis);
  const pa = reject(a, n);
  const pb = reject(b, n);
  if (length(pa) < 1e-9 || length(pb) < 1e-9) return NaN;
  const angle = angleBetween(pa, pb);
  return dot(cross(pa, pb), n) < 0 ? -angle : angle;
}
