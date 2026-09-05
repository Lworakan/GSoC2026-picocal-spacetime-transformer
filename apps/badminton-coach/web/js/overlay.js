/**
 * Everything drawn on top of the video.
 *
 * The overlay is the app's main explanation of itself: an angle is far easier to
 * trust when you can see the arc drawn on the arm than when it is a number in a
 * table. So the joint angles that drive the coaching are drawn *at the joint*,
 * and the court model is drawn back onto the floor so a bad calibration is
 * obvious at a glance rather than silently wrong.
 */

import { CONNECTIONS, LM, side } from './core/landmarks.js';
import { COURT, CALIBRATION_ORDER } from './core/court.js';

const COLORS = {
  bone: 'rgba(226, 232, 240, 0.85)',
  racket: '#f97316',
  free: '#38bdf8',
  joint: 'rgba(15, 23, 42, 0.9)',
  good: '#22c55e',
  warn: '#facc15',
  bad: '#ef4444',
  box: 'rgba(56, 189, 248, 0.55)',
  court: 'rgba(250, 204, 21, 0.9)',
  courtFill: 'rgba(250, 204, 21, 0.08)',
  text: '#f8fafc',
  shadow: 'rgba(2, 6, 23, 0.75)',
};

/** Resize the overlay canvas to match its displayed size, allowing for DPR. */
export function fitCanvas(canvas, dpr = window.devicePixelRatio || 1) {
  const rect = canvas.getBoundingClientRect();
  const w = Math.round(rect.width * dpr);
  const h = Math.round(rect.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return { width: rect.width, height: rect.height, dpr };
}

/**
 * The rectangle a `object-fit: contain` video actually occupies inside its box.
 *
 * Without this, every overlay is stretched relative to the video on any device
 * whose screen is not the same shape as the camera -- which is most of them, and
 * all of them in portrait.
 */
export function contentRect(boxW, boxH, srcW, srcH) {
  if (!srcW || !srcH) return { x: 0, y: 0, width: boxW, height: boxH };
  const scale = Math.min(boxW / srcW, boxH / srcH);
  const width = srcW * scale;
  const height = srcH * scale;
  return { x: (boxW - width) / 2, y: (boxH - height) / 2, width, height };
}

/** Map a normalised video point to canvas pixels. */
export const project = (rect, [x, y]) => [rect.x + x * rect.width, rect.y + y * rect.height];

/** Map a canvas point (CSS pixels) back to normalised video coordinates. */
export function unproject(rect, x, y) {
  return [(x - rect.x) / rect.width, (y - rect.y) / rect.height];
}

export function clear(ctx, canvas, dpr) {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
}

/** The tracked crop, so it is visible what the model is actually being shown. */
export function drawWindow(ctx, rect, window) {
  if (!window) return;
  const [x0, y0] = project(rect, [window.x0, window.y0]);
  const [x1, y1] = project(rect, [window.x1, window.y1]);
  ctx.save();
  ctx.strokeStyle = COLORS.box;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 6]);
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.restore();
}

export function drawSkeleton(ctx, rect, image, { racketArm = 'right', scale = 1 } = {}) {
  if (!image) return;
  const p = (i) => project(rect, image[i]);
  const racketBones = new Set([
    `${side('SHOULDER', racketArm)}-${side('ELBOW', racketArm)}`,
    `${side('ELBOW', racketArm)}-${side('WRIST', racketArm)}`,
  ]);
  const freeArm = racketArm === 'right' ? 'left' : 'right';
  const freeBones = new Set([
    `${side('SHOULDER', freeArm)}-${side('ELBOW', freeArm)}`,
    `${side('ELBOW', freeArm)}-${side('WRIST', freeArm)}`,
  ]);

  ctx.save();
  ctx.lineCap = 'round';
  for (const [a, b] of CONNECTIONS) {
    const key = `${a}-${b}`;
    const racket = racketBones.has(key);
    const free = freeBones.has(key);
    ctx.strokeStyle = racket ? COLORS.racket : free ? COLORS.free : COLORS.bone;
    ctx.lineWidth = (racket ? 5 : 3.5) * scale;
    ctx.beginPath();
    ctx.moveTo(...p(a));
    ctx.lineTo(...p(b));
    ctx.stroke();
  }
  for (const i of [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP,
    LM.LEFT_ELBOW, LM.RIGHT_ELBOW, LM.LEFT_KNEE, LM.RIGHT_KNEE]) {
    const [x, y] = p(i);
    ctx.fillStyle = COLORS.text;
    ctx.beginPath();
    ctx.arc(x, y, 3 * scale, 0, Math.PI * 2);
    ctx.fill();
  }
  // The racket hand gets a bigger marker: it is the thing being measured.
  const [wx, wy] = p(side('WRIST', racketArm));
  ctx.fillStyle = COLORS.racket;
  ctx.beginPath();
  ctx.arc(wx, wy, 6 * scale, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/**
 * An arc at joint `b` spanning the a-b-c angle, labelled with its value.
 *
 * `status` colours it, so a bent elbow at contact turns red on the video itself.
 */
export function drawAngleArc(ctx, rect, image, [a, b, c], value, {
  status = 'good', radius = 26, scale = 1, label = null,
} = {}) {
  if (!Number.isFinite(value)) return;
  const pa = project(rect, image[a]);
  const pb = project(rect, image[b]);
  const pc = project(rect, image[c]);
  const a1 = Math.atan2(pa[1] - pb[1], pa[0] - pb[0]);
  const a2 = Math.atan2(pc[1] - pb[1], pc[0] - pb[0]);
  let delta = a2 - a1;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;

  const r = radius * scale;
  const colour = COLORS[status] || COLORS.good;
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.lineWidth = 3 * scale;
  ctx.beginPath();
  ctx.arc(pb[0], pb[1], r, a1, a1 + delta, delta < 0);
  ctx.stroke();

  const mid = a1 + delta / 2;
  const tx = pb[0] + Math.cos(mid) * (r + 16 * scale);
  const ty = pb[1] + Math.sin(mid) * (r + 16 * scale);
  const text = label ? `${label} ${Math.round(value)}°` : `${Math.round(value)}°`;
  ctx.font = `${Math.round(13 * scale)}px system-ui, -apple-system, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const w = ctx.measureText(text).width + 10 * scale;
  ctx.fillStyle = COLORS.shadow;
  roundRect(ctx, tx - w / 2, ty - 10 * scale, w, 20 * scale, 6 * scale);
  ctx.fill();
  ctx.fillStyle = colour;
  ctx.fillText(text, tx, ty);
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Court lines, reprojected from the model so a bad calibration is visible. */
export function drawCourt(ctx, rect, calibration) {
  if (!calibration?.valid) return;
  const half = COURT.width / 2;
  const singles = COURT.singlesWidth / 2;
  const lines = [
    // outer boundary
    [[-half, 0], [half, 0], [half, COURT.halfLength], [-half, COURT.halfLength], [-half, 0]],
    // singles side lines
    [[-singles, 0], [-singles, COURT.halfLength]],
    [[singles, 0], [singles, COURT.halfLength]],
    // short service line and the doubles long service line
    [[-half, COURT.shortServiceLine], [half, COURT.shortServiceLine]],
    [[-half, COURT.halfLength - COURT.doublesLongServiceLine],
      [half, COURT.halfLength - COURT.doublesLongServiceLine]],
    // centre line, from the short service line back
    [[0, COURT.shortServiceLine], [0, COURT.halfLength]],
  ];

  ctx.save();
  ctx.strokeStyle = COLORS.court;
  ctx.lineWidth = 1.5;
  for (const line of lines) {
    ctx.beginPath();
    line.forEach((pt, i) => {
      const image = calibration.imagePoint(pt);
      if (!image) return;
      const [x, y] = project(rect, image);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Label the net edge: it is the one thing a user can get backwards.
  const netMid = calibration.imagePoint([0, 0]);
  if (netMid) {
    const [x, y] = project(rect, netMid);
    ctx.fillStyle = COLORS.court;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('NET', x, y - 6);
  }
  ctx.restore();
}

/** Calibration taps in progress. */
export function drawCalibrationPoints(ctx, rect, points) {
  ctx.save();
  points.forEach((pt, i) => {
    const [x, y] = project(rect, pt);
    ctx.fillStyle = COLORS.court;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = COLORS.joint;
    ctx.font = 'bold 11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), x, y);
  });
  if (points.length > 1) {
    ctx.strokeStyle = COLORS.court;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    points.forEach((pt, i) => {
      const [x, y] = project(rect, pt);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (points.length === 4) ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();
}

/** A small plan view of the court with the player's position and trail. */
export function drawCourtMap(canvas, positions, { base = [0, 3.0], recent = 120 } = {}) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const pad = 12;
  // A canvas inside a hidden tab measures 0x0. Drawing into it produced a
  // negative radius, and the uncaught error took down the frame loop with it.
  if (rect.width < pad * 3 || rect.height < pad * 3) return;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const scale = Math.min(
    (rect.width - pad * 2) / COURT.width,
    (rect.height - pad * 2) / COURT.halfLength,
  );
  const ox = rect.width / 2;
  const oy = pad;
  const toXY = ([x, y]) => [ox + x * scale, oy + y * scale];

  ctx.strokeStyle = 'rgba(148, 163, 184, 0.8)';
  ctx.lineWidth = 1;
  const half = COURT.width / 2;
  const singles = COURT.singlesWidth / 2;
  const box = [[-half, 0], [half, 0], [half, COURT.halfLength], [-half, COURT.halfLength]];
  ctx.beginPath();
  box.forEach((p, i) => (i ? ctx.lineTo(...toXY(p)) : ctx.moveTo(...toXY(p))));
  ctx.closePath();
  ctx.stroke();
  for (const line of [
    [[-singles, 0], [-singles, COURT.halfLength]],
    [[singles, 0], [singles, COURT.halfLength]],
    [[-half, COURT.shortServiceLine], [half, COURT.shortServiceLine]],
    [[0, COURT.shortServiceLine], [0, COURT.halfLength]],
  ]) {
    ctx.beginPath();
    ctx.moveTo(...toXY(line[0]));
    ctx.lineTo(...toXY(line[1]));
    ctx.stroke();
  }
  ctx.fillStyle = 'rgba(148, 163, 184, 0.9)';
  ctx.font = '10px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('NET', ox, oy - 2);

  const trail = positions.filter(Boolean).slice(-recent);
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  trail.forEach((p, i) => {
    const [x, y] = toXY([p.x, p.y]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const [bx, by] = toXY(base);
  ctx.strokeStyle = 'rgba(34, 197, 94, 0.9)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.arc(bx, by, 1.0 * scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  const last = trail[trail.length - 1];
  if (last) {
    const [x, y] = toXY([last.x, last.y]);
    ctx.fillStyle = COLORS.racket;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

export { COLORS, CALIBRATION_ORDER };
