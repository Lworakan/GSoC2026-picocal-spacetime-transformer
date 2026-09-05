/**
 * Two-stage pose tracking: find the player once, then follow a cropped ROI.
 *
 * Filming a badminton court with a phone puts the player at 15-25% of the frame
 * height, which is where a single full-frame pass starts dropping detections. On
 * the reference clip a full-frame pass found the player in 58% of frames;
 * cropping a padded square around the previous pose and upscaling it before
 * inference took that to 96%. It is also *cheaper* -- a 384x384 crop is a small
 * fraction of a 1080p frame -- which is what makes this usable on a phone.
 *
 * The state machine holds no reference to any pose library: callers supply two
 * detector callbacks, which is what lets the Python pipeline
 * (`analyze/badminton_coach/roi.py`) share the design and lets the tests drive it
 * with a fake detector.
 */

/** Axis-aligned box in normalised image coordinates. */
export class Box {
  constructor(x0, y0, x1, y1) {
    this.x0 = x0;
    this.y0 = y0;
    this.x1 = x1;
    this.y1 = y1;
  }

  get width() { return this.x1 - this.x0; }
  get height() { return this.y1 - this.y0; }
  get cx() { return (this.x0 + this.x1) / 2; }
  get cy() { return (this.y0 + this.y1) / 2; }

  contains(x, y) {
    return x >= this.x0 && x <= this.x1 && y >= this.y0 && y <= this.y1;
  }

  distanceTo(x, y) {
    return Math.hypot(this.cx - x, this.cy - y);
  }

  toArray() { return [this.x0, this.y0, this.x1, this.y1]; }
}

/** Bounding box of a landmark list of `[x, y, ...]`. */
export function bounds(points) {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const p of points) {
    if (p[0] < x0) x0 = p[0];
    if (p[0] > x1) x1 = p[0];
    if (p[1] < y0) y0 = p[1];
    if (p[1] > y1) y1 = p[1];
  }
  return new Box(x0, y0, x1, y1);
}

/**
 * A square-ish window around `box`, padded for the arm swing.
 *
 * The padding is generous on purpose: at the top of a smash the racket arm
 * leaves the torso's box entirely, and a crop that clips the wrist is worse than
 * no crop at all. `aspect` is the source frame's width/height, used to keep the
 * window square in *pixels* rather than in normalised units.
 */
export function cropWindow(box, pad = 0.85, aspect = 1) {
  const half = Math.max(box.width * aspect, box.height) * pad;
  const halfX = half / aspect;
  return new Box(
    Math.max(0, box.cx - halfX),
    Math.max(0, box.cy - half),
    Math.min(1, box.cx + halfX),
    Math.min(1, box.cy + half),
  );
}

/** Map landmarks measured inside `window` back to full-frame coordinates. */
export function unproject(points, window) {
  return points.map((p) => {
    const out = p.slice();
    out[0] = window.x0 + p[0] * window.width;
    out[1] = window.y0 + p[1] * window.height;
    return out;
  });
}

export const DEFAULT_ROI_OPTIONS = {
  /** Crop padding as a multiple of the pose's larger dimension. */
  pad: 0.85,
  /** Consecutive empty crops before falling back to a full-frame search. */
  maxMisses: 8,
  /** Reject a crop result whose size jumps by more than this: it is a different
   *  person who wandered into the crop, not our player. */
  maxSizeRatio: 2.2,
  /** Reject a crop result whose centre moves further than this in one frame. */
  maxJump: 0.25,
  /** When re-acquiring, only accept a detection this close to the last sighting. */
  reacquireRadius: 0.35,
};

export class RoiTracker {
  /**
   * @param {object} config
   * @param {Function} config.detectFull frame -> array of poses in full-frame coords
   * @param {Function} config.detectCrop (frame, Box) -> one pose in crop-local coords, or null
   * @param {number} [config.aspect=1] source frame width/height
   * @param {object} [config.options] see {@link DEFAULT_ROI_OPTIONS}
   */
  constructor({ detectFull, detectCrop, aspect = 1, options = {} }) {
    this.detectFull = detectFull;
    this.detectCrop = detectCrop;
    this.aspect = aspect;
    this.options = { ...DEFAULT_ROI_OPTIONS, ...options };
    this.box = null;
    this.lastBox = null;
    this.misses = 0;
    this.targetHint = null;
    this.stats = { full: 0, crop: 0, hits: 0, reacquisitions: 0 };
  }

  get tracking() { return this.box !== null; }

  /** Ask the next full-frame search to lock on to whoever is at this point. */
  lockOn(x, y) {
    this.targetHint = [x, y];
    this.box = null;
    this.misses = 0;
  }

  reset() {
    this.box = null;
    this.lastBox = null;
    this.misses = 0;
    this.targetHint = null;
  }

  /** The crop that will be used next, for drawing it on the overlay. */
  get window() {
    return this.box ? cropWindow(this.box, this.options.pad, this.aspect) : null;
  }

  /** Returns this frame's pose for the tracked player, or null. */
  async step(frame) {
    if (this.box !== null) {
      const pose = await this._stepCrop(frame);
      if (pose) return pose;
      // A miss keeps the lock and reports nothing for this frame. Falling
      // straight through to a full-frame search here would spend a full-
      // resolution inference on every dropped frame -- precisely when the device
      // is already behind -- and would make `maxMisses` mean nothing. The crop
      // clears `box` itself once it has missed too often.
      if (this.box !== null) return null;
    }
    return this._stepFull(frame);
  }

  async _stepCrop(frame) {
    const window = cropWindow(this.box, this.options.pad, this.aspect);
    this.stats.crop += 1;
    const result = await this.detectCrop(frame, window);
    if (!result) return this._miss();

    const image = unproject(result.image, window);
    const box = bounds(image);
    if (!this._plausible(box)) return this._miss();

    this.misses = 0;
    this.box = box;
    this.lastBox = box;
    this.stats.hits += 1;
    return { ...result, image, source: 'crop', box, window };
  }

  _miss() {
    this.misses += 1;
    if (this.misses > this.options.maxMisses) {
      this.box = null;
      this.misses = 0;
    }
    return null;
  }

  /** Guard against the crop latching on to a different body. */
  _plausible(box) {
    const old = this.box;
    if (!old) return true;
    if (old.height > 1e-6 && box.height > 1e-6) {
      const ratio = Math.max(old.height / box.height, box.height / old.height);
      if (ratio > this.options.maxSizeRatio) return false;
    }
    return box.distanceTo(old.cx, old.cy) <= this.options.maxJump;
  }

  async _stepFull(frame) {
    this.stats.full += 1;
    const poses = (await this.detectFull(frame)) || [];
    if (!poses.length) return null;
    const pose = this._choose(poses);
    if (!pose) return null;
    this.box = bounds(pose.image);
    this.lastBox = this.box;
    this.misses = 0;
    this.stats.hits += 1;
    this.stats.reacquisitions += 1;
    return { ...pose, source: 'full', box: this.box, window: null };
  }

  /**
   * Pick which full-frame detection is our player.
   *
   * A tap from the user wins. Otherwise prefer whoever is nearest to where the
   * player was a moment ago -- never simply the biggest pose in frame, because
   * the biggest pose is often a bystander walking between the phone and the
   * court, which is exactly what happens near the end of the reference clip.
   */
  _choose(poses) {
    const hint = this.targetHint;
    if (hint) {
      const inside = poses.filter((p) => bounds(p.image).contains(hint[0], hint[1]));
      const pool = inside.length ? inside : poses;
      this.targetHint = null;
      return pool.reduce((best, p) =>
        bounds(p.image).distanceTo(hint[0], hint[1]) < bounds(best.image).distanceTo(hint[0], hint[1])
          ? p
          : best);
    }
    if (this.lastBox) {
      const near = poses.filter(
        (p) => bounds(p.image).distanceTo(this.lastBox.cx, this.lastBox.cy)
          <= this.options.reacquireRadius,
      );
      if (near.length) {
        return near.reduce((best, p) =>
          bounds(p.image).distanceTo(this.lastBox.cx, this.lastBox.cy)
            < bounds(best.image).distanceTo(this.lastBox.cx, this.lastBox.cy)
            ? p
            : best);
      }
    }
    return poses.reduce((best, p) => (bounds(p.image).height > bounds(best.image).height ? p : best));
  }
}
