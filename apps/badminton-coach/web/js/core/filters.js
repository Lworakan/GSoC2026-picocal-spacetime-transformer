/**
 * Signal conditioning for pose landmarks.
 *
 * Raw MediaPipe landmarks on a player filmed from across a court jitter badly:
 * on the reference clip the raw right-wrist speed peaked at 41 m/s, which is
 * roughly three times the fastest wrist speed ever measured in a badminton
 * smash, so it is noise rather than movement. Everything downstream reads
 * filtered values.
 */

/**
 * A single-channel 1-Euro filter.
 *
 * A plain low-pass filter forces a choice between jitter at rest and lag during
 * a swing, and a smash is over in about 100 ms, so lag is not affordable. The
 * 1-Euro filter (Casiez, Roussel & Vogel, CHI 2012) widens its own cutoff as the
 * signal speeds up, which keeps the racket arm sharp while the standing player
 * stays still.
 */
export class OneEuro {
  /**
   * @param {object} [options]
   * @param {number} [options.minCutoff=1.2] Cutoff in Hz at zero speed: lower is smoother at rest.
   * @param {number} [options.beta=0.25] How fast the cutoff opens with speed: higher means less lag.
   * @param {number} [options.dCutoff=1.0] Cutoff for the internal speed estimate.
   */
  constructor({ minCutoff = 1.2, beta = 0.25, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.reset();
  }

  reset() {
    this.x = null;
    this.dx = 0;
    this.t = null;
  }

  static alpha(dt, cutoff) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  /**
   * @param {number} value raw sample
   * @param {number} t sample time in seconds
   * @returns {number} filtered value
   */
  filter(value, t) {
    if (this.x === null || this.t === null || !(t > this.t)) {
      this.x = value;
      this.t = t;
      this.dx = 0;
      return value;
    }
    const dt = t - this.t;
    const dxRaw = (value - this.x) / dt;
    this.dx = this.dx + OneEuro.alpha(dt, this.dCutoff) * (dxRaw - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x = this.x + OneEuro.alpha(dt, cutoff) * (value - this.x);
    this.t = t;
    return this.x;
  }
}

/** A 1-Euro filter applied independently to every coordinate of a landmark set. */
export class LandmarkFilter {
  /**
   * @param {number} count number of landmarks
   * @param {number} dims coordinates per landmark
   * @param {object} [options] passed to each {@link OneEuro}
   */
  constructor(count, dims = 3, options = {}) {
    this.dims = dims;
    this.channels = Array.from(
      { length: count * dims },
      () => new OneEuro(options),
    );
  }

  reset() {
    for (const c of this.channels) c.reset();
  }

  /**
   * @param {number[][]} points landmarks as arrays of at least `dims` numbers
   * @param {number} t sample time in seconds
   * @returns {number[][]} filtered landmarks; extra columns (visibility) pass through
   */
  filter(points, t) {
    return points.map((p, i) => {
      const out = p.slice();
      for (let d = 0; d < this.dims; d += 1) {
        out[d] = this.channels[i * this.dims + d].filter(p[d], t);
      }
      return out;
    });
  }
}

/**
 * Median over a sliding window.
 *
 * Used for body-scale estimates, where a single bad frame would otherwise
 * rescale every normalised measurement on that frame. On the reference clip the
 * per-frame trunk length wanders between 0.36 m and 0.53 m for a player whose
 * trunk is, of course, one fixed length.
 */
export class RunningMedian {
  /** @param {number} size window length in samples */
  constructor(size = 31) {
    this.size = size;
    this.buffer = [];
  }

  reset() {
    this.buffer = [];
  }

  /** @returns {number} the median including `value`, or NaN if nothing valid yet */
  push(value) {
    if (Number.isFinite(value)) {
      this.buffer.push(value);
      if (this.buffer.length > this.size) this.buffer.shift();
    }
    return this.value;
  }

  get value() {
    if (this.buffer.length === 0) return NaN;
    const sorted = this.buffer.slice().sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }
}

/** Median of an array, ignoring non-finite entries. */
export function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Arithmetic mean, ignoring non-finite entries. */
export function mean(values) {
  const ok = values.filter(Number.isFinite);
  return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : NaN;
}

/**
 * Central difference of a scalar series, robust to uneven frame spacing.
 *
 * A central difference has half the noise gain of a forward difference and no
 * half-frame phase shift, which matters because the peak of this derivative is
 * what we call the moment of contact.
 */
export function derivative(values, times) {
  const out = new Array(values.length).fill(NaN);
  for (let i = 1; i < values.length - 1; i += 1) {
    const dt = times[i + 1] - times[i - 1];
    if (dt > 0) out[i] = (values[i + 1] - values[i - 1]) / dt;
  }
  if (values.length > 1) {
    const dt0 = times[1] - times[0];
    if (dt0 > 0) out[0] = (values[1] - values[0]) / dt0;
    const n = values.length - 1;
    const dtn = times[n] - times[n - 1];
    if (dtn > 0) out[n] = (values[n] - values[n - 1]) / dtn;
  }
  return out;
}
