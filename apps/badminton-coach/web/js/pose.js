/**
 * MediaPipe Pose, wired to the ROI tracker.
 *
 * The wrapper exists so that everything above it -- the analysis core, the UI --
 * sees plain arrays of numbers and never a MediaPipe type. That keeps the maths
 * testable in Node, and it means swapping the pose backend later touches one
 * file.
 *
 * One landmarker instance serves both stages. Running two would double the
 * memory a phone has to find for the model, and the detector cost is dominated
 * by the input size, not by how many people it is allowed to return.
 */

import { RoiTracker } from './core/roitracker.js';

const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';

/**
 * A vendored copy takes precedence over the CDN.
 *
 * `npm run vendor` writes the MediaPipe runtime and a pose model into
 * `web/vendor/`, after which the app needs no network at all -- which is the
 * normal state of affairs inside a sports hall.
 */
const LOCAL_RUNTIME = 'vendor/tasks-vision';

/**
 * Resolve an app-relative path against the *page*, not against this module.
 *
 * `fetch('./x')` resolves against the document while `import('./x')` resolves
 * against the importing module, and this file lives one directory down. Left
 * implicit, the same string means two different URLs and the vendored runtime
 * silently 404s.
 */
const appUrl = (path) => new URL(path, document.baseURI).href;

async function exists(url) {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}

async function resolveRuntime() {
  const bundle = appUrl(`${LOCAL_RUNTIME}/vision_bundle.mjs`);
  if (await exists(bundle)) {
    return { bundle, wasm: appUrl(`${LOCAL_RUNTIME}/wasm`), local: true };
  }
  return { bundle: `${CDN}/vision_bundle.mjs`, wasm: `${CDN}/wasm`, local: false };
}

const MODEL_URLS = {
  lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
  heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
};

/** A local `.task` file in `web/vendor/models/` takes precedence over the CDN. */
async function resolveModelUrl(quality) {
  const local = appUrl(`vendor/models/pose_landmarker_${quality}.task`);
  if (await exists(local)) return local;
  return MODEL_URLS[quality] || MODEL_URLS.full;
}

export const CROP_SIZE = 384;

export class PoseEngine {
  constructor({ quality = 'full', numPoses = 3, cropSize = CROP_SIZE } = {}) {
    this.quality = quality;
    this.numPoses = numPoses;
    this.cropSize = cropSize;
    this.landmarker = null;
    this.tracker = null;
    this.lastTimestamp = -1;
    this.canvas = null;
    this.ctx = null;
  }

  async load(onProgress = () => {}) {
    onProgress('vision');
    const runtime = await resolveRuntime();
    this.runtimeSource = runtime.local ? 'local' : 'cdn';
    const { FilesetResolver, PoseLandmarker } = await import(
      /* @vite-ignore */ runtime.bundle
    );
    const fileset = await FilesetResolver.forVisionTasks(runtime.wasm);
    onProgress('model');
    const modelAssetPath = await resolveModelUrl(this.quality);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath, delegate },
      runningMode: 'VIDEO',
      numPoses: this.numPoses,
      minPoseDetectionConfidence: 0.35,
      minPosePresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
      outputSegmentationMasks: false,
    });
    try {
      this.landmarker = await PoseLandmarker.createFromOptions(fileset, options('GPU'));
      this.delegate = 'GPU';
    } catch (error) {
      // Older phones, locked-down browsers and headless test runners can all fail
      // to hand WebGL to the task's worker. CPU is slower but always available,
      // and a slow app beats a broken one.
      this.landmarker = await PoseLandmarker.createFromOptions(fileset, options('CPU'));
      this.delegate = 'CPU';
    }
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.cropSize;
    this.canvas.height = this.cropSize;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: false });
    onProgress('ready');
    return this;
  }

  /** Build the tracker once the source's aspect ratio is known. */
  attach({ aspect }) {
    this.tracker = new RoiTracker({
      aspect,
      detectFull: (frame) => this._detectFull(frame),
      detectCrop: (frame, window) => this._detectCrop(frame, window),
    });
    return this.tracker;
  }

  /**
   * MediaPipe requires strictly increasing timestamps and throws otherwise, which
   * happens for real whenever a video is scrubbed or a camera repeats a frame.
   */
  _stamp(timeMs) {
    const t = Math.max(timeMs, this.lastTimestamp + 1);
    this.lastTimestamp = t;
    return t;
  }

  _toPoses(result) {
    const poses = [];
    const image = result.landmarks || [];
    const world = result.worldLandmarks || [];
    for (let i = 0; i < image.length; i += 1) {
      poses.push({
        image: image[i].map((p) => [p.x, p.y, p.z, p.visibility ?? 0]),
        world: world[i] ? world[i].map((p) => [p.x, p.y, p.z]) : null,
      });
    }
    return poses.filter((p) => p.world);
  }

  _detectFull(frame) {
    const result = this.landmarker.detectForVideo(frame.source, this._stamp(frame.timeMs));
    return this._toPoses(result);
  }

  /**
   * Draw the crop into an offscreen canvas at a fixed size and run inference on
   * that. Upscaling a distant player is the whole point: the model sees a person
   * filling the frame instead of one 200 px tall.
   */
  _detectCrop(frame, window) {
    const { videoWidth: w, videoHeight: h } = frame;
    const sx = window.x0 * w;
    const sy = window.y0 * h;
    const sw = window.width * w;
    const sh = window.height * h;
    if (sw < 24 || sh < 24) return null;
    this.ctx.drawImage(frame.source, sx, sy, sw, sh, 0, 0, this.cropSize, this.cropSize);
    const result = this.landmarker.detectForVideo(this.canvas, this._stamp(frame.timeMs));
    const poses = this._toPoses(result);
    if (!poses.length) return null;
    // With more than one person in the crop, keep whoever is nearest its centre:
    // the crop was built around our player, so they are the central one.
    return poses.reduce((best, p) => (centreDistance(p) < centreDistance(best) ? p : best));
  }

  /** One frame through the tracker. */
  async detect(frame) {
    if (!this.tracker) throw new Error('call attach() before detect()');
    return this.tracker.step(frame);
  }

  close() {
    this.landmarker?.close?.();
    this.landmarker = null;
  }
}

function centreDistance(pose) {
  let sx = 0;
  let sy = 0;
  for (const p of pose.image) {
    sx += p[0];
    sy += p[1];
  }
  const n = pose.image.length;
  return Math.hypot(sx / n - 0.5, sy / n - 0.5);
}
