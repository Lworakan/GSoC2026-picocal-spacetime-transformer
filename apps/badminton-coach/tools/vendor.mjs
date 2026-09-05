/**
 * Download the MediaPipe runtime and a pose model into `web/vendor/`.
 *
 * The app works straight from the CDN, but a badminton hall is exactly the sort
 * of place with no usable signal, and the first load pulls tens of megabytes.
 * After running this, `web/` is entirely self-contained and can be copied to a
 * phone, a USB stick, or any static host.
 *
 *   node tools/vendor.mjs [lite|full|heavy]
 *
 * `web/vendor/` is git-ignored: these are large third-party binaries, fetched on
 * demand rather than committed.
 */

import { mkdir, writeFile, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const VERSION = '0.10.18';
const WEB = join(dirname(fileURLToPath(import.meta.url)), '..', 'web');
const OUT = join(WEB, 'vendor');

const RUNTIME_FILES = [
  'vision_bundle.mjs',
  'wasm/vision_wasm_internal.js',
  'wasm/vision_wasm_internal.wasm',
  'wasm/vision_wasm_nosimd_internal.js',
  'wasm/vision_wasm_nosimd_internal.wasm',
];

const MODEL_URL = (quality) =>
  `https://storage.googleapis.com/mediapipe-models/pose_landmarker/`
  + `pose_landmarker_${quality}/float16/1/pose_landmarker_${quality}.task`;

async function download(url, target) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText} for ${url}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, bytes);
  process.stdout.write(`  ${target.replace(WEB + '/', '')}  ${(bytes.length / 1e6).toFixed(1)} MB\n`);
}

const quality = process.argv[2] || 'full';
if (!['lite', 'full', 'heavy'].includes(quality)) {
  process.stderr.write(`unknown model quality: ${quality}\n`);
  process.exit(2);
}

process.stdout.write(`vendoring MediaPipe tasks-vision ${VERSION} and the ${quality} pose model\n`);
await rm(join(OUT, 'tasks-vision'), { recursive: true, force: true });
for (const file of RUNTIME_FILES) {
  await download(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/${file}`,
    join(OUT, 'tasks-vision', file),
  );
}
await download(MODEL_URL(quality), join(OUT, 'models', `pose_landmarker_${quality}.task`));
process.stdout.write('done — the app now runs with no network access\n');
