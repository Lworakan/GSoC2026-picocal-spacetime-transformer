/**
 * Offline support.
 *
 * A badminton hall is exactly the kind of place with no usable signal, so the
 * app shell and the MediaPipe runtime are cached on first run and served from
 * cache afterwards. The pose model itself is tens of megabytes and is left to
 * the browser's own HTTP cache; to guarantee it offline, drop the `.task` file
 * into `vendor/models/` and it is picked up as a same-origin asset.
 */

const VERSION = 'badminton-coach-v1';
const SHELL = [
  './',
  './index.html',
  './styles.css',
  './manifest.webmanifest',
  './js/app.js',
  './js/pose.js',
  './js/overlay.js',
  './js/i18n.js',
  './js/core/vec3.js',
  './js/core/filters.js',
  './js/core/landmarks.js',
  './js/core/biomech.js',
  './js/core/strokes.js',
  './js/core/court.js',
  './js/core/coach.js',
  './js/core/session.js',
  './js/core/roitracker.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(VERSION)
      // One missing file must not fail the whole install, or the app silently
      // loses offline support after any rename.
      .then((cache) => Promise.allSettled(SHELL.map((url) => cache.add(url))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  const sameOrigin = url.origin === self.location.origin;
  const isRuntime = url.hostname === 'cdn.jsdelivr.net';
  if (!sameOrigin && !isRuntime) return;

  // Cache-first: the shell and the wasm runtime never change within a version,
  // and on a court the cache is the only copy that exists.
  event.respondWith(
    caches.match(request).then((hit) => hit || fetch(request).then((response) => {
      if (response.ok && (sameOrigin || isRuntime)) {
        const copy = response.clone();
        caches.open(VERSION).then((cache) => cache.put(request, copy)).catch(() => {});
      }
      return response;
    }).catch(() => hit)),
  );
});
