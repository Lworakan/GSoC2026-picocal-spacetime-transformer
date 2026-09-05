/**
 * The app: camera or video in, live angles and coaching out.
 *
 * Structure is deliberately flat -- one controller wiring the DOM to the pure
 * analysis core -- because the interesting logic all lives in `js/core/`, which
 * has no DOM in it and is unit-tested in Node. This file is the part that would
 * be rewritten to put the same analysis in a different shell.
 */

import { PoseEngine } from './pose.js';
import { PoseSession, detectRacketArm } from './core/session.js';
import { coachStroke, coachSession, rankCues } from './core/coach.js';
import {
  CourtCalibration, CALIBRATION_ORDER, courtTrack, distanceCovered,
  zoneOccupancy, recoveryTimes, COURT,
} from './core/court.js';
import { LM } from './core/landmarks.js';
import { t, setLanguage, getLanguage, applyTranslations, shotName, localiseCue } from './i18n.js';
import * as overlay from './overlay.js';

const STORAGE_KEY = 'badminton-coach:v1';

const state = {
  settings: {
    lang: 'th',
    racketHand: 'right',
    quality: 'full',
    sensitivity: 6.0,
  },
  court: null,
  calibrating: false,
  calibrationPoints: [],
  engine: null,
  session: null,
  source: null,          // 'camera' | 'file'
  stream: null,
  running: false,
  busy: false,
  startedAt: 0,
  frameCount: 0,
  detectedCount: 0,
  fps: 0,
  lastFpsAt: 0,
  fpsFrames: 0,
  strokes: [],
  cues: [],
  positions: [],
  armDecided: false,
};

const el = {};

/* -- boot ------------------------------------------------------------------ */

function cacheElements() {
  const ids = [
    'screen-start', 'screen-analyse', 'video', 'overlay', 'stage', 'status-chip',
    'status-text', 'fps-chip', 'stage-hint', 'metrics', 'shot-list', 'summary',
    'court-map', 'court-metrics', 'court-status', 'tabs', 'toast',
    'btn-camera', 'file-input', 'btn-stop', 'btn-settings', 'btn-export',
    'btn-reset', 'btn-calibrate', 'btn-swap-sides', 'btn-clear-court',
    'lang-th', 'lang-en', 'racket-hand', 'quality', 'sensitivity', 'sensitivity-value',
  ];
  for (const id of ids) el[camel(id)] = document.getElementById(id);
}

const camel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

function loadSettings() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    Object.assign(state.settings, raw.settings || {});
    if (raw.court) state.court = CourtCalibration.fromJSON(raw.court);
    if (state.court && !state.court.valid) state.court = null;
  } catch {
    // A corrupt or blocked localStorage is not worth failing the app over.
  }
}

function saveSettings() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      settings: state.settings,
      court: state.court ? state.court.toJSON() : null,
    }));
  } catch {
    // Private browsing: settings just will not persist.
  }
}

function toast(message, ms = 3200) {
  el.toast.textContent = message;
  el.toast.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { el.toast.hidden = true; }, ms);
}

/* -- settings UI ----------------------------------------------------------- */

function wireSettings() {
  el.langTh.addEventListener('click', () => switchLanguage('th'));
  el.langEn.addEventListener('click', () => switchLanguage('en'));

  for (const button of el.racketHand.querySelectorAll('button')) {
    button.addEventListener('click', () => {
      state.settings.racketHand = button.dataset.value;
      for (const b of el.racketHand.querySelectorAll('button')) {
        b.classList.toggle('is-active', b === button);
      }
      if (state.session && button.dataset.value !== 'auto') {
        state.session.setRacketArm(button.dataset.value);
        recomputeAnalysis();
      }
      state.armDecided = button.dataset.value !== 'auto';
      saveSettings();
    });
  }

  el.quality.addEventListener('change', () => {
    state.settings.quality = el.quality.value;
    saveSettings();
  });

  el.sensitivity.addEventListener('input', () => {
    state.settings.sensitivity = Number(el.sensitivity.value);
    el.sensitivityValue.textContent = state.settings.sensitivity.toFixed(1);
    saveSettings();
    if (state.session) recomputeAnalysis();
  });

  el.btnSettings.addEventListener('click', () => {
    // The settings live on the start screen; from mid-session, going back there
    // would drop the buffer, so show them as a scroll target instead.
    if (state.running) {
      toast(t('settings') + ': ' + t('racket_hand') + ' · ' + t('sensitivity'));
      showScreen('start', { keepSession: true });
    } else {
      showScreen('start');
    }
  });
}

function switchLanguage(lang) {
  state.settings.lang = setLanguage(lang);
  el.langTh.classList.toggle('is-active', lang === 'th');
  el.langEn.classList.toggle('is-active', lang === 'en');
  saveSettings();
  renderAll();
}

function applySettingsToUi() {
  setLanguage(state.settings.lang);
  el.langTh.classList.toggle('is-active', state.settings.lang === 'th');
  el.langEn.classList.toggle('is-active', state.settings.lang === 'en');
  for (const b of el.racketHand.querySelectorAll('button')) {
    b.classList.toggle('is-active', b.dataset.value === state.settings.racketHand);
  }
  el.quality.value = state.settings.quality;
  el.sensitivity.value = String(state.settings.sensitivity);
  el.sensitivityValue.textContent = state.settings.sensitivity.toFixed(1);
}

/* -- screens and tabs ------------------------------------------------------ */

function showScreen(name, { keepSession = false } = {}) {
  el.screenStart.classList.toggle('is-active', name === 'start');
  el.screenAnalyse.classList.toggle('is-active', name === 'analyse');
  if (name === 'start' && !keepSession) stop();
}

function wireTabs() {
  el.tabs.addEventListener('click', (event) => {
    const tab = event.target.closest('.tab');
    if (!tab) return;
    for (const b of el.tabs.querySelectorAll('.tab')) b.classList.toggle('is-active', b === tab);
    for (const p of document.querySelectorAll('.panel')) {
      p.classList.toggle('is-active', p.dataset.panel === tab.dataset.tab);
    }
    if (tab.dataset.tab === 'court') renderCourt();
    if (tab.dataset.tab === 'summary') renderSummary();
  });
}

/* -- sources --------------------------------------------------------------- */

async function ensureEngine() {
  if (state.engine && state.engine.quality === state.settings.quality) return state.engine;
  state.engine?.close();
  toast(t('loading_model'), 8000);
  const engine = new PoseEngine({ quality: state.settings.quality });
  try {
    await engine.load();
  } catch (error) {
    toast(t('error_model'), 6000);
    throw error;
  }
  state.engine = engine;
  el.toast.hidden = true;
  return engine;
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
      audio: false,
    });
    state.stream = stream;
    el.video.srcObject = stream;
    el.video.removeAttribute('src');
    await el.video.play();
    await begin('camera');
  } catch (error) {
    toast(t('error_camera'), 5000);
  }
}

async function startFile(file) {
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  el.video.srcObject = null;
  el.video.src = URL.createObjectURL(file);
  el.video.loop = false;
  // No native controls: the app steps the file frame by frame, and a scrub bar
  // would fight the stepper. It also costs a chunk of a phone screen.
  el.video.controls = false;
  // Deliberately not played here. Loading the pose model takes seconds on a
  // first run, and a clip that plays during that window is half over before
  // anything is measured.
  await begin('file');
}

async function begin(source) {
  await ensureEngine();
  await waitForMetadata(el.video);

  state.source = source;
  state.session = new PoseSession({
    racketArm: state.settings.racketHand === 'auto' ? 'right' : state.settings.racketHand,
  });
  state.armDecided = state.settings.racketHand !== 'auto';
  state.strokes = [];
  state.cues = [];
  state.positions = [];
  state.frameCount = 0;
  state.detectedCount = 0;
  state.startedAt = performance.now();
  state.running = true;

  const aspect = el.video.videoWidth / el.video.videoHeight;
  state.engine.attach({ aspect });
  // Match the stage to the source so a landscape clip is not letterboxed into a
  // portrait box with black bars taking half the screen.
  el.stage.style.aspectRatio = `${el.video.videoWidth} / ${el.video.videoHeight}`;
  showScreen('analyse');
  // Rendering must never be able to stop the capture loop from starting: an
  // exception here once left the app sitting on a live video detecting nothing.
  try {
    renderAll();
  } catch (error) {
    console.error('initial render failed', error);
  }

  if (source === 'file') {
    stepThroughFile();
  } else {
    loop();
  }
}

function waitForMetadata(video) {
  if (video.readyState >= 1) return Promise.resolve();
  return new Promise((resolve) => video.addEventListener('loadedmetadata', resolve, { once: true }));
}

function stop() {
  state.running = false;
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  el.video.pause();
  el.video.srcObject = null;
  if (el.video.src) {
    URL.revokeObjectURL(el.video.src);
    el.video.removeAttribute('src');
    el.video.load();
  }
}

/* -- the frame loop -------------------------------------------------------- */

function timeOf(video) {
  // A file has a real timeline; a camera stream does not, so wall-clock stands
  // in. Both must be monotonic for the filters and the derivative to mean
  // anything.
  return state.source === 'file'
    ? video.currentTime
    : (performance.now() - state.startedAt) / 1000;
}

/**
 * Live capture: analyse whatever frame is on screen, as fast as the device can.
 *
 * Frames are necessarily skipped when inference is slower than the camera, which
 * is fine -- the timestamps carried with each sample are the real ones, so speeds
 * and derivatives stay correct at whatever rate is achieved.
 */
function loop() {
  if (!state.running) return;
  const video = el.video;
  const schedule = video.requestVideoFrameCallback
    ? (fn) => video.requestVideoFrameCallback(fn)
    : (fn) => requestAnimationFrame(fn);

  schedule(async () => {
    if (!state.running) return;
    if (!state.busy && video.readyState >= 2 && !video.paused && !video.ended) {
      state.busy = true;
      try {
        await processFrame(video);
      } catch (error) {
        console.error('frame failed', error);
      } finally {
        state.busy = false;
      }
    }
    loop();
  });
}

/**
 * Recorded clips are stepped frame by frame rather than played.
 *
 * Playing a file and sampling it means dropping most of it whenever inference is
 * slower than real time -- which it is on a phone -- and stroke detection needs
 * the frames around a peak to find the peak at all. Seeking costs wall-clock
 * time but analyses every frame, and a recorded clip has no reason to be
 * analysed in real time.
 */
async function stepThroughFile() {
  const video = el.video;
  video.pause();
  const step = 1 / (state.settings.frameRate || 30);
  const duration = Number.isFinite(video.duration) ? video.duration : 0;

  await seekTo(video, 0);
  while (state.running && video.currentTime < duration - step / 2) {
    try {
      await processFrame(video);
    } catch (error) {
      console.error('frame failed', error);
    }
    const next = Math.min(duration, video.currentTime + step);
    if (!(await seekTo(video, next))) break;
    setProgress(video.currentTime / duration);
  }
  setProgress(1);
  if (state.running) finish();
}

function seekTo(video, time) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (ok) => {
      if (settled) return;
      settled = true;
      video.removeEventListener('seeked', onSeeked);
      clearTimeout(timer);
      resolve(ok);
    };
    const onSeeked = () => done(true);
    video.addEventListener('seeked', onSeeked);
    // A seek that never completes (a truncated file, a codec hiccup) must not
    // hang the analysis for ever.
    const timer = setTimeout(() => done(false), 3000);
    try {
      video.currentTime = time;
    } catch {
      done(false);
    }
  });
}

function setProgress(fraction) {
  const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  el.statusText.textContent = pct >= 100 ? t('tracking') : `${pct}%`;
}

async function processFrame(video) {
  const timeMs = timeOf(video) * 1000;
  const pose = await state.engine.detect({
    source: video,
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
    timeMs,
  });

  state.frameCount += 1;
  tickFps();

  let analysed = null;
  if (pose) {
    state.detectedCount += 1;
    analysed = state.session.push({
      t: timeMs / 1000,
      world: pose.world,
      image: pose.image,
      frame: state.frameCount,
    });
    maybeDetectArm();
    const fresh = state.session.recentStrokes({ peakSpeed: state.settings.sensitivity });
    if (fresh.length) {
      state.strokes.push(...fresh);
      for (const stroke of fresh) state.cues.push(...coachStroke(stroke));
      renderShots();
    }
  }

  draw(pose, analysed);
  renderMetrics(analysed);
}

function tickFps() {
  const now = performance.now();
  state.fpsFrames += 1;
  if (now - state.lastFpsAt >= 500) {
    state.fps = (state.fpsFrames * 1000) / (now - state.lastFpsAt);
    state.lastFpsAt = now;
    state.fpsFrames = 0;
    el.fpsChip.innerHTML = `${state.fps.toFixed(0)} <span>${t('fps')}</span>`;
  }
}

/**
 * Decide the racket arm once there is enough movement to tell.
 *
 * Guessing from a handful of frames of someone standing still is worse than
 * using the default, so this waits for a clear margin between the two wrists.
 */
function maybeDetectArm() {
  if (state.armDecided || state.settings.racketHand !== 'auto') return;
  if (state.session.frames.length < 90) return;
  const guess = detectRacketArm(state.session.frames);
  if (!guess || guess.margin < 0.15) return;
  state.armDecided = true;
  state.session.setRacketArm(guess.arm);
  recomputeAnalysis();
  toast(`${t('racket_hand')}: ${t(guess.arm)}`);
}

function finish() {
  state.running = false;
  recomputeAnalysis();
  selectTab('summary');
}

function selectTab(name) {
  const tab = el.tabs.querySelector(`[data-tab="${name}"]`);
  tab?.click();
}

/** Re-run the analysis over the whole buffer, after a setting changed. */
function recomputeAnalysis() {
  if (!state.session) return;
  state.strokes = state.session.strokes({ peakSpeed: state.settings.sensitivity });
  state.cues = state.strokes.flatMap((s) => coachStroke(s));
  state.session._strokeCount = state.strokes.length;
  renderShots();
  renderSummary();
}

/* -- drawing --------------------------------------------------------------- */

function stageRect() {
  const { width, height, dpr } = overlay.fitCanvas(el.overlay);
  const rect = overlay.contentRect(width, height, el.video.videoWidth, el.video.videoHeight);
  return { rect, width, height, dpr };
}

function draw(pose, analysed) {
  const ctx = el.overlay.getContext('2d');
  const { rect, dpr } = stageRect();
  overlay.clear(ctx, el.overlay, dpr);

  if (state.court?.valid) overlay.drawCourt(ctx, rect, state.court);
  if (state.calibrating) overlay.drawCalibrationPoints(ctx, rect, state.calibrationPoints);

  const tracker = state.engine?.tracker;
  if (tracker?.window) overlay.drawWindow(ctx, rect, tracker.window);

  const image = analysed?.image || pose?.image;
  if (image) {
    const arm = state.session.racketArm;
    const scale = Math.max(0.7, Math.min(1.6, rect.height / 640));
    overlay.drawSkeleton(ctx, rect, image, { racketArm: arm, scale });

    if (analysed) {
      const m = analysed.metrics;
      const S = (name) => LM[`${arm.toUpperCase()}_${name}`];
      overlay.drawAngleArc(
        ctx, rect, image,
        [S('SHOULDER'), S('ELBOW'), S('WRIST')],
        m.elbow,
        { status: elbowStatus(m.elbow), scale, radius: 24 },
      );
      const kneeSide = m.kneeLeft < m.kneeRight ? 'LEFT' : 'RIGHT';
      overlay.drawAngleArc(
        ctx, rect, image,
        [LM[`${kneeSide}_HIP`], LM[`${kneeSide}_KNEE`], LM[`${kneeSide}_ANKLE`]],
        Math.min(m.kneeLeft, m.kneeRight),
        { status: 'good', scale, radius: 20 },
      );
    }
  }

  setStatus(Boolean(pose));
}

// A live hint only, not a coaching verdict: the coaching rules judge the elbow
// at contact, where the number actually means something.
const elbowStatus = (value) => (value >= 150 ? 'good' : value >= 120 ? 'warn' : 'bad');

function setStatus(tracking) {
  el.statusChip.classList.toggle('is-tracking', tracking);
  // While stepping through a file the chip shows progress instead, which is the
  // more useful thing to look at.
  if (state.source !== 'file') {
    el.statusText.textContent = tracking ? t('tracking') : t('searching');
  }
  const coverage = state.frameCount ? state.detectedCount / state.frameCount : 0;
  el.statusChip.title = `${(coverage * 100).toFixed(0)}% ${t('coverage')}`;
  el.stageHint.hidden = tracking && !state.calibrating;
  if (state.calibrating) {
    const next = CALIBRATION_ORDER[state.calibrationPoints.length];
    el.stageHint.textContent = next
      ? `${t('hint_calibrate')} — ${t(`corner_${next}`)}`
      : t('hint_calibrate');
    el.stageHint.hidden = false;
  } else {
    el.stageHint.textContent = t('hint_tap_player');
  }
}

/* -- taps on the video ----------------------------------------------------- */

function wireStage() {
  el.overlay.addEventListener('click', (event) => {
    const box = el.overlay.getBoundingClientRect();
    const { rect } = stageRect();
    const [x, y] = overlay.unproject(
      rect, event.clientX - box.left, event.clientY - box.top,
    );
    if (x < 0 || x > 1 || y < 0 || y > 1) return;

    if (state.calibrating) {
      state.calibrationPoints.push([x, y]);
      if (state.calibrationPoints.length === 4) finishCalibration();
      setStatus(true);
      return;
    }
    state.engine?.tracker?.lockOn(x, y);
  });
}

function finishCalibration() {
  const calibration = new CourtCalibration(state.calibrationPoints);
  state.calibrating = false;
  if (!calibration.valid) {
    state.calibrationPoints = [];
    toast(t('court_invalid'), 5000);
    return;
  }
  state.court = calibration;
  state.calibrationPoints = [];
  saveSettings();
  toast(t('court_ready'));
  renderCourt();
}

function wireCourtControls() {
  el.btnCalibrate.addEventListener('click', () => {
    state.calibrating = true;
    state.calibrationPoints = [];
    selectTabIfHidden();
    setStatus(true);
    toast(t('hint_calibrate'), 5000);
  });
  el.btnSwapSides.addEventListener('click', () => {
    if (!state.court) return;
    state.court = state.court.swapSides();
    saveSettings();
    renderCourt();
  });
  el.btnClearCourt.addEventListener('click', () => {
    state.court = null;
    saveSettings();
    renderCourt();
  });
}

/** Calibration needs the video visible, so leave the court tab while tapping. */
function selectTabIfHidden() {
  selectTab('live');
}

/* -- rendering ------------------------------------------------------------- */

function metricCard(label, value, unit, status = null) {
  const cls = status ? ` is-${status}` : '';
  const text = Number.isFinite(value) ? value : '–';
  return `<div class="metric${cls}">
    <div class="metric-label">${label}</div>
    <div class="metric-value">${text}<span class="unit">${unit}</span></div>
  </div>`;
}

function renderMetrics(analysed) {
  if (!analysed) return;
  const m = analysed.metrics;
  const speed = state.session.speedNow();
  const n = (v, d = 0) => (Number.isFinite(v) ? Number(v.toFixed(d)) : NaN);
  el.metrics.innerHTML = [
    metricCard(t('m_elbow'), n(m.elbow), '°', elbowStatus(m.elbow)),
    metricCard(t('m_elevation'), n(m.shoulderElevation), '°'),
    metricCard(t('m_azimuth'), n(m.shoulderAzimuth), '°'),
    metricCard(t('m_lean'), n(m.trunkLean.total), '°'),
    metricCard(t('m_separation'), n(m.separation), '°'),
    metricCard(t('m_knee'), n(Math.min(m.kneeLeft, m.kneeRight)), '°'),
    metricCard(t('m_stance'), n(m.stanceWidth, 2), '×'),
    metricCard(t('m_speed'), n(speed * m.trunkLength, 1), ' m/s'),
    metricCard(t('m_hand_height'), n(m.hand.height, 2), ''),
  ].join('');
}

function cueItem(raw) {
  const cue = localiseCue(raw);
  const value = Number.isFinite(cue.value)
    ? (Math.abs(cue.value) >= 10 ? cue.value.toFixed(0) : cue.value.toFixed(2))
    : '–';
  return `<li class="cue is-${cue.status}">
    <div class="cue-head">
      <span class="cue-label">${cue.label}</span>
      <span class="cue-value">${value}${cue.unit}</span>
    </div>
    <div class="cue-target">${t('target')}: ${cue.target}</div>
    <div class="cue-why">${cue.why}</div>
  </li>`;
}

function renderShots() {
  if (!state.strokes.length) {
    el.shotList.innerHTML = `<p class="empty">${t('no_shots')}</p>`;
    return;
  }
  const lang = getLanguage();
  const cards = state.strokes.slice().reverse().map((stroke) => {
    const cues = state.cues.filter((c) => c.strokeIndex === stroke.index);
    const stats = [
      `${t('m_elbow')} ${stroke.contact.elbow.toFixed(0)}°`,
      `${t('m_speed')} ${stroke.peakSpeedMs.toFixed(1)} m/s`,
      `${t('m_elevation')} ${stroke.contact.shoulderElevation.toFixed(0)}°`,
      `${t('m_separation')} ${stroke.backswing.maxSeparation.toFixed(0)}°`,
    ];
    return `<article class="shot">
      <div class="shot-head">
        <span class="shot-name">${shotName(stroke.shot, lang)}</span>
        <span class="shot-time">${stroke.t.toFixed(2)} s</span>
      </div>
      <div class="shot-stats">${stats.map((s) => `<span class="stat">${s}</span>`).join('')}</div>
      ${cues.length ? `<ul class="cues">${cues.map(cueItem).join('')}</ul>` : ''}
    </article>`;
  });
  el.shotList.innerHTML = cards.join('');
}

function computeCourt() {
  if (!state.court?.valid || !state.session) return null;
  const positions = courtTrack(state.session.frames, state.court, LM);
  const recovery = recoveryTimes(state.strokes, positions);
  return {
    positions,
    recovery,
    distance: distanceCovered(positions),
    occupancy: zoneOccupancy(positions),
  };
}

function renderCourt() {
  const ready = Boolean(state.court?.valid);
  el.courtStatus.textContent = ready ? t('court_ready') : t('court_none');
  el.btnCalibrate.textContent = ready ? t('calibrate_redo') : t('calibrate');
  el.btnSwapSides.hidden = !ready;
  el.btnClearCourt.hidden = !ready;

  const court = computeCourt();
  state.positions = court?.positions || [];
  overlay.drawCourtMap(el.courtMap, state.positions);

  if (!court) {
    el.courtMetrics.innerHTML = '';
    return;
  }
  const times = court.recovery.map((r) => r.recoverySeconds).filter(Number.isFinite);
  const median = times.length
    ? times.slice().sort((a, b) => a - b)[Math.floor(times.length / 2)]
    : NaN;
  el.courtMetrics.innerHTML = [
    metricCard(t('distance_covered'), Number(court.distance.toFixed(1)), ' m'),
    metricCard(t('recovery'), Number.isFinite(median) ? Number(median.toFixed(2)) : NaN, ' s'),
  ].join('');
}

function renderSummary() {
  if (!state.session) {
    el.summary.innerHTML = '';
    return;
  }
  const analysis = { strokes: state.strokes };
  const summary = state.strokes.length
    ? {
      count: state.strokes.length,
      forehand: state.strokes.filter((s) => s.side === 'forehand').length,
      backhand: state.strokes.filter((s) => s.side === 'backhand').length,
      roundhead: state.strokes.filter((s) => s.side === 'roundhead').length,
      maxSpeed: Math.max(...state.strokes.map((s) => s.peakSpeedMs)),
    }
    : null;

  const court = computeCourt();
  const sessionCues = coachSession(state.session.frames, state.strokes, {
    recovery: court?.recovery || null,
  });
  // Stroke cues only: session cues have their own section below, and ranking
  // them together listed each one twice.
  const ranked = rankCues(state.cues);

  const parts = [];
  parts.push(`<section>
    <h2>${t('summary_shots')}</h2>
    ${summary ? `<div class="metrics">
      ${metricCard(t('shots_found'), summary.count, '')}
      ${metricCard(t('forehand'), summary.forehand, '')}
      ${metricCard(t('backhand'), summary.backhand, '')}
      ${metricCard(t('summary_speed'), Number(summary.maxSpeed.toFixed(1)), ' m/s')}
    </div>` : `<p class="empty">${t('no_shots')}</p>`}
  </section>`);

  if (ranked.length) {
    parts.push(`<section>
      <h2>${t('summary_faults')}</h2>
      <ul class="cues">${ranked.map(localiseCue).map((g) => `<li class="cue is-${g.bad ? 'bad' : 'warn'}">
        <div class="cue-head">
          <span class="cue-label">${g.label}</span>
          <span class="cue-value">${g.count} ${t('times')}</span>
        </div>
        <div class="cue-target">${t('target')}: ${g.target}</div>
        <div class="cue-why">${g.why}</div>
      </li>`).join('')}</ul>
    </section>`);
  }

  if (sessionCues.length) {
    parts.push(`<section>
      <h2>${t('tab_summary')}</h2>
      <ul class="cues">${sessionCues.map(cueItem).join('')}</ul>
    </section>`);
  }

  if (court) {
    const zones = Object.entries(court.occupancy).sort((a, b) => b[1] - a[1]);
    const total = zones.reduce((sum, [, v]) => sum + v, 0) || 1;
    parts.push(`<section>
      <h2>${t('summary_court')}</h2>
      <div class="metrics">
        ${metricCard(t('distance_covered'), Number(court.distance.toFixed(1)), ' m')}
      </div>
      <div class="bars">${zones.map(([name, seconds]) => `<div class="bar-row">
        <span>${name}</span>
        <span class="bar"><span style="width:${((seconds / total) * 100).toFixed(1)}%"></span></span>
        <span>${seconds.toFixed(1)} s</span>
      </div>`).join('')}</div>
    </section>`);
  }

  el.summary.innerHTML = parts.join('');
  void analysis;
}

function renderAll() {
  applyTranslations();
  renderShots();
  renderCourt();
  renderSummary();
}

/* -- export ---------------------------------------------------------------- */

function exportJson() {
  if (!state.session) return;
  const court = computeCourt();
  const round = (v, d = 5) => Number(v.toFixed(d));
  const payload = {
    format: 1,
    tool: 'badminton-coach web',
    exportedAt: new Date().toISOString(),
    settings: state.settings,
    racketArm: state.session.racketArm,
    video: {
      width: el.video.videoWidth,
      height: el.video.videoHeight,
      source: state.source,
    },
    court: state.court ? state.court.toJSON() : null,
    strokes: state.strokes,
    cues: state.cues,
    sessionCues: coachSession(state.session.frames, state.strokes, {
      recovery: court?.recovery || null,
    }),
    recovery: court?.recovery || null,
    // The landmark track is included so the same session can be re-analysed by
    // the Python pipeline without re-recording anything.
    frames: state.session.frames.map((f) => ({
      t: round(f.t),
      frame: f.frame,
      image: f.image ? f.image.map((p) => p.map((v) => round(v, 4))) : null,
      // Stored back in MediaPipe's own y-down convention so the file matches
      // what `analyze/` writes and reads.
      world: f.world.map(([x, y, z]) => [round(x, 4), round(-y, 4), round(-z, 4)]),
    })),
  };
  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `badminton-session-${Date.now()}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* -- init ------------------------------------------------------------------ */

function init() {
  cacheElements();
  loadSettings();
  applySettingsToUi();
  applyTranslations();
  wireSettings();
  wireTabs();
  wireStage();
  wireCourtControls();

  el.btnCamera.addEventListener('click', startCamera);
  el.fileInput.addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    if (file) startFile(file);
  });
  el.btnStop.addEventListener('click', () => {
    if (state.running) finish();
    else showScreen('start');
  });
  el.btnExport.addEventListener('click', exportJson);
  el.btnReset.addEventListener('click', () => {
    stop();
    state.session = null;
    state.strokes = [];
    state.cues = [];
    showScreen('start');
  });

  window.addEventListener('resize', () => {
    if (state.session) overlay.drawCourtMap(el.courtMap, state.positions);
  });

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {
      // Offline support is a bonus; the app works without it.
    });
  }
  void COURT;
}

init();
