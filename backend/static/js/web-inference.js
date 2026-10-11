/**
 * Lazy, cached, on-device inference for the three modes. Camera frames never leave the browser:
 * objects and faces run in Web Workers (ONNX Runtime Web), hands in MediaPipe Tasks Vision.
 * Only the active mode's model is kept in memory.
 */
import { $ } from './util.js';
import { createObjectEngine, OBJECT_SIZES, OBJECT_MODEL_BYTES } from './web-objects.js';
import { createFaceEngine } from './web-faces-client.js';
import { createSignEngine, HAND_MODEL_BYTES } from './web-sign.js';
import { FACE_MODEL_INFO } from './web-faces.js';
import { megabytes } from './web-fetch.js';

const names = { objects: 'YOLO11n object detector', faces: 'expression models', sign: 'hand model' };
const engines = new Map();
const enginesReady = new Map();   // mode -> engine, once loaded
let displayedMode = null;

/** Input size request: 320 for phones, 'auto' otherwise (the worker picks 640 only with WebGPU); `?objects=` overrides. */
export function chooseObjectSize() {
  const forced = Number(new URLSearchParams(location.search).get('objects'));
  if (OBJECT_SIZES.includes(forced)) return forced;
  const phone = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)
    || (matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820);
  return phone ? 320 : 'auto';
}

/** Parallel WASM sessions: leave cores for the page and the camera; phones stay gentler on battery/heat. */
function workerBudget() {
  const cores = navigator.hardwareConcurrency || 4;
  const phone = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
  return Math.max(1, Math.min(phone ? 2 : 3, Math.floor(cores / 2)));
}

function sizeLabel(mode) {
  if (mode === 'objects') return `${megabytes(OBJECT_MODEL_BYTES)} MB`;
  if (mode === 'faces') return `${megabytes(FACE_MODEL_INFO.totalBytes)} MB`;
  return `${megabytes(HAND_MODEL_BYTES)} MB`;
}

function progress(mode, { detail, loaded, total } = {}) {
  if (displayedMode !== mode) return;
  $('modelLoadTitle').textContent = `Preparing the ${names[mode]}`;
  $('modelLoadDetail').textContent = detail || `${sizeLabel(mode)} · saved on this device after the first visit`;
  const bar = $('modelLoadBar');
  if (total) { bar.max = total; bar.value = loaded; bar.hidden = false; } else bar.hidden = true;
  $('modelLoad').classList.remove('hidden');
}

export function hideModelProgress() {
  displayedMode = null;
  $('modelLoad').classList.add('hidden');
}

function closeOthers(keep) {
  for (const [mode, entry] of engines) {
    if (mode === keep) continue;
    engines.delete(mode); enginesReady.delete(mode);
    entry.loading.then((engine) => engine.close?.(), () => {});
  }
}

function getEngine(mode) {
  closeOthers(mode);
  if (engines.has(mode)) return engines.get(mode).loading;
  displayedMode = mode;
  const entry = { loading: null };
  engines.set(mode, entry);
  const onProgress = (p) => progress(mode, p);
  entry.loading = (async () => {
    progress(mode);
    if (mode === 'objects') {
      const engine = createObjectEngine({ size: chooseObjectSize(), onProgress, maxWorkers: workerBudget() });
      await engine.ready;
      return engine;
    }
    if (mode === 'faces') { const engine = createFaceEngine({ onProgress, maxWorkers: workerBudget() }); await engine.ready; return engine; }
    if (mode === 'sign') return createSignEngine({ onProgress });
    throw new Error(`Unknown browser model: ${mode}`);
  })();
  entry.loading.then((engine) => { if (engines.get(mode) === entry) enginesReady.set(mode, engine); }, () => {});
  entry.loading.catch(() => { if (engines.get(mode) === entry) engines.delete(mode); if (displayedMode === mode) hideModelProgress(); });
  return entry.loading;
}

/** Run one video frame and return the payload shape the UI renders (same as the Python API's JSON). */
export async function analyzeFrame(mode, video, signMode = 'spell') {
  const engine = await getEngine(mode);
  if (displayedMode === mode) hideModelProgress();
  if (mode !== 'sign') return engine.detect(video);
  const start = performance.now();
  const detections = engine.detect(video, signMode);
  return { detections, image_size: { width: video.videoWidth, height: video.videoHeight },
    model_used: signMode === 'spell' ? 'mediapipe-hands+asl-letters' : 'mediapipe-gestures', device: engine.device,
    inference_ms: performance.now() - start };
}

/** How many frames the loop may keep in flight for a mode (more than 1 only for pooled CPU inference). */
export function frameConcurrency(mode) {
  const engine = enginesReady.get(mode);
  return engine?.concurrency ?? 1;
}

/** Test/diagnostic hook: the engine for a mode once loaded. */
export const loadedEngine = (mode) => engines.get(mode)?.loading;
