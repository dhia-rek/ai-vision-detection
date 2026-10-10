/** Live-camera capture, lifecycle and the stale-safe /api/analyze frame loop. */
import { $, sleep } from './util.js';
import { state, emit, on } from './store.js';
import { MAX_WIDTH, JPEG_QUALITY, REQUEST_TIMEOUT_MS } from './config.js';
import { activeMode } from './modes.js';
import { updateTracks, clearTracks, requestDraw } from './overlay.js';
import { toast } from './toast.js';
import { resetStats, recordFrame, recordLatency } from './stats.js';
import { resetAnnounce } from './insights.js';
import { resetSpell } from './sign.js';

const video = $('video'), viewport = $('viewport'), grab = $('grab');
const ctx = grab.getContext('2d');
let stream = null, camEpoch = 0, generation = 0, feedEpoch = 0, controller = null;
let inflight = false, loopRunning = false, firstResult = false, lastSize = { w: 4, h: 3 };

function setCamera(status) {
  state.camera = status;
  document.body.dataset.camera = status;
  emit('camera', { status, facing: state.facing });
}
function setMediaShown(shown, w = 0, h = 0) {
  document.body.dataset.media = shown ? 'live' : 'none';
  if (shown && w && h) {
    lastSize = { w, h };
    viewport.style.setProperty('--ar', `${w} / ${h}`);
    document.body.dataset.orient = w / h < 0.9 ? 'portrait' : 'landscape';
    document.body.dataset.mirror = String(state.mirror);
  }
  emit('media', { shown, w, h });
}
export function mediaShown() { return !!stream && video.videoWidth > 0; }

/** Abort prior work and fence any response that cannot be aborted in time. */
function invalidate() {
  feedEpoch++;
  if (controller) controller.abort();
  controller = null;
  inflight = false;
}
export function resetFeed() {
  invalidate();
  firstResult = false;
  clearTracks();
  resetAnnounce();
  resetSpell();
  const mode = activeMode();
  if (mode?.onReset) mode.onReset({ state });
  resetStats();
}

export function applyResult(data) {
  const size = data.image_size?.width ? { w: data.image_size.width, h: data.image_size.height } : lastSize;
  lastSize = size;
  let dets = Array.isArray(data.detections) ? data.detections.slice() : [];
  if (state.mode === 'faces') {
    dets.sort((a, b) => a.bbox.x1 - b.bbox.x1);
    dets.forEach((d, i) => { d._index = dets.length > 1 ? i + 1 : null; });
  }
  updateTracks(dets, size);
  activeMode()?.onResult?.(dets, data, { state });
  emit('result', { dets, data, size });
  return dets;
}
function sourceReady() { return mediaShown() && !video.paused && !document.hidden && state.camera === 'live'; }
async function sendFrame(gen) {
  if (inflight || !sourceReady()) return;
  inflight = true;
  const epoch = feedEpoch, ctl = new AbortController();
  controller = ctl;
  const timer = setTimeout(() => ctl.abort(), REQUEST_TIMEOUT_MS);
  try {
    const t0 = performance.now(), vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    const w = Math.min(vw, MAX_WIDTH), h = Math.round(vh * w / vw);
    grab.width = w; grab.height = h;
    ctx.drawImage(video, 0, 0, w, h);
    const blob = await new Promise((resolve) => grab.toBlob(resolve, 'image/jpeg', JPEG_QUALITY));
    if (!blob) throw new Error('Could not encode camera frame.');
    const fd = new FormData(); fd.append('file', blob, 'frame.jpg');
    const mode = activeMode();
    const query = mode.query ? mode.query(state) : `mode=${encodeURIComponent(mode.id)}`;
    const url = `/api/analyze?${query}${query ? '&' : ''}annotate=false`;
    const res = await fetch(url, { method: 'POST', body: fd, signal: ctl.signal });
    if (!res.ok) throw new Error(`Backend error: ${res.status} ${(await res.text().catch(() => '')).slice(0, 180)}`);
    const data = await res.json(), t1 = performance.now();
    if (epoch !== feedEpoch || gen !== generation) return;
    firstResult = true;
    const dets = applyResult(data);
    recordFrame(t1, t1 - t0, data);
    $('errorBox').classList.add('hidden');
  } catch (err) {
    if (epoch !== feedEpoch || gen !== generation || state.camera !== 'live') return;   // stale or stopped: say nothing
    $('errorText').textContent = err.name === 'AbortError' ? 'The server took too long to answer.' : (err.message || String(err));
    $('errorBox').classList.remove('hidden');
    recordLatency(0, {});
    await sleep(700);
  } finally {
    clearTimeout(timer);
    if (epoch === feedEpoch) { inflight = false; controller = null; }
  }
}
async function frameLoop(gen) {
  if (loopRunning) return;
  loopRunning = true;
  try {
    while (gen === generation && state.camera === 'live') {
      if (!sourceReady() || inflight) { await sleep(100); continue; }
      await sendFrame(gen);
    }
  } finally {
    loopRunning = false;
    if (gen !== generation && state.camera === 'live') startLoop();
  }
}
function startLoop() { generation++; void frameLoop(generation); }
function stopLoop() { generation++; invalidate(); }
function classify(err) {
  if (err?.name === 'InsecureError') return 'insecure';
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) return 'insecure';
  if (['NotAllowedError', 'SecurityError'].includes(err?.name)) return 'blocked';
  if (['NotFoundError', 'DevicesNotFoundError', 'OverconstrainedError'].includes(err?.name)) return 'nocamera';
  if (['NotReadableError', 'TrackStartError', 'AbortError'].includes(err?.name)) return 'busy';
  return 'error';
}
async function refreshFlip() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    $('flipBtn').classList.toggle('hidden', devices.filter((d) => d.kind === 'videoinput').length < 2);
  } catch { $('flipBtn').classList.add('hidden'); }
}
function releaseStream() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
}
/** Start the camera. `keepFrame` (used when flipping) leaves the last frame on screen while the new stream opens. */
export async function startCamera({ keepFrame = false } = {}) {
  if (!keepFrame && ['starting', 'live', 'paused'].includes(state.camera)) return;
  const mine = ++camEpoch;
  stopLoop();
  $('errorBox').classList.add('hidden');
  setCamera('starting');
  if (!keepFrame) setMediaShown(false);
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('Camera access requires HTTPS or localhost.'), { name: 'InsecureError' });
    const next = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: state.facing }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    if (mine !== camEpoch) { next.getTracks().forEach((t) => t.stop()); return; }
    releaseStream(); stream = next; video.srcObject = stream;
    const actual = stream.getVideoTracks()[0]?.getSettings?.().facingMode;
    if (actual === 'user' || actual === 'environment') state.facing = actual;
    state.mirror = state.facing === 'user';
    document.body.dataset.mirror = String(state.mirror);
    await video.play();
    if (mine !== camEpoch) { releaseStream(); return; }
    const meta = stream.getVideoTracks()[0];
    meta?.addEventListener('ended', () => {
      if (mine !== camEpoch) return;
      stopLoop(); releaseStream(); resetFeed(); setCamera('nocamera'); setMediaShown(false);
      toast('The camera was disconnected.', { kind: 'error' });
    }, { once: true });
    const w = video.videoWidth, h = video.videoHeight;
    setMediaShown(true, w, h); setCamera('live');
    resetStats(); resetFeed(); startLoop(); void refreshFlip();
  } catch (err) {
    if (mine !== camEpoch) return;
    releaseStream();
    const status = classify(err);
    document.body.dataset.cameraMessage = status === 'error' ? (err.message || '') : '';   // read by main.js when it renders the help card
    setMediaShown(false);
    setCamera(status);
  }
}
export function stopCamera() {
  camEpoch++;
  stopLoop(); releaseStream(); resetFeed();
  setCamera('off'); setMediaShown(false);
  $('flipBtn').classList.add('hidden');
  toast('Camera stopped.', { kind: 'info' });
}
export async function togglePause() {
  if (state.camera === 'live') {
    video.pause(); stopLoop(); setCamera('paused'); setMediaShown(true, video.videoWidth, video.videoHeight);
  } else if (state.camera === 'paused') {
    try { await video.play(); setCamera('live'); setMediaShown(true, video.videoWidth, video.videoHeight); startLoop(); }
    catch (err) { $('errorText').textContent = err.message; $('errorBox').classList.remove('hidden'); }
  }
}
export async function flipCamera() {
  if (!stream || !['live', 'paused'].includes(state.camera)) return;
  const previous = { facing: state.facing, mirror: state.mirror, paused: state.camera === 'paused' };
  state.facing = state.facing === 'user' ? 'environment' : 'user';
  state.mirror = state.facing === 'user';
  stopLoop(); releaseStream();
  await startCamera({ keepFrame: true });
  if (state.camera !== 'live') {
    const failure = state.camera;
    state.facing = previous.facing; state.mirror = previous.mirror;
    await startCamera({ keepFrame: true });
    toast(`Could not switch cameras (${failure}).`, { kind: 'error' });
  } else if (previous.paused) await togglePause();
}

on('mode', () => { resetFeed(); if (state.camera === 'live') startLoop(); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden && state.camera === 'live') { stopLoop(); }
  else if (!document.hidden && state.camera === 'live') startLoop();
});
/** Start optional permission-state observation after the shell has subscribed to camera events. */
export function initMedia() {
  try {
    navigator.permissions?.query({ name: 'camera' }).then((permission) => {
      permission.addEventListener('change', () => { if (permission.state === 'granted' && state.camera === 'blocked') void startCamera(); });
    }).catch(() => {});
  } catch { /* Permissions API is optional. */ }
}
