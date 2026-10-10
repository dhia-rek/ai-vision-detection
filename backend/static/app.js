'use strict';
/*
 * Vision web UI. Plain JS, no build step.
 *
 * Frames (webcam / video) and images are POSTed to /api/analyze?annotate=false. The server returns JSON
 * only; boxes, labels and hand skeletons are drawn here on a <canvas> over the media, so they stay crisp
 * at any size and can be animated between frames.
 */

const $ = (id) => document.getElementById(id);
const MAX_WIDTH = 640, JPEG_QUALITY = 0.7, MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const GROUPS = ['objects', 'faces', 'sign'];
const EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised'];
const MAX_FACES = 4, MAX_TRACKS = 60, SPARK_POINTS = 40;

/** Fixed colour per object class (hash of the class name picks the slot, so a class keeps its colour). */
const PALETTE = ['#38bdf8', '#f472b6', '#facc15', '#4ade80', '#fb923c', '#a78bfa',
  '#2dd4bf', '#f87171', '#60a5fa', '#a3e635', '#e879f9', '#fbbf24'];
const EMOTION_COLORS = {
  angry: '#f87171', disgust: '#a3e635', fear: '#a78bfa', happy: '#facc15',
  neutral: '#94a3b8', sad: '#60a5fa', surprised: '#fb923c',
};
/** Hands are drawn over photos on a dark viewport, so they use a fixed bright colour in both themes. */
const SIGN_COLOR = '#45d6ad';
const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

const MODE_META = {
  objects: { title: 'Objects', sub: '80 everyday object classes' },
  faces: { title: 'Expressions', sub: 'Seven expressions, scored per face' },
  sign: { title: 'Sign language', sub: 'ASL fingerspelling and hand gestures' },
};
const EMPTY_META = {
  webcam: { icon: 'camera', title: 'Start the camera', text: 'Frames are analysed on the server and never stored.' },
  video: { icon: 'film', title: 'Analyse a video', text: 'Pick a file or one of the samples below.', button: 'Choose a video' },
  image: { icon: 'image', title: 'Analyse an image', text: 'Pick a file or one of the samples below.', button: 'Choose an image' },
};
const STATUS_TEXT = {
  idle: 'Idle', busy: 'Analysing', done: 'Analysed', running: 'Live', paused: 'Paused', ended: 'Ended', error: 'Error',
};

const el = {
  body: document.body, viewport: $('viewport'), video: $('video'), still: $('still'), overlay: $('overlay'),
  grab: $('grab'), empty: $('empty'), panel: $('panel'),
};
const octx = el.overlay.getContext('2d');
const gctx = el.grab.getContext('2d');

const state = { mode: 'objects', sign: 'spell', source: 'webcam', overlay: true };
let stream = null;
let loopGen = 0;            // bumping this cancels the running frame loop
let loopRunning = false;
let inflight = false;       // one request in flight at a time
let imageGen = 0;
let stillFile = null;       // the image currently shown (re-analysed when the mode changes)
let samples = { objects: [], faces: [], sign: [], video: [] };
let objectUrls = [];        // blob: URLs we created for input media
let lastSize = { w: 4, h: 3 };   // pixel size of the image the last detections refer to
let firstResult = false;

// ---------- small helpers ----------
const pct = (v) => `${Math.round((Number(v) || 0) * 100)}%`;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function qs() { return `mode=${state.mode}&sign_mode=${state.sign}&annotate=false`; }
function trackUrl(u) { objectUrls.push(u); return u; }
function revokeUrls() { objectUrls.forEach((u) => URL.revokeObjectURL(u)); objectUrls = []; }
function svgUse(name) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(ns, 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.appendChild(use);
  return svg;
}
function h(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function hash(s) { let x = 2166136261; for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 16777619); } return x >>> 0; }
function spellActive() { return state.mode === 'sign' && state.sign === 'spell' && state.source !== 'image'; }
function liveSource() { return state.source === 'webcam' || state.source === 'video'; }
function labelOf(d) { return String(d.class); }

function setStatus(kind, text) {
  el.body.dataset.state = kind;
  $('statusPill').dataset.kind = kind;
  $('statusText').textContent = text || STATUS_TEXT[kind] || kind;
}
function showError(msg) {
  $('errorText').textContent = msg;
  $('errorBox').classList.remove('hidden');
  setStatus('error');
}
function clearError() { $('errorBox').classList.add('hidden'); }

// ---------- colours ----------
let accent = '#38bdf8';
function readAccent() {
  const cs = getComputedStyle(el.body);
  const v = (cs.getPropertyValue('--accent-canvas') || cs.getPropertyValue('--accent')).trim();
  if (/^#[0-9a-f]{6}$/i.test(v)) accent = v;   // canvas colours must be plain #rrggbb
}
function colorFor(d) {
  const label = labelOf(d);
  if (state.mode === 'faces') return EMOTION_COLORS[label] || accent;
  if (state.mode === 'sign') return SIGN_COLOR;
  return PALETTE[hash(label) % PALETTE.length];
}
function inkOn(hex) {
  const n = parseInt(hex.slice(1), 16);
  const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const l = 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
  return l > 0.4 ? '#0b0f14' : '#ffffff';
}

// ---------- overlay tracks (client-side drawing with smooth transitions) ----------
let tracks = [];
let raf = 0, lastTick = 0;

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

/** Merge a new set of detections into the tracks: matched ones glide, new ones fade in, lost ones fade out. */
function updateTracks(dets, size) {
  const items = dets.map((d) => ({
    d,
    box: [d.bbox.x1 / size.w, d.bbox.y1 / size.h, d.bbox.x2 / size.w, d.bbox.y2 / size.h],
    pts: Array.isArray(d.points) ? d.points.map((p) => [p[0] / size.w, p[1] / size.h]) : null,
  }));
  const live = tracks.filter((t) => t.fade === 1);
  const pairs = [];
  items.forEach((it, i) => live.forEach((t, j) => {
    if (state.mode === 'objects' && t.label !== labelOf(it.d)) return;
    const v = iou(it.box, t.target);
    if (v > 0.15) pairs.push([v, i, j]);
  }));
  pairs.sort((a, b) => b[0] - a[0]);
  const usedItem = new Set(), usedTrack = new Set();
  for (const [, i, j] of pairs) {
    if (usedItem.has(i) || usedTrack.has(j)) continue;
    usedItem.add(i); usedTrack.add(j);
    assign(live[j], items[i]);
  }
  live.forEach((t, j) => { if (!usedTrack.has(j)) t.fade = 0; });
  items.forEach((it, i) => {
    if (usedItem.has(i) || tracks.length >= MAX_TRACKS) return;
    const t = { box: it.box.slice(), pts: it.pts ? it.pts.map((p) => p.slice()) : null, alpha: 0, fade: 1 };
    assign(t, it);
    tracks.push(t);
  });
  requestDraw();
}
function assign(t, it) {
  t.target = it.box;
  t.ptsTarget = it.pts;
  if (!it.pts) t.pts = null;
  else if (!t.pts || t.pts.length !== it.pts.length) t.pts = it.pts.map((p) => p.slice());
  t.label = labelOf(it.d);
  t.conf = Number(it.d.confidence) || 0;
  t.index = it.d._index;
  t.color = colorFor(it.d);
  t.low = !!it.d.low_confidence;
}
function clearTracks() { tracks = []; requestDraw(); }
function requestDraw() { if (!raf) { lastTick = performance.now(); raf = requestAnimationFrame(tick); } }

function tick(now) {
  raf = 0;
  const dt = Math.min(100, now - lastTick);
  lastTick = now;
  const kPos = 1 - Math.exp(-dt / 70), kAlpha = 1 - Math.exp(-dt / 110);
  let moving = false;
  for (const t of tracks) {
    for (let i = 0; i < 4; i++) {
      const d = t.target[i] - t.box[i];
      if (Math.abs(d) > 0.0004) { t.box[i] += d * kPos; moving = true; } else t.box[i] = t.target[i];
    }
    if (t.pts && t.ptsTarget) {
      for (let i = 0; i < t.pts.length; i++) {
        for (let k = 0; k < 2; k++) {
          const d = t.ptsTarget[i][k] - t.pts[i][k];
          if (Math.abs(d) > 0.0004) { t.pts[i][k] += d * kPos; moving = true; } else t.pts[i][k] = t.ptsTarget[i][k];
        }
      }
    }
    const goal = t.fade;
    if (Math.abs(goal - t.alpha) > 0.01) { t.alpha += (goal - t.alpha) * kAlpha; moving = true; } else t.alpha = goal;
  }
  tracks = tracks.filter((t) => t.fade === 1 || t.alpha > 0.03);
  draw();
  if (moving) raf = requestAnimationFrame(tick);
}

/** The rectangle (CSS px) the media occupies inside the viewport: object-fit: contain, centred. */
function contentRect(w, hgt) {
  const ar = lastSize.w / lastSize.h;
  let cw = w, ch = w / ar;
  if (ch > hgt) { ch = hgt; cw = hgt * ar; }
  return { x: (w - cw) / 2, y: (hgt - ch) / 2, w: cw, h: ch };
}

function roundRectPath(c, x, y, w, hgt, r) {
  r = Math.max(0, Math.min(r, w / 2, hgt / 2));
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + hgt, r);
  c.arcTo(x + w, y + hgt, x, y + hgt, r);
  c.arcTo(x, y + hgt, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

function draw() {
  const cssW = el.overlay.clientWidth, cssH = el.overlay.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  const pw = Math.round(cssW * dpr), ph = Math.round(cssH * dpr);
  if (el.overlay.width !== pw || el.overlay.height !== ph) { el.overlay.width = pw; el.overlay.height = ph; }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, cssW, cssH);
  if (!state.overlay || !tracks.length) return;
  const r = contentRect(cssW, cssH);
  const big = Math.min(r.w, r.h);
  const lw = clamp(big / 220, 1.5, 3);
  const font = clamp(big / 38, 11, 15);
  octx.font = `600 ${font}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  octx.textBaseline = 'middle';
  const order = tracks.slice().sort((a, b) => area(b) - area(a));
  const placed = [];   // label chips already drawn this frame
  for (const t of order) {
    const x = r.x + t.box[0] * r.w, y = r.y + t.box[1] * r.h;
    const bw = (t.box[2] - t.box[0]) * r.w, bh = (t.box[3] - t.box[1]) * r.h;
    octx.globalAlpha = clamp(t.alpha, 0, 1);
    if (t.pts) drawHand(t, r, lw);
    roundRectPath(octx, x, y, bw, bh, Math.min(10, Math.min(bw, bh) / 6));
    octx.fillStyle = t.color + '1f';
    octx.fill();
    octx.lineWidth = lw;
    octx.strokeStyle = t.color;
    octx.setLineDash(t.low ? [6, 4] : []);
    octx.stroke();
    octx.setLineDash([]);
    drawChip(t, x, y, bw, font, r, placed);
  }
  octx.globalAlpha = 1;
}
function area(t) { return (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]); }

/** Draw a label chip above its box. In crowded scenes a small box skips its chip when it would cover another. */
function drawChip(t, x, y, bw, font, r, placed) {
  const text = `${t.index != null ? `${t.index} · ` : ''}${t.label} ${pct(t.conf)}`;
  const padX = 7, chipH = font + 9;
  const tw = octx.measureText(text).width;
  const cw = Math.min(tw + padX * 2, r.w);
  let cx = clamp(x, r.x, r.x + r.w - cw);
  let cy = y - chipH - 3;
  if (cy < r.y) cy = Math.min(y + 3, r.y + r.h - chipH);
  const hit = placed.some((p) => cx < p.x + p.w && cx + cw > p.x && cy < p.y + p.h && cy + chipH > p.y);
  if (hit && bw < 80) return;
  placed.push({ x: cx, y: cy, w: cw, h: chipH });
  roundRectPath(octx, cx, cy, cw, chipH, 6);
  octx.fillStyle = t.color;
  octx.fill();
  octx.fillStyle = inkOn(t.color);
  octx.save();
  octx.beginPath();
  octx.rect(cx, cy, cw, chipH);
  octx.clip();
  octx.fillText(text, cx + padX, cy + chipH / 2 + 0.5);
  octx.restore();
}

function drawHand(t, r, lw) {
  const P = t.pts.map((p) => [r.x + p[0] * r.w, r.y + p[1] * r.h]);
  octx.lineCap = 'round';
  octx.lineJoin = 'round';
  octx.lineWidth = lw + 1;
  octx.strokeStyle = 'rgba(0,0,0,.55)';
  octx.beginPath();
  for (const [a, b] of HAND_LINKS) { octx.moveTo(P[a][0], P[a][1]); octx.lineTo(P[b][0], P[b][1]); }
  octx.stroke();
  octx.lineWidth = lw;
  octx.strokeStyle = '#ffffff';
  octx.beginPath();
  for (const [a, b] of HAND_LINKS) { octx.moveTo(P[a][0], P[a][1]); octx.lineTo(P[b][0], P[b][1]); }
  octx.stroke();
  for (const p of P) {
    octx.beginPath();
    octx.arc(p[0], p[1], lw + 1.2, 0, Math.PI * 2);
    octx.fillStyle = t.color;
    octx.fill();
    octx.lineWidth = 1;
    octx.strokeStyle = '#ffffff';
    octx.stroke();
  }
}

new ResizeObserver(() => requestDraw()).observe(el.viewport);

// ---------- insight panel ----------
function renderInsights(dets) {
  if (state.mode === 'objects') renderObjects(dets);
  else if (state.mode === 'faces') renderFaces(dets);
  else renderSign(dets);
  announce(dets);
}

function renderObjects(dets) {
  const groups = new Map();
  for (const d of dets) {
    const g = groups.get(d.class) || { name: d.class, n: 0, sum: 0 };
    g.n += 1; g.sum += Number(d.confidence) || 0;
    groups.set(d.class, g);
  }
  const rows = [...groups.values()].sort((a, b) => b.n - a.n || b.sum / b.n - a.sum / a.n);
  const max = rows.length ? rows[0].n : 1;
  const list = $('objBars');
  const existing = new Map([...list.children].map((li) => [li.dataset.key, li]));
  rows.forEach((g, i) => {
    let li = existing.get(g.name);
    if (!li) {
      li = h('li', 'bar-row');
      li.dataset.key = g.name;
      li.append(h('span', 'bar-name', g.name), h('span', 'bar-track'), h('span', 'bar-count'), h('span', 'bar-conf'));
      li.children[1].appendChild(h('span', 'bar-fill'));
      li.style.setProperty('--c', PALETTE[hash(g.name) % PALETTE.length]);
    }
    existing.delete(g.name);
    li.children[1].firstChild.style.setProperty('--w', `${Math.round((g.n / max) * 100)}%`);
    li.children[2].textContent = `×${g.n}`;
    li.children[3].textContent = pct(g.sum / g.n);
    li.children[3].title = 'Average confidence';
    if (list.children[i] !== li) list.insertBefore(li, list.children[i] || null);
  });
  existing.forEach((li) => li.remove());
  const total = dets.length;
  $('objCount').textContent = String(total);
  $('objClasses').textContent = String(rows.length);
  $('objConf').textContent = total ? pct(dets.reduce((s, d) => s + (Number(d.confidence) || 0), 0) / total) : '–';
  $('objNone').classList.toggle('hidden', total > 0);
}

function emotionScores(d) {
  if (d.scores) return d.scores;
  return d.top3 || { [d.class]: d.confidence };   // older servers only send the top three
}

function renderFaces(dets) {
  const list = $('faceList');
  const faces = dets.slice().sort((a, b) => a.bbox.x1 - b.bbox.x1).slice(0, MAX_FACES);
  while (list.children.length > faces.length) list.lastChild.remove();
  faces.forEach((d, i) => {
    let card = list.children[i];
    if (!card) {
      card = h('article', 'face');
      const head = h('header', 'face-head');
      head.append(h('span', 'face-name'), h('span', 'face-top'));
      const ul = h('ul', 'emo');
      for (const e of EXPRESSIONS) {
        const li = h('li', 'emo-row');
        li.dataset.emo = e;
        li.style.setProperty('--c', EMOTION_COLORS[e]);
        const fill = h('span', 'emo-fill');
        const track = h('span', 'emo-track');
        track.appendChild(fill);
        li.append(h('span', 'emo-name', e), track, h('span', 'emo-val'));
        ul.appendChild(li);
      }
      card.append(head, ul);
      list.appendChild(card);
    }
    const scores = emotionScores(d);
    card.children[0].children[0].textContent = faces.length > 1 ? `Face ${i + 1}` : 'Face';
    card.children[0].children[1].textContent = `${d.class} ${pct(d.confidence)}`;
    card.children[0].children[1].style.setProperty('--c', EMOTION_COLORS[d.class] || accent);
    for (const li of card.children[1].children) {
      const v = Number(scores[li.dataset.emo]) || 0;
      li.children[1].firstChild.style.setProperty('--w', `${Math.round(v * 100)}%`);
      li.children[2].textContent = pct(v);
      li.classList.toggle('is-top', li.dataset.emo === d.class);
    }
  });
  $('faceNone').classList.toggle('hidden', faces.length > 0);
  const more = dets.length - faces.length;
  $('faceNone').textContent = more > 0 ? `+${more} more` : 'No face in view.';
  if (more > 0) $('faceNone').classList.remove('hidden');
}

function renderSign(dets) {
  const best = dets.reduce((a, d) => (!a || d.confidence > a.confidence ? d : a), null);
  const spell = state.sign === 'spell';
  const main = $('signMain');
  $('signLabel').textContent = spell ? 'Current letter' : 'Current gesture';
  if (!best) {
    main.textContent = '–';
    $('signConf').textContent = 'No hand in view';
    if (!spellActive()) setRing(0);
  } else {
    main.textContent = best.class;
    $('signConf').textContent = `${pct(best.confidence)} confidence${best.low_confidence ? ' (low)' : ''}`;
    if (spell && state.source === 'image') setRing(best.confidence >= builder.minConf ? 1 : best.confidence);
  }
  main.dataset.long = String(main.textContent.length > 2);
  main.classList.toggle('is-empty', !best);
}

const RING_C = 2 * Math.PI * 52;
function setRing(p) {
  const v = $('ringValue');
  v.style.strokeDasharray = RING_C.toFixed(2);
  v.style.strokeDashoffset = (RING_C * (1 - clamp(p, 0, 1))).toFixed(2);
}

let lastAnnounce = 0, lastSummary = '';
function announce(dets) {
  const now = performance.now();
  if (now - lastAnnounce < 1500) return;
  let text;
  if (!dets.length) text = state.mode === 'sign' ? 'No hand in view' : state.mode === 'faces' ? 'No face in view' : 'Nothing detected';
  else if (state.mode === 'objects') {
    const c = new Map();
    dets.forEach((d) => c.set(d.class, (c.get(d.class) || 0) + 1));
    text = [...c].map(([k, n]) => `${n} ${k}`).join(', ');
  } else {
    const b = dets.reduce((a, d) => (d.confidence > a.confidence ? d : a));
    text = `${b.class}, ${pct(b.confidence)}`;
  }
  if (text === lastSummary) return;
  lastSummary = text; lastAnnounce = now;
  $('liveSummary').textContent = text;
}

// ---------- status bar ----------
let fpsTimes = [], fpsSeries = [];
function resetStats() {
  fpsTimes = []; fpsSeries = [];
  $('fps').textContent = '–';
  $('sparkLine').setAttribute('points', '');
  $('latency').textContent = '–';
}
function recordFrame(t1, roundTrip, data) {
  fpsTimes.push(t1);
  if (fpsTimes.length > 9) fpsTimes.shift();
  if (fpsTimes.length > 1) {
    const fps = (fpsTimes.length - 1) * 1000 / (t1 - fpsTimes[0]);
    $('fps').textContent = fps.toFixed(1);
    fpsSeries.push(fps);
    if (fpsSeries.length > SPARK_POINTS) fpsSeries.shift();
    const top = Math.max(5, ...fpsSeries) * 1.15;
    const pts = fpsSeries.map((v, i) => `${((i / (SPARK_POINTS - 1)) * 80).toFixed(1)},${(21 - (v / top) * 20).toFixed(1)}`);
    $('sparkLine').setAttribute('points', pts.join(' '));
  }
  recordLatency(roundTrip, data);
}
function recordLatency(roundTrip, data) {
  $('latency').textContent = `${Math.round(roundTrip)} ms`;
  $('latencyStat').title = `Round trip ${Math.round(roundTrip)} ms, model ${data.inference_ms != null ? Math.round(data.inference_ms) : '?'} ms`;
  if (data.model_used) $('modelName').textContent = data.model_used;
  $('deviceText').textContent = String(data.device || 'cpu').toUpperCase();
}

/** Apply one /api/analyze response: overlay, insight panel and stats. */
function applyResult(data) {
  const size = data.image_size && data.image_size.width ? { w: data.image_size.width, h: data.image_size.height } : lastSize;
  lastSize = size;
  let dets = Array.isArray(data.detections) ? data.detections : [];
  if (state.mode === 'faces') {
    dets = dets.slice().sort((a, b) => a.bbox.x1 - b.bbox.x1);
    dets.forEach((d, i) => { d._index = dets.length > 1 ? i + 1 : null; });
  }
  updateTracks(dets, size);
  renderInsights(dets);
  return dets;
}

// ---------- fingerspelling text builder (mirrors WordBuilder in app/sign.py) ----------
const builder = {
  holdS: 0.8, spaceS: 1.5, minConf: 0.6,
  text: '', current: null, since: 0, committed: false, lastSeen: null,
  reset() { this.current = null; this.since = 0; this.committed = false; this.lastSeen = null; },
  /**
   * Feed one frame's prediction. A letter held for holdS seconds is committed once;
   * no letter for spaceS seconds inserts a space. Keep in sync with WordBuilder in app/sign.py.
   * @param {?string} letter predicted letter or null if no hand
   * @param {number} conf prediction confidence 0..1
   * @param {number} t timestamp in seconds
   */
  feed(letter, conf, t) {
    if (letter == null || conf < this.minConf) {
      this.current = null; this.committed = false;
      if (this.lastSeen !== null && t - this.lastSeen >= this.spaceS && this.text && !this.text.endsWith(' ')) this.text += ' ';
    } else {
      this.lastSeen = t;
      if (letter !== this.current) { this.current = letter; this.since = t; this.committed = false; }
      else if (!this.committed && t - this.since >= this.holdS) { this.committed = true; this.text += letter; pulseRing(); }
    }
    this.render(t);
  },
  render(t) {
    $('wordText').textContent = this.text;
    $('wordBox').classList.toggle('is-empty', !this.text);
    if (spellActive()) {
      let p = 0;
      if (this.current) p = this.committed ? 1 : Math.min(1, (t - this.since) / this.holdS);
      setRing(p);
    }
  },
  edit(fn) { this.text = fn(this.text); this.render(performance.now() / 1000); },
  clear() { this.text = ''; this.reset(); this.render(performance.now() / 1000); },
};
function pulseRing() {
  const r = $('holdRing');
  r.classList.remove('pulse');
  void r.offsetWidth;
  r.classList.add('pulse');
}
$('spaceBtn').onclick = () => { if (builder.text && !builder.text.endsWith(' ')) builder.edit((s) => s + ' '); };
$('backBtn').onclick = () => builder.edit((s) => s.slice(0, -1));
$('clearBtn').onclick = () => builder.clear();

// ---------- frame loop (webcam + video file) ----------
function sourceReady() {
  return liveSource() && el.video.videoWidth > 0 && !el.video.paused && !el.video.ended && (state.source !== 'webcam' || !!stream);
}

/** Grab the current video frame, send it, and apply the response. Resolves when the round trip is over. */
async function sendFrame(gen) {
  if (inflight) return;
  inflight = true;
  const modeKey = `${state.mode}/${state.sign}`;
  try {
    const t0 = performance.now();
    const vw = el.video.videoWidth, vh = el.video.videoHeight;
    const w = Math.min(vw, MAX_WIDTH), hh = Math.round(vh * w / vw);
    el.grab.width = w; el.grab.height = hh;
    gctx.drawImage(el.video, 0, 0, w, hh);
    const blob = await new Promise((r) => el.grab.toBlob(r, 'image/jpeg', JPEG_QUALITY));
    if (!blob) throw new Error('Could not encode frame');
    const fd = new FormData();
    fd.append('file', blob, 'frame.jpg');
    const spellAtSend = spellActive();
    if (!firstResult) setStatus('busy', 'Loading model');
    const res = await fetch(`/api/analyze?${qs()}`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`Backend error: ${res.status} ${(await res.text().catch(() => '')).slice(0, 200)}`);
    const data = await res.json();
    const t1 = performance.now();
    if (gen !== loopGen || modeKey !== `${state.mode}/${state.sign}`) return;   // stale: mode or source changed
    firstResult = true;
    const dets = applyResult(data);
    recordFrame(t1, t1 - t0, data);
    if (spellAtSend && spellActive()) {
      const best = dets.reduce((a, d) => (!a || d.confidence > a.confidence ? d : a), null);
      builder.feed(best ? best.class : null, best ? best.confidence : 0, t1 / 1000);
    }
    clearError();
    if (!el.video.paused) setStatus('running', state.source === 'video' ? 'Playing' : 'Live');
  } catch (err) {
    if (gen !== loopGen) return;
    showError(err.message || String(err));
    await sleep(1000);
  } finally {
    inflight = false;
  }
}

async function frameLoop(gen) {
  if (loopRunning) return;
  loopRunning = true;
  try {
    while (gen === loopGen) {
      if (inflight || !sourceReady()) {
        if (state.source === 'video' && (el.video.ended || !el.video.getAttribute('src'))) break;
        await sleep(60);
        continue;
      }
      await sendFrame(gen);
    }
  } finally {
    loopRunning = false;
    if (gen !== loopGen && liveSource() && mediaLive()) startLoop();   // a newer loop was requested meanwhile
  }
}
function mediaLive() { return (state.source === 'webcam' && !!stream) || (state.source === 'video' && !!el.video.getAttribute('src') && !el.video.ended); }
function startLoop() { loopGen++; frameLoop(loopGen); }
function stopLoop() { loopGen++; }

// ---------- webcam ----------
async function startCamera() {
  clearError();
  if (stream) return;
  const btn = $('startBtn');
  try {
    btn.disabled = true;
    setStatus('busy', 'Starting camera');
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    if (state.source !== 'webcam') { stopCameraTracks(); return; }
    el.video.srcObject = stream;
    await el.video.play().catch(() => {});
    resetStats();
    firstResult = false;
    builder.reset();
    setStatus('running', 'Live');
    syncUI();
    startLoop();
    el.viewport.focus({ preventScroll: true });
  } catch (err) {
    stream = null;
    const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
    showError(denied ? 'Camera access was blocked. Allow it in the browser, or use a video or an image.' : (err.message || String(err)));
  } finally {
    btn.disabled = false;
  }
}
function stopCameraTracks() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  el.video.srcObject = null;
}
function stopCamera() {
  stopLoop();
  stopCameraTracks();
  clearTracks();
  setStatus('idle');
  syncUI();
}
$('startBtn').onclick = startCamera;
$('stopBtn').onclick = stopCamera;

// ---------- video ----------
function loadVideo(url, isObjectUrl) {
  stopLoop();
  el.video.pause();
  revokeUrls();
  if (isObjectUrl) trackUrl(url);
  clearError();
  resetStats();
  clearTracks();
  builder.reset();
  firstResult = false;
  setStatus('busy', 'Loading video');
  el.video.src = url;
  el.video.load();
  el.video.play().catch(() => setStatus('paused'));
  syncUI();
  el.viewport.focus({ preventScroll: true });
}
el.video.addEventListener('play', () => {
  if (state.source === 'webcam') return;
  if (state.source !== 'video') return;
  setStatus('running', 'Playing');
  syncTransport();
  startLoop();
});
el.video.addEventListener('pause', () => { if (liveSource() && !el.video.ended) { setStatus('paused'); syncTransport(); } });
el.video.addEventListener('ended', () => {
  if (state.source !== 'video') return;
  stopLoop();
  setStatus('ended');
  syncTransport();
});
el.video.addEventListener('error', () => {
  if (state.source === 'video' && el.video.getAttribute('src')) showError('Could not play this video');
});
el.video.addEventListener('timeupdate', () => {
  if (state.source !== 'video' || !el.video.duration) return;
  $('seek').value = String(Math.round((el.video.currentTime / el.video.duration) * 1000));
});
el.video.addEventListener('seeked', () => {
  if (liveSource() && el.video.paused && !inflight && el.video.videoWidth > 0) sendFrame(loopGen);
});
$('seek').addEventListener('input', (e) => {
  if (el.video.duration) el.video.currentTime = (Number(e.target.value) / 1000) * el.video.duration;
});
$('videoInput').addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) openFile(f);
  e.target.value = '';
});

function togglePause() {
  if (!liveSource() || !mediaLive()) return;
  if (el.video.paused) {
    if (el.video.ended) el.video.currentTime = 0;
    el.video.play().catch(() => {});
    if (state.source === 'webcam') { setStatus('running', 'Live'); syncTransport(); }
  } else {
    el.video.pause();
    if (state.source === 'webcam') { setStatus('paused'); syncTransport(); }
  }
}
$('playBtn').onclick = togglePause;

// ---------- still images ----------
async function analyzeStill() {
  if (!stillFile) return;
  const gen = ++imageGen;
  clearError();
  setStatus('busy');
  try {
    const t0 = performance.now();
    const fd = new FormData();
    fd.append('file', stillFile);
    const res = await fetch(`/api/analyze?${qs()}`, { method: 'POST', body: fd });
    if (gen !== imageGen) return;
    if (!res.ok) throw new Error(`Backend error: ${res.status} ${(await res.text().catch(() => '')).slice(0, 200)}`);
    const data = await res.json();
    if (gen !== imageGen) return;
    applyResult(data);
    recordLatency(performance.now() - t0, data);
    setStatus('done');
  } catch (err) {
    if (gen === imageGen) showError(err.message || String(err));
  }
}

function loadImage(file) {
  if (!file.type || !file.type.startsWith('image/')) return showError('Please choose an image file.');
  if (file.size > MAX_IMAGE_BYTES) return showError('The image must be smaller than 10 MB.');
  releaseMedia();
  stillFile = file;
  el.still.src = trackUrl(URL.createObjectURL(file));
  resetStats();
  clearTracks();
  syncUI();
  analyzeStill();
}

/** Route a user-supplied file (picker or drop) to the right source by its type. */
function openFile(file) {
  const isVideo = (file.type || '').startsWith('video/');
  const isImage = (file.type || '').startsWith('image/');
  if (!isVideo && !isImage) return showError('Drop an image or a video file.');
  setSource(isVideo ? 'video' : 'image');
  if (isVideo) loadVideo(URL.createObjectURL(file), true);
  else loadImage(file);
}

async function openSample(group, name, kind) {
  clearError();
  const url = `/demo/samples/${encodeURIComponent(group)}/${encodeURIComponent(name)}`;
  if (kind === 'video') {
    setSource('video');
    loadVideo(url, false);
    return;
  }
  setSource('image');
  try {
    setStatus('busy');
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load sample (${res.status})`);
    const blob = await res.blob();
    loadImage(new File([blob], name, { type: blob.type || 'image/jpeg' }));
  } catch (err) { showError(err.message || String(err)); }
}

/** Free the current media and its results (not the source selection). */
function releaseMedia() {
  stopLoop();
  imageGen++;
  if (stream) stopCameraTracks();
  el.video.pause();
  el.video.removeAttribute('src');
  el.video.load();
  el.still.removeAttribute('src');
  stillFile = null;
  revokeUrls();
}

// ---------- drag and drop (whole window) ----------
let dragDepth = 0;
function hasFiles(e) { return e.dataTransfer && [...e.dataTransfer.types].includes('Files'); }
window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { e.preventDefault(); dragDepth++; el.body.classList.add('dragging'); } });
window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener('dragleave', (e) => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; el.body.classList.remove('dragging'); } });
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0; el.body.classList.remove('dragging');
  if (e.dataTransfer.files.length) openFile(e.dataTransfer.files[0]);
});
$('chooseBtn').onclick = () => (state.source === 'video' ? $('videoInput') : $('fileInput')).click();
$('fileInput').addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) openFile(f);
  e.target.value = '';
});
$('errorClose').onclick = () => { clearError(); if (el.body.dataset.state === 'error') setStatus('idle'); };

// ---------- samples ----------
function sampleLabel(name) { return name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' '); }
function renderSamples() {
  const box = $('samples');
  box.textContent = '';
  const items = [];
  if (state.source === 'image') (samples[state.mode] || []).forEach((n) => items.push({ group: state.mode, name: n, kind: 'image' }));
  if (state.source === 'video') (samples.video || []).forEach((n) => items.push({ group: 'video', name: n, kind: 'video' }));
  box.classList.toggle('hidden', !items.length);
  if (!items.length) return;
  box.appendChild(h('span', 'samples-label', 'Samples'));
  for (const it of items) {
    const b = h('button', `sample sample-${it.kind}`);
    b.type = 'button';
    b.dataset.group = it.group;
    b.dataset.name = it.name;
    b.setAttribute('aria-label', `Try sample: ${sampleLabel(it.name)}`);
    const url = `/demo/samples/${encodeURIComponent(it.group)}/${encodeURIComponent(it.name)}`;
    if (it.kind === 'image') {
      const img = h('img');
      img.src = url; img.alt = ''; img.loading = 'lazy'; img.decoding = 'async';
      b.appendChild(img);
    } else {
      const film = h('span', 'sample-film');
      film.appendChild(svgUse('film'));
      b.appendChild(film);
    }
    b.appendChild(h('span', 'sample-name', sampleLabel(it.name)));
    b.addEventListener('click', () => openSample(it.group, it.name, it.kind));
    box.appendChild(b);
  }
}

// ---------- UI sync ----------
function mediaShown() {
  if (state.source === 'image') return !!stillFile;
  if (state.source === 'video') return !!el.video.getAttribute('src');
  return !!stream;
}
function syncTransport() {
  const show = mediaShown() && liveSource();
  $('transport').classList.toggle('hidden', !show);
  const paused = el.video.paused;
  $('playBtn').dataset.paused = String(paused);
  $('playBtn').setAttribute('aria-label', paused ? 'Resume' : 'Pause');
  $('seek').classList.toggle('hidden', state.source !== 'video');
  $('stopBtn').classList.toggle('hidden', state.source !== 'webcam');
}
function syncUI() {
  const shown = mediaShown();
  el.body.dataset.mode = state.mode;
  el.body.dataset.source = state.source;
  el.panel.dataset.view = state.mode;
  el.panel.dataset.sign = state.sign;
  el.video.classList.toggle('hidden', !(liveSource() && shown));
  el.still.classList.toggle('hidden', !(state.source === 'image' && shown));
  el.empty.classList.toggle('hidden', shown);
  const meta = EMPTY_META[state.source];
  el.empty.querySelector('use').setAttribute('href', `#i-${meta.icon}`);
  $('emptyTitle').textContent = meta.title;
  $('emptyText').textContent = meta.text;
  $('startBtn').classList.toggle('hidden', state.source !== 'webcam');
  $('chooseBtn').classList.toggle('hidden', state.source === 'webcam');
  $('dropHintText').classList.toggle('hidden', state.source === 'webcam');
  if (meta.button) $('chooseLabel').textContent = meta.button;
  const mm = MODE_META[state.mode];
  $('panelTitle').textContent = mm.title;
  $('panelSub').textContent = state.mode === 'sign' ? (state.sign === 'spell' ? 'ASL fingerspelling, letters A to Z' : 'Seven static hand gestures') : mm.sub;
  $('insightObjects').classList.toggle('hidden', state.mode !== 'objects');
  $('insightFaces').classList.toggle('hidden', state.mode !== 'faces');
  $('insightSign').classList.toggle('hidden', state.mode !== 'sign');
  $('signSwitch').classList.toggle('hidden', state.mode !== 'sign');
  $('speller').classList.toggle('hidden', !spellActive());
  const note = $('signImageNote');
  note.classList.toggle('hidden', !(state.mode === 'sign' && (state.source === 'image' || state.sign === 'gestures')));
  note.textContent = state.sign === 'spell'
    ? 'Letters are read from a single static hand shape. J and Z need motion and are unreliable.'
    : 'Gestures are static hand shapes: thumbs up or down, open palm, victory, pointing up, closed fist, I love you.';
  setChecked($('modeSwitch'), 'mode', state.mode);
  setChecked($('sourceSwitch'), 'source', state.source);
  setChecked($('signSwitch'), 'sign', state.sign);
  $('overlayToggle').setAttribute('aria-pressed', String(state.overlay));
  el.viewport.classList.toggle('overlay-off', !state.overlay);
  syncTransport();
  renderSamples();
  readAccent();
  syncUrl();
  requestDraw();
}
function syncUrl() {
  const p = new URLSearchParams(location.search);
  p.set('mode', state.mode); p.set('source', state.source);
  if (state.mode === 'sign') p.set('sign_mode', state.sign); else p.delete('sign_mode');
  p.delete('sample');
  try { history.replaceState(null, '', `?${p}`); } catch (e) { /* file:// or sandboxed frames */ }
}

function setMode(mode, sign) {
  const changed = mode !== state.mode || (sign && sign !== state.sign);
  state.mode = mode;
  if (sign) state.sign = sign;
  if (!changed) return syncUI();
  imageGen++;
  clearTracks();
  lastSummary = ''; lastAnnounce = 0;
  builder.clear();
  firstResult = false;
  renderInsights([]);
  syncUI();
  if (state.source === 'image' && stillFile) analyzeStill();
}

function setSource(source) {
  if (source === state.source) return syncUI();
  releaseMedia();
  state.source = source;
  clearTracks();
  resetStats();
  clearError();
  renderInsights([]);
  setStatus('idle');
  syncUI();
}

// ---------- segmented controls (radio groups with roving tabindex and arrow keys) ----------
function setChecked(group, attr, value) {
  for (const b of group.querySelectorAll('[role="radio"]')) {
    const on = b.dataset[attr] === value;
    b.setAttribute('aria-checked', String(on));
    b.tabIndex = on ? 0 : -1;
  }
}
function wireSegmented(group, attr, onPick) {
  const radios = [...group.querySelectorAll('[role="radio"]')];
  radios.forEach((b) => b.addEventListener('click', () => onPick(b.dataset[attr])));
  group.addEventListener('keydown', (e) => {
    const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
    if (!(e.key in keys)) return;
    e.preventDefault();
    const i = radios.indexOf(document.activeElement);
    const next = radios[(i + keys[e.key] + radios.length) % radios.length];
    next.focus();
    onPick(next.dataset[attr]);
  });
}
wireSegmented($('modeSwitch'), 'mode', (m) => setMode(m));
wireSegmented($('signSwitch'), 'sign', (s) => setMode('sign', s));
wireSegmented($('sourceSwitch'), 'source', (s) => setSource(s));

// ---------- overlay toggle, theme, help, shortcuts ----------
function toggleOverlay() { state.overlay = !state.overlay; syncUI(); }
$('overlayToggle').onclick = toggleOverlay;

function applyTheme(t, store) {
  document.documentElement.dataset.theme = t;
  if (store) { try { localStorage.setItem('avd-theme', t); } catch (e) { /* private mode */ } }
  $('themeToggle').setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  readAccent();
  requestDraw();
}
function toggleTheme() { applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark', true); }
$('themeToggle').onclick = toggleTheme;
window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
  let stored = null;
  try { stored = localStorage.getItem('avd-theme'); } catch (err) { /* ignore */ }
  if (!stored) applyTheme(e.matches ? 'light' : 'dark', false);
});

const helpDialog = $('helpDialog');
$('helpBtn').onclick = () => helpDialog.showModal();
helpDialog.addEventListener('click', (e) => { if (e.target === helpDialog) helpDialog.close(); });

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || helpDialog.open) return;
  const t = e.target;
  if (t.closest('input, textarea, select, [contenteditable]')) return;
  const onControl = !!t.closest('button, a, [role="radio"]');
  switch (e.key) {
    case '1': setMode('objects'); break;
    case '2': setMode('faces'); break;
    case '3': setMode('sign'); break;
    case ' ':
      if (onControl) return;
      e.preventDefault(); togglePause(); break;
    case 'o': case 'O': toggleOverlay(); break;
    case 't': case 'T': toggleTheme(); break;
    case '?': helpDialog.showModal(); break;
    case 'Backspace':
      if (spellActive() && !onControl) { e.preventDefault(); builder.edit((s) => s.slice(0, -1)); }
      break;
    default: return;
  }
});

window.addEventListener('beforeunload', () => { stopCameraTracks(); revokeUrls(); });

// ---------- init ----------
(async function init() {
  const p = new URLSearchParams(location.search);
  const m = p.get('mode'), s = p.get('sign_mode'), src = p.get('source'), sample = p.get('sample');
  if (GROUPS.includes(m)) state.mode = m;
  if (['spell', 'gestures'].includes(s)) state.sign = s;
  if (['webcam', 'video', 'image'].includes(src)) state.source = src;
  applyTheme(document.documentElement.dataset.theme, false);
  setStatus('idle');
  setRing(0);
  resetStats();
  builder.render(performance.now() / 1000);
  renderInsights([]);
  syncUI();
  try {
    const r = await fetch('/demo/samples');
    if (r.ok) { samples = Object.assign(samples, await r.json()); renderSamples(); }
  } catch (e) { /* samples are optional */ }
  if (sample && state.source !== 'webcam') {
    const group = state.source === 'video' ? 'video' : (samples[state.mode].includes(sample) ? state.mode : GROUPS.find((g) => samples[g].includes(sample)));
    if (group) openSample(group, sample, state.source === 'video' ? 'video' : 'image');
  }
})();
