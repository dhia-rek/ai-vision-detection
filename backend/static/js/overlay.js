/** Live detection tracks, filters, smart labels, and canvas painting shared with snapshots. */
import { state, on } from './store.js';
import { $, clamp, pct, hash } from './util.js';
import { PALETTE, EMOTION_COLORS, SIGN_COLOR, HAND_LINKS, MAX_TRACKS } from './config.js';

let viewport = null, canvas = null, ctx = null, size = { w: 4, h: 3 };
let tracks = [], raf = 0, lastTick = 0;

function labelOf(d) { return String(d.class ?? ''); }
function colorFor(d) {
  if (state.mode === 'faces') return EMOTION_COLORS[labelOf(d)] || '#2563eb';
  if (state.mode === 'sign') return SIGN_COLOR;
  return PALETTE[hash(labelOf(d)) % PALETTE.length];
}
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}
function area(t) { return Math.max(0, t.box[2] - t.box[0]) * Math.max(0, t.box[3] - t.box[1]); }

/** Choose non-overlapping label chips in a bounded rectangle; exported for deterministic unit checks. */
export function placeLabels(boxes, rect, measure) {
  const placed = [], results = new Array(boxes.length);
  const sorted = boxes.map((b, i) => ({ ...b, _i: i })).sort((a, b) =>
    ((b.box.w ?? b.box[2] - b.box[0]) * (b.box.h ?? b.box[3] - b.box[1])) - ((a.box.w ?? a.box[2] - a.box[0]) * (a.box.h ?? a.box[3] - a.box[1])));
  const overlapArea = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  for (const item of sorted) {
    const b = item.box, x = b.x ?? b[0], y = b.y ?? b[1], w = b.w ?? b[2] - b[0], h = b.h ?? b[3] - b[1];
    const font = item.font || 13, pad = item.pad ?? Math.max(4, font * 0.48), full = String(item.text ?? ''), chipH = font + pad;
    const maxW = Math.max(12, rect.w), textW = Math.min(measure(full, font), maxW - pad * 2);
    const chipW = Math.min(maxW, textW + pad * 2);
    const candidates = [
      { x, y: y - chipH - 3 }, { x: x + w - chipW, y: y - chipH - 3 },
      { x: x + 2, y: y + 2 }, { x: x + 2, y: y + h - chipH - 2 },
      { x, y: y + h + 3 }, { x: x + (w - chipW) / 2, y: y + (h - chipH) / 2 },
    ].map((c) => ({ x: clamp(c.x, rect.x, rect.x + rect.w - chipW), y: clamp(c.y, rect.y, rect.y + rect.h - chipH), w: chipW, h: chipH }));
    const anchors = boxes.filter((other, i) => i !== item._i).map((other) => {
      const ob = other.box, ox = ob.x ?? ob[0], oy = ob.y ?? ob[1], ow = ob.w ?? ob[2] - ob[0];
      return { x: ox + ow / 2, y: oy };
    });
    const score = (c) => placed.reduce((sum, p) => sum + overlapArea(c, p), 0) + anchors.reduce((sum, p) => sum + (p.x >= c.x && p.x <= c.x + c.w && p.y >= c.y && p.y <= c.y + c.h ? 1e6 : 0), 0);
    let chosen = candidates.find((c) => score(c) === 0);
    if (!chosen) {
      const base = candidates.reduce((best, c) => score(c) < score(best) ? c : best, candidates[0]);
      let nearest = null, nearestDistance = Infinity;
      for (let py = rect.y; py <= rect.y + rect.h - chipH; py += Math.max(4, Math.floor(font / 2))) {
        for (let px = rect.x; px <= rect.x + rect.w - chipW; px += Math.max(4, Math.floor(font / 2))) {
          const candidate = { x: px, y: py, w: chipW, h: chipH };
          if (score(candidate) !== 0) continue;
          const distance = Math.abs(px - base.x) + Math.abs(py - base.y);
          if (distance < nearestDistance) { nearest = candidate; nearestDistance = distance; }
        }
      }
      chosen = nearest || base;
    }
    const overlaps = placed.reduce((sum, p) => sum + overlapArea(chosen, p), 0);
    const badge = overlaps / Math.max(1, chosen.w * chosen.h) > 0.4;
    if (badge) {
      const text = String(item.badge ?? item.index ?? full.slice(0, 3));
      const bw = Math.min(chosen.w, measure(text, font) + pad * 2);
      chosen = { ...chosen, x: clamp(x, rect.x, rect.x + rect.w - bw), w: bw, text, badge: true };
    } else chosen = { ...chosen, text: full, badge: false };
    placed.push(chosen);
    results[item._i] = chosen;
  }
  return results;
}

export function isVisible(det) {
  return state.mode !== 'objects' || ((Number(det.confidence) || 0) >= state.minConf && !state.hiddenClasses.has(labelOf(det)));
}
export function visibleDets(dets) { return dets.filter(isVisible); }
export function hiddenCount(dets) { return dets.length - visibleDets(dets).length; }
export function getSize() { return { ...size }; }

export function initOverlay({ viewport: vp, canvas: cv }) {
  viewport = vp; canvas = cv; ctx = cv.getContext('2d');
  new ResizeObserver(requestDraw).observe(viewport);
  on('filter', requestDraw); on('overlay', requestDraw); on('theme', requestDraw); on('mode', requestDraw);
  window.addEventListener('resize', requestDraw);
  requestDraw();
}
export function updateTracks(dets, frameSize) {
  if (frameSize?.w > 0 && frameSize?.h > 0) size = { w: frameSize.w, h: frameSize.h };
  const items = dets.slice(0, MAX_TRACKS).map((d) => ({ d, box: [d.bbox.x1 / size.w, d.bbox.y1 / size.h, d.bbox.x2 / size.w, d.bbox.y2 / size.h], pts: Array.isArray(d.points) ? d.points.map((p) => [p[0] / size.w, p[1] / size.h]) : null }));
  const live = tracks.filter((t) => t.fade === 1), pairs = [];
  items.forEach((it, i) => live.forEach((t, j) => {
    if (state.mode === 'objects' && t.label !== labelOf(it.d)) return;
    const score = iou(it.box, t.target); if (score > 0.15) pairs.push([score, i, j]);
  }));
  pairs.sort((a, b) => b[0] - a[0]);
  const usedI = new Set(), usedT = new Set();
  for (const [, i, j] of pairs) { if (usedI.has(i) || usedT.has(j)) continue; usedI.add(i); usedT.add(j); assign(live[j], items[i]); }
  live.forEach((t, j) => { if (!usedT.has(j)) t.fade = 0; });
  items.forEach((it, i) => { if (usedI.has(i) || tracks.length >= MAX_TRACKS) return; const t = { box: it.box.slice(), pts: null, alpha: 0, fade: 1 }; assign(t, it); tracks.push(t); });
  requestDraw();
}
function assign(t, it) {
  t.target = it.box; t.ptsTarget = it.pts;
  if (!it.pts) t.pts = null; else if (!t.pts || t.pts.length !== it.pts.length) t.pts = it.pts.map((p) => p.slice());
  t.label = labelOf(it.d); t.conf = Number(it.d.confidence) || 0; t.index = it.d._index;
  t.color = colorFor(it.d); t.low = !!it.d.low_confidence; t.detection = it.d;
}
export function clearTracks() { tracks = []; requestDraw(); }
export function requestDraw() { if (!raf && canvas) { lastTick = performance.now(); raf = requestAnimationFrame(tick); } }
function tick(now) {
  raf = 0; const dt = Math.min(100, now - lastTick); lastTick = now;
  const kp = 1 - Math.exp(-dt / 70), ka = 1 - Math.exp(-dt / 110); let moving = false;
  for (const t of tracks) {
    for (let i = 0; i < 4; i++) { const d = t.target[i] - t.box[i]; if (Math.abs(d) > 0.0004) { t.box[i] += d * kp; moving = true; } else t.box[i] = t.target[i]; }
    if (t.pts && t.ptsTarget) for (let i = 0; i < t.pts.length; i++) for (let k = 0; k < 2; k++) { const d = t.ptsTarget[i][k] - t.pts[i][k]; if (Math.abs(d) > 0.0004) { t.pts[i][k] += d * kp; moving = true; } else t.pts[i][k] = t.ptsTarget[i][k]; }
    if (Math.abs(t.fade - t.alpha) > 0.01) { t.alpha += (t.fade - t.alpha) * ka; moving = true; } else t.alpha = t.fade;
  }
  tracks = tracks.filter((t) => t.fade === 1 || t.alpha > 0.03); draw(); if (moving) raf = requestAnimationFrame(tick);
}
function roundRect(c, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2)); c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
}
function mediaRect(w, h, frame = size) {
  const ar = frame.w / frame.h; let mw = w, mh = w / ar;
  if (mh > h) { mh = h; mw = h * ar; }
  return { x: (w - mw) / 2, y: (h - mh) / 2, w: mw, h: mh };
}
function draw() {
  if (!ctx || !canvas) return;
  const dpr = window.devicePixelRatio || 1, cw = canvas.clientWidth, ch = canvas.clientHeight;
  const pw = Math.round(cw * dpr), ph = Math.round(ch * dpr);
  if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, cw, ch);
  if (!state.overlay) return;
  const r = mediaRect(cw, ch); paintOverlay(ctx, r, { animated: true });
}
function drawHand(c, t, r, lw, mirror) {
  const pts = t.pts.map((p) => [r.x + (mirror ? 1 - p[0] : p[0]) * r.w, r.y + p[1] * r.h]);
  c.lineCap = 'round'; c.lineJoin = 'round';
  for (const [color, width] of [['rgba(0,0,0,.65)', lw + 2], ['#fff', lw]]) {
    c.beginPath(); for (const [a, b] of HAND_LINKS) if (pts[a] && pts[b]) { c.moveTo(...pts[a]); c.lineTo(...pts[b]); }
    c.lineWidth = width; c.strokeStyle = color; c.stroke();
  }
  for (const p of pts) if (p) { c.beginPath(); c.arc(p[0], p[1], lw + 1, 0, Math.PI * 2); c.fillStyle = t.color; c.fill(); c.lineWidth = 1; c.strokeStyle = '#fff'; c.stroke(); }
}
/** Paint tracks in CSS pixel coordinates; both live overlay and snapshots call this function. */
export function paintOverlay(c, rect, { scale = 1, animated = false, mirror = state.mirror } = {}) {
  if (!state.overlay && c === ctx) return;
  const big = Math.min(rect.w, rect.h), lw = clamp(big / 220, 1.5, 3) * scale, font = clamp(big / 38, 11, 15) * scale;
  const source = tracks.filter((t) => isVisible(t.detection));
  const labels = source.map((t) => {
    const box = animated ? t.box : t.target, x0 = rect.x + box[0] * rect.w, y = rect.y + box[1] * rect.h;
    const w = (box[2] - box[0]) * rect.w, h = (box[3] - box[1]) * rect.h;
    const x = mirror ? rect.x + rect.w - (x0 - rect.x) - w : x0;
    return { t, x, y, w, h, text: `${t.index != null ? `${t.index} · ` : ''}${t.label} ${pct(t.conf)}`, badge: t.index ?? t.label.slice(0, 3), font, pad: Math.max(5, font * .48) };
  });
  c.save(); c.font = `600 ${font}px "Figtree", system-ui, sans-serif`; c.textBaseline = 'middle';
  const placements = placeLabels(labels.map((a) => ({ box: { x: a.x, y: a.y, w: a.w, h: a.h }, text: a.text, badge: a.badge, font, pad: a.pad })), rect, (text, f) => { c.font = `600 ${f}px "Figtree", system-ui, sans-serif`; return c.measureText(text).width; });
  labels.forEach((a, i) => {
    const t = a.t, box = animated ? t.box : t.target; c.globalAlpha = clamp(animated ? t.alpha : 1, 0, 1);
    if (t.pts) drawHand(c, t, rect, lw, mirror);
    roundRect(c, a.x, a.y, a.w, a.h, Math.min(10, a.w / 6, a.h / 6)); c.fillStyle = `${t.color}30`; c.fill();
    c.lineWidth = lw; c.strokeStyle = t.color; c.setLineDash(t.low ? [6 * scale, 4 * scale] : []); c.stroke(); c.setLineDash([]);
    const p = placements[i]; if (!p) return;
    roundRect(c, p.x, p.y, p.w, p.h, 5 * scale); c.fillStyle = t.color; c.fill();
    const n = parseInt(t.color.slice(1), 16), lum = .2126 * ((n >> 16) & 255) + .7152 * ((n >> 8) & 255) + .0722 * (n & 255);
    c.fillStyle = lum > 145 ? '#0a0e1f' : '#fff'; c.font = `600 ${font}px "Figtree", system-ui, sans-serif`;
    c.save(); c.beginPath(); c.rect(p.x, p.y, p.w, p.h); c.clip();
    let text = p.text, max = p.w - p.h * .55;
    while (text.length > 1 && c.measureText(text).width > max) text = `${text.slice(0, -2)}…`;
    c.fillText(text, p.x + p.h * .28, p.y + p.h / 2); c.restore();
  });
  c.globalAlpha = 1; c.restore();
}
