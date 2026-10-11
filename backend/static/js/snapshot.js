/** Save the current live frame and its client-side overlay as a PNG. */
import { state } from './store.js';
import { toast } from './toast.js';
import { paintOverlay } from './overlay.js';
let video = null;
export function initSnapshot({ button, video: media }) { video = media; button.addEventListener('click', takeSnapshot); }
export async function takeSnapshot() {
  if (!video || !video.videoWidth || !['live', 'paused'].includes(state.camera)) { toast('Start the camera before taking a snapshot.', { kind: 'error' }); return; }
  const cap = 1920, ratio = Math.min(1, cap / Math.max(video.videoWidth, video.videoHeight));
  const width = Math.max(1, Math.round(video.videoWidth * ratio)), height = Math.max(1, Math.round(video.videoHeight * ratio));
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  try {
    if (state.mirror) { ctx.translate(width, 0); ctx.scale(-1, 1); }
    ctx.drawImage(video, 0, 0, width, height); ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (state.overlay) paintOverlay(ctx, { x: 0, y: 0, w: width, h: height }, { scale: ratio, mirror: state.mirror });
    drawCaption(ctx, width, height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('Could not create PNG image');
    const now = new Date(), pad = (n) => String(n).padStart(2, '0');
    const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = `vision-${String(state.mode).replace(/[^a-z0-9_-]/gi, '-')}-${stamp}.png`;
    document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast('Snapshot saved', { kind: 'success' });
  } catch (error) { toast(error.message || 'Could not save the snapshot.', { kind: 'error' }); }
}
function drawCaption(ctx, width, height) {
  const now = new Date(), time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
  const text = `${state.mode} · ${time}`, pad = Math.max(8, width / 100), font = Math.max(12, width / 80);
  ctx.save(); ctx.font = `${font}px "JetBrains Mono", monospace`; const w = ctx.measureText(text).width + pad * 2, h = font + pad * 1.3;
  const x = pad, y = height - h - pad, r = Math.min(6, h / 3);
  ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.fillStyle = 'rgba(10,14,31,.68)'; ctx.fill();
  ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle'; ctx.fillText(text, x + pad, y + h / 2); ctx.restore();
}
