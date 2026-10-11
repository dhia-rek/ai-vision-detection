/* YOLO11n inference off the UI thread. The only input is a transferred camera bitmap; nothing is uploaded. */
import { loadOrt, createSession } from './web-ort.js';
import { fetchModel, megabytes } from './web-fetch.js';

const MODEL_ROOT = new URL('../models/objects/', import.meta.url);
const CONF = 0.25, NMS_IOU = 0.7, MAX_DET = 300;
const STRIDE = 32;
let ort, session, size = 0, labels = null, backend = '';
let canvas = null, context = null, tensor = null;

const report = (stage, detail, extra = {}) => self.postMessage({ type: 'progress', stage, detail, ...extra });

/** `requested` is 320, 640 or 'auto' (640 only when the GPU build is in use, 320 otherwise). */
async function init(requested) {
  if (session) return;
  report('runtime', 'Loading ONNX Runtime');
  const runtime = await loadOrt({ onProgress: (p) => report('runtime', p.detail, p) });
  ort = runtime.ort;
  const buffer = await fetchModel(new URL('yolo11n.onnx', MODEL_ROOT), ({ loaded, total, cached }) =>
    report('model', cached ? `YOLO11n · ${megabytes(total)} MB (saved on this device)` : `YOLO11n · ${megabytes(loaded)} / ${megabytes(total)} MB`,
      { loaded, total }));
  if (!labels) labels = await (await fetch(new URL('labels.json', MODEL_ROOT))).json();
  report('compile', 'Preparing the model');
  ({ session, backend } = await createSession(runtime, buffer));
  size = requested === 'auto' ? (backend === 'WebGPU' ? 640 : 320) : requested;
}

/**
 * Letterbox exactly like Ultralytics' rectangular inference (`LetterBox(auto=True)`): scale to fit
 * `size`, pad only up to the next multiple of 32 with grey 114, centred. The network is dynamic-shape,
 * so a 4:3 camera costs 480 rows instead of 640. Fills the shared input tensor.
 */
function prepare(bitmap) {
  const { width, height } = bitmap;
  const scale = Math.min(size / width, size / height);
  const drawnWidth = Math.round(width * scale), drawnHeight = Math.round(height * scale);
  const padW = (size - drawnWidth) % STRIDE / 2, padH = (size - drawnHeight) % STRIDE / 2;
  const left = Math.round(padW - 0.1), top = Math.round(padH - 0.1);
  const netW = drawnWidth + left + Math.round(padW + 0.1), netH = drawnHeight + top + Math.round(padH + 0.1);
  if (!canvas || canvas.width !== netW || canvas.height !== netH) {
    canvas = new OffscreenCanvas(netW, netH);
    context = canvas.getContext('2d', { willReadFrequently: true });
    tensor = new Float32Array(3 * netW * netH);
  }
  context.fillStyle = 'rgb(114,114,114)';
  context.fillRect(0, 0, netW, netH);
  context.drawImage(bitmap, left, top, drawnWidth, drawnHeight);
  bitmap.close();
  const rgba = context.getImageData(0, 0, netW, netH).data, pixels = netW * netH;
  for (let i = 0, j = 0; i < pixels; i++, j += 4) {
    tensor[i] = rgba[j] / 255; tensor[pixels + i] = rgba[j + 1] / 255; tensor[2 * pixels + i] = rgba[j + 2] / 255;
  }
  return { width, height, scale, padX: left, padY: top, netW, netH };
}

function overlap(a, b) {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1), h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

/** Decode the (1, 4+classes, anchors) output, then class-aware NMS (what Ultralytics does by default). */
function decode(output, frame) {
  const { data } = output, channels = output.dims[1], anchors = output.dims[2];
  if (channels !== labels.length + 4) throw new Error(`Unexpected YOLO output shape: ${output.dims}`);
  const clampX = (v) => Math.max(0, Math.min(frame.width, (v - frame.padX) / frame.scale));
  const clampY = (v) => Math.max(0, Math.min(frame.height, (v - frame.padY) / frame.scale));
  const candidates = [];
  for (let i = 0; i < anchors; i++) {
    let best = CONF, classId = -1;
    for (let c = 4; c < channels; c++) {
      const score = data[c * anchors + i];
      if (score > best) { best = score; classId = c - 4; }
    }
    if (classId < 0) continue;
    const cx = data[i], cy = data[anchors + i], w = data[2 * anchors + i], h = data[3 * anchors + i];
    candidates.push({ classId, confidence: best, x1: clampX(cx - w / 2), y1: clampY(cy - h / 2), x2: clampX(cx + w / 2), y2: clampY(cy + h / 2) });
  }
  candidates.sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const box of candidates) {
    if (box.x2 <= box.x1 || box.y2 <= box.y1) continue;
    if (kept.some((prior) => prior.classId === box.classId && overlap(prior, box) > NMS_IOU)) continue;
    kept.push(box);
    if (kept.length >= MAX_DET) break;
  }
  return kept.map(({ classId, confidence, x1, y1, x2, y2 }) => ({ class: labels[classId], confidence, bbox: { x1, y1, x2, y2 } }));
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      await init(data.size);
      report('ready', `${backend} · ${size}px`);
      self.postMessage({ type: 'result', id: data.id, data: { backend, size } });
    } else if (data.type === 'frame') {
      const start = performance.now();
      const frame = prepare(data.bitmap);
      const prepared = performance.now();
      const outputs = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, frame.netH, frame.netW]) });
      const ran = performance.now();
      const detections = decode(outputs[session.outputNames[0]], frame);
      const end = performance.now();
      self.postMessage({ type: 'result', id: data.id, data: {
        detections, image_size: { width: frame.width, height: frame.height }, mode: 'objects',
        model_used: `yolo11n-${size}`, device: backend, inference_ms: end - start,
        timing_ms: { prepare: prepared - start, model: ran - prepared, decode: end - ran },
      } });
    }
  } catch (error) {
    if (data.bitmap) data.bitmap.close();
    self.postMessage({ type: 'error', id: data.id, message: error?.message || String(error) });
  }
};
