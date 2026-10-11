/* YuNet + MobileFaceNet expression inference off the UI thread. Input: a transferred camera bitmap. */
import { loadOrt, createSession } from './web-ort.js';
import { createFaceDetector } from './web-faces.js';

let detectMaxSide = 640;         // longest side of the YuNet input; tests raise it to the source size
let alignMaxSide = 1280;         // resolution of the aligned expression crops and the reported boxes
let detector = null, backend = '';
const surfaces = {};

const report = (stage, detail, extra = {}) => self.postMessage({ type: 'progress', stage, detail, ...extra });

/** RGBA pixels of `bitmap` scaled to w x h, on a reusable canvas per purpose. */
function pixels(bitmap, w, h, purpose) {
  let surface = surfaces[purpose];
  if (!surface || surface.canvas.width !== w || surface.canvas.height !== h) {
    const canvas = new OffscreenCanvas(w, h);
    surface = surfaces[purpose] = { canvas, context: canvas.getContext('2d', { willReadFrequently: true }) };
  }
  surface.context.drawImage(bitmap, 0, 0, w, h);
  return surface.context.getImageData(0, 0, w, h);
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      if (data.detectMaxSide) detectMaxSide = data.detectMaxSide;
      if (data.alignMaxSide) alignMaxSide = data.alignMaxSide;
      report('runtime', 'Loading ONNX Runtime');
      const runtime = await loadOrt({ onProgress: (p) => report('runtime', p.detail, p) });
      detector = await createFaceDetector({
        runtime, createSession,
        onProgress: ({ label, loadedBytes, totalBytes }) => report('model', `${label} · ${(loadedBytes / 1e6).toFixed(1)} / ${(totalBytes / 1e6).toFixed(1)} MB`, { loaded: loadedBytes, total: totalBytes }),
      });
      backend = runtime.backend;
      report('ready', backend);
      self.postMessage({ type: 'result', id: data.id, data: { backend } });
    } else if (data.type === 'frame') {
      const start = performance.now();
      const { width, height } = data.bitmap;
      const fullScale = Math.min(1, alignMaxSide / Math.max(width, height));
      const fullImage = pixels(data.bitmap, Math.round(width * fullScale), Math.round(height * fullScale), 'full');
      const detectScale = Math.min(1, detectMaxSide / Math.max(fullImage.width, fullImage.height));
      const detectImage = detectScale === 1 ? fullImage
        : pixels(data.bitmap, Math.round(fullImage.width * detectScale), Math.round(fullImage.height * detectScale), 'detect');
      data.bitmap.close();
      const timing = {};
      const detections = await detector.detect(detectImage, { full: fullImage, timing });
      self.postMessage({ type: 'result', id: data.id, data: {
        detections, image_size: { width: fullImage.width, height: fullImage.height }, mode: 'faces',
        model_used: 'yunet+mobilefacenet-fer', device: backend, inference_ms: performance.now() - start, timing_ms: timing,
      } });
    }
  } catch (error) {
    if (data.bitmap) data.bitmap.close();
    self.postMessage({ type: 'error', id: data.id, message: error?.message || String(error) });
  }
};
