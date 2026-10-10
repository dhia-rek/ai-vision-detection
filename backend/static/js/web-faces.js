/**
 * Browser YuNet + MobileFaceNet expression inference.
 *
 * This module is intentionally standalone so the camera worker can import it without
 * depending on the UI shell. Input is an ImageData in RGB channel order (as produced
 * by Canvas); output matches the Python Detection JSON shape:
 * {class, confidence, bbox:{x1,y1,x2,y2}, face_confidence, top3, scores}.
 *
 * YuNet output decoding, five-point similarity alignment, RGB normalization and FER
 * softmax mirror backend/app/faces.py. Model files are checked into ../models; ORT is
 * loaded lazily from the shared, vendored 1.22.0 runtime when createFaceDetector is called.
 */

export const FACE_EXPRESSIONS = Object.freeze([
  'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised',
]);
export const FACE_MODEL_INFO = Object.freeze({
  yunet: { path: '../models/face_detection_yunet_2023mar.onnx', bytes: 232589, license: 'MIT' },
  fer: { path: '../models/facial_expression_recognition_mobilefacenet_2022july.onnx', bytes: 4790157, license: 'Apache-2.0' },
  ort: { version: '1.22.0', license: 'MIT' },
  totalBytes: 5024481,
});

const STRIDES = [8, 16, 32];
const YUNET_CONF = 0.7;
const NMS_IOU = 0.3;
const TOP_K = 50;
const MIN_FACE = 24;
const ALIGNED = 112;
const REFERENCE = [
  [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
  [41.5493, 92.3655], [70.7299, 92.2041],
];
function loadOrt() {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  const scriptUrl = new URL('../vendor/ort/ort.webgpu.min.js', import.meta.url).href;
  if (typeof importScripts === 'function') {
    importScripts(scriptUrl);
    return Promise.resolve(globalThis.ort);
  }
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = scriptUrl;
    script.onload = () => resolve(globalThis.ort);
    script.onerror = () => reject(new Error('Could not load the vendored ONNX Runtime Web runtime'));
    document.head.append(script);
  });
}

function getOutput(outputs, name) {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`YuNet output missing: ${name}`);
  return tensor.data;
}

function iou(a, b) {
  const iw = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const ih = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const intersection = iw * ih;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return intersection / Math.max(1e-9, areaA + areaB - intersection);
}

function decodeFaces(outputs, netWidth, netHeight, scale, originalWidth, originalHeight) {
  const candidates = [];
  for (const stride of STRIDES) {
    const rows = netWidth / stride, cols = netHeight / stride;
    const cls = getOutput(outputs, `cls_${stride}`);
    const obj = getOutput(outputs, `obj_${stride}`);
    const boxes = getOutput(outputs, `bbox_${stride}`);
    const points = getOutput(outputs, `kps_${stride}`);
    const count = rows * cols;
    for (let i = 0; i < count; i++) {
      const score = cls[i] * obj[i];
      if (score < YUNET_CONF) continue;
      const cx = ((i % rows) + 0.5) * stride, cy = (Math.floor(i / rows) + 0.5) * stride;
      const dx = boxes[i * 4], dy = boxes[i * 4 + 1];
      const width = Math.exp(boxes[i * 4 + 2]) * stride;
      const height = Math.exp(boxes[i * 4 + 3]) * stride;
      const x = (cx + dx * stride) / scale, y = (cy + dy * stride) / scale;
      const landmarks = [];
      for (let p = 0; p < 5; p++) {
        landmarks.push([(cx + points[i * 10 + p * 2] * stride) / scale,
          (cy + points[i * 10 + p * 2 + 1] * stride) / scale]);
      }
      const x1 = Math.max(0, x), y1 = Math.max(0, y);
      const x2 = Math.min(originalWidth, (cx + dx * stride + width) / scale);
      const y2 = Math.min(originalHeight, (cy + dy * stride + height) / scale);
      if (Math.min(x2 - x1, y2 - y1) < MIN_FACE) continue;
      candidates.push({ score, x1, y1, x2, y2, landmarks });
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const candidate of candidates) {
    if (kept.every((other) => iou(candidate, other) <= NMS_IOU)) kept.push(candidate);
    if (kept.length >= TOP_K) break;
  }
  return kept;
}

/** Solve the least-squares similarity transform mapping source landmark to reference. */
function similarityTransform(source) {
  const n = source.length;
  const sx = source.reduce((sum, p) => sum + p[0], 0) / n;
  const sy = source.reduce((sum, p) => sum + p[1], 0) / n;
  const tx = REFERENCE.reduce((sum, p) => sum + p[0], 0) / n;
  const ty = REFERENCE.reduce((sum, p) => sum + p[1], 0) / n;
  let denominator = 0, aNumerator = 0, bNumerator = 0;
  for (let i = 0; i < n; i++) {
    const x = source[i][0] - sx, y = source[i][1] - sy;
    const u = REFERENCE[i][0] - tx, v = REFERENCE[i][1] - ty;
    denominator += x * x + y * y;
    aNumerator += x * u + y * v;
    bNumerator += x * v - y * u;
  }
  const a = aNumerator / denominator, b = bNumerator / denominator;
  return [a, -b, tx - a * sx + b * sy, b, a, ty - b * sx - a * sy];
}

function alignedRgb(image, landmarks) {
  const [a, c, e, b, d, f] = similarityTransform(landmarks);
  const det = a * d - b * c;
  const inv = [d / det, -c / det, (c * f - d * e) / det,
    -b / det, a / det, (b * e - a * f) / det];
  const pixels = new Float32Array(3 * ALIGNED * ALIGNED);
  const { data, width, height } = image;
  for (let y = 0; y < ALIGNED; y++) {
    for (let x = 0; x < ALIGNED; x++) {
      const srcX = inv[0] * x + inv[1] * y + inv[2];
      const srcY = inv[3] * x + inv[4] * y + inv[5];
      if (srcX < 0 || srcY < 0 || srcX >= width - 1 || srcY >= height - 1) continue;
      const x0 = Math.floor(srcX), y0 = Math.floor(srcY), fx = srcX - x0, fy = srcY - y0;
      const dst = y * ALIGNED + x;
      for (let channel = 0; channel < 3; channel++) {
        const i00 = (y0 * width + x0) * 4 + channel;
        const top = data[i00] * (1 - fx) + data[i00 + 4] * fx;
        const bottom = data[i00 + width * 4] * (1 - fx) + data[i00 + (width + 1) * 4] * fx;
        // cv2.cvtColor(RGB,...) followed by float/255 and (x-.5)/.5.
        pixels[channel * ALIGNED * ALIGNED + dst] = (top * (1 - fy) + bottom * fy) / 127.5 - 1;
      }
    }
  }
  return pixels;
}

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < FACE_EXPRESSIONS.length; i++) max = Math.max(max, logits[i]);
  const values = FACE_EXPRESSIONS.map((_, i) => Math.exp(logits[i] - max));
  const sum = values.reduce((total, value) => total + value, 0);
  return values.map((value) => value / sum);
}
function rounded(value) { return Math.round(value * 10000) / 10000; }

/**
 * Lazily create inference sessions. `onProgress({stage, loadedBytes, totalBytes})` may
 * be used to update the per-mode model download indicator. `ort` injection simplifies
 * local testing; production reuses the pinned shared ORT bundle.
 */
export async function createFaceDetector({ ort: ortOverride, onProgress = () => {}, modelBaseUrl } = {}) {
  const ort = ortOverride || await loadOrt();
  ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
  // GitHub Pages is not cross-origin isolated by default, so threaded WASM is unavailable.
  ort.env.wasm.numThreads = 1;
  const base = modelBaseUrl || new URL('../models/', import.meta.url).href;
  const specs = [
    ['yunet', 'face_detection_yunet_2023mar.onnx'],
    ['fer', 'facial_expression_recognition_mobilefacenet_2022july.onnx'],
  ];
  let loaded = 0;
  const totalBytes = FACE_MODEL_INFO.totalBytes;
  const sessions = {};
  for (const [key, file] of specs) {
    onProgress({ stage: key, loadedBytes: loaded, totalBytes, label: key === 'yunet' ? 'Loading face detector' : 'Loading expression model' });
    const modelUrl = new URL(file, base).href;
    const providers = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
    try {
      sessions[key] = await ort.InferenceSession.create(modelUrl, { executionProviders: providers, graphOptimizationLevel: 'all' });
    } catch (error) {
      if (!navigator.gpu) throw error;
      sessions[key] = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    }
    loaded += FACE_MODEL_INFO[key].bytes;
    onProgress({ stage: key, loadedBytes: loaded, totalBytes, label: 'Face models ready' });
  }
  const yunetInput = sessions.yunet.inputNames[0];
  const ferInput = sessions.fer.inputNames[0];
  const ferOutput = sessions.fer.outputNames.includes('label') ? 'label' : sessions.fer.outputNames[0];

  /** Infer detections from RGB ImageData; safe to call from a Web Worker. */
  async function detect(image) {
    if (!image?.data || !Number.isInteger(image.width) || !Number.isInteger(image.height)) {
      throw new TypeError('Face detection expects an ImageData-like {data,width,height}');
    }
    const { width, height, data } = image;
    const scale = Math.min(1, 640 / Math.max(width, height));
    const netWidth = Math.ceil(width * scale / 32) * 32;
    const netHeight = Math.ceil(height * scale / 32) * 32;
    const scaledWidth = width * scale, scaledHeight = height * scale;
    const chw = new Float32Array(3 * netWidth * netHeight);
    // ORT input matches OpenCV's RGB, [0,255] blob. Pad right/bottom with black.
    for (let y = 0; y < scaledHeight; y++) {
      const sy = Math.min(height - 1, Math.floor(y / scale));
      for (let x = 0; x < scaledWidth; x++) {
        const sx = Math.min(width - 1, Math.floor(x / scale));
        const src = (sy * width + sx) * 4, dst = y * netWidth + x;
        chw[dst] = data[src]; chw[netWidth * netHeight + dst] = data[src + 1];
        chw[2 * netWidth * netHeight + dst] = data[src + 2];
      }
    }
    const tensor = new ort.Tensor('float32', chw, [1, 3, netHeight, netWidth]);
    const raw = await sessions.yunet.run({ [yunetInput]: tensor });
    const faces = decodeFaces(raw, netWidth, netHeight, scale, width, height);
    const results = [];
    for (const face of faces) {
      const aligned = alignedRgb(image, face.landmarks);
      const input = new ort.Tensor('float32', aligned, [1, 3, ALIGNED, ALIGNED]);
      const ferResult = await sessions.fer.run({ [ferInput]: input });
      const logits = ferResult[ferOutput].data;
      const probabilities = softmax(logits);
      const ranked = probabilities.map((probability, i) => ({ name: FACE_EXPRESSIONS[i], probability }))
        .sort((a, b) => b.probability - a.probability);
      const scores = Object.fromEntries(FACE_EXPRESSIONS.map((name, i) => [name, rounded(probabilities[i])]));
      results.push({
        class: ranked[0].name,
        confidence: rounded(probabilities[FACE_EXPRESSIONS.indexOf(ranked[0].name)]),
        bbox: { x1: Math.round(face.x1 * 10) / 10, y1: Math.round(face.y1 * 10) / 10,
          x2: Math.round(face.x2 * 10) / 10, y2: Math.round(face.y2 * 10) / 10 },
        face_confidence: rounded(face.score),
        top3: Object.fromEntries(ranked.slice(0, 3).map(({ name, probability }) => [name, rounded(probability)])),
        scores,
      });
    }
    return results;
  }
  return { detect, dispose: async () => Promise.all(Object.values(sessions).map((session) => session.release())) };
}
