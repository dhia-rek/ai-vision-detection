/**
 * Browser YuNet + MobileFaceNet expression inference.
 *
 * This module is intentionally standalone so the camera worker can import it without
 * depending on the UI shell. Input is an ImageData in RGB channel order (as produced
 * by Canvas); output matches the Python Detection JSON shape:
 * {class, confidence, bbox:{x1,y1,x2,y2}, face_confidence, top3, scores}.
 *
 * YuNet output decoding, five-point similarity alignment, RGB normalization and FER
 * softmax mirror backend/app/faces.py. Model files are checked into ../models; the ORT runtime
 * comes from web-ort.js (the caller passes it in, so this module also runs in Node/tests).
 */
import { fetchModel } from './web-fetch.js';

export const FACE_EXPRESSIONS = Object.freeze([
  'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised',
]);
export const FACE_MODEL_INFO = Object.freeze({
  yunet: { file: 'face_detection_yunet_2023mar_dynamic.onnx', bytes: 229582, license: 'MIT' },
  fer: { file: 'facial_expression_recognition_mobilefacenet_2022july.onnx', bytes: 4791892, license: 'Apache-2.0' },
  totalBytes: 229582 + 4791892,
});

const STRIDES = [8, 16, 32];
const YUNET_CONF = 0.7;
const NMS_IOU = 0.3;
const TOP_K = 50;
const MAX_EXPRESSIONS = 4;   // expressions are computed for the 4 most confident faces (the UI shows 4)
const MIN_FACE = 24;
const ALIGNED = 112;
const REFERENCE = [
  [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
  [41.5493, 92.3655], [70.7299, 92.2041],
];
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

/** Decode YuNet's three stride heads exactly like cv2.FaceDetectorYN (face_detect.cpp). */
function decodeFaces(outputs, netWidth, netHeight) {
  const candidates = [];
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  for (const stride of STRIDES) {
    const cols = netWidth / stride, rows = netHeight / stride;
    const cls = getOutput(outputs, `cls_${stride}`);
    const obj = getOutput(outputs, `obj_${stride}`);
    const boxes = getOutput(outputs, `bbox_${stride}`);
    const points = getOutput(outputs, `kps_${stride}`);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const score = Math.sqrt(clamp01(cls[i]) * clamp01(obj[i]));
        if (score < YUNET_CONF) continue;
        const cx = (c + boxes[i * 4]) * stride, cy = (r + boxes[i * 4 + 1]) * stride;
        const w = Math.exp(boxes[i * 4 + 2]) * stride, h = Math.exp(boxes[i * 4 + 3]) * stride;
        const landmarks = [];
        for (let p = 0; p < 5; p++) {
          landmarks.push([(points[i * 10 + p * 2] + c) * stride, (points[i * 10 + p * 2 + 1] + r) * stride]);
        }
        candidates.push({ score, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, landmarks });
      }
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
 * Create the two inference sessions. `runtime` is `{ort, providers, backend}` from web-ort.js and
 * `createSession(runtime, buffer, options)` builds a session (with the WASM fallback).
 * `onProgress({stage, loadedBytes, totalBytes, label})` drives the model download indicator.
 */
export async function createFaceDetector({ runtime, createSession, onProgress = () => {}, modelBaseUrl } = {}) {
  const ort = runtime.ort;
  const base = modelBaseUrl || new URL('../models/', import.meta.url).href;
  const totalBytes = FACE_MODEL_INFO.totalBytes;
  const sessions = {};
  let done = 0;
  for (const [key, label] of [['yunet', 'Face detector'], ['fer', 'Expression model']]) {
    const info = FACE_MODEL_INFO[key];
    const buffer = await fetchModel(new URL(info.file, base), ({ loaded }) =>
      onProgress({ stage: key, loadedBytes: done + loaded, totalBytes, label }));
    done += info.bytes;
    sessions[key] = (await createSession(runtime, buffer, { graphOptimizationLevel: 'all' })).session;
  }
  onProgress({ stage: 'ready', loadedBytes: totalBytes, totalBytes, label: 'Face models ready' });
  const yunetInput = sessions.yunet.inputNames[0];
  const ferInput = sessions.fer.inputNames[0];
  const ferOutput = sessions.fer.outputNames.includes('label') ? 'label' : sessions.fer.outputNames[0];

  /** Infer detections from RGB ImageData; safe to call from a Web Worker. */
  async function detect(image, { full = image, timing = null } = {}) {
    const t0 = performance.now();
    if (!image?.data || !Number.isInteger(image.width) || !Number.isInteger(image.height)) {
      throw new TypeError('Face detection expects an ImageData-like {data,width,height}');
    }
    const { width, height, data } = image;
    // YuNet is fully convolutional (the shipped graph declares dynamic H/W): feed the frame as is,
    // padded right/bottom with black to a multiple of 32, as OpenCV does.
    const netWidth = Math.ceil(width / 32) * 32, netHeight = Math.ceil(height / 32) * 32;
    const plane = netWidth * netHeight;
    const chw = new Float32Array(3 * plane);
    // Same blob as the Python backend: cv2.FaceDetectorYN gets the BGR image unchanged ([0,255], no mean).
    for (let y = 0; y < height; y++) {
      for (let x = 0, src = y * width * 4, dst = y * netWidth; x < width; x++, src += 4, dst++) {
        chw[dst] = data[src + 2]; chw[plane + dst] = data[src + 1]; chw[2 * plane + dst] = data[src];
      }
    }
    const tensor = new ort.Tensor('float32', chw, [1, 3, netHeight, netWidth]);
    const t1 = performance.now();
    const raw = await sessions.yunet.run({ [yunetInput]: tensor });
    const t2 = performance.now();
    // YuNet runs on `image` (<= 640 px); boxes, landmarks and the aligned crops use `full`, the same
    // frame at higher resolution (the Python backend works on the full frame as well).
    const k = full.width / width;
    const faces = decodeFaces(raw, netWidth, netHeight).map((face) => ({
      score: face.score, x1: face.x1 * k, y1: face.y1 * k, x2: face.x2 * k, y2: face.y2 * k,
      landmarks: face.landmarks.map(([x, y]) => [x * k, y * k]),
    })).filter((face) => Math.min(face.x2 - face.x1, face.y2 - face.y1) >= MIN_FACE);
    const results = [];
    let alignMs = 0, ferMs = 0;
    for (const face of faces.slice(0, MAX_EXPRESSIONS)) {
      const a0 = performance.now();
      const aligned = alignedRgb(full, face.landmarks);
      const input = new ort.Tensor('float32', aligned, [1, 3, ALIGNED, ALIGNED]);
      const a1 = performance.now();
      const ferResult = await sessions.fer.run({ [ferInput]: input });
      alignMs += a1 - a0; ferMs += performance.now() - a1;
      const logits = ferResult[ferOutput].data;
      const probabilities = softmax(logits);
      const ranked = probabilities.map((probability, i) => ({ name: FACE_EXPRESSIONS[i], probability }))
        .sort((a, b) => b.probability - a.probability);
      const scores = Object.fromEntries(FACE_EXPRESSIONS.map((name, i) => [name, rounded(probabilities[i])]));
      results.push({
        class: ranked[0].name,
        confidence: rounded(probabilities[FACE_EXPRESSIONS.indexOf(ranked[0].name)]),
        bbox: { x1: Math.round(Math.max(0, face.x1) * 10) / 10, y1: Math.round(Math.max(0, face.y1) * 10) / 10,
          x2: Math.round(Math.min(full.width, face.x2) * 10) / 10, y2: Math.round(Math.min(full.height, face.y2) * 10) / 10 },
        face_confidence: rounded(face.score),
        top3: Object.fromEntries(ranked.slice(0, 3).map(({ name, probability }) => [name, rounded(probability)])),
        scores,
      });
    }
    if (timing) Object.assign(timing, { prepare: t1 - t0, detector: t2 - t1, decode: performance.now() - t2 - alignMs - ferMs, align: alignMs, expression: ferMs });
    return results;
  }
  return { detect, dispose: async () => Promise.all(Object.values(sessions).map((session) => session.release())) };
}
