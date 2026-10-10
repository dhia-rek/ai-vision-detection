/**
 * Browser-only MediaPipe hands + ASL classifier. No image leaves the device.
 *
 * `createSignEngine()` lazily loads the vendored MediaPipe Tasks Vision runtime (Apache-2.0, pinned
 * 0.10.21) and its Gesture Recognizer task, plus the ASL classifier weights. `detect()` returns the
 * same detection objects as backend/app/sign.py (`class`, `confidence`, `bbox`, `points`, `top3`,
 * `scores`, `handedness`, `landmarks`), so the UI and the parity tests treat both alike.
 */
import { fetchModel, megabytes } from './web-fetch.js';
import { parseWeights, predictProba, LetterSmoother, LETTERS } from './web-asl.js';

export const TASKS_VISION_VERSION = '0.10.21';
const VENDOR = new URL('../vendor/mediapipe/', import.meta.url);
export const RUNTIME_URL = new URL('vision_bundle.mjs', VENDOR).href;
export const HAND_MODEL_URL = new URL('gesture_recognizer.task', VENDOR).href;
export const HAND_MODEL_BYTES = 8373440;
export const ASL_WEIGHTS_URL = new URL('../models/asl_letters_web.json', import.meta.url).href;

const GESTURES = {
  Thumb_Up: 'thumbs up (yes)', Thumb_Down: 'thumbs down (no)',
  Open_Palm: 'open palm (hello / stop)', ILoveYou: 'I love you',
  Victory: 'victory / peace', Pointing_Up: 'pointing up (wait)',
  Closed_Fist: 'closed fist',
};
const round = (n, digits) => Number(n.toFixed(digits));

function toDetection(landmarks, width, height, label, confidence, extra) {
  const xs = landmarks.map((p) => p.x * width), ys = landmarks.map((p) => p.y * height);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 0.1 * Math.max(maxX - minX, maxY - minY);
  return {
    class: label, confidence: round(confidence, 4),
    bbox: { x1: Math.max(0, minX - pad), y1: Math.max(0, minY - pad), x2: Math.min(width, maxX + pad), y2: Math.min(height, maxY + pad) },
    points: xs.map((x, i) => [x, ys[i]]),
    ...extra,
  };
}

/**
 * @param {object} [options]
 * @param {'VIDEO'|'IMAGE'} [options.runningMode] VIDEO tracks across frames (live camera); IMAGE treats
 *   every call independently, like the Python backend (used by the parity tests).
 * @param {(p: {stage: string, detail: string}) => void} [options.onProgress]
 */
export async function createSignEngine({ runningMode = 'VIDEO', maxHands = 2, onProgress = () => {},
  delegate = 'CPU', runtimeUrl = RUNTIME_URL, wasmRoot = VENDOR.href, modelUrl = HAND_MODEL_URL, weightsUrl = ASL_WEIGHTS_URL } = {}) {
  onProgress({ stage: 'runtime', detail: 'Loading MediaPipe' });
  const vision = await import(/* @vite-ignore */ runtimeUrl);
  // MediaPipe's wasm goes through the download-once cache too (served to it as a blob URL).
  const files = await vision.FilesetResolver.forVisionTasks(wasmRoot);
  const wasmBuffer = await fetchModel(files.wasmBinaryPath, ({ loaded, total, cached }) => onProgress({
    stage: 'runtime', detail: cached ? 'Loading MediaPipe' : `MediaPipe · ${megabytes(loaded)} / ${megabytes(total)} MB`, loaded, total }));
  files.wasmBinaryPath = URL.createObjectURL(new Blob([wasmBuffer], { type: 'application/wasm' }));
  const [weightsDoc] = await Promise.all([
    fetchModel(weightsUrl).then((buffer) => JSON.parse(new TextDecoder().decode(buffer))),
  ]);
  const weights = parseWeights(weightsDoc);
  const modelBuffer = await fetchModel(modelUrl, ({ loaded, total, cached }) => onProgress({
    stage: 'model', detail: cached ? `Hand model · ${megabytes(total)} MB (saved on this device)` : `Hand model · ${megabytes(loaded)} / ${megabytes(total || HAND_MODEL_BYTES)} MB`, loaded, total,
  }));
  onProgress({ stage: 'compile', detail: 'Preparing the model' });
  const options = { runningMode, numHands: maxHands, minHandDetectionConfidence: 0.5 };
  const create = (delegate) => vision.GestureRecognizer.createFromOptions(files, {
    ...options, baseOptions: { modelAssetBuffer: new Uint8Array(modelBuffer), delegate },
  });
  // XNNPACK (CPU) is the default: it is deterministic and works everywhere, while the WebGL delegate is
  // fast on good GPUs but flaky on some phones and unusably slow without a real GPU. `delegate: 'GPU'` opts in.
  let recognizer, device = delegate;
  try { recognizer = await create(delegate); } catch { recognizer = await create('CPU'); device = 'CPU'; }
  onProgress({ stage: 'ready', detail: device });

  const smoother = new LetterSmoother();
  let lastTimestamp = -1;
  return {
    device,
    reset() { smoother.reset(); },
    /**
     * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement|ImageBitmap} source
     * @param {'spell'|'gestures'} mode
     * @param {{timestamp?: number, smooth?: boolean}} [opts] `smooth` (VIDEO default) applies temporal smoothing to the main hand
     */
    detect(source, mode = 'spell', { timestamp = performance.now(), smooth = runningMode === 'VIDEO' } = {}) {
      const width = source.videoWidth || source.naturalWidth || source.width, height = source.videoHeight || source.naturalHeight || source.height;
      if (!width || !height) return [];
      let result;
      if (runningMode === 'VIDEO') {
        lastTimestamp = Math.max(timestamp, lastTimestamp + 1);   // MediaPipe requires strictly increasing timestamps
        result = recognizer.recognizeForVideo(source, lastTimestamp);
      } else result = recognizer.recognize(source);
      const hands = (result.landmarks || []).map((landmarks, i) => {
        const handedness = result.handedness?.[i]?.[0]?.categoryName ?? null;
        const gesture = result.gestures?.[i]?.[0] ?? null;
        const extra = { handedness, landmarks: landmarks.map((p) => [round(p.x, 4), round(p.y, 4), round(p.z, 4)]) };
        let label, confidence;
        if (mode === 'spell') {
          const proba = predictProba(weights, landmarks, width / height);
          const order = proba.map((_, k) => k).sort((a, b) => proba[b] - proba[a]);
          label = weights.classes[order[0]]; confidence = proba[order[0]];
          extra.top3 = Object.fromEntries(order.slice(0, 3).map((k) => [weights.classes[k], round(proba[k], 4)]));
          extra.scores = Object.fromEntries(weights.classes.map((c, k) => [c, round(proba[k], 3)]));
        } else {
          const raw = gesture?.categoryName;
          label = raw && GESTURES[raw] ? GESTURES[raw] : 'no gesture';
          confidence = gesture?.score ?? 0;
          extra.gesture = raw ?? 'None';
        }
        const detection = toDetection(landmarks, width, height, label, confidence, extra);
        detection._proba = mode === 'spell' ? extra.scores : null;
        return detection;
      });
      if (mode === 'spell' && smooth) {
        const main = hands.reduce((a, d) => (!a || d.confidence > a.confidence ? d : a), null);
        const [letter, smoothed] = smoother.feed(main ? main._proba : null, timestamp / 1000);
        if (main) { main.class = letter ?? '–'; main.confidence = round(smoothed, 4); }
      }
      for (const d of hands) { delete d._proba; if (d.confidence < 0.5) d.low_confidence = true; }
      return hands;
    },
    close() { recognizer.close(); },
  };
}
