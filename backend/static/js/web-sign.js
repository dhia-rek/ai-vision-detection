/** Browser-only MediaPipe hands + ASL classifier. No image is sent off-device.
 *
 * `createSignEngine()` lazy-loads the pinned MediaPipe Tasks Vision runtime and its
 * gesture-recognizer task. `detect()` returns the same detection object shape as the
 * Python API (`class`, `confidence`, `bbox`, optional `points` and extras).
 */

export const TASKS_VISION_VERSION = '0.10.21';
export const TASKS_VISION_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/vision_bundle.mjs`;
export const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
export const HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';
export const ASL_WEIGHTS_URL = new URL('../models/asl_letters.json', import.meta.url).href;

const GESTURES = {
  Thumb_Up: 'thumbs up (yes)', Thumb_Down: 'thumbs down (no)',
  Open_Palm: 'open palm (hello / stop)', ILoveYou: 'I love you',
  Victory: 'victory / peace', Pointing_Up: 'pointing up (wait)',
  Closed_Fist: 'closed fist',
};
const TIPS = [4, 8, 12, 16, 20];

/** Exact port of backend/app/sign_features.py (`FEATURE_VERSION == 1`). */
export function landmarkFeatures(points, handedness = null) {
  if (!points || points.length !== 21) throw new TypeError('Expected 21 hand landmarks');
  const p = points.map((q) => [Number(q.x ?? q[0]), Number(q.y ?? q[1]), Number(q.z ?? q[2])]);
  // Copy before mutation (NumPy broadcasts the original wrist row across all rows).
  const origin = [...p[0]];
  for (const q of p) {
    q[0] -= origin[0]; q[1] -= origin[1]; q[2] -= origin[2];
    if (handedness && String(handedness).toLowerCase().startsWith('l')) q[0] = -q[0];
  }
  let scale = 0;
  for (const q of p) scale = Math.max(scale, Math.hypot(q[0], q[1]));
  const divisor = scale > 1e-9 ? scale : 1;
  for (const q of p) { q[0] /= divisor; q[1] /= divisor; q[2] /= divisor; }
  const out = [];
  for (let i = 1; i < 21; i++) out.push(...p[i]);
  for (let i = 0; i < TIPS.length; i++) for (let j = i + 1; j < TIPS.length; j++) {
    const a = p[TIPS[i]], b = p[TIPS[j]];
    out.push(Math.hypot(a[0] - b[0], a[1] - b[1]));
  }
  // NumPy's reference implementation returns a float32 feature vector.
  return out.map(Math.fround);
}

function validateWeights(model) {
  if (!model || model.format !== 'dhia-asl-mlp' || model.format_version !== 1 || model.feature_version !== 1) {
    throw new Error('Unsupported ASL weight format or landmark feature version');
  }
  const n = model.input_size;
  if (n !== 70 || !Array.isArray(model.classes) || !model.classes.length ||
      !Array.isArray(model.mean) || model.mean.length !== n ||
      !Array.isArray(model.std) || model.std.length !== n ||
      !Array.isArray(model.layers) || !model.layers.length) throw new Error('Invalid ASL weight dimensions');
  let width = n;
  for (const [i, layer] of model.layers.entries()) {
    if (layer.input_size !== width || !Number.isInteger(layer.output_size) || layer.output_size < 1 ||
        !Array.isArray(layer.weights) || layer.weights.length !== width * layer.output_size ||
        !Array.isArray(layer.bias) || layer.bias.length !== layer.output_size ||
        !['relu', 'linear'].includes(layer.activation)) throw new Error(`Invalid ASL layer ${i + 1}`);
    width = layer.output_size;
  }
  if (width !== model.classes.length || model.layers.at(-1).activation !== 'linear') {
    throw new Error('ASL output layer does not match classes');
  }
  if (model.std.some((v) => !Number.isFinite(v) || v === 0)) throw new Error('ASL standard deviations must be finite and nonzero');
  return model;
}

export async function loadAslWeights(url = ASL_WEIGHTS_URL, fetchImpl = fetch) {
  const response = await fetchImpl(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Could not load ASL weights (${response.status})`);
  return validateWeights(await response.json());
}

/** Inference supports any dense-layer depth/width described by the JSON export schema. */
export function classifyLandmarks(points, handedness, model) {
  validateWeights(model);
  let x = landmarkFeatures(points, handedness).map((v, i) => Math.fround(Math.fround(v - model.mean[i]) / model.std[i]));
  for (const layer of model.layers) {
    const y = new Array(layer.output_size).fill(0);
    for (let col = 0; col < layer.output_size; col++) {
      let sum = Math.fround(layer.bias[col]);
      for (let row = 0; row < layer.input_size; row++) {
        sum = Math.fround(sum + Math.fround(x[row] * layer.weights[row * layer.output_size + col]));
      }
      y[col] = layer.activation === 'relu' ? Math.max(0, sum) : sum;
    }
    x = y;
  }
  const max = Math.max(...x), exps = x.map((v) => Math.fround(Math.exp(Math.fround(v - max))));
  const total = exps.reduce((a, b) => Math.fround(a + b), 0);
  const ranked = exps.map((v, i) => [model.classes[i], Math.fround(v / total)]).sort((a, b) => b[1] - a[1]).slice(0, 3);
  return { letter: ranked[0][0], confidence: ranked[0][1], top3: ranked.map(([label, score]) => [label, score]) };
}

function toDetection(points, width, height, label, confidence, extra) {
  const pixelPoints = points.map((p) => [p.x * width, p.y * height]);
  const xs = pixelPoints.map(([x]) => x), ys = pixelPoints.map(([, y]) => y);
  const pad = 0.1 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const round = (n, digits = 1) => Number(n.toFixed(digits));
  const bbox = {
    x1: round(Math.max(0, Math.min(...xs) - pad)), y1: round(Math.max(0, Math.min(...ys) - pad)),
    x2: round(Math.min(width, Math.max(...xs) + pad)), y2: round(Math.min(height, Math.max(...ys) + pad)),
  };
  const det = {
    class: label, confidence: round(confidence, 4), bbox,
    points: pixelPoints.map(([x, y]) => [round(x), round(y)]), ...extra,
  };
  return det;
}

/**
 * Create a lazy hand-inference engine. `onProgress({stage, loaded, total})` is
 * suitable for a loading indicator; browser HTTP caching handles repeat visits.
 */
export async function createSignEngine({ mode = 'spell', maxHands = 2, onProgress = () => {},
  fetchImpl = fetch, runtimeUrl = TASKS_VISION_CDN, wasmRoot = WASM_ROOT, modelUrl = HAND_MODEL_URL,
  weightsUrl = ASL_WEIGHTS_URL } = {}) {
  if (!['spell', 'gestures'].includes(mode)) throw new Error('Sign mode must be spell or gestures');
  onProgress({ stage: 'runtime', loaded: 0, total: 0 });
  const vision = await import(/* @vite-ignore */ runtimeUrl);
  const [visionFiles, weights] = await Promise.all([
    vision.FilesetResolver.forVisionTasks(wasmRoot),
    loadAslWeights(weightsUrl, fetchImpl),
  ]);
  onProgress({ stage: 'model', loaded: 0, total: 0 });
  const options = { runningMode: 'VIDEO', numHands: maxHands, minHandDetectionConfidence: 0.5 };
  let recognizer;
  try {
    recognizer = await vision.GestureRecognizer.createFromOptions(visionFiles, {
      ...options, baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' },
    });
  } catch {
    // Unsupported GPU delegates (common in headless Chromium and older phones)
    // must not make the camera mode unusable.
    recognizer = await vision.GestureRecognizer.createFromOptions(visionFiles, {
      ...options, baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
    });
  }
  onProgress({ stage: 'ready', loaded: 1, total: 1 });
  return {
    mode,
    detect(source, timestamp = performance.now(), selectedMode = mode) {
      const width = source.videoWidth || source.width, height = source.videoHeight || source.height;
      if (!width || !height) return [];
      const result = recognizer.recognizeForVideo(source, timestamp);
      return (result.landmarks || []).map((landmarks, i) => {
        const handed = result.handedness?.[i]?.[0]?.categoryName ?? null;
        const gesture = result.gestures?.[i]?.[0] ?? null;
        let label, confidence, extra = { handedness: handed };
        if (selectedMode === 'spell') {
          const prediction = classifyLandmarks(landmarks, handed, weights);
          label = prediction.letter; confidence = prediction.confidence;
          extra.top3 = Object.fromEntries(prediction.top3.map(([k, v]) => [k, Number(v.toFixed(4))]));
        } else {
          const raw = gesture?.categoryName;
          label = raw && GESTURES[raw] ? GESTURES[raw] : 'no gesture';
          confidence = gesture?.score ?? 0;
          extra.gesture = raw ?? 'None';
        }
        if (confidence < 0.5) extra.low_confidence = true;
        return toDetection(landmarks, width, height, label, confidence, extra);
      });
    },
    close() { recognizer.close(); },
  };
}
