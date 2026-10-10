/**
 * ASL letter classifier in plain JavaScript: a port of backend/app/sign_features.py (feature
 * version 3, 287 features), backend/app/asl_classifier.py (mirror-averaged MLP) and
 * `LetterSmoother` from backend/app/sign.py. No DOM, no network: it runs in a page, a worker or Node.
 *
 * The weights file is the `dhia-asl-mlp` JSON written by scripts/export_asl_web.py. Layer count and
 * widths come from the file, so retrained weights with the same `feature_version` drop in unchanged.
 */

export const FEATURE_VERSION = 3;
export const N_FEATURES = 287;
export const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

const TIPS = [4, 8, 12, 16, 20];
const PALM = [5, 9, 13, 17];
const PAIRS = [];
for (let i = 0; i < 21; i++) for (let j = i + 1; j < 21; j++) PAIRS.push([i, j]);

const asXyz = (q) => (Array.isArray(q) ? [q[0], q[1], q[2] ?? 0] : [q.x, q.y, q.z ?? 0]);

/**
 * Features for one hand, identical to `landmark_features(points, aspect, mirror)`.
 * @param {Array<{x:number,y:number,z:number}|number[]>} points 21 MediaPipe landmarks (x, y normalised).
 * @param {number} aspect image width / height
 * @param {boolean} mirror flip the hand left <-> right before computing features
 * @returns {Float32Array} 287 values
 */
export function landmarkFeatures(points, aspect = 1, mirror = false) {
  if (!points || points.length !== 21) throw new TypeError('Expected 21 hand landmarks');
  const p = points.map((q) => {
    const [x, y, z] = asXyz(q);
    return [(mirror ? 1 - x : x) * aspect, y, z * aspect];
  });
  const wrist = p[0].slice();
  for (const q of p) { q[0] -= wrist[0]; q[1] -= wrist[1]; q[2] -= wrist[2]; }
  let scale = 0;
  for (const i of PALM) scale += Math.hypot(p[i][0], p[i][1], p[i][2]);
  scale /= PALM.length;
  const divisor = scale > 1e-9 ? scale : 1;
  for (const q of p) { q[0] /= divisor; q[1] /= divisor; q[2] /= divisor; }
  // In-plane rotation: wrist -> middle knuckle (landmark 9) points up (image y grows downwards).
  const angle = Math.atan2(p[9][0], -p[9][1]);
  const c = Math.cos(angle), s = Math.sin(angle);
  for (const q of p) { const x = q[0], y = q[1]; q[0] = c * x + s * y; q[1] = -s * x + c * y; }

  const out = new Float32Array(N_FEATURES);
  let k = 0;
  for (let i = 1; i < 21; i++) { out[k++] = p[i][0]; out[k++] = p[i][1]; out[k++] = p[i][2]; }
  out[k++] = c; out[k++] = s;
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) {
    const a = p[TIPS[i]], b = p[TIPS[j]];
    out[k++] = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  }
  for (const t of TIPS) out[k++] = Math.hypot(p[t][0], p[t][1], p[t][2]);
  for (const [i, j] of PAIRS) out[k++] = Math.hypot(p[i][0] - p[j][0], p[i][1] - p[j][1]);
  return out;
}

/** Validate a `dhia-asl-mlp` document and return a ready-to-run model (typed arrays). */
export function parseWeights(doc) {
  if (!doc || doc.format !== 'dhia-asl-mlp' || doc.format_version !== 1) {
    throw new Error('Unsupported ASL weight format');
  }
  if (doc.feature_version !== FEATURE_VERSION || doc.input_size !== N_FEATURES) {
    throw new Error(`ASL weights use feature version ${doc.feature_version}; this build reads version ${FEATURE_VERSION}`);
  }
  const n = doc.input_size;
  if (!Array.isArray(doc.classes) || !doc.classes.length || doc.mean?.length !== n || doc.std?.length !== n ||
      !Array.isArray(doc.layers) || !doc.layers.length) throw new Error('Invalid ASL weight dimensions');
  if (doc.std.some((v) => !Number.isFinite(v) || v <= 0)) throw new Error('ASL std must be finite and positive');
  let width = n;
  const layers = doc.layers.map((layer, i) => {
    const { input_size: rows, output_size: cols } = layer;
    if (rows !== width || !(cols >= 1) || layer.weights?.length !== rows * cols || layer.bias?.length !== cols ||
        !['relu', 'linear'].includes(layer.activation)) throw new Error(`Invalid ASL layer ${i + 1}`);
    width = cols;
    return { rows, cols, weights: Float32Array.from(layer.weights), bias: Float32Array.from(layer.bias), relu: layer.activation === 'relu' };
  });
  if (width !== doc.classes.length || doc.layers.at(-1).activation !== 'linear') {
    throw new Error('ASL output layer does not match classes');
  }
  return { classes: doc.classes, mean: Float32Array.from(doc.mean), std: Float32Array.from(doc.std), layers };
}

function softmaxMlp(model, features) {
  let x = new Float64Array(N_FEATURES);
  for (let i = 0; i < N_FEATURES; i++) x[i] = (features[i] - model.mean[i]) / model.std[i];
  for (const { rows, cols, weights, bias, relu } of model.layers) {
    const y = new Float64Array(cols);
    for (let j = 0; j < cols; j++) y[j] = bias[j];
    for (let i = 0; i < rows; i++) {
      const xi = x[i], base = i * cols;
      if (xi === 0) continue;
      for (let j = 0; j < cols; j++) y[j] += xi * weights[base + j];
    }
    if (relu) for (let j = 0; j < cols; j++) if (y[j] < 0) y[j] = 0;
    x = y;
  }
  let max = -Infinity;
  for (const v of x) if (v > max) max = v;
  const e = Array.from(x, (v) => Math.exp(v - max));
  const total = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / total);
}

/**
 * Mirror-averaged class probabilities for one hand, `classes` order (= `predict_proba`).
 * @param {object} model result of `parseWeights`
 */
export function predictProba(model, points, aspect = 1) {
  const a = softmaxMlp(model, landmarkFeatures(points, aspect, false));
  const b = softmaxMlp(model, landmarkFeatures(points, aspect, true));
  return a.map((v, i) => (v + b[i]) / 2);
}

/** Best letter, its probability and the three most likely letters (= `AslLetterClassifier.predict`). */
export function predict(model, points, aspect = 1) {
  const proba = predictProba(model, points, aspect);
  const order = proba.map((v, i) => i).sort((i, j) => proba[j] - proba[i]);
  const top3 = order.slice(0, 3).map((i) => [model.classes[i], proba[i]]);
  return { letter: top3[0][0], confidence: top3[0][1], top3, proba };
}

/** Temporal smoothing of per-frame letter probabilities (port of `LetterSmoother`). */
export class LetterSmoother {
  constructor({ alpha = 0.45, switchMargin = 0.12, minConf = 0.5, gapResetS = 0.6 } = {}) {
    Object.assign(this, { alpha, switchMargin, minConf, gapResetS });
    this.reset();
  }

  reset() { this.ema = null; this.current = null; this.lastSeen = null; }

  /** @param {Object<string, number>|null} scores letter -> probability, null when no hand is visible */
  feed(scores, t) {
    if (this.lastSeen !== null && t - this.lastSeen > this.gapResetS) this.reset();
    if (scores == null) return [null, 0];
    if (this.ema === null) this.ema = Object.fromEntries([...LETTERS].map((c) => [c, Number(scores[c]) || 0]));
    else for (const c of LETTERS) this.ema[c] = this.alpha * (Number(scores[c]) || 0) + (1 - this.alpha) * this.ema[c];
    this.lastSeen = t;
    const winner = [...LETTERS].reduce((a, c) => (this.ema[c] > this.ema[a] ? c : a), 'A');
    if (this.current === null || this.ema[winner] > this.ema[this.current] + this.switchMargin) this.current = winner;
    const confidence = this.ema[this.current];
    return [confidence >= this.minConf ? this.current : null, confidence];
  }
}
