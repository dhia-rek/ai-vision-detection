/* YOLO11n inference off the UI thread. The only input is a transferred camera bitmap. */
/* global ort */
const root = new URL('../', self.location.href);
const modelRoot = new URL('models/objects/', root);
let session, inputSize, labels, provider;
let canvas, context;

function report(stage, detail = '') { self.postMessage({ type: 'progress', stage, detail }); }

async function load(size) {
  if (session && inputSize === size) return;
  report('runtime', 'Loading ONNX Runtime');
  if (!self.ort) importScripts(new URL('vendor/ort/ort.webgpu.min.js', root).href);
  ort.env.wasm.wasmPaths = new URL('vendor/ort/', root).href;
  ort.env.wasm.numThreads = 1;
  const modelUrl = new URL(`yolo11n-${size}.onnx`, modelRoot);
  report('model', `YOLO11n · ${(size === 320 ? 10.6 : 10.7).toFixed(1)} MB`);
  const response = await fetch(modelUrl);
  if (!response.ok) throw new Error(`Could not load object model (${response.status})`);
  const model = await response.arrayBuffer();
  if (!labels) {
    const labelResponse = await fetch(new URL('labels.json', modelRoot));
    if (!labelResponse.ok) throw new Error('Could not load object labels');
    labels = await labelResponse.json();
  }
  const options = { executionProviders: self.navigator?.gpu ? ['webgpu', 'wasm'] : ['wasm'] };
  try {
    session = await ort.InferenceSession.create(model, options);
    provider = self.navigator?.gpu ? 'WebGPU' : 'WASM';
  } catch (error) {
    if (!self.navigator?.gpu) throw error;
    report('runtime', 'WebGPU unavailable; using WASM');
    session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
    provider = 'WASM';
  }
  inputSize = size;
  canvas = new OffscreenCanvas(size, size);
  context = canvas.getContext('2d', { willReadFrequently: true });
  report('ready', `${provider} · ${size}px`);
}

function prepare(bitmap) {
  const { width, height } = bitmap;
  const scale = Math.min(inputSize / width, inputSize / height);
  const drawnWidth = Math.round(width * scale), drawnHeight = Math.round(height * scale);
  const padX = Math.round((inputSize - drawnWidth) / 2 - 0.1);
  const padY = Math.round((inputSize - drawnHeight) / 2 - 0.1);
  context.fillStyle = 'rgb(114,114,114)';
  context.fillRect(0, 0, inputSize, inputSize);
  context.drawImage(bitmap, padX, padY, drawnWidth, drawnHeight);
  bitmap.close();
  const rgba = context.getImageData(0, 0, inputSize, inputSize).data;
  const pixels = inputSize * inputSize, tensor = new Float32Array(pixels * 3);
  for (let i = 0; i < pixels; i++) {
    const j = i * 4;
    tensor[i] = rgba[j] / 255;
    tensor[pixels + i] = rgba[j + 1] / 255;
    tensor[pixels * 2 + i] = rgba[j + 2] / 255;
  }
  return { tensor, width, height, scale, padX, padY };
}

function overlap(a, b) {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const intersection = w * h;
  return intersection / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection || 1);
}

function decode(output, frame) {
  const data = output.data, channels = output.dims[1], anchors = output.dims[2];
  if (channels !== labels.length + 4) throw new Error(`Unexpected YOLO output: ${output.dims}`);
  const candidates = [];
  for (let i = 0; i < anchors; i++) {
    let confidence = 0, classId = -1;
    for (let c = 4; c < channels; c++) {
      const score = data[c * anchors + i];
      if (score > confidence) { confidence = score; classId = c - 4; }
    }
    if (confidence < 0.25) continue;
    const cx = data[i], cy = data[anchors + i], w = data[2 * anchors + i], h = data[3 * anchors + i];
    const clampX = (v) => Math.max(0, Math.min(frame.width, (v - frame.padX) / frame.scale));
    const clampY = (v) => Math.max(0, Math.min(frame.height, (v - frame.padY) / frame.scale));
    const x1 = clampX(cx - w / 2), y1 = clampY(cy - h / 2);
    const x2 = clampX(cx + w / 2), y2 = clampY(cy + h / 2);
    if (x2 <= x1 || y2 <= y1) continue;
    candidates.push({ classId, confidence, x1, y1, x2, y2 });
  }
  candidates.sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const box of candidates.slice(0, 30000)) {
    if (kept.some((prior) => prior.classId === box.classId && overlap(prior, box) > 0.7)) continue;
    kept.push(box);
    if (kept.length >= 300) break;
  }
  return kept.map(({ classId, confidence, x1, y1, x2, y2 }) => ({
    class: labels[classId], confidence,
    bbox: { x1, y1, x2, y2 },
  }));
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      await load(data.size);
      self.postMessage({ type: 'ready', provider, size: inputSize });
    } else if (data.type === 'frame') {
      const start = performance.now();
      const frame = prepare(data.bitmap);
      const input = new ort.Tensor('float32', frame.tensor, [1, 3, inputSize, inputSize]);
      const outputs = await session.run({ [session.inputNames[0]]: input });
      const detections = decode(outputs[session.outputNames[0]], frame);
      self.postMessage({ type: 'result', id: data.id, data: {
        detections, image_size: { width: frame.width, height: frame.height },
        mode: 'objects', model_used: `yolo11n-${inputSize}`, device: provider,
        inference_ms: performance.now() - start,
      } });
    }
  } catch (error) {
    if (data.bitmap) data.bitmap.close();
    self.postMessage({ type: 'error', id: data.id, message: error?.message || String(error) });
  }
};
