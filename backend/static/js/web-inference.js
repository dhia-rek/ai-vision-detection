/** Lazy, cached browser inference. No camera image leaves the browser. */
import { $ } from './util.js';
import { createObjectEngine } from './web-objects.js';
import { createSignEngine } from './web-sign.js';
import { createFaceDetector } from './web-faces.js';

const engines = new Map();
const names = { objects: 'YOLO11n objects', faces: 'Face expressions', sign: 'Hand signs' };
const sizes = { objects: '10.6 MB on phone · 10.7 MB on desktop', faces: '5.0 MB', sign: '8.4 MB' };
let displayedMode = null;
const faceCanvas = document.createElement('canvas');
const faceContext = faceCanvas.getContext('2d', { willReadFrequently: true });

function progress(mode, detail) {
  if (displayedMode !== mode) return;
  $('modelLoadTitle').textContent = `Preparing ${names[mode]}`;
  $('modelLoadDetail').textContent = detail || `${sizes[mode]} · saved by your browser after the first visit`;
  $('modelLoad').classList.remove('hidden');
}

export function hideModelProgress() {
  displayedMode = null;
  $('modelLoad').classList.add('hidden');
}

function getEngine(mode) {
  if (engines.has(mode)) return engines.get(mode);
  displayedMode = mode;
  progress(mode);
  let loading;
  if (mode === 'objects') {
    const phone = matchMedia('(max-width: 699px)').matches || (navigator.hardwareConcurrency || 8) <= 4;
    const size = phone ? 320 : 640;
    const object = createObjectEngine({ size, onProgress: ({ detail }) => progress(mode, detail) });
    loading = object.ready.then(() => object);
  } else if (mode === 'faces') {
    loading = createFaceDetector({ onProgress: ({ label, loadedBytes, totalBytes }) => {
      const amount = totalBytes ? ` · ${((loadedBytes || 0) / 1e6).toFixed(1)}/${(totalBytes / 1e6).toFixed(1)} MB` : '';
      progress(mode, `${label || 'Loading face models'}${amount}`);
    } });
  } else if (mode === 'sign') {
    loading = createSignEngine({ onProgress: ({ stage }) => progress(mode, `${stage === 'runtime' ? 'Loading MediaPipe' : stage === 'model' ? 'Loading hand model · 8.4 MB' : 'Ready'}`) });
  } else throw new Error(`Unknown browser model: ${mode}`);
  const guarded = loading.catch((error) => { engines.delete(mode); hideModelProgress(); throw error; });
  engines.set(mode, guarded);
  return guarded;
}

/** Run one video frame and return the existing UI's detection payload shape. */
export async function analyzeFrame(mode, video, signMode = 'spell') {
  const engine = await getEngine(mode);
  if (displayedMode === mode) hideModelProgress();
  if (mode === 'objects') return engine.detect(video);
  if (mode === 'sign') {
    const start = performance.now();
    const detections = engine.detect(video, start, signMode);
    return { detections, image_size: { width: video.videoWidth, height: video.videoHeight },
      model_used: signMode === 'spell' ? 'mediapipe-hands+asl' : 'mediapipe-gestures', device: 'browser',
      inference_ms: performance.now() - start };
  }
  const start = performance.now();
  const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
  faceCanvas.width = Math.round(video.videoWidth * scale);
  faceCanvas.height = Math.round(video.videoHeight * scale);
  faceContext.drawImage(video, 0, 0, faceCanvas.width, faceCanvas.height);
  const image = faceContext.getImageData(0, 0, faceCanvas.width, faceCanvas.height);
  const detections = await engine.detect(image);
  return { detections, image_size: { width: image.width, height: image.height },
    model_used: 'yunet+mobilefacenet-fer', device: 'browser', inference_ms: performance.now() - start };
}
