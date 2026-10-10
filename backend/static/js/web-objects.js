/** Main-thread handle for the YOLO11n object-detection workers. */
import { createWorkerPool } from './web-pool.js';

/** Longest input side the dynamic-shape model runs at: 320 on phones and CPU-only devices, 640 with WebGPU. */
export const OBJECT_SIZES = Object.freeze([320, 640]);
export const OBJECT_MODEL_BYTES = 11027808;

/**
 * `detect(source)` resolves with the Python API's payload shape:
 * `{detections: [{class, confidence, bbox:{x1,y1,x2,y2}}], image_size, model_used, device, inference_ms}`.
 * @param {{size?: 320|640|'auto', onProgress?: Function, maxWorkers?: number}} options
 */
export function createObjectEngine({ size = 'auto', onProgress = () => {}, maxWorkers = 1 } = {}) {
  const pool = createWorkerPool(new URL('./web-object-worker.js', import.meta.url), { size }, { maxWorkers, onProgress });
  return {
    ready: pool.ready,
    get concurrency() { return pool.concurrency; },
    async detect(source) { return pool.frame(await createImageBitmap(source)); },
    close: pool.close,
  };
}
