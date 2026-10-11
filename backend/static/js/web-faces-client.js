/** Main-thread handle for the face-expression workers (YuNet + MobileFaceNet). */
import { createWorkerPool } from './web-pool.js';

/** `detectMaxSide`: longest side fed to YuNet (640 live; raise for full-resolution stills);
 * `alignMaxSide`: longest side of the frame used for the expression crops and the reported boxes.
 * `detect(source)` resolves with `{detections, image_size, ...}`; `detections` use the Python API's shape. */
export function createFaceEngine({ onProgress = () => {}, maxWorkers = 1, detectMaxSide = 640, alignMaxSide = 1280 } = {}) {
  const pool = createWorkerPool(new URL('./web-face-worker.js', import.meta.url), { detectMaxSide, alignMaxSide }, { maxWorkers, onProgress });
  return {
    ready: pool.ready,
    get concurrency() { return pool.concurrency; },
    async detect(source) { return pool.frame(await createImageBitmap(source)); },
    close: pool.close,
  };
}
