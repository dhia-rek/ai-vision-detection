/**
 * A pool of identical inference workers.
 *
 * A WASM session is single-threaded (threads need cross-origin isolation, which static hosting cannot
 * give), so a CPU-only device runs several sessions side by side and the frame loop keeps that many
 * frames in flight. A WebGPU device keeps one worker: the GPU is the bottleneck, not the thread.
 */
import { createWorkerClient } from './web-worker-client.js';

/**
 * @param {URL} url module worker script; answers `init` with `{backend}` and `frame` with a result
 * @param {object} init message sent to every worker on start-up
 * @param {{maxWorkers?: number, onProgress?: Function}} options
 *   `maxWorkers` extra workers are only started when the first one reports a WASM backend.
 */
export function createWorkerPool(url, init, { maxWorkers = 1, onProgress = () => {} } = {}) {
  const slots = [];      // {client, busy, ready}
  const waiting = [];    // resolvers of callers waiting for a free worker
  const release = (slot) => {
    slot.busy = false;
    const next = waiting.shift();
    if (next) { slot.busy = true; next(slot); }
  };
  const spawn = (progress) => {
    const slot = { busy: false, ready: false, client: createWorkerClient(url, init, { onProgress: progress }) };
    slot.client.ready.then(() => { slot.ready = true; release(slot); }, () => {});
    slots.push(slot);
    return slot;
  };
  const first = spawn(onProgress);
  const close = () => { for (const slot of slots) slot.client.close(); waiting.length = 0; };
  const ready = first.client.ready.then((info) => {
    if (info.backend === 'WASM') for (let i = 1; i < maxWorkers; i++) spawn(() => {});
    return info;
  }, (error) => { close(); throw error; });   // a failed start must not leave a worker behind
  const acquire = () => {
    const free = slots.find((slot) => slot.ready && !slot.busy);
    if (free) { free.busy = true; return Promise.resolve(free); }
    return new Promise((resolve) => waiting.push(resolve));
  };
  return {
    ready,
    /** Number of frames that can usefully be in flight right now. */
    get concurrency() { return Math.max(1, slots.filter((slot) => slot.ready).length); },
    /** Run a frame on the next free worker. `bitmap` is transferred. */
    async frame(bitmap, extra) {
      try { await ready; } catch (error) { bitmap.close(); throw error; }
      const slot = await acquire();
      try { return await slot.client.frame(bitmap, extra); } finally { release(slot); }
    },
    close,
  };
}
