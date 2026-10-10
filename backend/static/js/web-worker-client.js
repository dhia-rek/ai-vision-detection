/** Promise-based wrapper around a module Web Worker that serves one inference engine. */
export function createWorkerClient(url, init, { onProgress = () => {} } = {}) {
  const worker = new Worker(url, { type: 'module' });
  const pending = new Map();
  let nextId = 0, closed = false;
  const failAll = (error) => { for (const item of pending.values()) item.reject(error); pending.clear(); };
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') { onProgress(data); return; }
    const item = pending.get(data.id);
    if (!item) return;
    pending.delete(data.id);
    if (data.type === 'error') item.reject(new Error(data.message)); else item.resolve(data.data);
  };
  worker.onerror = (event) => failAll(new Error(event.message || 'Inference worker failed'));
  function call(message, transfer = []) {
    if (closed) return Promise.reject(new Error('Inference engine closed'));
    const id = ++nextId;
    return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); worker.postMessage({ ...message, id }, transfer); });
  }
  const ready = call({ type: 'init', ...init });
  return {
    ready,
    /** Send a frame (an ImageBitmap is transferred) and resolve with the worker's result. */
    async frame(bitmap, extra = {}) {
      try { await ready; } catch (error) { bitmap.close(); throw error; }
      return call({ type: 'frame', bitmap, ...extra }, [bitmap]);
    },
    close() { closed = true; worker.terminate(); failAll(new Error('Inference engine closed')); },
  };
}
