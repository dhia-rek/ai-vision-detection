/** Main-thread handle for the object inference Web Worker. */
export function createObjectEngine({ size = 640, onProgress = () => {} } = {}) {
  const worker = new Worker(new URL('./web-object-worker.js', import.meta.url));
  let nextId = 0, ready = false;
  const pending = new Map();
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') { onProgress(data); return; }
    if (data.type === 'ready') ready = true;
    const item = pending.get(data.id ?? 'init');
    if (!item) return;
    pending.delete(data.id ?? 'init');
    if (data.type === 'error') item.reject(new Error(data.message));
    else item.resolve(data.data ?? data);
  };
  worker.onerror = (event) => {
    for (const item of pending.values()) item.reject(new Error(event.message || 'Object worker failed'));
    pending.clear();
  };
  const init = new Promise((resolve, reject) => {
    pending.set('init', { resolve, reject });
    worker.postMessage({ type: 'init', size });
  });
  return {
    async detect(video) {
      await init;
      if (!ready) throw new Error('Object model is not ready');
      const bitmap = await createImageBitmap(video);
      const id = ++nextId;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ type: 'frame', id, bitmap }, [bitmap]);
      });
    },
    ready: init,
    close() { worker.terminate(); for (const item of pending.values()) item.reject(new Error('Object engine closed')); pending.clear(); },
  };
}
