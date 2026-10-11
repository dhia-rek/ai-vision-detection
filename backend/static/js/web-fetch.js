/** Model download with progress and a persistent Cache API entry (works on the page and in workers). */

/** Bump when a bundled model file changes: old entries are then ignored and deleted. */
export const MODEL_CACHE = 'avd-models-v1';

let cachePromise = null;
function openCache() {
  if (typeof caches === 'undefined') return Promise.resolve(null);
  cachePromise ??= (async () => {
    try {
      for (const name of await caches.keys()) if (name.startsWith('avd-models-') && name !== MODEL_CACHE) await caches.delete(name);
      return await caches.open(MODEL_CACHE);
    } catch { return null; }   // Cache API is unavailable in some private modes: fall back to HTTP caching
  })();
  return cachePromise;
}

/**
 * Fetch a file as an ArrayBuffer, reporting `onProgress({loaded, total, cached})`.
 * A cached copy is returned without touching the network.
 */
export async function fetchModel(url, onProgress = () => {}) {
  const href = new URL(url, self.location.href).href;
  const cache = await openCache();
  const hit = cache && await cache.match(href);
  if (hit) {
    const buffer = await hit.arrayBuffer();
    onProgress({ loaded: buffer.byteLength, total: buffer.byteLength, cached: true });
    return buffer;
  }
  const response = await fetch(href);
  if (!response.ok) throw new Error(`Could not download ${href.split('/').pop()} (${response.status})`);
  const total = Number(response.headers.get('content-length')) || 0;
  const keep = cache ? response.clone() : null;
  const chunks = [];
  let loaded = 0;
  if (response.body) {
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); loaded += value.byteLength;
      // content-length is the encoded size when the host compresses; never report more than 100%.
      onProgress({ loaded: total ? Math.min(loaded, total) : loaded, total: total || loaded, cached: false });
    }
  }
  const buffer = chunks.length ? await new Blob(chunks).arrayBuffer() : await response.arrayBuffer();
  onProgress({ loaded: buffer.byteLength, total: buffer.byteLength, cached: false });
  if (keep) { try { await cache.put(href, new Response(buffer, { headers: { 'content-type': keep.headers.get('content-type') || 'application/octet-stream' } })); } catch { /* quota */ } }
  return buffer;
}

export const megabytes = (bytes) => (bytes / 1e6).toFixed(1);
