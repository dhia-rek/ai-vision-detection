/** Loads the pinned ONNX Runtime Web build that suits this device (module workers and the page). */
import { fetchModel, megabytes } from './web-fetch.js';

const ORT_ROOT = new URL('../vendor/ort/', import.meta.url);

/** True when a WebGPU adapter is really available (the API alone is not enough). */
export async function hasWebGpu() {
  try { return !!(self.navigator?.gpu && await self.navigator.gpu.requestAdapter()); } catch { return false; }
}

/**
 * @returns {Promise<{ort: object, providers: string[], backend: 'WebGPU'|'WASM'}>}
 * WebGPU devices get the JSEP build; everything else the smaller plain WASM build.
 * Threaded WASM needs cross-origin isolation, which static hosting cannot provide, so one thread is used.
 */
export async function loadOrt({ forceWasm = false, onProgress = () => {} } = {}) {
  const gpu = !forceWasm && await hasWebGpu();
  const ort = await import(new URL(gpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs', ORT_ROOT).href);
  ort.env.wasm.wasmPaths = ORT_ROOT.href;
  // The runtime binary goes through the same download-once cache as the models.
  const wasm = new URL(gpu ? 'ort-wasm-simd-threaded.jsep.wasm' : 'ort-wasm-simd-threaded.wasm', ORT_ROOT);
  ort.env.wasm.wasmBinary = new Uint8Array(await fetchModel(wasm, ({ loaded, total, cached }) =>
    onProgress({ detail: cached ? 'Loading ONNX Runtime' : `ONNX Runtime · ${megabytes(loaded)} / ${megabytes(total)} MB`, loaded, total })));
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';
  return { ort, providers: gpu ? ['webgpu', 'wasm'] : ['wasm'], backend: gpu ? 'WebGPU' : 'WASM' };
}

/** Create a session; if the GPU path fails on this device, retry on WASM so the mode still works. */
export async function createSession(runtime, model, options = {}) {
  try {
    const session = await runtime.ort.InferenceSession.create(model, { executionProviders: runtime.providers, logSeverityLevel: 3, ...options });
    return { session, backend: runtime.backend };
  } catch (error) {
    if (runtime.providers[0] !== 'webgpu') throw error;
    const wasm = await loadOrt({ forceWasm: true });
    const session = await wasm.ort.InferenceSession.create(model, { executionProviders: ['wasm'], logSeverityLevel: 3, ...options });
    Object.assign(runtime, wasm);
    return { session, backend: 'WASM' };
  }
}
