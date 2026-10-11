# Browser UI modules

The static experience is vanilla JavaScript ES modules. `../app.js` is the entry point. Every asset and model URL is
resolved relative to its importing module (`new URL(..., import.meta.url)`), so the built artifact can be mounted at
`/ai-vision-detection/` or any other sub-path. Live inference runs on the device; nothing here calls the Python API.

| Module | Role |
| --- | --- |
| `util.js`, `store.js`, `config.js` | DOM helpers, app state/events, shared constants |
| `modes.js` | Mode registry and extension point |
| `overlay.js`, `snapshot.js` | Canvas overlay, filters and annotated PNG capture |
| `insights.js`, `sign.js` | Objects/Expressions panels; Sign panel (word builder, chart, challenge) |
| `media.js` | Camera lifecycle, device switching and the frame loop (pipelined frames, stale-result fencing) |
| `main.js` | Shell, mode/theme/embed controls, keyboard shortcuts |
| `stats.js`, `toast.js` | Runtime stats and notifications |
| `web-inference.js` | Lazy per-mode engine loading, model-load progress UI, object-size and worker-count policy; `analyzeFrame()` returns the UI's detection payload |
| `web-fetch.js` | Model download with progress, stored in the Cache API (`avd-models-v1`) |
| `web-ort.js` | Picks and loads the pinned ONNX Runtime Web build (WebGPU JSEP or plain WASM, 1 thread); session creation with WASM fallback |
| `web-worker-client.js`, `web-pool.js` | Promise wrapper for a module worker; pool of identical workers (more than one only on WASM) |
| `web-objects.js`, `web-object-worker.js` | YOLO11n: rectangular letterbox, ONNX run, class-aware NMS |
| `web-faces-client.js`, `web-face-worker.js`, `web-faces.js` | Expressions: YuNet decode (as `cv2.FaceDetectorYN`), five-point alignment, MobileFaceNet FER |
| `web-sign.js` | MediaPipe Tasks Vision gesture recognizer; builds detections like `backend/app/sign.py` |
| `web-asl.js` | ASL feature extraction (feature version 3, 287 values), mirror-averaged MLP from the JSON weights, `LetterSmoother` |

## Inference pipeline

- Objects and expressions run in module Web Workers: the page transfers an `ImageBitmap` per frame and gets the result back, so
  the UI thread only draws. Objects use the dynamic-shape `yolo11n.onnx` at a 320 px (phones, CPU-only) or 640 px
  (WebGPU desktops) longest side, `?objects=320|640` overrides. On WASM, `web-pool.js` starts up to 3 workers (2 on phones,
  at most half the cores) and `media.js` keeps `engine.concurrency` frames in flight, dropping results that arrive out of order.
- Hands run on the main thread (MediaPipe Tasks Vision, XNNPACK CPU delegate by default; `createSignEngine({delegate: 'GPU'})`
  opts into WebGL). `media.js` yields to the event loop between frames so the page keeps painting.
- ASL letters: `web-asl.js` mirrors `backend/app/sign_features.py` and `asl_classifier.py`; `parseWeights()` rejects a weights
  file whose `feature_version` or `input_size` differs from the code (3 / 287). Layer count and widths come from the JSON.
- Only the active mode's engine is kept: switching modes closes the previous worker/recognizer. A load failure clears the
  cached promise so the next frame retries.
- Model files are fetched through `fetchModel()`: progress events, Cache API storage, and ONNX Runtime / MediaPipe WASM
  binaries go through the same path.

Static hosting and model licences: [`../models/README.md`](../models/README.md), [`../../../THIRD_PARTY.md`](../../../THIRD_PARTY.md).
Tests: `tests/test_web_parity.py` (browser vs Python), `tests/test_web_asl_parity.py` (JS vs NumPy), `tests/test_web_e2e.py`
(fake camera, desktop and phone).

## Adding a mode (future Record panel)

The shell has an extension point for a fourth **Record** panel (landmark capture, built separately in PR #5 on the older UI).
Keep it isolated so it can merge without touching the inference modules:

1. Add a radio control to `index.html` (`data-mode="record"`) and a panel inside `#panelBody` (see the `RECORD HOOK` comment).
2. Add a small `record.js` and register mode `record` with `order: 4` through `registerMode(...)` in `modes.js`.
3. Import it in `../app.js` so registration happens before `main.js` initialises the shell.
4. Take landmarks from `web-sign.js` detections (`landmarks`, 21 x [x, y, z] normalised); do not add a server route.

Avoid circular imports: utilities, state, config and the registry are low-level; panels and inference modules may depend on
them; `media.js` and `main.js` compose the application.
