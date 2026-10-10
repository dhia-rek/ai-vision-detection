# Browser UI modules

The static experience is vanilla JavaScript ES modules. `../app.js` is the entry point; assets and models resolve relative to their importing module so the artifact can be mounted at `/vision/` or another subpath. Live camera inference runs in the browser and does not call the Python API.

| Module | Role |
| --- | --- |
| `util.js`, `store.js`, `config.js` | DOM helpers, app state/events, shared constants |
| `modes.js` | Mode registry and extension point |
| `overlay.js`, `snapshot.js` | Canvas overlay, filters and annotated PNG capture |
| `insights.js`, `sign.js` | Objects/Expressions and Sign panels |
| `web-objects.js` / `web-object-worker.js` | Lazy YOLO11 ONNX Runtime Web worker, preprocessing and NMS |
| `web-faces.js` | YuNet + MobileFaceNet inference using ONNX Runtime Web |
| `web-sign.js` | MediaPipe Tasks Vision hand/gesture inference and exported ASL classifier |
| `stats.js`, `toast.js` | Runtime stats and notifications |
| `media.js` | Camera lifecycle, device switching and frame scheduling |
| `main.js` | Shell, mode/theme/embed controls, keyboard shortcuts |

## Inference and static hosting

Model modules load lazily when a mode is selected. ONNX Runtime Web is pinned and vendored in `../vendor/ort/`; MediaPipe Tasks Vision is version-pinned to its CDN distribution and the gesture task model is retrieved from Google storage. The current model assets and licenses are listed in `../models/README.md`. Browser HTTP caching applies to static and model assets. No frame is sent to the Python backend.

The GitHub Pages artifact contains `index.html` at artifact root and the rest of `backend/static/` below `static/`. Relative references such as `static/app.js` and model paths resolved with `new URL(..., import.meta.url)` are required for subpath hosting at `/vision/`. The artifact workflow builds and uploads only; publishing is a separate operation.

## Adding a mode (future Record panel)

- `js/web-inference.js`: lazy, cached browser inference coordinator for objects, faces, and sign modes.

The shell has a documented extension point for a fourth **Record** panel, intended for the separately maintained landmark-capture work. Keep it isolated so it can merge without changing inference modules:

1. Add a radio control to `index.html` (`data-mode="record"`) and a corresponding panel inside `#panelBody` (see the `RECORD HOOK` comment).
2. Add a small `record.js` panel module and register mode `record` with `order: 4` through `registerMode(...)` in `modes.js`.
3. Import the module in `../app.js` so registration occurs before `main.js` initializes the shell.
4. Use shared store events where appropriate; do not make the browser mode depend on a Python API route.

Mode-specific inference should live in its own module/worker. Avoid circular imports: utilities, state, config, and the registry are low-level; panels and inference modules may depend on them; `media.js` and `main.js` compose the application.
