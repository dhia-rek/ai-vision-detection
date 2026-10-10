# AI Vision Detection

Live object detection, facial-expression classification and ASL hand-shape recognition that run **entirely in your browser**: no inference server, no account, and camera frames never leave the device. The static app is built for Dhia's portfolio at **https://dhiarekik.me/vision/**.

![CI](https://github.com/dhia-rek/ai-vision-detection/actions/workflows/ci.yml/badge.svg)

| Objects | Expressions | Sign language |
| --- | --- | --- |
| ![Objects](docs/screenshot-objects.jpg) | ![Faces](docs/screenshot-faces.jpg) | ![Sign language](docs/screenshot-sign-letters.jpg) |

The screenshots use Chromium's fake **camera**; the app is live-camera only (no image or video upload).

## How it works

Training and evaluation live in Python; live inference lives in the browser.

| | Python (`backend/`, `scripts/`) | Browser (`backend/static/`) |
| --- | --- | --- |
| Role | Train the ASL classifier, evaluate models, reference inference, tests, API/CLI demos | The product: camera, inference, UI |
| Objects | Ultralytics YOLO11n (`.pt`) | YOLO11n ONNX in Web Workers, ONNX Runtime Web; letterbox + class-aware NMS in JS |
| Expressions | OpenCV YuNet + MobileFaceNet FER | The same two models as ONNX in Web Workers |
| Sign | MediaPipe Python + NumPy ASL MLP | MediaPipe Tasks Vision (hands + gestures) + a JS port of the ASL feature code and MLP |

- **Device-aware.** Objects run at a 320 px input on phones and CPU-only devices and at 640 px where a real WebGPU adapter exists (`?objects=320|640` overrides). WebGPU devices use the GPU build of ONNX Runtime; everything else uses the smaller WASM build. WASM is single-threaded (threads need cross-origin isolation, which GitHub Pages cannot send), so CPU-only devices run a small pool of workers and keep several frames in flight.
- **Lazy and cached.** A mode's model downloads when the mode is first opened, with a progress bar and its size. Downloads are kept in the browser's Cache API (`avd-models-v1`), so later visits start without network; only the active mode's model stays in memory.
- **Static.** `scripts/build_pages_artifact.py` produces `index.html` + `static/`, every URL relative, so the folder can be mounted at `/vision/` or anywhere else. The [workflow](.github/workflows/pages-artifact.yml) only builds and uploads that artifact; it does not deploy.

### Embed

Same-origin iframe on `dhiarekik.me`; the camera needs `allow="camera"` and HTTPS:

```html
<iframe src="/vision/?embed=1&mode=objects&theme=light"
        title="AI Vision Detection" allow="camera" loading="lazy"
        style="width:100%;height:560px;border:0;border-radius:16px"></iframe>
```

Deep links: `mode=objects|faces|sign`, `sign_mode=spell|gestures`, `theme=light|dark`, `embed=1`.

## Browser performance

Measured with `scripts/web_bench.py`: headless Chromium 153 on a 6-vCPU Debian VPS (Haswell-class cores, **no GPU**, so WebGPU is not used), Chromium's fake camera at 30 fps, FPS read from the app's own counter over 10 s. "Phone" is a Pixel 7 emulation (user agent, viewport, touch: phone code path, 320 px input, 2 workers) on the *same* CPU; the 2-core row pins the browser to two cores (`taskset -c 0,1`). Real devices differ: a laptop or recent phone core is typically faster than this VPS, and a WebGPU laptop runs objects at 640 px on the GPU (not measured here).

| Mode | Device class | Model (input) | FPS | Latency / frame | Model + runtime download (raw / gzip) |
| --- | --- | --- | ---: | ---: | ---: |
| Objects | Desktop, 6 cores (CPU/WASM) | YOLO11n 320 | **29.2** | 100 ms | 22.2 / 12.2 MB |
| Objects | Phone emulation, 6 cores | YOLO11n 320 | **19.4** | 98 ms | 22.2 / 12.2 MB |
| Objects | Phone emulation, 2 cores | YOLO11n 320 | **9.6** | 99 ms | 22.2 / 12.2 MB |
| Objects | Desktop, forced 640 on WASM | YOLO11n 640 | 8.7 | 357 ms | 22.2 / 12.2 MB |
| Expressions | Desktop / phone / phone 2-core | YuNet + MobileFaceNet | 13.4 / 8.5 / 4.1 | 232 ms | 16.2 / 7.6 MB |
| Sign letters | Desktop / phone / phone 2-core | MediaPipe hands + ASL MLP | 10.8 / 10.5 / 9.6 | 88 ms | 19.3 / 9.8 MB |
| Sign gestures | Desktop / phone / phone 2-core | MediaPipe gesture recognizer | 10.8 / 10.6 / 9.5 | 88 ms | 19.3 / 9.8 MB |

Targets: objects at least 15 fps on a laptop (met on CPU alone) and at least 8 fps on a phone (met, including the 2-core pin). Expressions are the slowest mode (the expression network costs about 170 ms per face in single-threaded WASM); on the 2-core pin they reach 4 fps. The app shell (HTML, CSS, JS, fonts, ASL charts) is about 0.4 MB. Download figures are the files each mode stores in the cache; "gzip" is a best-case estimate, because static hosts do not always compress `.onnx`/`.wasm`. First result appears 2.3 to 4.3 s after pressing Start on a cold cache and 1.0 to 2.0 s afterwards (local server, no network latency).

## Parity with the Python backend

`tests/test_web_parity.py` runs the browser pipeline in Chromium on the bundled sample images and compares it with the Python backend (`YoloDetector`/Ultralytics at the same input size, `ExpressionDetector`, `SignDetector`). Criteria: identical labels, box IoU of at least 0.90, scores within 0.05. All 22 cases pass:

| Samples | Detections matched | Min box IoU | Max score delta |
| --- | ---: | ---: | ---: |
| Objects, 640 px (5 images) | 31 / 31 | 0.919 | 0.037 |
| Objects, 320 px (5 images) | 22 / 22 | 0.951 | 0.040 |
| Expressions, source resolution (3 images) | 3 / 3 | 1.000 | 0.001 |
| Sign letters (4 images: B, L, V, Y) | 4 / 4 | 1.000 | 0.000 |
| Sign gestures (2 images) | 2 / 2 | 1.000 | 0.000 |

The ASL classifier is additionally checked against NumPy (`tests/test_web_asl_parity.py`): the 287 features and their mirrored variant agree to 2e-5 and the mirror-averaged probabilities to 1e-4, and the `LetterSmoother` port matches frame by frame.

Honest caveats: (1) the live expression pipeline detects faces on a frame of at most 640 px (the strict test above uses source resolution like Python does). That keeps labels and boxes (IoU at least 0.87 on the samples) but can move expression probabilities, by up to 0.31 on `angry.jpg` (label unchanged). (2) The WebGPU path could not be timed here; it was checked for correctness on a software WebGPU adapter and gave the same detections as WASM.

## Build and run

```bash
uv sync                                           # Python env (training, tests, API)
uv run python scripts/export_web_models.py --only vendor   # pinned ONNX Runtime Web + MediaPipe (git-ignored)
uv run python scripts/build_pages_artifact.py --out dist/vision --serve 8013
# open http://127.0.0.1:8013/  (camera access works on localhost)
```

`export_web_models.py` with no `--only` also re-exports YOLO11n, derives the face models and exports the ASL JSON (`uv run --with onnx --with onnxslim python scripts/export_web_models.py`); the model files and the ASL JSON are committed, the vendored runtimes are not. Retrained ASL weights only need `--only asl`: the JSON describes feature version, scaler, classes and arbitrary dense layers, so new weights drop in with no JavaScript change (a different `feature_version` is rejected with a clear error).

Tests: `uv run pytest -q` (Python, parity, browser end-to-end), `uv run pytest -m browser` for the Chromium suites only (needs `uv run playwright install chromium` and `ffmpeg`), `uv run python scripts/web_bench.py` for the measurements above. Module guide: [backend/static/js/README.md](backend/static/js/README.md). Model files, sizes and checksums: [backend/static/models/README.md](backend/static/models/README.md).

### Python API and tools

The FastAPI backend (`uv run uvicorn app.main:app --app-dir backend --reload`) still serves the same static page plus `POST /api/analyze`, `/api/frame`, `/api/video` and `/demo/*` for tests, evaluation, CLI demos and the Docker/Hugging Face Space. The browser app never calls them. Training and evaluation entry points are in [scripts/README.md](scripts/README.md) and [docs/](docs/).

## Licenses

- Project code: MIT, see [LICENSE](LICENSE).
- ONNX Runtime Web 1.22.0 (MIT) and MediaPipe Tasks Vision 0.10.21 with the gesture model (Apache-2.0) are downloaded at build time and not committed.
- YOLO11 (Ultralytics) is **AGPL-3.0**: the exported weights and this app's source stay public in this repository; there is no Ultralytics enterprise licence.
- YuNet (MIT) and the MobileFaceNet expression model (Apache-2.0) come from the OpenCV Zoo.
- Full list with versions and sources: [THIRD_PARTY.md](THIRD_PARTY.md). ASL illustration attribution: [backend/static/asl/README.md](backend/static/asl/README.md); fonts use SIL OFL 1.1.

## Limitations and responsible use

Portfolio/demo project, not an interpreter or a system for consequential decisions. Expression labels are a guess about a face, not a statement about a person's feelings. Sign mode reads static hand shapes: letters J and Z need motion and are unreliable, and it is not sign-language translation. Obtain consent before using the camera around other people. Do not use it for hiring, proctoring, policing, insurance or other decisions about people. Speed depends on the device; the first visit downloads 16 to 22 MB depending on the mode.

Author: Dhia ([@dhia-rek](https://github.com/dhia-rek)).
