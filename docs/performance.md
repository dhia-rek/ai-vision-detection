# Performance and parity

Measured numbers behind the live site. How the pieces fit: [architecture.md](architecture.md).

## Speed in the browser

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
