# Browser model assets

Everything the static app loads at run time. Models and the ASL JSON are **committed**; third-party runtimes are
**downloaded at build time** into `../vendor/` (git-ignored) by `scripts/export_web_models.py --only vendor`,
which checks npm SRI hashes and the model-zoo SHA-256. Only the active mode's files are downloaded, and the
browser keeps them in the Cache API (`avd-models-v1`, bump the name in `js/web-fetch.js` when a file changes).

## Committed models

| File | Mode | Size (bytes) | Licence / source | SHA-256 |
| --- | --- | ---: | --- | --- |
| `objects/yolo11n.onnx` | Objects | 11,027,808 | **AGPL-3.0**, Ultralytics YOLO11n (`yolo11n.pt` v8.3.0 exported by `export_web_models.py`: opset 17, dynamic H/W, simplified, FP32) | `14cb1c8a44ad0627efae21a9c3bd3383f909b14c5c550fbacad03d70677874a3` |
| `objects/labels.json` | Objects | 862 | COCO class names, from the checkpoint | `d85cce85cf16eba13db489fc2aca0571cc453f95e2e787e2c13fce269426d955` |
| `face_detection_yunet_2023mar_dynamic.onnx` | Expressions | 229,582 | MIT, [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet); the zoo file (SHA-256 `8f2383e4...2552fa4`) with its declared H/W made symbolic, no weight or operator changed | `850e50ed4400c2f3a3ff1d9d6b388b920ebfb1e806977d14055a5f9ac1ac1796` |
| `facial_expression_recognition_mobilefacenet_2022july.onnx` | Expressions | 4,791,892 | Apache-2.0, [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition), unmodified | `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9` |
| `asl_letters_web.json` | Sign | 1,367,196 (541,574 gzip) | Project MIT; weights of the NumPy MLP trained in this repo (`dhia-asl-mlp`, feature version 3, 287 inputs, 26 classes) | `95303b421d5e966315a8b96df3b61a76e04e1332c94cd55feb615c40c54661a2` |

The dynamic YOLO11n graph serves both the 320 px (phones, CPU-only) and 640 px (WebGPU desktops) inputs and
Ultralytics-style rectangular inputs, so one file is shipped. `asl_letters_web.json` is a copy of
`backend/app/models/asl_letters_web.json` (a test keeps them identical); regenerate with
`uv run python scripts/export_web_models.py --only asl`. Format: [`scripts/README.md`](../../../scripts/README.md).

## Downloaded runtimes (`../vendor/`, not committed)

| Asset | Used for | Version / licence | Size (bytes) |
| --- | --- | --- | ---: |
| `ort/ort.webgpu.min.mjs`, `ort-wasm-simd-threaded.jsep.mjs`, `.jsep.wasm` | ONNX Runtime Web, WebGPU build (devices with a WebGPU adapter) | 1.22.0, MIT | 357,582 + 44,677 + 21,872,216 (5.1 MB gzip) |
| `ort/ort.wasm.min.mjs`, `ort-wasm-simd-threaded.mjs`, `.wasm` | ONNX Runtime Web, plain WASM build (everything else) | 1.22.0, MIT | 48,259 + 20,856 + 11,210,254 (2.9 MB gzip) |
| `mediapipe/vision_bundle.mjs` + `vision_wasm_internal.{js,wasm}` | MediaPipe Tasks Vision (hands, gestures) | 0.10.21, Apache-2.0 | 137,735 + 204,284 + 9,574,032 |
| `mediapipe/vision_wasm_nosimd_internal.{js,wasm}` | The same, only for CPUs without WASM SIMD | 0.10.21, Apache-2.0 | 204,137 + 9,448,638 |
| `mediapipe/gesture_recognizer.task` | MediaPipe hand landmarks + canned gestures (float16) | Apache-2.0, Google | 8,373,440 (SHA-256 `97952348...0482`) |

## Download per mode (what a first visit fetches, raw / gzip estimate)

| Mode | Files | Size |
| --- | --- | ---: |
| Objects | `yolo11n.onnx` + ORT WASM runtime (WebGPU devices: JSEP runtime instead) | 22.2 MB / 12.2 MB |
| Expressions | YuNet + FER + ORT WASM runtime | 16.2 MB / 7.6 MB |
| Sign | MediaPipe wasm + gesture task + ASL JSON | 19.3 MB / 9.8 MB |

Plus about 0.4 MB of app shell. Numbers come from `scripts/web_bench.py` (the files each mode stores in the cache).

## Licence notes

- ONNX Runtime Web: MIT. MediaPipe Tasks Vision and the gesture task: Apache-2.0.
- Ultralytics YOLO11 is AGPL-3.0. The ONNX export, the export script and the app source stay public in this repository.
- Pre-processing, decoding and thresholds mirror `backend/app/faces.py` (YuNet decode, five-point alignment,
  RGB normalisation, expression order) and Ultralytics' defaults (conf 0.25, NMS IoU 0.7, rectangular letterbox).
- Full table: [`THIRD_PARTY.md`](../../../THIRD_PARTY.md).
