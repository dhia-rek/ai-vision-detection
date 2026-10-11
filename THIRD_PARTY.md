# Third-party components

| Component | Version | Licence | Where | Source |
| --- | --- | --- | --- | --- |
| ONNX Runtime Web | 1.22.0 | MIT | `backend/static/vendor/ort/` (downloaded at build time) | https://www.npmjs.com/package/onnxruntime-web |
| MediaPipe Tasks Vision | 0.10.21 | Apache-2.0 | `backend/static/vendor/mediapipe/` (downloaded at build time) | https://www.npmjs.com/package/@mediapipe/tasks-vision |
| MediaPipe Gesture Recognizer model | float16 v1 | Apache-2.0 | `backend/static/vendor/mediapipe/gesture_recognizer.task` | https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task |
| Ultralytics YOLO11n (weights, ONNX export) | 8.3.0 assets, ultralytics 8.4 | **AGPL-3.0** | `backend/static/models/objects/yolo11n.onnx` | https://github.com/ultralytics/ultralytics |
| YuNet face detector | 2023mar | MIT | `backend/static/models/face_detection_yunet_2023mar_dynamic.onnx` | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |
| MobileFaceNet expression model | 2022july | Apache-2.0 | `backend/static/models/facial_expression_recognition_mobilefacenet_2022july.onnx` | https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition |
| ASL letter classifier weights | feature v3 | MIT (this project) | `backend/static/models/asl_letters_web.json` | trained in this repository, see `docs/asl-datasets.md` for data sources and licences |
| ASL chart illustrations | | see `backend/static/asl/README.md` | `backend/static/asl/` | |
| Lexend, Figtree, JetBrains Mono | | SIL OFL 1.1 | `backend/static/fonts/` | Google Fonts |
| Lucide icons | | ISC | inline SVG sprite in `backend/static/index.html` | https://lucide.dev |
| Sample images | | CC0 / credits in `backend/samples/CREDITS.md` | `backend/samples/` | |
| Preview clips | | Recorded from this app on the sample images above (same licences) | `backend/static/previews/` | `scripts/record_previews.py` |

**YOLO11 and the AGPL.** Ultralytics YOLO11 weights and exports are AGPL-3.0. This repository does not use an
Ultralytics enterprise licence, so everything that serves the model, including this app's source, must stay public
under compatible terms. Keep the repository public and keep this notice with any redistribution.
