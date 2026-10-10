# AI Vision Detection
Real-time **object detection**, **facial expression recognition** and **ASL fingerspelling / gesture recognition** in the browser: live webcam, video file or image, CPU only, served by FastAPI.

[![CI](https://github.com/dhia-rek/ai-vision-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/dhia-rek/ai-vision-detection/actions/workflows/ci.yml)
![Code: MIT](https://img.shields.io/badge/code-MIT-blue)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![YOLOv8 weights: AGPL-3.0](https://img.shields.io/badge/YOLOv8%20weights-AGPL--3.0-orange)

![Demo](docs/demo/demo.gif)
*Offline demo on the five bundled object samples (YOLOv8n, CPU).*

| Objects | Facial expression |
| --- | --- |
| ![Objects](docs/screenshot-objects.jpg) | ![Faces](docs/screenshot-faces.jpg) |
| *Bicycles: 8 detections, 26-89 %.* | *Happy, 100 %.* |

| Sign: gestures | Sign: letters |
| --- | --- |
| ![Gestures](docs/screenshot-sign-gestures.jpg) | ![Letters](docs/screenshot-sign-letters.jpg) |
| *"I love you", 90 %.* | *Letter Y, 100 %.* |

![Fingerspelling a video](docs/screenshot-sign-video.jpg)
*The demo video spelled "CAB" live in the browser.*

## Modes

| Mode | What it does | Models |
| --- | --- | --- |
| **Objects** | Boxes + labels for the 80 COCO classes | YOLOv8n |
| **Facial expression** | Finds faces, reads one of 7 expressions per face | YuNet + MobileFaceNet |
| **Sign language** | Two sub-modes: **Fingerspelling** (ASL letters A-Z, with a word builder) and **Gestures** (7 hand shapes mapped to meanings) | MediaPipe hands + our letter classifier |

Sources: **Webcam**, **Video file**, **Image**. Bundled sample images and sample videos need no camera.
Deep links: `/?mode=objects|faces|sign&sign_mode=spell|gestures&source=webcam|video|image&sample=<file>`.

This was built as a demo of how a browser can help communicate with non-speaking people. It is **not** a replacement for an interpreter (see [Limitations](#limitations)).

## Architecture

```mermaid
flowchart LR
    UI[Browser<br/>webcam / video / image] -->|JPEG frames over HTTP| API[FastAPI<br/>app/main.py]
    API --> R{mode router}
    R -->|objects| Y[YOLOv8n]
    R -->|faces| F[YuNet detector] --> AL[5-point align 112x112] --> M[MobileFaceNet expression]
    R -->|sign| H[MediaPipe hands<br/>21 landmarks]
    H -->|spell| L[letter classifier<br/>numpy]
    H -->|gestures| G[canned gesture -> meaning]
    Y --> A[annotate<br/>OpenCV]
    M --> A
    L --> A
    G --> A
    A -->|annotated JPEG / JSON| UI
    UI --> S[speller in JS<br/>letters -> text]
```

Images and frames are decoded in memory and never written to disk. Detectors are created lazily per mode and guarded by a lock.

## How each mode works

**Objects.** Letterbox resize, one YOLOv8 forward pass, confidence threshold (`AVD_CONF`, default 0.25), NMS, COCO labels, drawn with OpenCV. No training or fine-tuning here.

**Facial expression.** (`faces.py`)
1. **YuNet** (OpenCV `FaceDetectorYN`) finds faces and 5 landmarks (eyes, nose, mouth corners). Faces smaller than 24 px are skipped.
2. The face is **aligned** to 112x112 with a similarity transform onto reference landmark positions.
3. **MobileFaceNet** (OpenCV DNN, no torch) outputs a softmax over 7 classes: angry, disgust, fear, happy, neutral, sad, surprised. The top class and a top-3 are returned.

**Sign language.** (`sign.py`) One MediaPipe Gesture Recognizer pass gives 21 landmarks per hand (up to 2 hands) plus a canned gesture.
- **Fingerspelling:** landmarks are turned into 70 features (wrist-centred, scale-normalised coordinates + 10 fingertip distances, left hands mirrored; `sign_features.py`) and fed to a small classifier (`asl_classifier.py`, pure numpy, <50 KB). It was trained with `scripts/train_asl.py` on the MIT-licensed HF dataset `sid220/asl-now-fingerspelling` (MediaPipe landmarks, ~2,100 samples, 26 letters) with mirror augmentation. Only landmarks are used, not images.
- **Gestures:** MediaPipe's canned gestures are mapped to a meaning: thumbs up -> yes, thumbs down -> no, open palm -> hello / stop, I love you, victory / peace, pointing up -> wait, closed fist.
- **Speller (browser, same rules as `WordBuilder` in `sign.py`):** hold a letter **0.8 s** to commit it (confidence >= 0.6); change shape or drop the hand to repeat a letter; **1.5 s** without a hand adds a space. **Space / Backspace / Clear** buttons are in the UI.

> This recognises the ASL fingerspelling alphabet and a few hand gestures from **static hand shapes**. It is **not** sign-language translation: ASL is a full language with motion, facial grammar and two-handed signs, none of which this models. "Thank you" and the real ASL signs for yes / no / hello involve motion and are not supported.

## Results

Measured with the real models on the bundled samples. These are qualitative checks, not benchmarks.

**Objects** (YOLOv8n)

| Image | Detections |
| --- | --- |
| `bicycles.jpg` | 8x bicycle |
| `cats.jpg` | 2x cat |
| `dog.jpg` | 1x dog (0.95) |
| `livingroom.jpg` | 3x chair, 2x couch, 1x tv, 1x potted plant |
| `street.jpg` | 6x car, 2x truck, 1x traffic light, 1x person |

**Facial expression**

| Image | Result |
| --- | --- |
| `happy.jpg` | happy 1.00 |
| `angry.jpg` | angry 0.91 |
| `neutral.jpg` | neutral 1.00 |

Known failures: a CC BY photo of a surprised young woman with open mouth read as *happy* 0.59 (surprised was not top 1). The angry photo downscaled from 960 to 640 px wide flips to *neutral* 0.59 (angry 0.33): borderline expressions are resolution-sensitive. In the 640 px `faces.mp4` demo the angry segment is read as neutral. Drawings and line-art faces are not detected.

**Sign gestures:** "I love you" 0.82 / 0.90 on two photos; thumbs up 0.56-0.69 on three photos. Open palm, victory and thumbs-down photos taken from a distance (small hands) were **not detected at all**: MediaPipe needs the hand reasonably large in frame (webcam distance).

**Sign letters:** trained on `sid220/asl-now-fingerspelling` (MIT, 2,122 MediaPipe landmark samples).

| Test set | Accuracy |
| --- | --- |
| Held-out 20 % of the training dataset (same recipe trained on 80 %) | **96.5 %** |
| Independent real photos, [EitanG98/asl_letters](https://huggingface.co/datasets/EitanG98/asl_letters) (Unlicense, 720 photos; hand found in 686) | **48.8 %** of detected hands (46.5 % of all) |

The real-photo number is the honest one, and it is only fair: it was also used to choose between
four training recipes, so it is slightly optimistic. Strong letters on real photos: B, K, L, M, V, W, Y
(85-96 %). Weak: D, H, P, Q, T, X (0-12 %). The six letter fixtures in `tests/fixtures/sign/`
(A, B, C, L, V, Y) are all recognised at >= 0.999, but they were picked from photos the model gets right,
so they are regression tests, not an accuracy estimate.
J and Z are motion letters and are classified from a single pose, so they are unreliable.

**Video:** `fingerspelling_cab.mp4` (built from three letter photos) -> `/api/video` text **"CAB"** (C 0.93-0.95, A 0.999, B 0.997); in the browser UI the speller produced "CAB" live at 8-12.6 fps.

The sample videos are **slideshows built from openly licensed still photos**, not real recordings.

Reproduce the image results:

```bash
cd backend
uv run python -m app.demo --mode objects --out ../docs/demo   # or faces / sign
```

## Performance

Measured on this project's 6-vCPU VPS, CPU only, 640x480 JPEG frames posted one at a time to
`/api/frame` over localhost (40 frames after warm-up):

| Mode | Server inference (median) | Round trip (median) | Throughput |
| --- | --- | --- | --- |
| Objects (YOLOv8n) | 45 ms | 53 ms | **18 fps** |
| Facial expression | 36 ms | 42 ms | **24 fps** |
| Sign: gestures | 37 ms | 43 ms | **23 fps** |
| Sign: fingerspelling | 37 ms | 43 ms | **23 fps** |

In the real browser UI (headless Chromium, video source, same machine) the full loop (canvas grab,
JPEG encode, upload, annotated JPEG back) ran at **8-12.6 fps**. Over a real network expect less;
the UI sends one frame at a time and never queues. Numbers vary with CPU load: with other heavy
jobs on the same VPS, YOLO dropped below 1 fps.

## Project structure

```text
backend/app/
  main.py           FastAPI app: routes, upload limits, mode routing, video analysis
  detector.py       Detection dataclass, Detector protocol, YOLO + stub detectors, drawing, JPEG helpers
  faces.py          YuNet face detector + MobileFaceNet expression classifier
  sign.py           MediaPipe hands -> ASL letters or gestures; WordBuilder for fingerspelling
  sign_features.py  Landmark normalisation shared by training and inference
  asl_classifier.py Pure-numpy ASL letter classifier (loads models/asl_letters.npz)
  zoo.py            Model registry: checksummed downloads and the local cache
  demo.py           CLI that runs a mode over the bundled samples
backend/static/     Single-page web UI (vanilla JS)
scripts/train_asl.py  Trains the ASL letter classifier
tests/              pytest suite (stub detector by default; `slow` marker runs real models)
```

## Development

```bash
uv sync                         # install runtime + dev dependencies
uv run pytest -q -m "not slow"  # fast tests (drop the marker to run real models)
uv run ruff check .             # lint, including Google-style docstring rules
uv run ruff format --check .    # formatting
uv run mypy                     # type check (basic mode)
uv run interrogate              # docstring coverage, fails under 90%
```

CI runs the same checks on every pull request.

## Quick start

```bash
# 1. Install (uv; CPU-only torch is used on Linux)
uv sync

# 2. (optional) pre-download the models; otherwise they download on first use
cd backend && uv run python -m app.zoo

# 3. Run the app
uv run uvicorn app.main:app --reload
# open http://localhost:8000

# Tests (from the repo root)
uv run pytest -q
```

Models are cached in `~/.cache/ai-vision-detection/models` (or `AVD_MODEL_DIR`) and **verified by SHA-256**. Nothing is committed except the tiny ASL classifier.

Docker (also targets a Hugging Face Space, see `DEPLOY.md`):

```bash
docker build -t ai-vision-detection .
docker run -p 7860:7860 ai-vision-detection
```

The image installs `libegl1`, `libgles2` and `libgl1` (needed by MediaPipe) and pre-downloads the models at build time. Webcam access needs `localhost` or HTTPS (a browser requirement).

CLI demo: `uv run python -m app.demo --mode objects|faces|sign` (stub mode: `AVD_DETECTOR=stub`).

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Web UI |
| GET | `/health` | Liveness check |
| GET | `/api/modes` | Modes and the model registry (size, licence, source) |
| POST | `/api/analyze?mode=objects\|faces\|sign&sign_mode=spell\|gestures` | Multipart `file`: image in, JSON out: `result_image` (JPEG data URL), `detections` (class, confidence, bbox; `points` for hands; `top3`, `handedness`, `gesture` where relevant), `inference_ms` |
| POST | `/api/frame` | Multipart `frame` (+ `mode`, `sign_mode`, `quality` 10-95): annotated JPEG; headers `X-Detections` (top 25 by confidence; `X-Detections-Truncated`), `X-Total-Detections`, `X-Inference-Ms` |
| POST | `/api/video?mode=&sign_mode=&fps=5&max_seconds=60` | Multipart `file` (video; `fps` <= 10, `max_seconds` <= 120, at most 600 frames analysed): per-frame detections (class, confidence, bbox), `label_counts`, `avg_inference_ms`, and `text` in fingerspelling mode |
| POST | `/detect/`, `/detect/frame` | Legacy object-detection endpoints |
| GET | `/demo/samples` | Bundled samples per group |
| GET | `/demo/samples/{group}/{name}` | Fetch a sample |
| POST | `/demo/analyze/{group}/{name}` | Run a mode on a bundled sample image |

Errors: `400` not an image / video or undecodable, `413` too large, `404` unknown sample. Interactive docs at `/docs`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `AVD_DETECTOR` | `real` | `real` for the models, `stub` for a fake detector (tests, no models needed) |
| `AVD_WEIGHTS` | downloaded `yolov8n.pt` | Path to YOLO weights for Objects mode |
| `AVD_MODEL_DIR` | `~/.cache/ai-vision-detection/models` | Model cache directory |
| `AVD_CONF` | `0.25` | Object confidence threshold |
| `AVD_MAX_UPLOAD_MB` | `10` | Maximum image upload size |
| `AVD_MAX_VIDEO_MB` | `50` | Maximum video upload size |
| `AVD_MAX_PIXELS` | `40` | Maximum decoded image size in megapixels (decompression-bomb guard) |
| `AVD_CORS_ORIGINS` | *(empty)* | Comma-separated allowed origins; CORS is off unless set |

## Models & licences

| Model | Size | Licence | Source / use |
| --- | --- | --- | --- |
| `yolov8n.pt` | 6.2 MB | **AGPL-3.0** | Ultralytics YOLOv8n, COCO 80 classes (objects) |
| YuNet face detector | 0.2 MB | MIT | OpenCV Zoo (faces) |
| MobileFaceNet expression | 4.6 MB | Apache-2.0 | OpenCV Zoo, trained on RAF-DB (faces). RAF-DB is a research dataset with its own non-commercial terms; the weights are Apache-2.0 |
| `gesture_recognizer.task` | 8.0 MB | Apache-2.0 | Google MediaPipe: hand landmarks + 7 canned gestures |
| `asl_letters.npz` | <50 KB | MIT (ours) | Letter classifier, trained with `scripts/train_asl.py` on [sid220/asl-now-fingerspelling](https://huggingface.co/datasets/sid220/asl-now-fingerspelling) (MIT) |

Evaluation of the letter classifier used [EitanG98/asl_letters](https://huggingface.co/datasets/EitanG98/asl_letters) (Unlicense). Test photos in `tests/fixtures` and samples in `backend/samples` are openly licensed; see `tests/fixtures/*/README.md` and [backend/samples/CREDITS.md](backend/samples/CREDITS.md).

## Project structure

```
backend/
  app/
    main.py            FastAPI app factory, routes, mode router, video analysis
    detector.py        YoloDetector, StubDetector, annotate, JPEG helpers
    faces.py           YuNet + MobileFaceNet expression detector
    sign.py            MediaPipe hands, gesture map, WordBuilder
    sign_features.py   Landmark -> 70 features
    asl_classifier.py  Pure-numpy letter classifier
    zoo.py             Model registry: URLs, SHA-256, download + cache
    demo.py            Offline demo CLI (--mode objects|faces|sign)
    models/asl_letters.npz   The only committed model (<50 KB)
  samples/             Openly licensed images + 2 slideshow videos + CREDITS.md
  static/index.html    Single-page UI (HTML/JS, speller)
scripts/train_asl.py   Train + export the ASL letter classifier
tests/                 pytest suite and photo fixtures
docs/                  Demo GIF and screenshots
Dockerfile, DEPLOY.md  Container and Hugging Face Space deployment
```

## Engineering notes

- **Tooling**: `uv` + `pyproject.toml` + lockfile, CPU-only PyTorch on Linux, ruff.
- **Model zoo**: models are downloaded on demand and SHA-256 verified (atomic temp-file rename), so the repo stays small.
- **Design**: one `Detector` interface per mode behind an app factory; a torch-free `StubDetector` lets tests run in seconds without models. Face models run on OpenCV DNN, no extra framework.
- **Small classifier, small risk**: the ASL classifier is plain numpy (no sklearn at runtime) and has a feature-version check.
- **Security & privacy**: uploads decoded in memory; size and pixel limits; CORS off by default; internal errors not echoed; client filenames never used as paths.
- **Tests and CI**: pytest (API, detectors, WordBuilder, real-model checks) and GitHub Actions.
- **Deployment**: Dockerfile and Hugging Face Space preparation (`DEPLOY.md`).

## Large files

The repo history carries about 59 MB of old YOLO weights (`yolov8m.pt` 52 MB, `yolov8n.pt` 6.5 MB, committed several times). The code **no longer uses them**: models are fetched by `app.zoo`. `AVD_WEIGHTS` can still point at one. History was not rewritten; the files can be dropped from the tree, or the history cleaned if size matters.

## Limitations

**All modes**
- CPU latency plus JPEG/network round trips limit the webcam frame rate. No tracking: each frame is independent, so results can flicker.
- Quality was only checked qualitatively on a few images per mode.

**Objects**
- 80 COCO classes only. Small or occluded objects are missed; false positives happen in unusual light or viewpoints.

**Facial expression**
- Reads **expressions, not emotions**. Seven coarse classes; blended or subtle expressions are poorly covered.
- Borderline cases are fragile and resolution-sensitive (see Results). Surprise and angry were misread in our tests.
- Training data (RAF-DB) is mostly web and posed images; accuracy across ages, skin tones, cultures, lighting and head poses is uneven and not measured here.
- Line art and drawn faces are not detected; very small faces (<24 px) are skipped.

**Sign language**
- Static hand shapes only: no motion, no facial grammar, no two-handed signs. It is not translation.
- Letters J and Z need motion; here they come from a single pose and are unreliable. Similar shapes (e.g. H/U, I/J, S/T/A/N) get confused.
- The letter classifier is demo-grade: it is far more accurate on its own held-out split (webcam-style landmarks) than on varied real photos; see Results. Training data is small (~2,100 samples).
- MediaPipe needs the hand reasonably large in frame; hands far from the camera are not detected. Gestures are MediaPipe's canned set only.
- The speller commits a letter only after a steady 0.8 s hold.

## Ethics & privacy

- **Privacy:** images and webcam frames are processed in memory and never stored. Video uploads to `/api/video` are written to a private temp file only for decoding and deleted when the request ends. There are no face embeddings, no identification or matching of people, and no logging of images.
- **Expression is not emotion.** The face model maps facial-muscle configuration to a label. It does not know how anyone feels, and a smile can be polite, ironic or masked. Treat output as a rough description of a picture, never as a fact about a person.
- **Bias.** The model was trained on a limited, mostly posed / web dataset and is likely to perform unevenly across demographic groups and cultures. Errors are not evenly distributed.
- **Do not use it** for hiring, education or exam proctoring, policing, border control, insurance, or any decision about a person. The EU AI Act prohibits emotion recognition in workplaces and education (with medical and safety exceptions); this is context, not legal advice, so check the rules where you are.
- **Consent.** Only run it on yourself or on people who know and agree. Do not run it covertly on others, and do not point it at images of people you have no right to process (e.g. under GDPR).
- Sign mode is an accessibility demo. It must not be relied on to understand a deaf or non-speaking person; use a qualified interpreter or ask the person.
- Detection output is not evidence of anything.

## License

- **Code**: MIT, see [LICENSE](LICENSE).
- **Ultralytics / YOLOv8 weights** are **AGPL-3.0**. Distributing this app or serving it over a network brings AGPL obligations (including offering source). Ultralytics sells an enterprise licence for proprietary use.
- Other models: see the table above (MIT / Apache-2.0). Mind the RAF-DB note.
- **Samples and fixtures**: CC0, public domain, Unlicense or CC licences as listed in [backend/samples/CREDITS.md](backend/samples/CREDITS.md) and `tests/fixtures/*/README.md`; follow attribution where required.

## Stack

Python 3.11+, FastAPI, Uvicorn, Ultralytics YOLOv8, PyTorch (CPU), MediaPipe, OpenCV (headless, DNN), NumPy, Pillow, vanilla HTML/JS front-end; uv, pytest, ruff, GitHub Actions, Docker.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and the [COCO dataset](https://cocodataset.org).
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (YuNet, MobileFaceNet) and the RAF-DB authors.
- [Google MediaPipe](https://github.com/google-ai-edge/mediapipe).
- Datasets: [sid220/asl-now-fingerspelling](https://huggingface.co/datasets/sid220/asl-now-fingerspelling) (training), [EitanG98/asl_letters](https://huggingface.co/datasets/EitanG98/asl_letters) (evaluation).
- Sample images: Wikimedia Commons and the sources in `backend/samples/CREDITS.md`.

Author: Dhia ([@dhia-rek](https://github.com/dhia-rek)).
