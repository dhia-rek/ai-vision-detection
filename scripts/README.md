# Scripts

| Script | What it does |
| --- | --- |
| `asl_extract.py` | Extracts MediaPipe hand landmarks (no pixels) from open ASL datasets into small `.npz` files |
| `train_asl.py` | Compares recipes (leave-one-source-out), trains the ASL letter MLP and evaluates it on a held-out collection |
| `export_asl_web.py` | Exports the trained `.npz` to the JSON weights the browser loads (format below) |
| `eval_objects.py` | Benchmarks YOLO models on a reproducible COCO subset (mAP@0.5, CPU fps) |
| `eval_expressions.py` | Scores the expression pipeline on a labelled image archive (GFFD-2025) |
| `export_web_models.py` | Exports or fetches every model and runtime the static site needs |
| `build_pages_artifact.py` | Assembles the static site exactly as GitHub Pages serves it |
| `web_bench.py` | Measures FPS, latency and download size per mode in headless Chromium |
| `ui_shots.py` | Captures and audits UI screenshots with a fake camera (desktop, phone, light, dark) |
| `record_previews.py` | Records the looping clips shown before the camera starts (`backend/static/previews/`) |

Results and method for the ASL model: [`docs/asl-eval.md`](../docs/asl-eval.md). How the pieces fit:
[`docs/architecture.md`](../docs/architecture.md).

## ASL browser model export

Run `uv run python scripts/export_asl_web.py` after training the NumPy MLP. The command
deterministically exports `backend/app/models/asl_letters.npz` to
`backend/app/models/asl_letters_web.json` (about 1.37 MB of JSON, typically much smaller
with HTTP gzip). The JSON holds no images, landmarks, or training samples.

Format `dhia-asl-mlp`, `format_version: 1`:

- `feature_version`: must equal `backend/app/sign_features.py`'s `FEATURE_VERSION`
  (currently 3). A browser implementation must port that feature function, including
  image aspect correction and its mirror-averaged prediction; the old 70-feature
  70-feature representation is **not** compatible with this export.
- `input_size`: 287; `classes`: A–Z in output-column order.
- `mean` and `std`: arrays of `input_size` floats. Start with
  `x[i] = (features[i] - mean[i]) / std[i]`.
- `layers`: dense layers in order. `weights` is a flat row-major array with
  `weights[inputIndex * output_size + outputIndex]`; `bias` has `output_size`
  floats. Compute `y[j] = bias[j] + sum_i(x[i] * weights[i * output_size + j])`.
  Apply `relu` (`max(0, y)`) after hidden layers and no activation (`linear`) on
  the final layer. Finish with stable softmax: subtract the maximum logit before
  exponentiating. The backend predicts each hand and its horizontal mirror and
  averages their **probabilities**, not logits.
- `source_sha256`: SHA-256 of the `.npz` checkpoint, to tie the web export to the
  Python model. Float values are rendered to eight significant digits; the parity
  test checks that this does not change outputs materially.

The browser loads the copy in `backend/static/models/asl_letters_web.json`; `backend/static/js/web-asl.js` ports the
v3 feature transform and inference, checked against NumPy by `tests/test_web_asl_parity.py`. The API also serves the
export at `GET /api/models/asl-letters-web`.

## Static browser app: `export_web_models.py`, `build_pages_artifact.py`, `web_bench.py`

- `uv run --with onnx --with onnxslim python scripts/export_web_models.py [--only asl objects faces vendor] [--force]`
  - `asl`: runs `export_asl_web.py`, then copies the JSON to `backend/static/models/asl_letters_web.json` (the file the
    browser fetches). Retrained weights with the same `feature_version` drop in without a JavaScript change.
  - `objects`: exports `yolo11n.pt` to one dynamic-shape ONNX (opset 17) used at 320 and 640 px, plus `labels.json`.
  - `faces`: copies the verified FER model and derives the dynamic-input YuNet from the zoo file.
  - `vendor`: downloads the pinned ONNX Runtime Web 1.22.0, MediaPipe Tasks Vision 0.10.21 and the gesture task into
    `backend/static/vendor/` (git-ignored), verifying the npm SRI hashes / zoo SHA-256.
- `uv run python scripts/build_pages_artifact.py --out dist/vision [--serve 8013]` assembles `index.html` + `static/` with
  relative URLs (served at `/ai-vision-detection/`) and optionally serves it.
- `uv run python scripts/record_previews.py --base http://127.0.0.1:8013` records the preview clips from the running
  site (fake camera fed with the bundled sample photos) and writes `backend/static/previews/<mode>.mp4` and `.jpg`.
- `uv run python scripts/web_bench.py [--only objects/phone] [--out bench.json]` measures FPS, latency and download size per
  mode and device class in headless Chromium with a fake 30 fps camera (`taskset -c 0,1 ...` pins the CPU count).
