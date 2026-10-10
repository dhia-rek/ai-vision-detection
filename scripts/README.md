# ASL browser model export

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

The API serves the committed export at `GET /api/models/asl-letters-web`. A browser
implementation can fetch that URL; the separate UI branch still needs to port the
v3 feature transform and inference before this becomes live browser-side detection.

## Static browser app: `export_web_models.py`, `build_pages_artifact.py`, `web_bench.py`

- `uv run --with onnx --with onnxslim python scripts/export_web_models.py [--only asl objects faces vendor] [--force]`
  - `asl`: runs `export_asl_web.py`, then copies the JSON to `backend/static/models/asl_letters_web.json` (the file the
    browser fetches). Retrained weights with the same `feature_version` drop in without a JavaScript change.
  - `objects`: exports `yolo11n.pt` to one dynamic-shape ONNX (opset 17) used at 320 and 640 px, plus `labels.json`.
  - `faces`: copies the verified FER model and derives the dynamic-input YuNet from the zoo file.
  - `vendor`: downloads the pinned ONNX Runtime Web 1.22.0, MediaPipe Tasks Vision 0.10.21 and the gesture task into
    `backend/static/vendor/` (git-ignored), verifying the npm SRI hashes / zoo SHA-256.
- `uv run python scripts/build_pages_artifact.py --out dist/vision [--serve 8013]` assembles `index.html` + `static/` with
  relative URLs (mountable at `/vision/`) and optionally serves it.
- `uv run python scripts/web_bench.py [--only objects/phone] [--out bench.json]` measures FPS, latency and download size per
  mode and device class in headless Chromium with a fake 30 fps camera (`taskset -c 0,1 ...` pins the CPU count).
