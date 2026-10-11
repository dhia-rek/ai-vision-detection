"""Build every model and runtime file the static browser app needs.

Steps (all idempotent; ``--only`` runs a subset, ``--force`` redoes them):

``asl``      Export ``backend/app/models/asl_letters.npz`` to the browser JSON (format
             ``dhia-asl-mlp``, see ``scripts/README.md``) and copy it to
             ``backend/static/models/asl_letters_web.json``. Retrained weights only need this step.
``objects``  Export YOLO11n to ONNX (Ultralytics, opset 17, dynamic H/W: one file serves 320 and 640 px
             and rectangular inputs, matching Ultralytics' own rect inference).
``faces``    Copy the SHA-256-verified MobileFaceNet FER ONNX file from the model zoo and derive a
             dynamic-input YuNet from the verified OpenCV Zoo file (only the declared H/W change).
``vendor``   Download the pinned ONNX Runtime Web, MediaPipe Tasks Vision and MediaPipe gesture
             model into ``backend/static/vendor/`` (git-ignored; the Pages workflow runs this step).

Usage::

    uv run --with onnx --with onnxslim python scripts/export_web_models.py
    uv run python scripts/export_web_models.py --only asl
    uv run python scripts/export_web_models.py --only vendor
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

STATIC = ROOT / "backend" / "static"
MODELS = STATIC / "models"
VENDOR = STATIC / "vendor"
ASL_NPZ = ROOT / "backend" / "app" / "models" / "asl_letters.npz"
ASL_CANONICAL = ROOT / "backend" / "app" / "models" / "asl_letters_web.json"
OPSET = 17
YUNET_DYNAMIC = "face_detection_yunet_2023mar_dynamic.onnx"

# Pinned npm packages (integrity = the registry's SRI sha512).
ORT_VERSION = "1.22.0"
ORT_TARBALL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"
ORT_INTEGRITY = (
    "sha512-Ud/+EBo6mhuaQWt/OjaOk0iNWjXqJoeeMFr6xQEERZdIZH2OWpGzuujz7lfuOBjUa6TEE/sc4nb7Da5dNL34fg=="
)
# WebGPU bundle (+ its JSEP wasm) for GPUs; the plain wasm bundle is ~10 MB smaller for CPU-only devices.
ORT_FILES = (
    "dist/ort.webgpu.min.mjs",
    "dist/ort-wasm-simd-threaded.jsep.mjs",
    "dist/ort-wasm-simd-threaded.jsep.wasm",
    "dist/ort.wasm.min.mjs",
    "dist/ort-wasm-simd-threaded.mjs",
    "dist/ort-wasm-simd-threaded.wasm",
)
MEDIAPIPE_VERSION = "0.10.21"
MEDIAPIPE_TARBALL = (
    f"https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{MEDIAPIPE_VERSION}.tgz"
)
MEDIAPIPE_INTEGRITY = (
    "sha512-TuhKH+credq4zLksGbYrnvJ1aLIWMc5r0UHwzxzql4BHECJwIAoBR61ZrqwGOW6ZmSBIzU1t4VtKj8hbxFaKeA=="
)
MEDIAPIPE_FILES = (
    "vision_bundle.mjs",
    "wasm/vision_wasm_internal.js",
    "wasm/vision_wasm_internal.wasm",
    "wasm/vision_wasm_nosimd_internal.js",
    "wasm/vision_wasm_nosimd_internal.wasm",
)


def _log(message: str) -> None:
    print(message, flush=True)


def export_asl(force: bool = False) -> None:
    """Export the ASL MLP checkpoint and place the JSON where the static app fetches it."""
    from export_asl_web import export

    document = export(ASL_NPZ, ASL_CANONICAL)
    target = MODELS / "asl_letters_web.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ASL_CANONICAL, target)
    _log(
        f"asl: feature_version={document['feature_version']} input={document['input_size']} "
        f"classes={len(document['classes'])} -> {target.relative_to(ROOT)} ({target.stat().st_size} bytes)"
    )


def export_objects(force: bool = False) -> None:
    """Export YOLO11n to one dynamic-shape ONNX file and write the COCO label list."""
    from app.zoo import DEFAULT_OBJECT_MODEL, ensure_model
    from ultralytics import YOLO

    out_dir = MODELS / "objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = ensure_model(DEFAULT_OBJECT_MODEL)
    model = YOLO(str(weights))
    names = [model.names[i] for i in range(len(model.names))]
    (out_dir / "labels.json").write_text(json.dumps(names) + "\n", encoding="utf-8")
    target = out_dir / "yolo11n.onnx"
    if target.exists() and not force:
        _log(f"objects: {target.name} exists (use --force to re-export)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "yolo11n.pt"
        shutil.copyfile(weights, copy)
        exported = YOLO(str(copy)).export(
            format="onnx", imgsz=640, opset=OPSET, simplify=True, dynamic=True, half=False, nms=False
        )
        shutil.move(str(exported), target)
    _log(f"objects: {target.relative_to(ROOT)} ({target.stat().st_size} bytes)")


def export_faces(force: bool = False) -> None:
    """Copy the checksum-verified FER model; derive the dynamic-input YuNet used in the browser."""
    import onnx
    from app.zoo import ensure_model

    MODELS.mkdir(parents=True, exist_ok=True)
    fer = "facial_expression_recognition_mobilefacenet_2022july.onnx"
    shutil.copyfile(ensure_model(fer), MODELS / fer)
    _log(f"faces: {MODELS.relative_to(ROOT) / fer} ({(MODELS / fer).stat().st_size} bytes)")
    # The zoo graph declares a fixed 640x640 input although every layer is convolutional; ORT Web
    # (unlike OpenCV) enforces the declared shape. Declaring H and W symbolic changes no weight or op.
    model = onnx.load(str(ensure_model("face_detection_yunet_2023mar.onnx")))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_param, dims[3].dim_param = "height", "width"
    del model.graph.value_info[:]
    for output in model.graph.output:
        for dim in output.type.tensor_type.shape.dim:
            dim.Clear()
    target = MODELS / YUNET_DYNAMIC
    onnx.save(model, str(target))
    _log(f"faces: {target.relative_to(ROOT)} ({target.stat().st_size} bytes)")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https URLs
        return response.read()


def _verify_integrity(data: bytes, integrity: str) -> None:
    algorithm, _, expected = integrity.partition("-")
    digest = base64.b64encode(hashlib.new(algorithm, data).digest()).decode()
    if digest != expected:
        raise RuntimeError(f"npm integrity mismatch for {algorithm}: got {digest}")


def _extract(tarball_url: str, integrity: str, files: tuple[str, ...], out_dir: Path) -> None:
    data = _fetch(tarball_url)
    _verify_integrity(data, integrity)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in files:
            handle = archive.extractfile(f"package/{member}")
            if handle is None:
                raise RuntimeError(f"{member} is missing from {tarball_url}")
            (out_dir / Path(member).name).write_bytes(handle.read())
            _log(f"vendor: {(out_dir / Path(member).name).relative_to(ROOT)}")


def export_vendor(force: bool = False) -> None:
    """Download pinned third-party runtimes (git-ignored; verified by npm SRI / SHA-256)."""
    from app.zoo import ensure_model

    ort_dir = VENDOR / "ort"
    if force or not all((ort_dir / Path(f).name).exists() for f in ORT_FILES):
        _extract(ORT_TARBALL, ORT_INTEGRITY, ORT_FILES, ort_dir)
    mp_dir = VENDOR / "mediapipe"
    if force or not all((mp_dir / Path(f).name).exists() for f in MEDIAPIPE_FILES):
        _extract(MEDIAPIPE_TARBALL, MEDIAPIPE_INTEGRITY, MEDIAPIPE_FILES, mp_dir)
    task = mp_dir / "gesture_recognizer.task"
    shutil.copyfile(ensure_model("gesture_recognizer.task"), task)
    _log(f"vendor: {task.relative_to(ROOT)} ({task.stat().st_size} bytes)")


STEPS = {"asl": export_asl, "objects": export_objects, "faces": export_faces, "vendor": export_vendor}


def main() -> None:
    """Run the selected export steps."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", choices=sorted(STEPS), help="run only these steps")
    parser.add_argument("--force", action="store_true", help="redo steps even if outputs exist")
    args = parser.parse_args()
    for name in args.only or STEPS:
        STEPS[name](args.force)


if __name__ == "__main__":
    main()
