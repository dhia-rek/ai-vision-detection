"""Pretrained model registry: downloaded on first use, verified by SHA-256, cached on disk.

Cache dir: $AVD_MODEL_DIR, default ~/.cache/ai-vision-detection/models.
Nothing here is committed to the repo; `python -m app.zoo` pre-downloads everything.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    size_mb: float
    license: str
    source: str


MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec(
            "yolov8n.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
            6.2,
            "AGPL-3.0",
            "Ultralytics YOLOv8n, COCO 80 classes",
        ),
        ModelSpec(
            "face_detection_yunet_2023mar.onnx",
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
            0.2,
            "MIT",
            "OpenCV Zoo YuNet face detector",
        ),
        ModelSpec(
            "facial_expression_recognition_mobilefacenet_2022july.onnx",
            "https://huggingface.co/opencv/facial_expression_recognition/resolve/main/"
            "facial_expression_recognition_mobilefacenet_2022july.onnx",
            "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9",
            4.6,
            "Apache-2.0",
            "OpenCV Zoo MobileFaceNet expression classifier (trained on RAF-DB)",
        ),
        ModelSpec(
            "gesture_recognizer.task",
            "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/"
            "gesture_recognizer.task",
            "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
            8.0,
            "Apache-2.0",
            "MediaPipe hand landmarks + canned gesture classifier",
        ),
    ]
}

_lock = threading.Lock()


def model_dir() -> Path:
    return Path(os.getenv("AVD_MODEL_DIR", Path.home() / ".cache" / "ai-vision-detection" / "models"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(name: str) -> Path:
    """Return the local path of a registered model, downloading it on first use."""
    spec = MODELS[name]
    path = model_dir() / name
    with _lock:
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} ({spec.size_mb} MB, {spec.license}) ...", flush=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            urllib.request.urlretrieve(spec.url, tmp_path)  # noqa: S310 - fixed https URLs
            digest = _sha256(tmp_path)
            if digest != spec.sha256:
                raise RuntimeError(f"Checksum mismatch for {name}: {digest}")
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    for model in MODELS:
        print(ensure_model(model))
