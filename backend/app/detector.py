"""Shared detection types, the YOLO object detector and drawing helpers.

`YoloDetector` wraps Ultralytics YOLOv8. `StubDetector` returns fixed boxes and
never imports torch, so tests and smoke checks run in seconds without weights.
Face and sign analyzers (faces.py, sign.py) return the same `Detection` type.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    # Optional keypoints in pixels (e.g. 21 hand landmarks), drawn as a skeleton.
    points: tuple[tuple[float, float], ...] = ()
    # Mode-specific extras (top-3 scores, gesture, letter, ...), passed through to JSON.
    extra: dict = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict:
        d = {
            "class": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": {k: round(getattr(self, k), 1) for k in ("x1", "y1", "x2", "y2")},
        }
        if self.points:
            d["points"] = [[round(x, 1), round(y, 1)] for x, y in self.points]
        d.update(self.extra)
        return d


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    """Ultralytics YOLOv8 detector (COCO, 80 classes)."""

    def __init__(self, weights: str | Path, conf: float = 0.25, device: str = "cpu"):
        from ultralytics import YOLO  # heavy import, only when actually used

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device
        self.name = Path(str(weights)).stem
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Detection]:
        with self._lock:
            result = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
        names = result.names
        out = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append(Detection(names[int(box.cls)], float(box.conf), x1, y1, x2, y2))
        return out


class StubDetector:
    """Deterministic fake detector: one box in the centre of every image."""

    name = "stub"

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection("object", 0.99, w * 0.25, h * 0.25, w * 0.75, h * 0.75)]


def _color(label: str) -> tuple[int, int, int]:
    # Stable per-class colour (BGR) without depending on Python's salted hash().
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(label))
    rng = np.random.default_rng(seed)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


# MediaPipe hand skeleton (pairs of landmark indices).
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip


def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw boxes and labels with plain OpenCV (no font download needed)."""
    out = image.copy()
    thickness = max(1, round(min(out.shape[:2]) / 400))
    for d in detections:
        color = _color(d.label)
        if len(d.points) == 21:
            pts = [(int(x), int(y)) for x, y in d.points]
            for a, b in HAND_EDGES:
                cv2.line(out, pts[a], pts[b], (255, 255, 255), thickness + 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(out, p, thickness + 2, color, -1, cv2.LINE_AA)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, thickness + 1)
        text = f"{d.label} {d.confidence:.2f}"
        scale = 0.5 * thickness
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        top = max(p1[1], th + base + 2)
        cv2.rectangle(out, (p1[0], top - th - base - 2), (p1[0] + tw + 2, top), color, -1)
        cv2.putText(
            out,
            text,
            (p1[0] + 1, top - base),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return out


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()


def decode_image(raw: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
