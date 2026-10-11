"""Shared detection types, the YOLO object detector and drawing helpers.

`YoloDetector` wraps an Ultralytics YOLO detector. `StubDetector` returns fixed boxes and
never imports torch, so tests and smoke checks run in seconds without weights.
Face and sign analyzers (faces.py, sign.py) return the same `Detection` type.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected object, face or hand, in pixel coordinates of the source image.

    ``extra`` is excluded from equality comparison.
    """

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
        """Serialise to the JSON shape returned by the API.

        Floats are rounded to keep payloads small; ``extra`` keys are merged at the top
        level and may override the standard keys.

        Returns:
            Dict with ``class``, ``confidence``, ``bbox`` and, if present, ``points``.
        """
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
    """Interface shared by all detectors (objects, faces, sign, stub)."""

    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect things in a BGR image and return them as ``Detection`` objects."""
        ...


class YoloDetector:
    """Ultralytics YOLO detector (COCO, 80 classes)."""

    def __init__(self, weights: str | Path, conf: float = 0.25, device: str = "cpu") -> None:
        """Load YOLO weights.

        Args:
            weights: Path to a YOLO ``.pt`` weights file.
            conf: Minimum confidence for a box to be returned.
            device: Torch device string, e.g. "cpu" or "cuda:0".
        """
        from ultralytics import YOLO  # heavy import, only when actually used

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device
        self.name = Path(str(weights)).stem
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run YOLO on a BGR image.

        The model call is serialised with a lock because one model instance is shared
        between concurrent requests and is not assumed to be thread-safe.

        Returns:
            One ``Detection`` per box, labelled with the COCO class name.
        """
        with self._lock:
            # ultralytics types predict() loosely; with a single ndarray input it is a list[Results].
            results: Any = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)
            result = results[0]
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
        """Return a single "object" box covering the central half of the image."""
        h, w = image.shape[:2]
        return [Detection("object", 0.99, w * 0.25, h * 0.25, w * 0.75, h * 0.75)]


def _color(label: str) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for a label.

    Seeded from the characters rather than ``hash()``, which is salted per process, so
    colours stay the same across runs. Channels are at least 60 so they never turn
    near-black on dark images.
    """
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(label))
    rng = np.random.default_rng(seed)
    b, g, r = (int(c) for c in rng.integers(60, 255, size=3))
    return (b, g, r)


# MediaPipe hand skeleton (pairs of landmark indices).
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip


def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw boxes, labels and hand skeletons with plain OpenCV (no font download needed).

    Line thickness scales with the shorter image side. Hand skeletons are drawn only
    for detections with exactly 21 points. The label sits above the box, clamped so it
    stays inside the image when the box touches the top edge.

    Args:
        image: BGR image; not modified.
        detections: Detections to draw.

    Returns:
        A new annotated BGR image.
    """
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
    """Encode a BGR image as JPEG bytes.

    Raises:
        RuntimeError: If OpenCV fails to encode the image.
    """
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()


def decode_image(raw: bytes) -> np.ndarray | None:
    """Decode image bytes to a BGR array, or return None if they are not a valid image."""
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
