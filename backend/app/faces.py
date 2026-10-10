"""Facial expression recognition: YuNet face detector + MobileFaceNet expression classifier.

Both models come from the OpenCV Zoo and run on OpenCV's DNN module (CPU, no torch).
Faces are aligned to 112x112 with a similarity transform from YuNet's 5 landmarks,
exactly as in the OpenCV Zoo reference demo. Everything stays in memory.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from .detector import Detection
from .zoo import ensure_model

ALIGNED_SIZE = 112  # side of the aligned face crop the expression model expects
YUNET_NMS_THRESHOLD = 0.3  # YuNet non-max-suppression IoU threshold
YUNET_TOP_K = 50  # YuNet candidate boxes kept before NMS

EXPRESSIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprised")

# Reference landmark positions (eyes, nose tip, mouth corners) in the 112x112 aligned crop.
_REFERENCE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


class ExpressionDetector:
    """Detects faces and labels each with one of seven ``EXPRESSIONS``."""

    name = "yunet+mobilefacenet-fer"

    def __init__(self, face_conf: float = 0.7, min_face: int = 24) -> None:
        """Load both models (downloading them on first use).

        Args:
            face_conf: YuNet score threshold for accepting a face.
            min_face: Faces whose shorter side is below this many pixels are skipped;
                the expression model is unreliable on tiny faces.
        """
        self.detector = cv2.FaceDetectorYN.create(
            str(ensure_model("face_detection_yunet_2023mar.onnx")),
            "",
            (320, 320),
            face_conf,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
        )
        self.classifier = cv2.dnn.readNet(
            str(ensure_model("facial_expression_recognition_mobilefacenet_2022july.onnx"))
        )
        self.min_face = min_face
        self._lock = threading.Lock()

    def _classify(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Return expression probabilities for one face.

        Args:
            image: Full BGR image.
            landmarks: (5, 2) float32 YuNet landmarks: eyes, nose tip, mouth corners.

        Returns:
            Softmax probabilities, ordered as ``EXPRESSIONS``.
        """
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, _REFERENCE, method=cv2.LMEDS)
        crop = cv2.warpAffine(image, matrix, (ALIGNED_SIZE, ALIGNED_SIZE))
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        self.classifier.setInput(cv2.dnn.blobFromImage(x), "data")
        logits = self.classifier.forward("label")[0].astype(np.float64)
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Find faces and classify their expressions.

        Holds a lock for the whole call since the OpenCV detector's input size is
        mutable shared state. Boxes are clamped to the image bounds.

        Returns:
            One ``Detection`` per face labelled with the top expression; ``extra`` holds
            ``face_confidence``, the ``top3`` expression scores and ``scores``, the probability of
            all seven expressions (in ``EXPRESSIONS`` order).
        """
        h, w = image.shape[:2]
        with self._lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
            if faces is None:
                return []
            out = []
            for face in faces:
                x, y, fw, fh = (float(v) for v in face[:4])
                if min(fw, fh) < self.min_face:
                    continue
                probs = self._classify(image, face[4:14].reshape(5, 2).astype(np.float32))
                top = np.argsort(-probs)[:3]
                out.append(
                    Detection(
                        EXPRESSIONS[top[0]],
                        float(probs[top[0]]),
                        max(0.0, x),
                        max(0.0, y),
                        min(float(w), x + fw),
                        min(float(h), y + fh),
                        extra={
                            "face_confidence": round(float(face[14]), 4),
                            "top3": {EXPRESSIONS[i]: round(float(probs[i]), 4) for i in top},
                            "scores": {
                                n: round(float(p), 4) for n, p in zip(EXPRESSIONS, probs, strict=True)
                            },
                        },
                    )
                )
            return out
