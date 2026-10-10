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

EXPRESSIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprised")

# Reference landmark positions (eyes, nose tip, mouth corners) in the 112x112 aligned crop.
_REFERENCE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


class ExpressionDetector:
    name = "yunet+mobilefacenet-fer"

    def __init__(self, face_conf: float = 0.7, min_face: int = 24):
        self.detector = cv2.FaceDetectorYN.create(
            str(ensure_model("face_detection_yunet_2023mar.onnx")), "", (320, 320), face_conf, 0.3, 50
        )
        self.classifier = cv2.dnn.readNet(
            str(ensure_model("facial_expression_recognition_mobilefacenet_2022july.onnx"))
        )
        self.min_face = min_face
        self._lock = threading.Lock()

    def _classify(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, _REFERENCE, method=cv2.LMEDS)
        crop = cv2.warpAffine(image, matrix, (112, 112))
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        self.classifier.setInput(cv2.dnn.blobFromImage(x), "data")
        logits = self.classifier.forward("label")[0].astype(np.float64)
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def detect(self, image: np.ndarray) -> list[Detection]:
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
                        },
                    )
                )
            return out
