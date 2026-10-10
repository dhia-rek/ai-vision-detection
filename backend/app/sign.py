"""Sign language: MediaPipe hand landmarks -> ASL fingerspelled letters or common gestures.

One MediaPipe Gesture Recognizer pass gives, per hand, 21 landmarks and a canned gesture.
- "spell" mode: landmarks go through our small ASL letter classifier (asl_classifier.py).
- "gestures" mode: the canned gesture is mapped to a meaning (thumbs up = yes, ...).

These are static hand shapes. Letters J and Z and most real ASL words involve motion,
so this is fingerspelling + a few gestures, not sign-language translation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import Detection
from .zoo import ensure_model

ASL_MODEL = Path(__file__).resolve().parent / "models" / "asl_letters.npz"

# MediaPipe canned gestures -> what we display. Honest names: these are hand shapes.
GESTURES = {
    "Thumb_Up": "thumbs up (yes)",
    "Thumb_Down": "thumbs down (no)",
    "Open_Palm": "open palm (hello / stop)",
    "ILoveYou": "I love you",
    "Victory": "victory / peace",
    "Pointing_Up": "pointing up (wait)",
    "Closed_Fist": "closed fist",
}

SIGN_MODES = ("spell", "gestures")


class SignDetector:
    def __init__(self, mode: str = "spell", min_confidence: float = 0.5, max_hands: int = 2):
        if mode not in SIGN_MODES:
            raise ValueError(f"sign mode must be one of {SIGN_MODES}")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model("gesture_recognizer.task"))),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.mode = mode
        self.min_confidence = min_confidence
        self.letters = None
        if mode == "spell":
            from .asl_classifier import AslLetterClassifier

            self.letters = AslLetterClassifier(ASL_MODEL)
        self.name = f"mediapipe-hands+{'asl-letters' if mode == 'spell' else 'gestures'}"
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self.recognizer.recognize(mp_image)
        out = []
        for i, landmarks in enumerate(result.hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            handed = result.handedness[i][0].category_name if result.handedness else None
            gesture = result.gestures[i][0] if result.gestures and result.gestures[i] else None
            extra = {"handedness": handed}
            if self.mode == "spell":
                letter, prob, top3 = self.letters.predict(pts, handed)
                label, conf = letter, prob
                extra["top3"] = {k: round(v, 4) for k, v in top3}
            else:
                if gesture is None or gesture.category_name not in GESTURES:
                    label, conf = "no gesture", float(gesture.score) if gesture else 0.0
                else:
                    label, conf = GESTURES[gesture.category_name], float(gesture.score)
                extra["gesture"] = gesture.category_name if gesture else "None"
            if conf < self.min_confidence:
                extra["low_confidence"] = True
            xs, ys = pts[:, 0] * w, pts[:, 1] * h
            pad = 0.1 * max(xs.max() - xs.min(), ys.max() - ys.min())
            out.append(
                Detection(
                    label,
                    float(conf),
                    float(max(0, xs.min() - pad)),
                    float(max(0, ys.min() - pad)),
                    float(min(w, xs.max() + pad)),
                    float(min(h, ys.max() + pad)),
                    points=tuple((float(x), float(y)) for x, y in zip(xs, ys, strict=True)),
                    extra=extra,
                )
            )
        return out


@dataclass
class WordBuilder:
    """Turns a stream of per-frame letters into text.

    A letter is committed once it has been held steadily for `hold_s` seconds. To type the same
    letter twice, change hand shape or drop the hand briefly. No hand for `space_s` seconds
    after a letter adds a space. The browser UI uses the same rules (static/index.html).
    """

    hold_s: float = 0.8
    space_s: float = 1.5
    min_confidence: float = 0.6
    text: str = ""
    _current: str | None = None
    _since: float = 0.0
    _committed: bool = False
    _last_seen: float | None = None

    def feed(self, letter: str | None, confidence: float, t: float) -> str | None:
        """Feed one frame at time `t` (seconds). Returns the character committed, if any."""
        if letter is None or confidence < self.min_confidence:
            self._current, self._committed = None, False
            if (
                self._last_seen is not None
                and t - self._last_seen >= self.space_s
                and self.text
                and not self.text.endswith(" ")
            ):
                self.text += " "
                return " "
            return None
        self._last_seen = t
        if letter != self._current:
            self._current, self._since, self._committed = letter, t, False
            return None
        if not self._committed and t - self._since >= self.hold_s:
            self._committed = True
            self.text += letter
            return letter
        return None
