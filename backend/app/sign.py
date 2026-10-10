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
from typing import TYPE_CHECKING

import numpy as np

from .detector import Detection
from .zoo import ensure_model

if TYPE_CHECKING:
    from .asl_classifier import AslLetterClassifier

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

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SIGN_MODES = ("spell", "gestures")


class SignDetector:
    """MediaPipe hand detector that labels each hand as an ASL letter or a gesture."""

    def __init__(self, mode: str = "spell", min_confidence: float = 0.5, max_hands: int = 2) -> None:
        """Load MediaPipe (and the letter classifier in spell mode).

        Args:
            mode: One of ``SIGN_MODES``: "spell" (ASL letters) or "gestures".
            min_confidence: Detections below this get ``extra["low_confidence"] = True``;
                they are still returned.
            max_hands: Maximum number of hands MediaPipe looks for.

        Raises:
            ValueError: If ``mode`` is not in ``SIGN_MODES``.
        """
        if mode not in SIGN_MODES:
            raise ValueError(f"sign mode must be one of {SIGN_MODES}")
        import mediapipe as mp  # heavy imports, only when actually used
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
        self.letters: AslLetterClassifier | None = None
        if mode == "spell":
            from .asl_classifier import AslLetterClassifier

            self.letters = AslLetterClassifier(ASL_MODEL)
        self.name = f"mediapipe-hands+{'asl-letters' if mode == 'spell' else 'gestures'}"
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect hands in a BGR image.

        In spell mode the label is the predicted letter and ``extra`` has ``top3``; in
        gestures mode the label is the mapped gesture meaning ("no gesture" if
        MediaPipe's gesture is missing or unmapped) and ``extra`` has the raw ``gesture``
        name. ``extra["handedness"]`` is always set. The box is the landmark bounds
        padded by 10% of the larger side and clamped to the image; ``points`` are pixels.

        Returns:
            One ``Detection`` per hand.
        """
        h, w = image.shape[:2]
        # MediaPipe expects RGB; reverse the channel axis and make it contiguous.
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self.recognizer.recognize(mp_image)
        out = []
        for i, landmarks in enumerate(result.hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            handed = result.handedness[i][0].category_name if result.handedness else None
            gesture = result.gestures[i][0] if result.gestures and result.gestures[i] else None
            extra = {"handedness": handed, "landmarks": np.round(pts, 4).tolist()}
            if self.mode == "spell":
                assert self.letters is not None  # set in __init__ when mode == "spell"
                probs = self.letters.predict_proba(pts, w / h)
                order = np.argsort(probs)[::-1]
                label, conf = self.letters.classes[order[0]], float(probs[order[0]])
                extra["top3"] = {self.letters.classes[i]: round(float(probs[i]), 4) for i in order[:3]}
                extra["scores"] = {
                    c: round(float(v), 3) for c, v in zip(self.letters.classes, probs, strict=True)
                }
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
        """Feed one frame observed at time `t` (seconds).

        A missing or low-confidence letter resets the current hold, so the same letter
        can be typed again afterwards.

        Args:
            letter: Predicted letter, or None when no hand is visible.
            confidence: Confidence of the prediction; below ``min_confidence`` counts as
                no letter.
            t: Frame timestamp in seconds, non-decreasing.

        Returns:
            The character appended to ``text`` by this frame (a letter or a space), or
            None if nothing was committed.
        """
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


@dataclass
class LetterSmoother:
    """Smooth per-frame ASL letter probabilities with temporal hysteresis.

    Args:
        alpha: Weight of the newest probability vector in the exponential average.
        switch_margin: Required probability advantage before changing the current letter.
        min_conf: Minimum smoothed confidence that yields a letter.
        gap_reset_s: Missing-hand duration that resets the probability state.
    """

    alpha: float = 0.45
    switch_margin: float = 0.12
    min_conf: float = 0.5
    gap_reset_s: float = 0.6
    ema: dict[str, float] | None = None
    current: str | None = None
    last_seen: float | None = None

    def reset(self) -> None:
        """Clear the averaged probabilities and current letter."""
        self.ema = None
        self.current = None
        self.last_seen = None

    def feed(self, scores: dict[str, float] | None, t: float) -> tuple[str | None, float]:
        """Feed letter probabilities at timestamp ``t`` and return the smoothed result.

        A missing hand returns ``(None, 0)`` and clears state after a sufficiently long gap.
        Probabilities are averaged by letter key; a new winner must exceed the current
        letter by ``switch_margin`` before it can replace it.

        Args:
            scores: Probability for each letter, or ``None`` when no hand is visible.
            t: Monotonic frame time in seconds.

        Returns:
            The selected letter (or ``None`` below the confidence threshold) and its
            smoothed probability.
        """
        if self.last_seen is not None and t - self.last_seen > self.gap_reset_s:
            self.reset()
        if scores is None:
            return None, 0.0
        if self.ema is None:
            self.ema = {letter: float(scores.get(letter, 0.0)) for letter in LETTERS}
        else:
            self.ema = {
                letter: self.alpha * float(scores.get(letter, 0.0)) + (1 - self.alpha) * self.ema[letter]
                for letter in LETTERS
            }
        self.last_seen = t
        winner = max(self.ema, key=self.ema.__getitem__)
        if self.current is None or self.ema[winner] > self.ema[self.current] + self.switch_margin:
            self.current = winner
        confidence = self.ema[self.current]
        return (self.current if confidence >= self.min_conf else None), confidence


def smooth_letters(
    frames: list[tuple[float, dict[str, float] | None]],
) -> list[tuple[str | None, float]]:
    """Smooth an ordered sequence of score vectors for video transcription.

    Args:
        frames: Ordered ``(timestamp, scores)`` pairs, with ``None`` scores when no hand is
            visible. Timestamps are seconds, so missing-hand reset timing matches video time.

    Returns:
        Smoothed letter and confidence pairs, one for each input frame.
    """
    smoother = LetterSmoother()
    return [smoother.feed(scores, t) for t, scores in frames]
