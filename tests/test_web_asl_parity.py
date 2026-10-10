"""The browser's ASL pipeline (backend/static/js/web-asl.js) must match the NumPy reference.

Runs the JavaScript under Node on synthetic and real (MediaPipe) landmarks and compares the 287
features, their mirrored variant and the mirror-averaged probabilities with ``sign_features.py`` and
``asl_classifier.py``. Also pins the ``LetterSmoother`` port to ``smooth_letters``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from app.asl_classifier import AslLetterClassifier
from app.sign import LETTERS, smooth_letters
from app.sign_features import FEATURE_VERSION, N_FEATURES, landmark_features

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "backend/static/models/asl_letters_web.json"
NPZ = ROOT / "backend/app/models/asl_letters.npz"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _node(script: str, payload: dict) -> list:
    done = subprocess.run(
        ["node", str(ROOT / "tests/js" / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(done.stdout)


def _synthetic_hands() -> list[tuple[np.ndarray, float]]:
    hands = []
    for seed, aspect in ((1, 1.0), (2, 1.78), (3, 0.56), (4, 1.33), (5, 1.0), (6, 1.5)):
        rng = np.random.default_rng(seed)
        points = rng.normal(0, 0.04, (21, 3)).astype(np.float32)
        points[:, :2] += rng.uniform(0.25, 0.75, 2)
        points[0, :2] = (0.5, 0.8)
        points[9, :2] = (0.5 + rng.uniform(-0.2, 0.2), 0.5)
        hands.append((points, aspect))
    return hands


def _real_hands() -> list[tuple[np.ndarray, float]]:
    """Landmarks from the bundled sign samples; skipped when MediaPipe cannot run."""
    import cv2
    from app.sign import SignDetector

    try:
        detector = SignDetector("spell")
    except Exception as error:  # noqa: BLE001 - e.g. missing EGL libraries on a bare CI image
        pytest.skip(f"MediaPipe unavailable: {error}")
    hands = []
    for path in sorted((ROOT / "backend/samples/sign").glob("letter_*.jpg")):
        image = cv2.imread(str(path))
        for detection in detector.detect(image):
            h, w = image.shape[:2]
            hands.append((np.array(detection.extra["landmarks"], dtype=np.float32), w / h))
    detector.recognizer.close()
    assert hands, "no hands found in the bundled samples"
    return hands


def test_static_weights_are_the_canonical_export():
    """The copy the browser fetches is byte-identical to the exporter's canonical output."""
    canonical = ROOT / "backend/app/models/asl_letters_web.json"
    assert WEIGHTS.read_bytes() == canonical.read_bytes()


def test_javascript_matches_numpy_features_and_probabilities():
    classifier = AslLetterClassifier(NPZ)
    hands = _synthetic_hands() + _real_hands()
    weights = json.loads(WEIGHTS.read_text())
    assert weights["feature_version"] == FEATURE_VERSION and weights["input_size"] == N_FEATURES
    results = _node(
        "asl_parity.mjs",
        {"weights": weights, "cases": [{"points": p.tolist(), "aspect": a} for p, a in hands]},
    )
    assert len(results) == len(hands)
    for (points, aspect), got in zip(hands, results, strict=True):
        np.testing.assert_allclose(got["features"], landmark_features(points, aspect), atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(
            got["mirrored"], landmark_features(points, aspect, mirror=True), atol=2e-5, rtol=1e-5
        )
        np.testing.assert_allclose(got["proba"], classifier.predict_proba(points, aspect), atol=1e-4)


def test_letter_smoother_matches_python():
    rng = np.random.default_rng(7)
    frames = []
    t = 0.0
    for i in range(60):
        t += 0.05 + (0.8 if i == 30 else 0.0)  # one gap longer than gap_reset_s
        if i in (10, 11, 40):
            frames.append((t, None))
            continue
        raw = rng.dirichlet(np.ones(26) * 0.3)
        frames.append((t, {c: float(v) for c, v in zip(LETTERS, raw, strict=True)}))
    expected = smooth_letters(frames)
    got = _node("smoother_parity.mjs", {"frames": frames})
    for (letter, conf), (js_letter, js_conf) in zip(expected, got, strict=True):
        assert js_letter == letter
        assert js_conf == pytest.approx(conf, abs=1e-9)
