"""The committed browser JSON must agree with the Python MLP checkpoint."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from app.asl_classifier import AslLetterClassifier
from app.sign_features import FEATURE_VERSION, N_FEATURES, landmark_features_batch

ROOT = Path(__file__).parents[1]
MODEL = ROOT / "backend/app/models/asl_letters.npz"
WEB = ROOT / "backend/app/models/asl_letters_web.json"
sys.path.insert(0, str(ROOT / "scripts"))
from export_asl_web import export  # noqa: E402


def _web_forward(features, document):
    """Port the browser's row-major dense/ReLU/softmax operations to NumPy."""
    x = np.asarray(features, dtype=np.float32) - np.asarray(document["mean"], dtype=np.float32)
    x = x / np.asarray(document["std"], dtype=np.float32)
    for layer in document["layers"]:
        weights = np.asarray(layer["weights"], dtype=np.float32).reshape(
            layer["input_size"], layer["output_size"]
        )
        bias = np.asarray(layer["bias"], dtype=np.float32)
        output = np.empty(layer["output_size"], dtype=np.float32)
        for col in range(layer["output_size"]):
            total = bias[col]
            for row in range(layer["input_size"]):
                total = np.float32(total + np.float32(x[row] * weights[row, col]))
            output[col] = max(0, total) if layer["activation"] == "relu" else total
        x = output
    exps = np.exp(x - x.max()).astype(np.float32)
    return exps / exps.sum(dtype=np.float32)


def test_browser_export_matches_python_and_is_reproducible(tmp_path):
    document = json.loads(WEB.read_text())
    assert document["format"] == "dhia-asl-mlp"
    assert document["format_version"] == 1
    assert document["feature_version"] == FEATURE_VERSION == 3
    assert document["input_size"] == N_FEATURES == 287
    assert document["classes"] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert document["source_sha256"] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert export(MODEL, tmp_path / "web.json") == document
    assert (tmp_path / "web.json").read_bytes() == WEB.read_bytes()

    classifier = AslLetterClassifier(MODEL)
    for seed, aspect in ((4, 1.0), (17, 1.5), (31, 0.8)):
        rng = np.random.default_rng(seed)
        points = rng.normal(0, 0.035, (21, 3))
        points[:, :2] += 0.5
        points[0] = (0.5, 0.8, 0.0)
        points[9] = (0.54, 0.47, -0.02)
        mirrored = points.copy()
        mirrored[:, 0] = 1.0 - mirrored[:, 0]
        features = landmark_features_batch(np.stack([points, mirrored]), aspect)
        browser_prob = np.mean([_web_forward(row, document) for row in features], axis=0)
        python_prob = classifier.predict_proba(points, aspect)
        np.testing.assert_allclose(browser_prob, python_prob, atol=1e-4, rtol=1e-4)
        assert document["classes"][browser_prob.argmax()] == classifier.predict(points, aspect)[0]


@pytest.mark.parametrize("invalid", ["zero_std", "missing_middle_layer"])
def test_export_rejects_invalid_checkpoint(tmp_path, invalid):
    with np.load(MODEL, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if invalid == "zero_std":
        arrays["std"] = arrays["std"].copy()
        arrays["std"][0] = 0
    else:
        del arrays["W2"]
        del arrays["b2"]
    bad = tmp_path / "invalid.npz"
    np.savez(bad, **arrays)
    with pytest.raises(ValueError):
        export(bad, tmp_path / "invalid.json")
