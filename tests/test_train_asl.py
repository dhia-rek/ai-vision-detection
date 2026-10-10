import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from app.asl_classifier import AslLetterClassifier
from app.sign_features import N_FEATURES, landmark_features_batch

SCRIPT = Path(__file__).parents[1] / "scripts/train_asl.py"
spec = importlib.util.spec_from_file_location("train_asl_test_module", SCRIPT)
train_asl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_asl
spec.loader.exec_module(train_asl)


def make_source(y, name="source", offset=0.0):
    src = train_asl.Source.__new__(train_asl.Source)
    src.y = np.asarray(y, dtype=int)
    n = len(src.y)
    rng = np.random.default_rng(91 + int(offset * 10))
    src.P = rng.normal(0, 0.03, (n, 21, 3))
    src.P[:, :, :2] += 0.5 + offset
    src.P[:, 0] = (0.5 + offset, 0.8, 0.0)
    src.P[:, 9] = (0.54 + offset, 0.47, -0.02)
    src.size = np.tile([640.0, 480.0], (n, 1))
    src.grp = np.arange(n)
    src.hand = np.array(["Right"] * n)
    src.name = name
    return src


def test_wilson_interval_sanity():
    lo0, hi0 = train_asl.wilson(0, 100)
    assert lo0 == 0.0 and hi0 == pytest.approx(0.037, abs=1e-4)
    lo, hi = train_asl.wilson(100, 100)
    assert 0.96 < lo < 0.97 and hi == pytest.approx(1.0)
    lo, hi = train_asl.wilson(50, 100)
    assert lo < 0.5 < hi
    assert lo == pytest.approx(1 - hi)
    assert lo <= 0.5 <= hi


def test_report_metrics_and_confusion_counts():
    src = make_source([0, 0, 9, 25, 1])
    prob = np.full((5, 26), 0.01 / 25)
    predictions = [0, 1, 9, 0, 1]
    confidences = [0.9, 0.7, 0.55, 0.8, 0.4]
    for row, (pred, conf) in enumerate(zip(predictions, confidences, strict=True)):
        prob[row, :] = (1 - conf) / 25
        prob[row, pred] = conf
    res = train_asl.report(src, prob)
    assert res["n"] == 5
    assert res["accuracy"] == 0.6
    assert res["per_letter"] == {"A": 0.5, "B": 1.0, "J": 1.0, "Z": 0.0}
    assert res["confusion"][0][0] == 1 and res["confusion"][0][1] == 1
    assert res["confusion"][25][0] == 1
    assert res["static_accuracy"] == 2 / 3
    assert res["n_static"] == 3
    assert res["confident_coverage"] == 0.6
    assert res["confident_accuracy"] == 1 / 3


def test_balanced_epoch_only_draws_present_classes():
    a = make_source([0, 0, 2], "a")
    b = make_source([2, 2, 5], "b", 0.1)
    P, aspect, y = train_asl.balanced_epoch([a, b], per_class=4, rng=np.random.default_rng(4))
    assert P.shape == (12, 21, 3)
    assert aspect.shape == (12,)
    assert np.isfinite(P).all() and np.isfinite(aspect).all()
    assert np.unique(y, return_counts=True)[0].tolist() == [0, 2, 5]
    assert np.unique(y, return_counts=True)[1].tolist() == [4, 4, 4]


def test_augment_features_is_seeded_finite_and_strength_zero_is_flip_only():
    src = make_source([0, 1, 2, 3, 4, 5, 6, 7])
    aspect = src.aspect()
    first = train_asl.augment_features(src.P, aspect, np.random.default_rng(123))
    repeated = train_asl.augment_features(src.P, aspect, np.random.default_rng(123))
    different = train_asl.augment_features(src.P, aspect, np.random.default_rng(124))
    assert first.shape == (len(src.y), N_FEATURES)
    assert first.dtype == np.float32 and np.isfinite(first).all()
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)

    rng = np.random.default_rng(32)
    flips = rng.random(len(src.P)) < 0.5
    expected = src.P.copy()
    expected[:, :, 0] *= aspect[:, None]
    expected[:, :, 2] *= aspect[:, None]
    expected -= expected[:, :1]
    scale = np.linalg.norm(expected[:, [5, 9, 13, 17]], axis=2).mean(1)
    expected /= np.where(scale > 1e-9, scale, 1.0)[:, None, None]
    expected[flips, :, 0] *= -1
    got = train_asl.augment_features(src.P, aspect, np.random.default_rng(32), strength=0)
    np.testing.assert_allclose(got, landmark_features_batch(expected), atol=1e-6)


def test_standardiser_and_export_round_trip(tmp_path):
    rng = np.random.default_rng(8)
    F = rng.normal(size=(12, N_FEATURES)).astype(np.float32)
    sc = train_asl.Standardiser(F)
    standard = sc(F)
    assert standard.dtype == np.float32
    np.testing.assert_allclose(standard.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(standard * sc.std + sc.mean, F, atol=3e-6)

    arrays = {"sc": sc}
    dims = [N_FEATURES, 8, 26]
    for i, (left, right) in enumerate(zip(dims, dims[1:], strict=False), 1):
        arrays[f"W{i}"] = rng.normal(0, 0.08, (left, right)).astype(np.float32)
        arrays[f"b{i}"] = rng.normal(0, 0.08, right).astype(np.float32)
    path = tmp_path / "tiny-asl.npz"
    train_asl.export(arrays, str(path))
    clf = AslLetterClassifier(path)
    x = rng.normal(size=(3, N_FEATURES)).astype(np.float32)
    h = sc(x) @ arrays["W1"] + arrays["b1"]
    logits = np.maximum(h, 0) @ arrays["W2"] + arrays["b2"]
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(clf.proba_batch(x), expected, rtol=1e-6, atol=1e-7)


def test_legacy_v1_probs_is_not_applicable():
    # The original v1 model artifact was removed; no committed-history file is available here.
    assert not (Path(__file__).parents[1] / "backend/app/models/asl_letters_v1.npz").exists()
