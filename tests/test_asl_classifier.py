from pathlib import Path

import numpy as np
import pytest
from app.asl_classifier import AslLetterClassifier
from app.sign_features import landmark_features

MODEL = Path(__file__).parents[1] / "backend/app/models/asl_letters.npz"


def hand():
    rng = np.random.default_rng(23)
    p = rng.normal(0, 0.035, (21, 3))
    p[:, :2] += 0.5
    p[0] = (0.5, 0.8, 0.0)
    p[9] = (0.54, 0.47, -0.02)
    return p


def test_committed_classifier_contract_and_predict_top3():
    clf = AslLetterClassifier(MODEL)
    assert clf.classes == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    features = np.stack([landmark_features(hand()), landmark_features(hand() + [0.01, -0.02, 0.0])])
    prob = clf.proba_batch(features)
    assert prob.shape == (2, 26)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-7)

    p = hand()
    best, confidence, top3 = clf.predict(p)
    expected = clf.predict_proba(p)
    assert len(top3) == 3
    assert top3 == sorted(top3, key=lambda item: item[1], reverse=True)
    assert best == top3[0][0]
    assert confidence == top3[0][1]
    for letter, value in top3:
        assert value == expected[clf.classes.index(letter)]


def test_predict_proba_is_mirror_averaged():
    clf = AslLetterClassifier(MODEL)
    p = hand()
    mirrored = p.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    np.testing.assert_allclose(clf.predict_proba(p), clf.predict_proba(mirrored), atol=1e-6)


def test_wrong_feature_version_is_rejected(tmp_path):
    with np.load(MODEL, allow_pickle=False) as original:
        arrays = {name: original[name] for name in original.files}
    arrays["feature_version"] = np.array(int(arrays["feature_version"]) + 1)
    bad = tmp_path / "wrong-version.npz"
    np.savez(bad, **arrays)
    with pytest.raises(ValueError, match="feature_version"):
        AslLetterClassifier(bad)
