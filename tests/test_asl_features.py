import numpy as np
from app.sign_features import (
    FEATURE_VERSION,
    N_FEATURES,
    landmark_features,
    landmark_features_batch,
)


def hand(seed=7):
    """A non-degenerate, deterministic MediaPipe-shaped hand."""
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.04, (21, 3))
    p[:, 0] += 0.5
    p[:, 1] += 0.5
    p[0] = (0.5, 0.8, 0.0)
    p[9] = (0.54, 0.47, -0.02)
    return p


def test_feature_contract_and_batch_matches_single():
    hands = np.stack([hand(7), hand(11), hand(19)])
    aspects = np.array([1.0, 1.4, 0.75])
    batch = landmark_features_batch(hands, aspects)
    assert FEATURE_VERSION == 3
    assert N_FEATURES == 287
    assert batch.shape == (3, N_FEATURES)
    assert batch.dtype == np.float32
    for i in range(3):
        np.testing.assert_allclose(batch[i], landmark_features(hands[i], aspects[i]), atol=1e-7)


def test_translation_and_uniform_scale_invariance():
    p = hand()
    base = landmark_features(p)
    translated = p + np.array([4.3, -2.1, 0.7])
    np.testing.assert_allclose(landmark_features(translated), base, atol=2e-6)
    scaled = p[0] + 3.7 * (p - p[0])
    np.testing.assert_allclose(landmark_features(scaled), base, atol=2e-6)


def test_in_plane_rotation_invariance_and_orientation_encoding():
    p = hand()
    angle = 0.63
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    turned = p.copy()
    turned[:, :2] = (p[:, :2] - p[0, :2]) @ rot.T + p[0, :2]
    base = landmark_features(p)
    actual = landmark_features(turned)
    np.testing.assert_allclose(actual[:60], base[:60], atol=2e-6)
    np.testing.assert_allclose(actual[62:], base[62:], atol=2e-6)
    # The removed hand angle is retained as cos/sin, and increases by the applied rotation.
    expected_angle = np.arctan2(base[61], base[60]) + angle
    np.testing.assert_allclose(actual[60:62], [np.cos(expected_angle), np.sin(expected_angle)], atol=2e-6)


def test_mirror_and_aspect_handling():
    p = hand()
    mirrored = p.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    np.testing.assert_allclose(landmark_features(p, mirror=True), landmark_features(mirrored), atol=1e-7)

    aspect = 1.6
    stretched = p.copy()
    stretched[:, 0] /= aspect
    stretched[:, 2] /= aspect
    np.testing.assert_allclose(landmark_features(stretched, aspect=aspect), landmark_features(p), atol=2e-6)


def test_degenerate_hand_has_finite_features():
    result = landmark_features(np.full((21, 3), [0.2, 0.4, -0.1]))
    assert result.shape == (N_FEATURES,)
    assert np.isfinite(result).all()
