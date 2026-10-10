"""Hand-landmark features for the ASL letter classifier (numpy only).

Input is 21 MediaPipe hand landmarks (x, y normalised to the image width/height, z relative to
the wrist, same scale as x). The features are

* **translation invariant** (the wrist is the origin),
* **scale invariant** (divided by the palm size),
* **aspect-correct** (x and z are multiplied by ``width / height`` so a stretched camera frame
  does not distort the hand),
* **in-plane rotation invariant** (the hand is rotated so wrist -> middle-finger knuckle points up);
  the removed angle is returned as ``cos``/``sin`` so orientation-dependent letters (K/P, G/Q)
  stay separable,
* described by all 210 pairwise landmark distances in the image plane (rotation and mirror
  invariant, and the single most useful addition in leave-one-source-out tests).

Handedness is deliberately *not* used: a letter looks the same on either hand, so the classifier is
trained on mirrored copies and averages its prediction over the hand and its mirror image.
"""

import numpy as np

FEATURE_VERSION = 3
N_FEATURES = 287
_PAIRS = np.triu_indices(21, 1)
TIPS = (4, 8, 12, 16, 20)
PALM = (5, 9, 13, 17)  # finger knuckles (MCP joints)


def landmark_features_batch(points: np.ndarray, aspect: np.ndarray | float = 1.0) -> np.ndarray:
    """Vectorised features for a batch of hands.

    Args:
        points: Landmarks, shape (n, 21, 3).
        aspect: Image width / height, a scalar or one value per hand.

    Returns:
        float32 array (n, 287): x, y, z of landmarks 1..20 after normalisation and rotation (60
        values), cos/sin of the removed in-plane angle (2), the 10 fingertip-fingertip and 5
        fingertip-wrist distances in 3D (15), then the 210 pairwise 2D distances (210).
    """
    p = np.array(points, dtype=np.float64).reshape(-1, 21, 3)
    a = np.broadcast_to(np.asarray(aspect, dtype=np.float64), (len(p),))
    p[:, :, 0] *= a[:, None]
    p[:, :, 2] *= a[:, None]
    p -= p[:, :1]
    scale = np.linalg.norm(p[:, PALM], axis=2).mean(axis=1)
    p /= np.where(scale > 1e-9, scale, 1.0)[:, None, None]
    # In-plane rotation: wrist -> middle knuckle (landmark 9) points up (image y grows downwards).
    ang = np.arctan2(p[:, 9, 0], -p[:, 9, 1])
    c, s = np.cos(ang), np.sin(ang)
    x, y = p[:, :, 0].copy(), p[:, :, 1].copy()
    p[:, :, 0] = c[:, None] * x + s[:, None] * y
    p[:, :, 1] = -s[:, None] * x + c[:, None] * y
    tips = p[:, TIPS]
    tt = [np.linalg.norm(tips[:, i] - tips[:, j], axis=1) for i in range(5) for j in range(i + 1, 5)]
    tw = [np.linalg.norm(tips[:, i], axis=1) for i in range(5)]
    dist = np.stack(tt + tw, axis=1)
    pair = np.linalg.norm(p[:, None, :, :2] - p[:, :, None, :2], axis=3)[:, _PAIRS[0], _PAIRS[1]]
    return np.concatenate([p[:, 1:].reshape(len(p), -1), c[:, None], s[:, None], dist, pair], axis=1).astype(
        np.float32
    )


def landmark_features(points: np.ndarray, aspect: float = 1.0, mirror: bool = False) -> np.ndarray:
    """Features for one hand (see ``landmark_features_batch``).

    Args:
        points: Array reshapeable to (21, 3) of MediaPipe landmarks.
        aspect: Image width / height.
        mirror: Flip the hand left <-> right (as in a mirrored image) before computing features.

    Returns:
        float32 vector of length ``N_FEATURES``.
    """
    p = np.array(points, dtype=np.float64).reshape(1, 21, 3)
    if mirror:
        p[:, :, 0] = 1.0 - p[:, :, 0]
    return landmark_features_batch(p, aspect)[0]
