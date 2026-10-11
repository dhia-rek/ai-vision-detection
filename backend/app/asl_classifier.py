"""Pure-numpy ASL letter classifier (inference only)."""

from pathlib import Path

import numpy as np

from .sign_features import FEATURE_VERSION, landmark_features_batch


class AslLetterClassifier:
    """Small MLP over rotation-invariant hand-landmark features, run in plain numpy.

    Weights come from an ``.npz`` file written by ``scripts/train_asl.py``. Avoiding
    torch/sklearn at inference keeps the backend light. Predictions are averaged over the
    hand and its mirror image, so left and right hands are treated alike.

    Attributes:
        classes: Letter labels, in output-column order.
        mean: Per-feature mean used to standardise inputs.
        std: Per-feature standard deviation used to standardise inputs.
        layers: List of ``(W, b)`` pairs, applied in order (ReLU between layers).
    """

    def __init__(self, path: str | Path) -> None:
        """Load the classifier weights.

        Args:
            path: Path to the ``.npz`` file (loaded with ``allow_pickle=False``).

        Raises:
            ValueError: If the file's ``feature_version`` differs from ``FEATURE_VERSION``,
                i.e. it was trained on features computed differently.
        """
        z = np.load(path, allow_pickle=False)
        if int(z["feature_version"]) != FEATURE_VERSION:
            raise ValueError("feature_version mismatch")
        self.classes = [str(c) for c in z["classes"]]
        self.mean, self.std = z["mean"], z["std"]
        self.layers = []
        i = 1
        while f"W{i}" in z:
            self.layers.append((z[f"W{i}"], z[f"b{i}"]))
            i += 1

    def proba_batch(self, features: np.ndarray) -> np.ndarray:
        """Class probabilities for a batch of feature vectors (no mirroring).

        Args:
            features: Array (n, N_FEATURES) from ``landmark_features_batch``.

        Returns:
            Array (n, len(classes)) of softmax probabilities.
        """
        x = (np.atleast_2d(features) - self.mean) / self.std
        for k, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if k < len(self.layers) - 1:
                x = np.maximum(x, 0)
        e = np.exp(x - x.max(axis=1, keepdims=True))  # subtract max for a stable softmax
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, points: np.ndarray, aspect: float = 1.0) -> tuple[str, float, list[tuple[str, float]]]:
        """Classify one hand.

        Args:
            points: 21 MediaPipe landmarks (x, y, z), anything reshapeable to (21, 3), with x/y
                normalised to the image width/height.
            aspect: Image width / height, so the hand keeps its true proportions.

        Returns:
            ``(best_letter, best_probability, top3)`` where ``top3`` is a list of the
            three most likely ``(letter, probability)`` pairs, best first.
        """
        p_mean = self.predict_proba(points, aspect)
        idx = np.argsort(p_mean)[::-1][:3]
        top3 = [(self.classes[i], float(p_mean[i])) for i in idx]
        return top3[0][0], top3[0][1], top3

    def predict_proba(self, points: np.ndarray, aspect: float = 1.0) -> np.ndarray:
        """Mirror-averaged probability vector for one hand, in ``classes`` order."""
        p = np.array(points, dtype=np.float64).reshape(1, 21, 3)
        both = np.concatenate([p, p])
        both[1, :, 0] = 1.0 - both[1, :, 0]
        return self.proba_batch(landmark_features_batch(both, aspect)).mean(axis=0)
