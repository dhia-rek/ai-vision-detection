"""Pure-numpy ASL letter classifier (inference only)."""

from pathlib import Path

import numpy as np

from .sign_features import FEATURE_VERSION, landmark_features


class AslLetterClassifier:
    """Small MLP/linear softmax classifier over hand-landmark features, run in plain numpy.

    Weights come from an ``.npz`` file written by ``scripts/train_asl.py``. Avoiding
    torch/sklearn at inference keeps the backend light.

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

    def predict(
        self, points: np.ndarray, handedness: str | None = None
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Classify one hand.

        Args:
            points: 21 MediaPipe landmarks (x, y, z), anything reshapeable to (21, 3).
            handedness: "Left"/"Right" label from MediaPipe, or None if unknown.

        Returns:
            ``(best_letter, best_probability, top3)`` where ``top3`` is a list of the
            three most likely ``(letter, probability)`` pairs, best first.
        """
        x = (landmark_features(points, handedness) - self.mean) / self.std
        for k, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if k < len(self.layers) - 1:
                x = np.maximum(x, 0)
        e = np.exp(x - x.max())  # subtract max for a numerically stable softmax
        p = e / e.sum()
        idx = np.argsort(p)[::-1][:3]
        top3 = [(self.classes[i], float(p[i])) for i in idx]
        return top3[0][0], top3[0][1], top3
