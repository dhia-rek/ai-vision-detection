"""Pure-numpy ASL letter classifier (inference only)."""

import numpy as np

from .sign_features import FEATURE_VERSION, landmark_features


class AslLetterClassifier:
    def __init__(self, path: str):
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

    def predict(self, points, handedness=None):
        x = (landmark_features(points, handedness) - self.mean) / self.std
        for k, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if k < len(self.layers) - 1:
                x = np.maximum(x, 0)
        e = np.exp(x - x.max())
        p = e / e.sum()
        idx = np.argsort(p)[::-1][:3]
        top3 = [(self.classes[i], float(p[i])) for i in idx]
        return top3[0][0], top3[0][1], top3
