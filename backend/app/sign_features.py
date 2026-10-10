"""Hand-landmark features for the ASL letter classifier (numpy only).

Convention: input is 21 MediaPipe hand landmarks (x, y normalised to image
width/height, z relative to wrist). Hands labelled "Left" are mirrored in x so
every hand looks like a right hand. ``None``/unknown handedness is left as-is
(the model is trained with mirror augmentation so it tolerates both).
Note: MediaPipe's handedness label assumes a mirrored (selfie) image; pass the
label consistently with how the image was fed to MediaPipe.
"""
import numpy as np

FEATURE_VERSION = 1
TIPS = (4, 8, 12, 16, 20)


def landmark_features(points: np.ndarray, handedness: str | None = None) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(21, 3).copy()
    p -= p[0]  # wrist -> origin
    if handedness is not None and handedness.lower().startswith("l"):
        p[:, 0] = -p[:, 0]
    scale = np.linalg.norm(p[:, :2], axis=1).max()
    p /= scale if scale > 1e-9 else 1.0
    tips = p[list(TIPS), :2]
    d = [np.linalg.norm(tips[i] - tips[j]) for i in range(5) for j in range(i + 1, 5)]
    return np.concatenate([p[1:].ravel(), d]).astype(np.float32)  # 60 + 10 = 70
