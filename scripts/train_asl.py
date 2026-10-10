"""Train the ASL letter classifier. Usage:

uv run --with numpy --with scikit-learn --with huggingface_hub \
    python scripts/train_asl.py --cache /tmp/asl-now
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.sign_features import FEATURE_VERSION, landmark_features  # noqa: E402


def load(cache):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="sid220/asl-now-fingerspelling", repo_type="dataset", local_dir=cache)
    X, y = [], []
    for d in sorted(Path(cache).iterdir()):
        if d.is_dir() and len(d.name) == 1 and d.name.isalpha():
            for f in sorted(d.glob("*.json")):
                pts = np.array([[q["x"], q["y"], q["z"]] for q in json.loads(f.read_text())])
                if pts.shape == (21, 3):
                    X.append(pts)
                    y.append(d.name)
    return np.array(X), np.array(y)


def feats(P, mirror=False):
    return np.array([landmark_features(p, "Left" if mirror else None) for p in P])


def augment(P, y, k=10, seed=0):
    """k augmented copies: in-plane rotation +-30deg, scale 0.85-1.15, gaussian noise (1% of hand size),
    x-vs-y aspect stretch 0.6-1.7 (camera aspect mismatch), random mirror (handedness)."""
    r = np.random.default_rng(seed)
    Fs, Ys = [], []
    for _ in range(k):
        Q = P - P[:, :1]
        n = len(Q)
        a = np.deg2rad(r.uniform(-30, 30, n))
        c, s = np.cos(a)[:, None], np.sin(a)[:, None]
        x, yy = Q[:, :, 0].copy(), Q[:, :, 1].copy()
        Q[:, :, 0] = c * x - s * yy
        Q[:, :, 1] = s * x + c * yy
        Q *= r.uniform(0.85, 1.15, n)[:, None, None]
        Q += r.normal(0, 0.01, Q.shape) * np.linalg.norm(Q[:, :, :2], axis=2).max(1)[:, None, None]
        Q[:, :, 0] *= np.exp(r.uniform(np.log(0.6), np.log(1.7), n))[:, None]
        m = r.random(n) < 0.5
        hands = ["Left" if mm else None for mm in m]
        Fs.append(np.array([landmark_features(q, h) for q, h in zip(Q, hands, strict=True)]))
        Ys.append(y)
    return np.vstack(Fs), np.concatenate(Ys)


def fit_export(F, Y, out, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(F)
    m = LogisticRegression(C=10, max_iter=3000).fit(sc.transform(F), Y)
    np.savez_compressed(
        out,
        classes=np.array(m.classes_),
        feature_version=np.array(FEATURE_VERSION),
        mean=sc.mean_.astype(np.float32),
        std=sc.scale_.astype(np.float32),
        W1=m.coef_.T.astype(np.float32),
        b1=m.intercept_.astype(np.float32),
    )
    return sc, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asl-now")
    ap.add_argument("--out", default=str(ROOT / "backend/app/models/asl_letters.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copies", type=int, default=10)
    a = ap.parse_args()
    from sklearn.model_selection import train_test_split

    P, y = load(a.cache)
    print("samples", len(P), "classes", len(set(y)))
    Ptr, Pte, ytr, yte = train_test_split(P, y, test_size=0.2, stratify=y, random_state=a.seed)
    sc, m = fit_export(*augment(Ptr, ytr, a.copies, a.seed), a.out, a.seed)
    print(f"held-out acc (80/20, augment train only): {m.score(sc.transform(feats(Pte)), yte):.4f}")
    fit_export(*augment(P, y, a.copies, a.seed), a.out, a.seed)
    print("wrote", a.out, Path(a.out).stat().st_size, "bytes; logreg + aug x", a.copies)


if __name__ == "__main__":
    main()
