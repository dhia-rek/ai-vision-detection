"""Train the ASL letter classifier. Usage:
uv run --with numpy --with scikit-learn --with huggingface_hub python scripts/train_asl.py --cache /tmp/asl-now
"""
import argparse, json, sys
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
                pts = np.array([[q["x"], q["y"], q["z"]] for q in json.load(open(f))])
                if pts.shape == (21, 3):
                    X.append(pts); y.append(d.name)
    return np.array(X), np.array(y)


def feats(P, mirror=False):
    return np.array([landmark_features(p, "Left" if mirror else None) for p in P])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/asl-now")
    ap.add_argument("--out", default=str(ROOT / "backend/app/models/asl_letters.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", choices=["logreg", "mlp128"], help="skip selection, export this model")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    P, y = load(a.cache)
    print("samples", len(P), "classes", len(set(y)))
    Ptr, Pte, ytr, yte = train_test_split(P, y, test_size=0.2, stratify=y, random_state=a.seed)
    Fte, Fte_m = feats(Pte), feats(Pte, True)

    def models():
        return {
            "logreg": LogisticRegression(C=10, max_iter=2000),
            "mlp128": MLPClassifier((128,), alpha=1e-3, max_iter=600, random_state=a.seed),
        }
    res = {}
    for aug in (False, True):
        Ftr = feats(Ptr); ytr_ = ytr
        if aug:
            Ftr = np.vstack([Ftr, feats(Ptr, True)]); ytr_ = np.concatenate([ytr, ytr])
        sc = StandardScaler().fit(Ftr)
        for n, m in models().items():
            m.fit(sc.transform(Ftr), ytr_)
            acc = m.score(sc.transform(Fte), yte)
            acc_m = m.score(sc.transform(Fte_m), yte)
            res[(n, aug)] = acc
            print(f"{n} mirror_aug={aug} test_acc={acc:.4f} test_acc_on_mirrored={acc_m:.4f}")
    # Mirror augmentation is used for the final model (robust to unknown handedness).
    best = max((k for k in res if k[1]), key=res.get)
    # prefer the simpler logistic regression if within 1 point of the best
    if res[("logreg", True)] >= res[best] - 0.01:
        best = ("logreg", True)
    if a.force:
        best = (a.force, True)
    print("chosen", best)
    F = np.vstack([feats(P), feats(P, True)]); Y = np.concatenate([y, y])
    sc = StandardScaler().fit(F)
    m = models()[best[0]].fit(sc.transform(F), Y)
    classes = np.array(m.classes_)
    if best[0] == "logreg":
        arrs = {"W1": m.coef_.T, "b1": m.intercept_}
    else:
        arrs = {f"W{i+1}": w for i, w in enumerate(m.coefs_)} | {f"b{i+1}": b for i, b in enumerate(m.intercepts_)}
    arrs = {k: v.astype(np.float32) for k, v in arrs.items()}
    np.savez_compressed(a.out, classes=classes, feature_version=np.array(FEATURE_VERSION),
                        mean=sc.mean_.astype(np.float32), std=sc.scale_.astype(np.float32), **arrs)
    print("wrote", a.out, Path(a.out).stat().st_size, "bytes; model", best[0])


if __name__ == "__main__":
    main()
