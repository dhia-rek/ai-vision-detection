"""Train, compare and evaluate the ASL letter classifier on signer-independent data.

Data comes from ``scripts/asl_extract.py`` (landmark ``.npz`` files, one per source). The
recipe is chosen by **leave-one-source-out** accuracy on the training sources; the final
numbers are reported on **test sources that are never used for training or selection**.

Sub-commands::

    train_asl.py compare --lm /tmp/asl-lm --out docs/asl-compare.json   # LOSO comparison of 3 models
    train_asl.py fit     --lm /tmp/asl-lm --model mlp --out backend/app/models/asl_letters.npz
    train_asl.py eval    --lm /tmp/asl-lm --model-file backend/app/models/asl_letters.npz \
                          --out docs/asl-eval.md
    train_asl.py eval    --lm /tmp/asl-lm --model-file old.npz --legacy-v1    # the original model

Run with ``uv run --with scikit-learn python scripts/train_asl.py ...``. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.sign_features import FEATURE_VERSION, N_FEATURES, landmark_features_batch  # noqa: E402

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TRAIN_SOURCES = ("asl_now", "kp", "cab_train")
TEST_SOURCES = ("eitan", "cab_test")  # never used for training or model selection
Proba = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- data


class Source:
    """One landmark source: normalised landmarks, image sizes, labels and group ids."""

    def __init__(self, path: Path) -> None:
        """Load a ``.npz`` written by ``asl_extract.py``."""
        z = np.load(path, allow_pickle=False)
        self.P: np.ndarray = z["P"].astype(np.float64)
        self.size: np.ndarray = z["size"].astype(np.float64)
        self.y: np.ndarray = np.array([LETTERS.index(c) for c in z["y"]])
        self.grp: np.ndarray = z["grp"]
        self.hand: np.ndarray = z["hand"]
        self.name = path.stem

    def subset(self, idx: np.ndarray) -> Source:
        """A copy restricted to the samples ``idx``."""
        t = Source.__new__(Source)
        t.P, t.size, t.y, t.grp, t.name = self.P[idx], self.size[idx], self.y[idx], self.grp[idx], self.name
        t.hand = self.hand[idx]
        return t

    def aspect(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Width / height per sample; unknown sizes get a plausible random camera aspect."""
        a = self.size[:, 0] / self.size[:, 1]
        bad = ~np.isfinite(a)
        if bad.any():
            r = rng or np.random.default_rng(0)
            wide = np.exp(r.uniform(np.log(1.2), np.log(1.8), bad.sum()))
            tall = np.exp(r.uniform(np.log(0.56), np.log(1.2), bad.sum()))
            a[bad] = np.where(r.random(bad.sum()) < 0.85, wide, tall)
        return a


def load_sources(lm: str) -> dict[str, Source]:
    """Load every available source (``cab_train`` is merged from its shards)."""
    root = Path(lm)
    out: dict[str, Source] = {}
    extra = tuple(sorted({f.stem.rsplit("_", 1)[0] for f in root.glob("rec_*.npz")}))  # user recordings
    for name in (*TRAIN_SOURCES, *TEST_SOURCES, *extra):
        files = sorted(root.glob(f"{name}_[0-9].npz")) or [root / f"{name}.npz"]
        parts = [Source(f) for f in files if f.exists()]
        if not parts:
            continue
        s = parts[0]
        for q in parts[1:]:
            s.P, s.size = np.concatenate([s.P, q.P]), np.concatenate([s.size, q.size])
            s.y, s.grp = np.concatenate([s.y, q.y]), np.concatenate([s.grp, q.grp])
            s.hand = np.concatenate([s.hand, q.hand])
        s.name = name
        out[name] = s
    return out


# ------------------------------------------------------------------- augmentation


def _rot(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Batch of 3D rotation matrices from Euler angles (radians)."""
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    one, zero = np.ones_like(ax), np.zeros_like(ax)
    rx = np.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], 1).reshape(-1, 3, 3)
    ry = np.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], 1).reshape(-1, 3, 3)
    rz = np.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], 1).reshape(-1, 3, 3)
    return rz @ rx @ ry


def augment_features(
    P: np.ndarray, aspect: np.ndarray, rng: np.random.Generator, strength: float = 1.0
) -> np.ndarray:
    """Features of randomly perturbed copies of the hands.

    Random mirror (left/right hand), in-plane rotation +-25 deg, out-of-plane tilt +-20 deg and
    per-landmark jitter (1.5 % of the palm). Hand size is irrelevant by construction.

    Args:
        P: Normalised landmarks (n, 21, 3).
        aspect: Image width / height per hand.
        rng: Random generator.
        strength: Scales the angles and noise.

    Returns:
        Feature matrix (n, N_FEATURES).
    """
    n = len(P)
    p = P.copy()
    p[:, :, 0] *= aspect[:, None]
    p[:, :, 2] *= aspect[:, None]
    p -= p[:, :1]
    scale = np.linalg.norm(p[:, [5, 9, 13, 17]], axis=2).mean(1)
    p /= np.where(scale > 1e-9, scale, 1.0)[:, None, None]
    flip = rng.random(n) < 0.5
    p[flip, :, 0] *= -1
    d = np.deg2rad
    r = _rot(
        rng.uniform(-d(20), d(20), n) * strength,
        rng.uniform(-d(20), d(20), n) * strength,
        rng.uniform(-d(25), d(25), n) * strength,
    )
    p = p @ r.transpose(0, 2, 1)
    p += rng.normal(0, 0.015 * strength, p.shape) * np.array([1.0, 1.0, 2.0])
    return landmark_features_batch(p, 1.0)


def plain_features(src: Source, mirror: bool = False) -> np.ndarray:
    """Un-augmented features of a source (optionally mirrored)."""
    p = src.P.copy()
    if mirror:
        p[:, :, 0] = 1.0 - p[:, :, 0]
    return landmark_features_batch(p, src.aspect())


def balanced_epoch(
    srcs: list[Source], per_class: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``per_class`` samples per letter, choosing the source uniformly among those that have it.

    Returns:
        ``(P, aspect, y)`` of the drawn hands.
    """
    Ps, As, ys = [], [], []
    aspects = [s.aspect(rng) for s in srcs]
    for c in range(26):
        have = [i for i, s in enumerate(srcs) if (s.y == c).any()]
        if not have:
            continue
        pick = rng.choice(have, per_class)
        for i in set(pick.tolist()):
            k = int((pick == i).sum())
            idx = rng.choice(np.flatnonzero(srcs[i].y == c), k)
            Ps.append(srcs[i].P[idx])
            As.append(aspects[i][idx])
            ys.append(np.full(k, c))
    return np.concatenate(Ps), np.concatenate(As), np.concatenate(ys)


# ------------------------------------------------------------------------ models


def make_training_set(
    srcs: list[Source], copies: int, per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Augmented, class- and source-balanced feature matrix for the sklearn models."""
    rng = np.random.default_rng(seed)
    F, Y = [], []
    for _ in range(copies):
        P, A, y = balanced_epoch(srcs, per_class, rng)
        F.append(augment_features(P, A, rng))
        Y.append(y)
    return np.concatenate(F), np.concatenate(Y)


class Standardiser:
    """Mean/std scaler that can be exported as plain arrays."""

    def __init__(self, F: np.ndarray) -> None:
        """Fit on a feature matrix."""
        self.mean = F.mean(0).astype(np.float32)
        self.std = (F.std(0) + 1e-6).astype(np.float32)

    def __call__(self, F: np.ndarray) -> np.ndarray:
        """Standardise ``F``."""
        return ((F - self.mean) / self.std).astype(np.float32)


def fit_logreg(srcs: list[Source], seed: int) -> tuple[Proba, dict[str, Any]]:
    """Multinomial logistic regression (linear baseline)."""
    from sklearn.linear_model import LogisticRegression

    F, Y = make_training_set(srcs, copies=4, per_class=500, seed=seed)
    sc = Standardiser(F)
    m = LogisticRegression(C=1.0, max_iter=400).fit(sc(F), Y)
    cls = m.classes_

    def proba(X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), 26))
        out[:, cls] = m.predict_proba(sc(X))
        return out

    return proba, {"sc": sc, "W1": m.coef_.T.astype(np.float32), "b1": m.intercept_.astype(np.float32)}


def fit_hgb(srcs: list[Source], seed: int) -> tuple[Proba, dict[str, Any]]:
    """Histogram gradient boosting (sklearn); not exportable to the numpy runtime."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    F, Y = make_training_set(srcs, copies=3, per_class=400, seed=seed)
    m = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_leaf_nodes=24, random_state=seed
    ).fit(F, Y)
    cls = m.classes_

    def proba(X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), 26))
        out[:, cls] = m.predict_proba(X)
        return out

    return proba, {}


def fit_mlp(
    srcs: list[Source],
    seed: int,
    epochs: int = 40,
    hidden: tuple[int, ...] = (256, 128),
    per_class: int = 500,
) -> tuple[Proba, dict[str, Any]]:
    """Small ReLU MLP trained with torch on freshly augmented, balanced epochs (exportable)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    P0, A0, _ = balanced_epoch(srcs, 200, rng)
    sc = Standardiser(augment_features(P0, A0, rng))
    dims = [N_FEATURES, *hidden, 26]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(0.2)]
    net = nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    per_epoch = -(-26 * per_class // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (per_epoch + 2))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for _ in range(epochs):
        P, A, y = balanced_epoch(srcs, per_class, rng)
        X = torch.from_numpy(sc(augment_features(P, A, rng)))
        Y = torch.from_numpy(y).long()
        perm = torch.randperm(len(X))
        net.train()
        for i in range(0, len(X), 256):
            b = perm[i : i + 256]
            opt.zero_grad()
            loss_fn(net(X[b]), Y[b]).backward()
            opt.step()
            sched.step()
    net.eval()

    def proba(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(net(torch.from_numpy(sc(X))), 1).numpy().astype(np.float64)

    arrays: dict[str, Any] = {"sc": sc}
    for i, m in enumerate([m for m in net if isinstance(m, nn.Linear)], 1):
        arrays[f"W{i}"] = m.weight.detach().numpy().T.astype(np.float32)
        arrays[f"b{i}"] = m.bias.detach().numpy().astype(np.float32)
    return proba, arrays


MODELS: dict[str, Callable[..., tuple[Proba, dict[str, Any]]]] = {
    "logreg": fit_logreg,
    "mlp": fit_mlp,
    "hgb": fit_hgb,
}


def export(arrays: dict[str, Any], path: str) -> None:
    """Write an exportable model in the layout read by ``AslLetterClassifier``."""
    sc: Standardiser = arrays["sc"]
    extra = {k: v for k, v in arrays.items() if k != "sc"}
    np.savez_compressed(
        path,
        classes=np.array(LETTERS),
        feature_version=np.array(FEATURE_VERSION),
        mean=sc.mean,
        std=sc.std,
        **extra,
    )


# --------------------------------------------------------------------- evaluation


def predict_tta(proba: Proba, src: Source) -> np.ndarray:
    """Class probabilities averaged over the hand and its mirror image."""
    return (proba(plain_features(src)) + proba(plain_features(src, mirror=True))) / 2


def report(src: Source, prob: np.ndarray) -> dict[str, Any]:
    """Accuracy, per-letter accuracy and confusion matrix of predictions on one source."""
    pred = prob.argmax(1)
    cm = np.zeros((26, 26), dtype=int)
    np.add.at(cm, (src.y, pred), 1)
    per = {LETTERS[c]: float(cm[c, c] / cm[c].sum()) for c in range(26) if cm[c].sum()}
    ok = pred == src.y
    static = ~np.isin(src.y, [LETTERS.index("J"), LETTERS.index("Z")])
    conf = prob.max(1)
    lo, hi = wilson(int(ok.sum()), len(ok))
    return {
        "n": len(src.y),
        "accuracy": float(ok.mean()),
        "ci95": [lo, hi],
        "static_accuracy": float(ok[static].mean()) if static.any() else None,
        "n_static": int(static.sum()),
        "confident_accuracy": float(ok[conf >= 0.6].mean()) if (conf >= 0.6).any() else None,
        "confident_coverage": float((conf >= 0.6).mean()),
        "macro_accuracy": float(np.mean(list(per.values()))),
        "per_letter": per,
        "confusion": cm.tolist(),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for ``k`` successes out of ``n``."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def loso(srcs: dict[str, Source], model: str, seed: int) -> dict[str, Any]:
    """Leave-one-source-out: train on the other training sources, test on the held-out one."""
    out: dict[str, Any] = {}
    for held in TRAIN_SOURCES:
        if held not in srcs:
            continue
        train = [srcs[n] for n in TRAIN_SOURCES if n != held and n in srcs]
        proba, _ = MODELS[model](train, seed)
        test = srcs[held]
        if len(test.y) > 3000:  # keep big folds fast; fixed subset
            test = test.subset(np.random.default_rng(1).choice(len(test.y), 3000, replace=False))
        out[held] = report(test, predict_tta(proba, test))
    folds = [v for v in out.values()]
    out["mean_accuracy"] = float(np.mean([v["accuracy"] for v in folds]))
    out["mean_macro_accuracy"] = float(np.mean([v["macro_accuracy"] for v in folds]))
    return out


def md_table(res: dict[str, Any], title: str) -> str:
    """Markdown rendering of a ``report``: per-letter accuracy and the confusion matrix."""
    per = res["per_letter"]
    lines = [
        f"### {title}",
        "",
        f"n = {res['n']}, accuracy **{100 * res['accuracy']:.1f} %** "
        f"(95 % CI {100 * res['ci95'][0]:.1f}-{100 * res['ci95'][1]:.1f}), "
        f"per-letter mean {100 * res['macro_accuracy']:.1f} %, "
        f"static letters only (no J/Z, n={res['n_static']}) "
        f"{100 * (res['static_accuracy'] or 0):.1f} %; "
        f"with confidence >= 0.6: {100 * (res['confident_accuracy'] or 0):.1f} % accurate on "
        f"{100 * res['confident_coverage']:.0f} % of hands",
        "",
        "| Letter | " + " | ".join(per) + " |",
        "|---|" + "---|" * len(per),
        "| Accuracy % | " + " | ".join(f"{100 * v:.0f}" for v in per.values()) + " |",
        "",
        "Confusion matrix (rows = true letter, columns = predicted):",
        "",
        "```",
        "    " + " ".join(f"{c:>3}" for c in LETTERS) + "  |   n",
    ]
    cm = np.array(res["confusion"])
    for c in (i for i in range(26) if cm[i].sum()):
        lines.append(
            f"{LETTERS[c]:>3} " + " ".join(f"{v:>3}" if v else "  ." for v in cm[c]) + f"  | {cm[c].sum():>3}"
        )
    return "\n".join([*lines, "```", ""])


def legacy_v1_probs(path: str, src: Source) -> np.ndarray:
    """Probabilities of the original v1 classifier (wrist-centred x/y/z + tip distances, softmax).

    The v1 recipe mirrored hands MediaPipe labelled "Left", using the label stored with the sample.
    """
    z = np.load(path, allow_pickle=False)
    classes = [LETTERS.index(str(c)) for c in z["classes"]]
    W, b, mean, std = z["W1"], z["b1"], z["mean"], z["std"]
    out = np.zeros((len(src.y), 26))
    for i, p in enumerate(src.P):
        q = p - p[0]
        if str(src.hand[i]).lower().startswith("l"):
            q[:, 0] = -q[:, 0]
        q = q / max(float(np.max(np.linalg.norm(q[:, :2], axis=1))), 1e-9)
        t = q[[4, 8, 12, 16, 20], :2]
        d = [np.linalg.norm(t[a] - t[c]) for a in range(5) for c in range(a + 1, 5)]
        f = (np.concatenate([q[1:].ravel(), d]) - mean) / std
        e = np.exp(f @ W + b - (f @ W + b).max())
        out[i, classes] = e / e.sum()
    return out


# ------------------------------------------------------------------------- CLI


def cmd_compare(a: argparse.Namespace) -> None:
    """Leave-one-source-out comparison of the model families; writes JSON."""
    srcs = load_sources(a.lm)
    results = {m: loso(srcs, m, a.seed) for m in a.models}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    for m, r in results.items():
        folds = {k: round(v["accuracy"], 4) for k, v in r.items() if isinstance(v, dict)}
        print(
            m,
            folds,
            "mean",
            round(r["mean_accuracy"], 4),
            "per-letter mean",
            round(r["mean_macro_accuracy"], 4),
        )


def cmd_fit(a: argparse.Namespace) -> None:
    """Train on all training sources, export the model and print accuracy on the test sources."""
    if a.model == "hgb":
        raise SystemExit("hgb cannot be exported to the numpy runtime")
    srcs = load_sources(a.lm)
    train = [s for n, s in srcs.items() if n not in TEST_SOURCES]  # includes user recordings (rec_*)
    print("training sources:", {s.name: len(s.y) for s in train})
    proba, arrays = MODELS[a.model](train, a.seed)
    export(arrays, a.out)
    print("wrote", a.out, Path(a.out).stat().st_size, "bytes")
    for n in TEST_SOURCES:
        if n in srcs:
            r = report(srcs[n], predict_tta(proba, srcs[n]))
            print(n, f"{100 * r['accuracy']:.1f} %", "n", r["n"])


def cmd_eval(a: argparse.Namespace) -> None:
    """Evaluate a saved model file on the test sources; optionally write the markdown report."""
    srcs = load_sources(a.lm)
    sections, summary = [], {}
    if not a.legacy_v1:
        from app.asl_classifier import AslLetterClassifier

        clf = AslLetterClassifier(a.model_file)
    for n in TEST_SOURCES:
        if n not in srcs:
            continue
        s = srcs[n]
        prob = legacy_v1_probs(a.model_file, s) if a.legacy_v1 else predict_tta(clf.proba_batch, s)
        summary[n] = report(s, prob)
        sections.append(md_table(summary[n], f"{n}" + (" (original v1 model)" if a.legacy_v1 else "")))
        print(n, f"{100 * summary[n]['accuracy']:.1f} %", "n", summary[n]["n"])
    if a.out:
        Path(a.out).write_text("\n".join(sections))
        Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    """Parse the sub-command and run it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("compare", cmd_compare), ("fit", cmd_fit), ("eval", cmd_eval)):
        p = sub.add_parser(name)
        p.add_argument("--lm", default="/tmp/asl-lm")
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(fn=fn)
        if name == "compare":
            p.add_argument("--models", nargs="+", default=["logreg", "mlp", "hgb"])
            p.add_argument("--out", required=True)
        if name == "fit":
            p.add_argument("--model", choices=sorted(MODELS), default="mlp")
            p.add_argument("--out", default=str(ROOT / "backend/app/models/asl_letters.npz"))
        if name == "eval":
            p.add_argument("--model-file", required=True)
            p.add_argument("--legacy-v1", action="store_true")
            p.add_argument("--out", default="")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
