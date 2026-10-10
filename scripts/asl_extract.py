"""Extract MediaPipe hand landmarks from ASL letter datasets into small ``.npz`` files.

Only landmarks are kept (no pixels). Every source gets one file with the arrays
``P`` (n, 21, 3) normalised landmarks, ``size`` (n, 2) the (width, height) of the image the
landmarks are relative to, ``y`` letter labels, ``grp`` a group id (video or file prefix; used
for group-aware splits) and ``hand`` MediaPipe's handedness label.

Sources (see ``docs/asl-datasets.md`` for licences):

* ``asl_now``  - JSON landmarks, ``sid220/asl-now-fingerspelling`` (MIT).
* ``kp``       - CSV landmarks, ``om192006/sign_language_keypoints`` (MIT).
* ``images``   - a folder ``<root>/<LETTER>/*.jpg`` run through the app's MediaPipe recogniser.

Usage::

    uv run python scripts/asl_extract.py asl_now --root /tmp/asl-now --out /tmp/asl-lm/asl_now.npz
    uv run python scripts/asl_extract.py images --root /tmp/x/Train --stride 8 --out /tmp/asl-lm/cab.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _save(out: str, P: list, size: list, y: list, grp: list, hand: list) -> None:
    """Write the arrays of one source to ``out``."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        P=np.asarray(P, dtype=np.float32).reshape(-1, 21, 3),
        size=np.asarray(size, dtype=np.float32).reshape(-1, 2),
        y=np.asarray(y),
        grp=np.asarray(grp),
        hand=np.asarray(hand),
    )
    print("wrote", out, "samples", len(y), "classes", len(set(y)))


def from_asl_now(root: str, out: str) -> None:
    """Read the ``sid220/asl-now-fingerspelling`` snapshot (one JSON file per sample)."""
    P, size, y, grp, hand = [], [], [], [], []
    for d in sorted(Path(root).iterdir()):
        if not (d.is_dir() and len(d.name) == 1 and d.name.isalpha()):
            continue
        for f in sorted(d.glob("*.json")):
            pts = np.array([[q["x"], q["y"], q["z"]] for q in json.loads(f.read_text())])
            if pts.shape == (21, 3):
                P.append(pts)
                size.append((np.nan, np.nan))  # webcam frame size is not recorded
                y.append(d.name.upper())
                grp.append("asl_now")
                hand.append("")
    _save(out, P, size, y, grp, hand)


def from_kp(root: str, out: str, stride: int) -> None:
    """Read the ``om192006/sign_language_keypoints`` CSVs (one file per video frame).

    Files are named ``<L>_<n>.csv``; consecutive ``n`` are consecutive frames, so the group id is
    ``<L>_<n // 200>`` (neighbouring frames share a group).
    """
    P, size, y, grp, hand = [], [], [], [], []
    for d in sorted(Path(root).iterdir()):
        if not (d.is_dir() and len(d.name) == 1 and d.name.isalpha()):
            continue
        files = sorted(d.glob("*.csv"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
        for i, f in enumerate(files):
            if i % stride:
                continue
            with f.open() as fh:
                rows = list(csv.DictReader(fh))
            pts = np.array([[float(r["X"]), float(r["Y"]), float(r["Z"])] for r in rows])
            if pts.shape != (21, 3):
                continue
            n = int(re.sub(r"\D", "", f.stem) or 0)
            P.append(pts)
            size.append((np.nan, np.nan))
            y.append(d.name.upper())
            grp.append(f"kp_{d.name}_{n // 200}")
            hand.append("")
    _save(out, P, size, y, grp, hand)


def from_images(root: str, out: str, stride: int, pad: float, min_side: int, shard: str) -> None:
    """Run the app's MediaPipe gesture recogniser over ``<root>/<LETTER>/*`` images.

    Small images are upscaled to ``min_side`` and padded by ``pad`` (fraction of the side) with
    grey, which helps the palm detector on tight crops. Landmarks are stored relative to the
    padded image together with its size, so pixel geometry stays exact.

    Args:
        root: Folder with one sub-folder per letter.
        out: Output ``.npz`` path.
        stride: Keep every ``stride``-th image per letter (frames of a video are near duplicates).
        pad: Border added on every side, as a fraction of the (resized) side.
        min_side: Upscale images whose short side is below this.
        shard: ``"i/n"``: process only every n-th kept image starting at i (for parallel runs).
    """
    import cv2
    import mediapipe as mp
    from app.zoo import ensure_model
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model("gesture_recognizer.task"))),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    rec = vision.GestureRecognizer.create_from_options(options)
    si, sn = (int(v) for v in shard.split("/"))
    P, size, y, grp, hand = [], [], [], [], []
    seen = miss = 0
    for d in sorted(Path(root).iterdir()):
        if not (d.is_dir() and len(d.name) == 1 and d.name.isalpha()):
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for i, f in enumerate(files[::stride]):
            if i % sn != si:
                continue
            img = cv2.imread(str(f))
            if img is None:
                continue
            seen += 1
            h, w = img.shape[:2]
            s = max(1.0, min_side / min(h, w))
            if s > 1.0:
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
            if pad > 0:
                b = round(pad * min(img.shape[:2]))
                img = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            H, W = img.shape[:2]
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            res = rec.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.hand_landmarks:
                miss += 1
                continue
            lm = res.hand_landmarks[0]
            P.append([[p.x, p.y, p.z] for p in lm])
            size.append((W, H))
            y.append(d.name.upper())
            grp.append(
                re.sub(r"[-_.]*(img)?[-_]*\d+(_test)?(_jpg)?.*$", "", f.stem.replace(".mp4", "")) or f.stem
            )
            hand.append(res.handedness[0][0].category_name if res.handedness else "")
    print(f"images seen {seen}, no hand {miss} ({100 * miss / max(seen, 1):.1f} %)")
    _save(out, P, size, y, grp, hand)


def from_recorded(root: str, out: str) -> None:
    """Read downloaded browser landmark recordings from a JSON file or folder."""
    source = Path(root)
    files = [source] if source.is_file() else sorted(source.glob("*.json"))
    P, size, y, grp, hand = [], [], [], [], []
    for file in files:
        payload = json.loads(file.read_text())
        if payload.get("format") != "avd-landmarks" or payload.get("version") != 1:
            continue
        for sample in payload.get("samples", []):
            letter = str(sample.get("letter", "")).upper()
            points = np.asarray(sample.get("landmarks", []), dtype=np.float32)
            dimensions = sample.get("image_size", [])
            if letter not in LETTERS or points.shape != (21, 3) or len(dimensions) != 2:
                continue
            P.append(points)
            size.append((float(dimensions[0]), float(dimensions[1])))
            y.append(letter)
            grp.append(f"{file.stem}_{letter}")
            hand.append(str(sample.get("handedness") or ""))
    _save(out, P, size, y, grp, hand)


def main() -> None:
    """Parse arguments and run one extractor."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["asl_now", "kp", "images", "recorded"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pad", type=float, default=0.3)
    ap.add_argument("--min-side", type=int, default=384)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.kind == "asl_now":
        from_asl_now(a.root, a.out)
    elif a.kind == "kp":
        from_kp(a.root, a.out, a.stride)
    elif a.kind == "images":
        from_images(a.root, a.out, a.stride, a.pad, a.min_side, a.shard)
    else:
        from_recorded(a.root, a.out)


if __name__ == "__main__":
    main()
