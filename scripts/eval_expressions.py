"""Evaluate the bundled facial-expression pipeline on a labeled image archive.

The archive must contain one directory per emotion, optionally nested beneath a
split directory. Download the dataset archive from its primary source first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.faces import EXPRESSIONS, ExpressionDetector  # noqa: E402

DATASET_ID = "wmfd4p3z32"
DATASET_VERSION = 1
DATASET_FILE = "Raw.zip"
DEFAULT_ROOT = Path("/tmp/expr-eval2")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_NAMES = {
    "angry": "angry",
    "anger": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
}
PREDICTIONS = (*EXPRESSIONS, "no_face")
MAX_EDGE = 1280


def _class_from_path(path: Path) -> str | None:
    """Return the canonical emotion for a directory component, if recognized."""
    for part in reversed(path.parts[:-1]):
        label = LABEL_NAMES.get(re.sub(r"[^a-z]", "", part.lower()))
        if label is not None:
            return label
    return None


def _subgroup_from_path(path: Path) -> str:
    """Return Genuine/Fake where encoded in path, otherwise Other."""
    parts = {part.lower() for part in path.parts}
    return next((name.title() for name in ("genuine", "fake") if name in parts), "Other")


def _download(destination: Path) -> tuple[Path, dict[str, Any]]:
    """Fetch the Mendeley listing, select Raw.zip, and download that archive."""
    api = (
        f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files"
        f"?folder_id=root&version={DATASET_VERSION}&$start=0&$limit=1000"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Referer": f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}",
        "Accept": "application/vnd.mendeley-public-dataset.1+json",
    }
    request = urllib.request.Request(api, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        listing = json.loads(response.read())
    archive_info = next(item for item in listing if item["filename"] == DATASET_FILE)
    details = archive_info["content_details"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != details["size"]:
        request = urllib.request.Request(
            details["download_url"], headers={"User-Agent": headers["User-Agent"]}
        )
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
            while chunk := response.read(1 << 20):
                output.write(chunk)
    digest = hashlib.sha256()
    with destination.open("rb") as archive_file:
        while chunk := archive_file.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != details["sha256_hash"]:
        raise ValueError(f"SHA-256 mismatch for {destination}: {digest.hexdigest()}")
    return destination, details


def _extract_images(archive: Path, root: Path, limit: int) -> list[tuple[str, str, Path]]:
    """Extract a deterministic, capped, balanced set of images from a ZIP archive."""
    with zipfile.ZipFile(archive) as zipped:
        entries: dict[str, dict[str, list[zipfile.ZipInfo]]] = {
            label: {"Genuine": [], "Fake": [], "Other": []} for label in EXPRESSIONS
        }
        for info in zipped.infolist():
            source = Path(info.filename)
            label = _class_from_path(source)
            if not info.is_dir() and source.suffix.lower() in IMAGE_SUFFIXES and label:
                entries[label][_subgroup_from_path(source)].append(info)
        chosen: list[tuple[str, str, zipfile.ZipInfo]] = []
        for label in EXPRESSIONS:
            groups = entries[label]
            genuine, fake = (
                sorted(groups["Genuine"], key=lambda x: x.filename),
                sorted(groups["Fake"], key=lambda x: x.filename),
            )
            if genuine and fake:
                n_genuine, n_fake = (limit + 1) // 2, limit // 2
                selected = [*genuine[:n_genuine], *fake[:n_fake]]
                if len(selected) < limit:
                    remainder = sorted(
                        [*genuine[n_genuine:], *fake[n_fake:], *groups["Other"]], key=lambda x: x.filename
                    )
                    selected.extend(remainder[: limit - len(selected)])
            else:
                selected = sorted([*genuine, *fake, *groups["Other"]], key=lambda x: x.filename)[:limit]
            for info in sorted(selected, key=lambda x: x.filename):
                chosen.append((label, _subgroup_from_path(Path(info.filename)), info))
        counts = {label: sum(map(len, groups.values())) for label, groups in entries.items()}
        if not chosen or any(not entries[label] for label in EXPRESSIONS):
            raise ValueError(f"Archive must have all seven labeled folders; discovered {counts}")
        root.mkdir(parents=True, exist_ok=True)
        result: list[tuple[str, str, Path]] = []
        for label, subgroup, info in chosen:
            target = root / label / f"{len(result):06d}_{Path(info.filename).name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zipped.read(info))
            result.append((label, subgroup, target))
    return result


def _upright_rgb(path: Path) -> Image.Image:
    """Load an image upright according to EXIF and cap its longest side."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    scale = min(1.0, MAX_EDGE / max(width, height))
    if scale < 1.0:
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    return image


def _montage(images: list[tuple[str, str, Path]], destination: Path) -> None:
    """Save a labeled three-by-seven montage after EXIF orientation correction."""
    from PIL import ImageDraw, ImageFont

    width, height, label_height = 300, 240, 26
    montage = Image.new("RGB", (width * len(EXPRESSIONS), (height + label_height) * 3), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for column, label in enumerate(EXPRESSIONS):
        class_images = [sample for sample in images if sample[0] == label]
        selected = [
            sample for subgroup in ("Genuine", "Fake") for sample in class_images if sample[1] == subgroup
        ][:2]
        selected = selected[:3]
        if len(selected) < 3:
            selected.extend(sample for sample in class_images if sample not in selected)
        selected = selected[:3]
        for row, (_, subgroup, path) in enumerate(selected):
            image = _upright_rgb(path)
            image.thumbnail((width - 8, height - 8))
            x0, y0 = column * width, row * (height + label_height)
            montage.paste(image, (x0 + (width - image.width) // 2, y0 + (height - image.height) // 2))
            draw.text((x0 + 4, y0 + height), f"{label} / {subgroup}", fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    montage.save(destination, quality=90)


def _scores(matrix: list[list[int]], actual: Counter[str]) -> dict[str, Any]:
    """Calculate per-class accuracy, overall accuracy, and macro-F1."""
    total = sum(actual.values())
    correct = sum(matrix[i][i] for i in range(len(EXPRESSIONS)))
    accuracy = {
        label: matrix[i][i] / actual[label] if actual[label] else 0.0 for i, label in enumerate(EXPRESSIONS)
    }
    f1: list[float] = []
    for i, label in enumerate(EXPRESSIONS):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(len(EXPRESSIONS)) if row != i)
        fn = actual[label] - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "per_class_accuracy": accuracy,
        "overall_accuracy_including_misses": correct / total if total else 0.0,
        "macro_f1_including_misses": sum(f1) / len(f1),
        "count": total,
    }


def _evaluate_subset(images: list[tuple[str, str, Path]], detector: ExpressionDetector) -> dict[str, Any]:
    """Run predictions and compute metrics for an already selected image subset."""
    matrix = [[0 for _ in PREDICTIONS] for _ in EXPRESSIONS]
    actual: Counter[str] = Counter()
    detected_correct = detected_count = 0
    for label, _, path in images:
        rgb = np.asarray(_upright_rgb(path))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detections = detector.detect(bgr)
        actual[label] += 1
        if not detections:
            matrix[EXPRESSIONS.index(label)][-1] += 1
            continue
        face = max(detections, key=lambda item: max(0.0, item.x2 - item.x1) * max(0.0, item.y2 - item.y1))
        prediction = face.label if face.label in EXPRESSIONS else "no_face"
        matrix[EXPRESSIONS.index(label)][PREDICTIONS.index(prediction)] += 1
        detected_count += 1
        detected_correct += int(prediction == label)
    result = _scores(matrix, actual)
    result.update(
        {
            "confusion_matrix": matrix,
            "no_face_count": sum(row[-1] for row in matrix),
            "detected_face_count": detected_count,
            "detected_face_accuracy": detected_correct / detected_count if detected_count else 0.0,
            "sample_counts": dict(actual),
        }
    )
    return result


def evaluate(images: list[tuple[str, str, Path]], model_name: str) -> dict[str, Any]:
    """Run the project's detector and report all-image, Genuine, and subgroup scores."""
    detector = ExpressionDetector()
    overall = _evaluate_subset(images, detector)
    genuine = _evaluate_subset([sample for sample in images if sample[1] == "Genuine"], detector)
    subgroup: dict[str, Any] = {}
    for name in ("Genuine", "Fake", "Other"):
        selected = [sample for sample in images if sample[1] == name]
        subgroup[name] = _evaluate_subset(selected, detector) if selected else {"count": 0}
    overall.update(
        {
            "model": model_name,
            "class_order": list(EXPRESSIONS),
            "prediction_order": list(PREDICTIONS),
            "genuine_only": genuine,
            "per_class_subgroup": {
                name: data.get("per_class_accuracy", {}) for name, data in subgroup.items()
            },
            "subgroup_metrics": subgroup,
            "selected": [
                {"label": label, "subgroup": group, "path": str(path)} for label, group, path in images
            ],
        }
    )
    return overall


def main() -> None:
    """Download, sample, evaluate, and save expression metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=100, help="Maximum images per emotion class")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Temporary data directory")
    parser.add_argument("--output", type=Path, default=Path(".agent-scratch/expressions-eval2.json"))
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    archive, details = _download(args.root / DATASET_FILE)
    images = _extract_images(archive, args.root / "images", args.per_class)
    _montage(images, args.root / "montage.jpg")
    result = evaluate(images, "yunet+mobilefacenet-fer")
    result["archive"] = {
        "filename": DATASET_FILE,
        "sha256_expected": details["sha256_hash"],
        "sha256_verified": True,
        "size": details["size"],
    }
    _write_result = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_write_result, encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Montage: {args.root / 'montage.jpg'}")


if __name__ == "__main__":
    main()
