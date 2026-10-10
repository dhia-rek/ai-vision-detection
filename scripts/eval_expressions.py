"""Evaluate the bundled facial-expression pipeline on a labeled image archive.

The archive must contain one directory per emotion, optionally nested beneath a
split directory. Download the dataset archive from its primary source first.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.faces import EXPRESSIONS, ExpressionDetector  # noqa: E402

DATASET_ID = "wmfd4p3z32"
DATASET_VERSION = 1
DATASET_FILE = "CroppedAugmentedDataset.zip"
DEFAULT_ROOT = Path("/tmp/expr-eval")
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


def _class_from_path(path: Path) -> str | None:
    """Return the canonical emotion for a directory component, if recognized."""
    for part in reversed(path.parts[:-1]):
        label = LABEL_NAMES.get(re.sub(r"[^a-z]", "", part.lower()))
        if label is not None:
            return label
    return None


def _download(destination: Path) -> Path:
    """Fetch the official Mendeley file listing and download its capped-eval archive."""
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
    url = archive_info["content_details"]["download_url"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == archive_info["content_details"]["size"]:
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": headers["User-Agent"]})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1 << 20):
            output.write(chunk)
    return destination


def _extract_images(archive: Path, root: Path, limit: int) -> list[tuple[str, Path]]:
    """Extract a deterministic, capped, balanced set of images from a ZIP archive."""
    with zipfile.ZipFile(archive) as zipped:
        entries: dict[str, dict[str, list[zipfile.ZipInfo]]] = {
            label: {"genuine": [], "fake": [], "other": []} for label in EXPRESSIONS
        }
        for info in zipped.infolist():
            source = Path(info.filename)
            label = _class_from_path(source)
            if not info.is_dir() and source.suffix.lower() in IMAGE_SUFFIXES and label:
                parts = {part.lower() for part in source.parts}
                subgroup = next((name for name in ("genuine", "fake") if name in parts), "other")
                entries[label][subgroup].append(info)
        chosen: list[tuple[str, zipfile.ZipInfo]] = []
        for label in EXPRESSIONS:
            subgroups = entries[label]
            if subgroups["genuine"] and subgroups["fake"]:
                per_subgroup = (limit + 1) // 2
                genuine = sorted(subgroups["genuine"], key=lambda item: item.filename)
                fake = sorted(subgroups["fake"], key=lambda item: item.filename)
                selected = [*genuine[:per_subgroup], *fake[: limit // 2]]
                if len(selected) < limit:
                    remainder = genuine[per_subgroup:] + fake[limit // 2 :]
                    selected.extend(remainder[: limit - len(selected)])
            else:
                selected = sorted(
                    [*subgroups["genuine"], *subgroups["fake"], *subgroups["other"]],
                    key=lambda item: item.filename,
                )[:limit]
            chosen.extend((label, info) for info in sorted(selected, key=lambda item: item.filename))
        if not chosen or any(not entries[label] for label in EXPRESSIONS):
            counts = {label: sum(map(len, entries[label].values())) for label in EXPRESSIONS}
            raise ValueError(f"Archive must have all seven labeled folders; discovered {counts}")
        root.mkdir(parents=True, exist_ok=True)
        result: list[tuple[str, Path]] = []
        for label, info in chosen:
            safe_name = Path(info.filename).name
            target = root / label / safe_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zipped.read(info))
            result.append((label, target))
    return result


def _metrics(confusion: list[list[int]], actual: Counter[str]) -> dict[str, Any]:
    """Compute class, total, and macro metrics from a confusion matrix."""
    total = sum(actual.values())
    correct = sum(confusion[index][index] for index in range(len(EXPRESSIONS)))
    class_accuracy = {
        label: confusion[index][index] / actual[label] if actual[label] else 0.0
        for index, label in enumerate(EXPRESSIONS)
    }
    f1_values: list[float] = []
    for index in range(len(EXPRESSIONS)):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(len(EXPRESSIONS)) if row != index)
        false_negative = actual[EXPRESSIONS[index]] - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "per_class_accuracy": class_accuracy,
        "overall_accuracy_including_misses": correct / total if total else 0.0,
        "macro_f1_including_misses": sum(f1_values) / len(f1_values),
        "count": total,
    }


def evaluate(images: list[tuple[str, Path]], model_name: str) -> dict[str, Any]:
    """Run the project's detector and return metrics, counting absent faces as misses."""
    detector = ExpressionDetector()
    confusion = [[0 for _ in PREDICTIONS] for _ in EXPRESSIONS]
    actual: Counter[str] = Counter()
    detected_correct = 0
    detected_count = 0
    small_upscaled = 0
    for label, path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unable to decode image: {path}")
        height, width = image.shape[:2]
        if min(height, width) < 112:
            scale = 224 / min(height, width)
            image = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
            small_upscaled += 1
        detections = detector.detect(image)
        actual[label] += 1
        if not detections:
            confusion[EXPRESSIONS.index(label)][len(EXPRESSIONS)] += 1
            continue
        # Use the largest detected face as the sample's subject.
        face = max(detections, key=lambda item: max(0.0, item.x2 - item.x1) * max(0.0, item.y2 - item.y1))
        prediction = face.label if face.label in EXPRESSIONS else "no_face"
        confusion[EXPRESSIONS.index(label)][PREDICTIONS.index(prediction)] += 1
        detected_count += 1
        detected_correct += int(prediction == label)
    metrics = _metrics(confusion, actual)
    metrics.update(
        {
            "model": model_name,
            "class_order": list(EXPRESSIONS),
            "prediction_order": list(PREDICTIONS),
            "confusion_matrix": confusion,
            "no_face_count": sum(row[-1] for row in confusion),
            "detected_face_count": detected_count,
            "detected_face_accuracy": detected_correct / detected_count if detected_count else 0.0,
            "tiny_images_upscaled_to_224": small_upscaled,
            "sample_counts": dict(actual),
        }
    )
    return metrics


def _write_report(result: dict[str, Any], path: Path) -> None:
    """Write machine-readable and human-readable evaluation results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Facial-expression evaluation results",
        "",
        f"Model: `{result['model']}`",
        f"Overall accuracy (no-face is a miss): {result['overall_accuracy_including_misses']:.4f}",
        f"Accuracy among detected faces: {result['detected_face_accuracy']:.4f}",
        f"Macro-F1 (no-face misses count as false negatives): {result['macro_f1_including_misses']:.4f}",
        f"No-face count: {result['no_face_count']} / {result['count']}",
        f"Images upscaled to 224x224: {result['tiny_images_upscaled_to_224']}",
        "",
        "| Emotion | N | Accuracy |",
        "| --- | ---: | ---: |",
    ]
    for label in EXPRESSIONS:
        lines.append(
            f"| {label} | {result['sample_counts'].get(label, 0)} | "
            f"{result['per_class_accuracy'][label]:.4f} |"
        )
    lines += ["", "Confusion matrix (rows=true; columns=predicted, final column=no_face):", "", "```text"]
    lines.extend(" ".join(str(value) for value in row) for row in result["confusion_matrix"])
    lines += ["```", ""]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Download, sample, evaluate, and save expression metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=100, help="Maximum images per emotion class")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Temporary data directory")
    parser.add_argument("--output", type=Path, default=Path(".agent-scratch/expressions-eval.json"))
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    archive = _download(args.root / "dataset.zip")
    images = _extract_images(archive, args.root / "images", args.per_class)
    result = evaluate(images, "yunet+mobilefacenet-fer")
    _write_report(result, args.output)
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
