"""Evaluate YOLO object models on a reproducible COCO subset and bundled samples.

Run with ``OMP_NUM_THREADS=3 nice -n 10 uv run python scripts/eval_objects.py``.
COCO data and outputs are kept outside source control.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
COCO_ROOT = Path("/tmp/coco-eval")
ANNOTATIONS = COCO_ROOT / "annotations" / "instances_val2017.json"
DATASET = COCO_ROOT / "yolo-val100"
OUTPUTS = ROOT / ".agent-scratch"
WEIGHTS = ("yolov8n.pt", "yolo11n.pt", "yolo11s.pt")
IMAGE_IDS = 100


def download(url: str, destination: Path) -> None:
    """Download a file only when it is not already present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)  # noqa: S310 - fixed public dataset URLs


def prepare_coco() -> tuple[list[int], dict[int, int]]:
    """Download annotations and the fixed first 100 annotated COCO validation images.

    Returns:
        The selected COCO image IDs and COCO-category-to-YOLO-class mapping.
    """
    if not ANNOTATIONS.exists():
        archive = COCO_ROOT / "annotations_trainval2017.zip"
        download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extract("annotations/instances_val2017.json", COCO_ROOT)
    data = json.loads(ANNOTATIONS.read_text())
    categories = sorted(data["categories"], key=lambda category: category["id"])
    category_map = {int(category["id"]): index for index, category in enumerate(categories)}
    annotated = {int(annotation["image_id"]) for annotation in data["annotations"]}
    selected = sorted(int(image["id"]) for image in data["images"] if int(image["id"]) in annotated)[
        :IMAGE_IDS
    ]
    image_info = {int(image["id"]): image for image in data["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in data["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    image_dir = DATASET / "images" / "val"
    label_dir = DATASET / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image_id in selected:
        filename = f"{image_id:012d}.jpg"
        image_path = image_dir / filename
        download(f"http://images.cocodataset.org/val2017/{filename}", image_path)
        info = image_info[image_id]
        labels: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            x, y, width, height = (float(value) for value in annotation["bbox"])
            center_x = (x + width / 2) / int(info["width"])
            center_y = (y + height / 2) / int(info["height"])
            labels.append(
                f"{category_map[int(annotation['category_id'])]} {center_x:.8f} {center_y:.8f} "
                f"{width / int(info['width']):.8f} {height / int(info['height']):.8f}"
            )
        (label_dir / f"{image_id:012d}.txt").write_text("\n".join(labels) + "\n")
    dataset_yaml: dict[str, Any] = {
        "path": str(DATASET),
        "train": "images/val",
        "val": "images/val",
        "names": {index: str(category["name"]) for index, category in enumerate(categories)},
    }
    (DATASET / "data.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False))
    return selected, category_map


def evaluate_map(weights: str, dataset_yaml: Path) -> dict[str, float]:
    """Measure COCO mAP metrics using Ultralytics validation on CPU."""
    model = YOLO(weights)
    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=640,
        device="cpu",
        conf=0.001,
        workers=0,
        verbose=False,
        plots=False,
    )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def measure_latency(weights: str, sample_paths: list[Path], conf: float) -> dict[str, float]:
    """Measure project YoloDetector latency across bundled object samples."""
    from app.detector import YoloDetector

    detector = YoloDetector(weights, conf=conf, device="cpu")
    images = [cv2.imread(str(path)) for path in sample_paths]
    if any(image is None for image in images):
        raise RuntimeError("Could not decode one or more bundled object samples")
    frames = [image for image in images if image is not None]
    for index in range(3):
        detector.detect(frames[index % len(frames)])
    durations: list[float] = []
    for index in range(40):
        frame = frames[index % len(frames)]
        tick = time.perf_counter()
        detector.detect(frame)
        durations.append((time.perf_counter() - tick) * 1000)
    latency = median(durations)
    return {"median_ms": latency, "fps": 1000.0 / latency}


def main() -> None:
    """Run all evaluations and write machine-readable and markdown reports."""
    torch.set_num_threads(3)
    os.environ["OMP_NUM_THREADS"] = "3"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    image_ids, _ = prepare_coco()
    sample_paths = sorted((ROOT / "backend" / "samples" / "objects").glob("*.jpg"))
    if len(sample_paths) != 5:
        raise RuntimeError(f"Expected 5 bundled object images, found {len(sample_paths)}")
    conf = float(os.getenv("AVD_CONF", "0.25"))
    results: dict[str, Any] = {
        "dataset": "COCO val2017 first 100 annotated IDs, ascending",
        "image_ids": image_ids,
        "map_settings": {"imgsz": 640, "device": "cpu", "conf": 0.001},
        "latency_settings": {
            "images": [path.name for path in sample_paths],
            "warmup": 3,
            "runs": 40,
            "conf": conf,
        },
        "models": {},
    }
    for weights in WEIGHTS:
        map_scores = evaluate_map(weights, DATASET / "data.yaml")
        latency = measure_latency(weights, sample_paths, conf)
        results["models"][weights] = {**map_scores, **latency}
        print(f"{weights}: {results['models'][weights]}", flush=True)
    (OUTPUTS / "objects-eval.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = ["| Weights | mAP@0.5 | mAP@0.5:0.95 | Median CPU ms | CPU fps |", "|---|---:|---:|---:|---:|"]
    for weights, values in results["models"].items():
        rows.append(
            f"| {weights} | {values['map50']:.4f} | {values['map50_95']:.4f} | "
            f"{values['median_ms']:.1f} | {values['fps']:.2f} |"
        )
    (OUTPUTS / "objects-eval.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
