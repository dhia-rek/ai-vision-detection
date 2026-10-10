"""Integration parity checks for the browser YOLO11n model and its Python oracle.

The browser ships YOLO11n ONNX. The training/evaluation API's default detector remains
YOLOv8n, so this test intentionally uses the exported YOLO11n source checkpoint as the
Python oracle instead of comparing unlike model generations.
"""

from __future__ import annotations

import base64
import os
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("AVD_STATIC_URL", "http://127.0.0.1:8013").rstrip("/")
PYTHON_ORACLE = ROOT / ".agent-scratch" / "phase2-export" / "yolo11n-source.pt"
OBJECT_SAMPLES = (
    "objects/street.jpg",
    "objects/cats.jpg",
    "objects/livingroom.jpg",
)


def _static_site_is_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/", timeout=2) as response:
            return response.status == 200 and b"AI Vision" in response.read(4096)
    except (OSError, urllib.error.URLError):
        return False


def _iou(a: dict, b: dict) -> float:
    left, top = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    right, bottom = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    return intersection / max(1e-9, area_a + area_b - intersection)


@pytest.mark.slow
@pytest.mark.parametrize("sample", OBJECT_SAMPLES)
def test_yolo11_browser_predictions_match_python_oracle(sample: str) -> None:
    """Match browser YOLO11n ONNX detections to Ultralytics YOLO11n on source images."""
    if not _static_site_is_ready():
        pytest.skip(f"static browser site is not serving at {BASE_URL}")
    if not PYTHON_ORACLE.is_file():
        pytest.skip("YOLO11n Python oracle checkpoint has not been exported")

    from ultralytics import YOLO

    image_path = ROOT / "backend" / "samples" / sample
    bgr = cv2.imread(str(image_path))
    assert bgr is not None, f"Could not read bundled sample {sample}"
    python_result = YOLO(str(PYTHON_ORACLE)).predict(
        bgr, imgsz=640, conf=0.25, iou=0.7, max_det=300, device="cpu", verbose=False
    )[0]
    names = python_result.names
    python_detections = [
        {
            "class": names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "bbox": dict(zip(("x1", "y1", "x2", "y2"), map(float, box.xyxy[0].tolist()), strict=True)),
        }
        for box in python_result.boxes
    ]
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
        browser_detections = page.evaluate(
            """async ({ encoded }) => {
              const { createObjectEngine } = await import('./static/js/web-objects.js');
              const engine = createObjectEngine({ size: 640 });
              await engine.ready;
              const image = new Image();
              image.src = `data:image/jpeg;base64,${encoded}`;
              await image.decode();
              const result = await engine.detect(image);
              engine.close();
              return result.detections;
            }""",
            {"encoded": encoded},
        )
        browser.close()
    assert not errors, f"Browser inference errors: {errors}"

    # Greedy, class-aware, score-ordered matching; every unmatched prediction is a parity failure.
    remaining = list(python_detections)
    mismatches = []
    for actual in sorted(browser_detections, key=lambda item: item["confidence"], reverse=True):
        candidates = [item for item in remaining if item["class"] == actual["class"]]
        if not candidates:
            mismatches.append(f"extra browser detection {actual}")
            continue
        best = max(candidates, key=lambda item: _iou(actual["bbox"], item["bbox"]))
        overlap = _iou(actual["bbox"], best["bbox"])
        score_delta = abs(actual["confidence"] - best["confidence"])
        if overlap < 0.9 or score_delta > 0.05:
            mismatches.append(
                f"{actual['class']}: IoU={overlap:.3f}, score delta={score_delta:.3f}; "
                f"browser={actual}, python={best}"
            )
        remaining.remove(best)
    mismatches.extend(f"missing browser detection {item}" for item in remaining)
    assert not mismatches, f"YOLO11n parity failed for {sample}:\n" + "\n".join(mismatches)
