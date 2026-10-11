"""Browser outputs must match the Python backend on the bundled sample images.

The static app runs ONNX Runtime Web / MediaPipe Tasks Vision in headless Chromium; the oracle is the
Python backend (`YoloDetector`, `ExpressionDetector`, `SignDetector`). Tolerances: identical labels,
box IoU >= 0.90, scores within 0.05. An unmatched detection only counts as a mismatch if it is not a
borderline one (confidence within 0.05 of the 0.25 detection threshold), because the two runtimes
resize pixels slightly differently.

Set ``AVD_PARITY_REPORT=path.json`` to write every measured number (used for the README table).
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import pytest
from web_support import chromium, collect_errors

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "backend" / "samples"
IOU_MIN, SCORE_TOL, BORDERLINE = 0.90, 0.05, 0.05
OBJECT_CONF = 0.25
RESULTS: dict[str, dict] = {}

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def _iou(a: dict, b: dict) -> float:
    left, top = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    right, bottom = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    return inter / max(1e-9, area_a + area_b - inter)


def compare(
    expected: list[dict],
    actual: list[dict],
    threshold: float = OBJECT_CONF,
    iou_min: float = IOU_MIN,
    score_tol: float = SCORE_TOL,
) -> dict:
    """Greedy, label-aware matching; returns the worst IoU/score delta and every mismatch."""
    remaining = list(expected)
    mismatches, ious, deltas = [], [], []
    borderline = 0
    for got in sorted(actual, key=lambda d: -d["confidence"]):
        candidates = [d for d in remaining if d["class"] == got["class"]]
        if not candidates:
            if got["confidence"] - threshold <= BORDERLINE:
                borderline += 1
            else:
                mismatches.append(f"extra browser detection {got['class']} {got['confidence']:.3f}")
            continue
        best = max(candidates, key=lambda d: _iou(got["bbox"], d["bbox"]))
        remaining.remove(best)
        overlap, delta = _iou(got["bbox"], best["bbox"]), abs(got["confidence"] - best["confidence"])
        ious.append(overlap)
        deltas.append(delta)
        if overlap < iou_min or delta > score_tol:
            mismatches.append(f"{got['class']}: IoU={overlap:.3f} score delta={delta:.3f}")
    for missing in remaining:
        if missing["confidence"] - threshold <= BORDERLINE:
            borderline += 1
        else:
            mismatches.append(f"missing browser detection {missing['class']} {missing['confidence']:.3f}")
    return {
        "matched": len(ious),
        "expected": len(expected),
        "actual": len(actual),
        "borderline": borderline,
        "min_iou": min(ious, default=1.0),
        "max_score_delta": max(deltas, default=0.0),
        "mismatches": mismatches,
    }


def _record(key: str, result: dict) -> None:
    RESULTS[key] = {k: v for k, v in result.items() if k != "mismatches"}
    report = os.environ.get("AVD_PARITY_REPORT")
    if report:
        Path(report).write_text(json.dumps(RESULTS, indent=2, sort_keys=True))


def _data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


@pytest.fixture(scope="module")
def page(static_site, playwright_session):
    """One Chromium page on the built site, shared by this module."""
    with chromium(playwright_session) as browser:
        errors: list[str] = []
        tab = browser.new_page()
        collect_errors(tab, errors)
        tab.set_default_timeout(180_000)
        tab.goto(f"{static_site}/", wait_until="domcontentloaded")
        yield tab
        assert not errors, f"browser errors: {errors}"


def _bgr(path: Path):
    image = cv2.imread(str(path))
    assert image is not None, path
    return image


def _as_dict(detection) -> dict:
    box = detection.bbox if hasattr(detection, "bbox") else None
    return {
        "class": detection.label,
        "confidence": detection.confidence,
        "bbox": {"x1": detection.x1, "y1": detection.y1, "x2": detection.x2, "y2": detection.y2}
        if box is None
        else box,
    }


OBJECT_JS = """async ({ url, size }) => {
  const { createObjectEngine } = await import('./static/js/web-objects.js');
  const engine = createObjectEngine({ size });
  await engine.ready;
  const image = new Image(); image.src = url; await image.decode();
  const result = await engine.detect(image);
  engine.close();
  return result.detections;
}"""


@pytest.mark.parametrize("size", [640, 320])
@pytest.mark.parametrize("sample", sorted(p.name for p in (SAMPLES / "objects").glob("*.jpg")))
def test_objects_match_python_yolo11n(page, sample: str, size: int) -> None:
    """Browser YOLO11n ONNX (letterbox + NMS in JS) vs Ultralytics on the same image and input size."""
    from app.zoo import DEFAULT_OBJECT_MODEL, ensure_model
    from ultralytics import YOLO

    path = SAMPLES / "objects" / sample
    result = YOLO(str(ensure_model(DEFAULT_OBJECT_MODEL))).predict(
        _bgr(path), imgsz=size, conf=OBJECT_CONF, device="cpu", verbose=False
    )[0]
    expected = [
        {
            "class": result.names[int(b.cls)],
            "confidence": float(b.conf),
            "bbox": dict(zip(("x1", "y1", "x2", "y2"), (float(v) for v in b.xyxy[0]), strict=True)),
        }
        for b in result.boxes
    ]
    actual = page.evaluate(OBJECT_JS, {"url": _data_url(path), "size": size})
    outcome = compare(expected, actual)
    _record(f"objects/{size}/{sample}", outcome)
    assert not outcome["mismatches"], f"{sample}@{size}: {outcome['mismatches']}"
    assert outcome["matched"] > 0, "no detections to compare"


FACE_JS = """async ({ url, maxSide }) => {
  const { createFaceEngine } = await import('./static/js/web-faces-client.js');
  const engine = createFaceEngine(maxSide ? { detectMaxSide: maxSide, alignMaxSide: maxSide } : {});
  await engine.ready;
  const image = new Image(); image.src = url; await image.decode();
  const result = await engine.detect(image);
  engine.close();
  return result.detections.map((d) => ({ ...d, frame_width: result.image_size.width }));
}"""


@pytest.mark.parametrize("sample", sorted(p.name for p in (SAMPLES / "faces").glob("*.jpg")))
def test_faces_match_python_backend(page, sample: str) -> None:
    """Browser YuNet + MobileFaceNet FER vs the OpenCV-based backend: same face, same expression."""
    from app.faces import ExpressionDetector

    path = SAMPLES / "faces" / sample
    expected = []
    for d in ExpressionDetector().detect(_bgr(path)):
        item = _as_dict(d)
        item["scores"] = d.extra["scores"]
        expected.append(item)
    # The Python backend runs YuNet at the source resolution; so does this strict comparison.
    actual = page.evaluate(FACE_JS, {"url": _data_url(path), "maxSide": 4096})
    width = _bgr(path).shape[1]
    for item in actual:  # the app reports boxes in its (possibly downscaled) frame; map back to the source
        k = width / item.pop("frame_width")
        item["bbox"] = {key: value * k for key, value in item["bbox"].items()}
    outcome = compare(expected, actual, threshold=0.0)
    for got, want in zip(
        sorted(actual, key=lambda d: d["bbox"]["x1"]),
        sorted(expected, key=lambda d: d["bbox"]["x1"]),
        strict=False,
    ):
        worst = max(abs(got["scores"][k] - want["scores"][k]) for k in want["scores"])
        outcome["max_score_delta"] = max(outcome["max_score_delta"], worst)
        if worst > SCORE_TOL:
            outcome["mismatches"].append(f"expression scores differ by {worst:.3f}")
    _record(f"faces/{sample}", outcome)
    assert expected, "the Python backend found no face in the sample"
    assert not outcome["mismatches"], f"{sample}: {outcome['mismatches']}"


@pytest.mark.parametrize("sample", sorted(p.name for p in (SAMPLES / "faces").glob("*.jpg")))
def test_faces_at_live_resolution_keep_label_and_box(page, sample: str) -> None:
    """With the live settings (YuNet input <= 640 px) the label and box (IoU >= 0.8) still match.

    Landmarks found on a downscaled frame shift the aligned crop by a few pixels, which moves the
    expression probabilities (up to ~0.3 on `angry.jpg`) without changing the winning label.
    """
    from app.faces import ExpressionDetector

    path = SAMPLES / "faces" / sample
    width = _bgr(path).shape[1]
    expected = [{**_as_dict(d), "scores": d.extra["scores"]} for d in ExpressionDetector().detect(_bgr(path))]
    actual = page.evaluate(FACE_JS, {"url": _data_url(path), "maxSide": 0})
    for item in actual:
        k = width / item.pop("frame_width")
        item["bbox"] = {key: value * k for key, value in item["bbox"].items()}
    outcome = compare(expected, actual, threshold=0.0, iou_min=0.8, score_tol=1.0)
    outcome["max_score_delta"] = max(
        abs(got["scores"][k] - want["scores"][k])
        for got, want in zip(actual, expected, strict=False)
        for k in want["scores"]
    )
    _record(f"faces-live/{sample}", outcome)
    assert not outcome["mismatches"], f"{sample}: {outcome['mismatches']}"
    assert [d["class"] for d in actual] == [d["class"] for d in expected]


SIGN_JS = """async ({ url, mode }) => {
  const { createSignEngine } = await import('./static/js/web-sign.js');
  const engine = await createSignEngine({ runningMode: 'IMAGE' });
  const image = new Image(); image.src = url; await image.decode();
  const result = engine.detect(image, mode);
  engine.close();
  return result;
}"""

SIGN_CASES = [
    ("letter_B.jpg", "spell"),
    ("letter_L.jpg", "spell"),
    ("letter_V.jpg", "spell"),
    ("letter_Y.jpg", "spell"),
    ("gesture_thumbs_up.jpg", "gestures"),
    ("gesture_i_love_you.jpg", "gestures"),
]


@pytest.mark.parametrize(("sample", "mode"), SIGN_CASES)
def test_sign_matches_python_backend(page, sample: str, mode: str) -> None:
    """Browser MediaPipe + JS ASL classifier vs the Python SignDetector (letters, top-3, gestures)."""
    from app.sign import SignDetector

    path = SAMPLES / "sign" / sample
    detector = SignDetector(mode)
    expected = [{**_as_dict(d), "top3": d.extra.get("top3")} for d in detector.detect(_bgr(path))]
    detector.recognizer.close()
    actual = page.evaluate(SIGN_JS, {"url": _data_url(path), "mode": mode})
    outcome = compare(expected, actual, threshold=0.0)
    for got, want in zip(actual, expected, strict=False):
        if want.get("top3"):
            top = set(want["top3"]) | set(got["top3"])
            worst = max(abs(got["top3"].get(k, 0.0) - want["top3"].get(k, 0.0)) for k in top)
            outcome["max_score_delta"] = max(outcome["max_score_delta"], worst)
            if worst > SCORE_TOL:
                outcome["mismatches"].append(f"top-3 probabilities differ by {worst:.3f}")
    _record(f"sign/{mode}/{sample}", outcome)
    assert expected, "the Python backend found no hand in the sample"
    assert not outcome["mismatches"], f"{sample}: {outcome['mismatches']}"
