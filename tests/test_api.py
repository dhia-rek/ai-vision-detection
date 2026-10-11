import base64

import cv2
import numpy as np
import pytest
from app.detector import StubDetector
from app.main import create_app
from fastapi.testclient import TestClient

MODE_PARAMS = [
    {"mode": "objects"},
    {"mode": "faces"},
    {"mode": "sign", "sign_mode": "spell"},
    {"mode": "sign", "sign_mode": "gestures"},
]


def post_image(client, path, jpeg, params=None):
    return client.post(path, params=params or {}, files={"file": ("a.jpg", jpeg, "image/jpeg")})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_modes_endpoint(client):
    body = client.get("/api/modes").json()
    assert set(body["modes"]) == {"objects", "faces", "sign"}
    assert len(body["modes"]) == 3
    assert body["models"]
    for info in body["models"].values():
        assert info["license"]
        assert info["size_mb"] > 0
        assert info["source"]


@pytest.mark.parametrize("params", MODE_PARAMS)
def test_analyze_every_mode(client, jpeg, params):
    r = post_image(client, "/api/analyze", jpeg, params)
    assert r.status_code == 200
    body = r.json()
    expected = params["mode"] if params["mode"] != "sign" else f"sign-{params['sign_mode']}"
    assert body["mode"] == expected
    assert body["model_used"] == "stub"
    assert body["total_detections"] == 1 == len(body["detections"])
    assert body["inference_ms"] >= 0
    det = body["detections"][0]
    assert det["class"] == "object"
    assert det["bbox"] == {"x1": 30.0, "y1": 20.0, "x2": 90.0, "y2": 60.0}
    prefix = "data:image/jpeg;base64,"
    assert body["result_image"].startswith(prefix)
    raw = base64.b64decode(body["result_image"][len(prefix) :])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (80, 120, 3)


def test_analyze_reports_image_size_and_device(client, jpeg):
    body = post_image(client, "/api/analyze", jpeg).json()
    assert body["image_size"] == {"width": 120, "height": 80}
    assert body["device"] == "cpu"


@pytest.mark.parametrize("params", MODE_PARAMS)
def test_analyze_without_annotation(client, jpeg, params):
    """annotate=false returns the same JSON minus the burned-in image."""
    full = post_image(client, "/api/analyze", jpeg, params).json()
    lean = post_image(client, "/api/analyze", jpeg, {**params, "annotate": "false"}).json()
    assert "result_image" not in lean
    assert lean == {k: v for k, v in full.items() if k not in ("result_image", "inference_ms")} | {
        "inference_ms": lean["inference_ms"]
    }


def test_analyze_annotate_defaults_to_true(client, jpeg):
    assert "result_image" in post_image(client, "/api/analyze", jpeg).json()


def test_analyze_default_mode_is_objects(client, jpeg):
    assert post_image(client, "/api/analyze", jpeg).json()["mode"] == "objects"


def test_invalid_mode(client, jpeg):
    assert post_image(client, "/api/analyze", jpeg, {"mode": "bogus"}).status_code == 400


def test_invalid_sign_mode(client, jpeg):
    params = {"mode": "sign", "sign_mode": "bogus"}
    assert post_image(client, "/api/analyze", jpeg, params).status_code == 400


def test_sign_mode_ignored_outside_sign(client, jpeg):
    r = post_image(client, "/api/analyze", jpeg, {"mode": "faces", "sign_mode": "bogus"})
    assert r.status_code == 200


def test_non_image_content_type(client):
    r = client.post("/api/analyze", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_garbage_bytes(client):
    r = client.post("/api/analyze", files={"file": ("a.jpg", b"not an image at all", "image/jpeg")})
    assert r.status_code == 400


def test_missing_file(client):
    assert client.post("/api/analyze").status_code == 422


def test_too_large(monkeypatch, jpeg):
    monkeypatch.setenv("AVD_MAX_UPLOAD_MB", "0.001")  # ~1 KB
    c = TestClient(create_app(detector=StubDetector()))
    big = jpeg + b"\0" * 5000
    assert post_image(c, "/api/analyze", big).status_code == 413


def test_too_many_pixels(monkeypatch):
    monkeypatch.setenv("AVD_MAX_PIXELS", "0.01")  # 10k pixels
    c = TestClient(create_app(detector=StubDetector()))
    ok, buf = cv2.imencode(".png", np.zeros((200, 200, 3), dtype=np.uint8))
    assert ok
    r = c.post("/api/analyze", files={"file": ("big.png", buf.tobytes(), "image/png")})
    assert r.status_code == 413


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/frame"),
        ("post", "/api/video"),
        ("post", "/detect/"),
        ("post", "/detect/frame"),
        ("get", "/demo/samples"),
        ("post", "/demo/analyze/objects/dog.jpg"),
        ("get", "/api/models/asl-letters-web"),
    ],
)
def test_removed_server_routes_are_gone(client, method, path):
    """The live app runs in the browser; the old frame, video and sample routes no longer exist."""
    assert getattr(client, method)(path).status_code in (404, 405)
