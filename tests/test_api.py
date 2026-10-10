import base64

import cv2
import numpy as np
import pytest
from app.detector import StubDetector
from app.main import create_app
from fastapi.testclient import TestClient


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_detect_happy_path(client, jpeg):
    r = client.post("/detect/", files={"file": ("a.jpg", jpeg, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "stub"
    assert body["total_detections"] == 1 == len(body["detections"])
    det = body["detections"][0]
    assert det["class"] == "object"
    assert 0 < det["confidence"] <= 1
    assert det["bbox"] == {"x1": 30.0, "y1": 20.0, "x2": 90.0, "y2": 60.0}

    prefix = "data:image/jpeg;base64,"
    assert body["result_image"].startswith(prefix)
    raw = base64.b64decode(body["result_image"][len(prefix) :])
    assert raw[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (80, 120, 3)


def test_frame_endpoint(client, jpeg):
    r = client.post("/detect/frame", files={"frame": ("f.jpg", jpeg, "image/jpeg")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["X-Total-Detections"] == "1"
    assert r.content[:2] == b"\xff\xd8"


@pytest.mark.parametrize("quality", [5, 96])
def test_frame_quality_out_of_range(client, jpeg, quality):
    r = client.post(
        "/detect/frame", params={"quality": quality}, files={"frame": ("f.jpg", jpeg, "image/jpeg")}
    )
    assert r.status_code == 422


def test_frame_quality_affects_size(client):
    from conftest import jpeg_bytes

    data = jpeg_bytes(300, 300)
    sizes = {}
    for q in (10, 95):
        r = client.post(
            "/detect/frame", params={"quality": q}, files={"frame": ("f.jpg", data, "image/jpeg")}
        )
        sizes[q] = len(r.content)
    assert sizes[10] < sizes[95]


def test_non_image_content_type(client):
    r = client.post("/detect/", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_garbage_bytes(client):
    r = client.post("/detect/", files={"file": ("a.jpg", b"not an image at all", "image/jpeg")})
    assert r.status_code == 400


def test_missing_file(client):
    assert client.post("/detect/").status_code == 422


def test_too_large(monkeypatch, jpeg):
    monkeypatch.setenv("AVD_MAX_UPLOAD_MB", "0.001")  # ~1 KB
    c = TestClient(create_app(detector=StubDetector()))
    big = jpeg + b"\0" * 5000
    r = c.post("/detect/", files={"file": ("a.jpg", big, "image/jpeg")})
    assert r.status_code == 413
    r = c.post("/detect/frame", files={"frame": ("a.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_demo_samples_listed(client):
    r = client.get("/demo/samples")
    assert r.status_code == 200
    samples = r.json()["samples"]
    assert samples
    assert "dog.jpg" in samples


def test_demo_sample_image_and_detect(client):
    r = client.get("/demo/samples/dog.jpg")
    assert r.status_code == 200
    assert r.content[:2] == b"\xff\xd8"
    r = client.post("/demo/detect/dog.jpg")
    assert r.status_code == 200
    assert r.json()["total_detections"] == 1


def test_unknown_sample(client):
    assert client.get("/demo/samples/nope.jpg").status_code == 404
    assert client.post("/demo/detect/nope.jpg").status_code == 404


@pytest.mark.parametrize(
    "name", ["../main.py", "..%2Fmain.py", "%2e%2e%2fmain.py", "..%2F..%2Fpyproject.toml"]
)
def test_path_traversal_rejected(client, name):
    r = client.get(f"/demo/samples/{name}")
    assert r.status_code == 404
    assert b"create_app" not in r.content
    r = client.post(f"/demo/detect/{name}")
    assert r.status_code == 404


def test_too_many_pixels(monkeypatch):
    import cv2
    import numpy as np
    from app.detector import StubDetector
    from app.main import create_app

    monkeypatch.setenv("AVD_MAX_PIXELS", "0.01")  # 10k pixels
    c = TestClient(create_app(detector=StubDetector()))
    ok, buf = cv2.imencode(".png", np.zeros((200, 200, 3), dtype=np.uint8))
    r = c.post("/detect/", files={"file": ("big.png", buf.tobytes(), "image/png")})
    assert r.status_code == 413
