import base64
import json

import cv2
import numpy as np
import pytest
from app.detector import StubDetector
from app.main import create_app
from fastapi.testclient import TestClient

AVI = "video/x-msvideo"
MODE_PARAMS = [
    {"mode": "objects"},
    {"mode": "faces"},
    {"mode": "sign", "sign_mode": "spell"},
    {"mode": "sign", "sign_mode": "gestures"},
]


def post_image(client, path, jpeg, params=None, field="file"):
    return client.post(path, params=params or {}, files={field: ("a.jpg", jpeg, "image/jpeg")})


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


def test_analyze_default_mode_is_objects(client, jpeg):
    assert post_image(client, "/api/analyze", jpeg).json()["mode"] == "objects"


@pytest.mark.parametrize("params", MODE_PARAMS)
def test_frame_headers_every_mode(client, jpeg, params):
    r = post_image(client, "/api/frame", jpeg, params, field="frame")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"
    assert r.headers["X-Total-Detections"] == "1"
    assert float(r.headers["X-Inference-Ms"]) >= 0
    dets = json.loads(r.headers["X-Detections"])
    assert isinstance(dets, list) and len(dets) == 1
    assert dets[0]["class"] == "object"
    assert 0 < dets[0]["confidence"] <= 1


def test_invalid_mode(client, jpeg):
    assert post_image(client, "/api/analyze", jpeg, {"mode": "bogus"}).status_code == 400
    assert post_image(client, "/api/frame", jpeg, {"mode": "bogus"}, field="frame").status_code == 400


def test_invalid_sign_mode(client, jpeg):
    params = {"mode": "sign", "sign_mode": "bogus"}
    assert post_image(client, "/api/analyze", jpeg, params).status_code == 400
    assert post_image(client, "/api/frame", jpeg, params, field="frame").status_code == 400


def test_sign_mode_ignored_outside_sign(client, jpeg):
    r = post_image(client, "/api/analyze", jpeg, {"mode": "faces", "sign_mode": "bogus"})
    assert r.status_code == 200


def test_legacy_detect(client, jpeg):
    r = post_image(client, "/detect/", jpeg)
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "objects"
    assert body["total_detections"] == 1


def test_legacy_detect_frame(client, jpeg):
    r = post_image(client, "/detect/frame", jpeg, field="frame")
    assert r.status_code == 200
    assert r.headers["X-Total-Detections"] == "1"
    assert r.content[:2] == b"\xff\xd8"


@pytest.mark.parametrize("path", ["/api/frame", "/detect/frame"])
@pytest.mark.parametrize("quality", [5, 96])
def test_frame_quality_out_of_range(client, jpeg, path, quality):
    r = post_image(client, path, jpeg, {"quality": quality}, field="frame")
    assert r.status_code == 422


def test_frame_quality_affects_size(client):
    from conftest import jpeg_bytes

    data = jpeg_bytes(300, 300)
    sizes = {
        q: len(post_image(client, "/api/frame", data, {"quality": q}, field="frame").content)
        for q in (10, 95)
    }
    assert sizes[10] < sizes[95]


@pytest.mark.parametrize("path", ["/api/analyze", "/detect/"])
def test_non_image_content_type(client, path):
    r = client.post(path, files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


@pytest.mark.parametrize("path", ["/api/analyze", "/detect/"])
def test_garbage_bytes(client, path):
    r = client.post(path, files={"file": ("a.jpg", b"not an image at all", "image/jpeg")})
    assert r.status_code == 400


def test_garbage_bytes_frame(client):
    r = client.post("/api/frame", files={"frame": ("a.jpg", b"nope", "image/jpeg")})
    assert r.status_code == 400


def test_missing_file(client):
    assert client.post("/api/analyze").status_code == 422
    assert client.post("/api/video").status_code == 422


def test_too_large(monkeypatch, jpeg):
    monkeypatch.setenv("AVD_MAX_UPLOAD_MB", "0.001")  # ~1 KB
    c = TestClient(create_app(detector=StubDetector()))
    big = jpeg + b"\0" * 5000
    assert post_image(c, "/api/analyze", big).status_code == 413
    assert post_image(c, "/api/frame", big, field="frame").status_code == 413
    assert post_image(c, "/detect/", big).status_code == 413


def test_too_many_pixels(monkeypatch):
    monkeypatch.setenv("AVD_MAX_PIXELS", "0.01")  # 10k pixels
    c = TestClient(create_app(detector=StubDetector()))
    ok, buf = cv2.imencode(".png", np.zeros((200, 200, 3), dtype=np.uint8))
    assert ok
    r = c.post("/api/analyze", files={"file": ("big.png", buf.tobytes(), "image/png")})
    assert r.status_code == 413


# ---- video ----


@pytest.mark.parametrize("params", [{"mode": "objects"}, {"mode": "sign", "sign_mode": "spell"}])
def test_video_happy_path(client, avi_bytes, params):
    r = client.post("/api/video", params={**params, "fps": 5}, files={"file": ("t.avi", avi_bytes, AVI)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["processed_frames"] > 0
    assert len(body["frames"]) == body["processed_frames"]
    assert body["label_counts"]["object"] == body["processed_frames"]
    assert body["model_used"] == "stub"
    frame = body["frames"][0]
    assert frame["t"] == 0
    assert frame["detections"][0]["class"] == "object"
    # Only the spell mode builds text.
    assert ("text" in body) == (params["mode"] == "sign")


def test_video_fps_subsamples(client, avi_bytes):
    full = client.post("/api/video", params={"fps": 10}, files={"file": ("t.avi", avi_bytes, AVI)}).json()
    half = client.post("/api/video", params={"fps": 5}, files={"file": ("t.avi", avi_bytes, AVI)}).json()
    assert half["processed_frames"] < full["processed_frames"]


def test_video_max_seconds(client, avi_bytes):
    r = client.post(
        "/api/video", params={"fps": 10, "max_seconds": 0.35}, files={"file": ("t.avi", avi_bytes, AVI)}
    )
    assert 0 < r.json()["processed_frames"] < 10


def test_video_bad_content_type(client, avi_bytes):
    r = client.post("/api/video", files={"file": ("t.avi", avi_bytes, "application/octet-stream")})
    assert r.status_code == 400
    r = client.post("/api/video", files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 400


def test_video_garbage(client):
    r = client.post("/api/video", files={"file": ("t.avi", b"definitely not a video", AVI)})
    assert r.status_code == 400


def test_video_invalid_modes(client, avi_bytes):
    files = {"file": ("t.avi", avi_bytes, AVI)}
    assert client.post("/api/video", params={"mode": "bogus"}, files=files).status_code == 400
    r = client.post("/api/video", params={"mode": "sign", "sign_mode": "x"}, files=files)
    assert r.status_code == 400


@pytest.mark.parametrize("fps", [0, 31])
def test_video_fps_out_of_range(client, avi_bytes, fps):
    r = client.post("/api/video", params={"fps": fps}, files={"file": ("t.avi", avi_bytes, AVI)})
    assert r.status_code == 422


def test_video_too_large(monkeypatch, avi_bytes):
    monkeypatch.setenv("AVD_MAX_VIDEO_MB", "0.001")  # ~1 KB
    assert len(avi_bytes) > 1048
    c = TestClient(create_app(detector=StubDetector()))
    r = c.post("/api/video", files={"file": ("t.avi", avi_bytes, AVI)})
    assert r.status_code == 413


# ---- demo samples ----


def test_demo_samples_listed(client):
    r = client.get("/demo/samples")
    assert r.status_code == 200
    samples = r.json()
    assert set(samples) == {"objects", "faces", "sign", "video"}
    assert "dog.jpg" in samples["objects"]
    assert "happy.jpg" in samples["faces"]
    assert samples["sign"]
    assert "fingerspelling_cab.mp4" in samples["video"]
    assert "faces.mp4" in samples["video"]


def test_demo_sample_file(client):
    r = client.get("/demo/samples/objects/dog.jpg")
    assert r.status_code == 200
    assert r.content[:2] == b"\xff\xd8"
    r = client.get("/demo/samples/video/faces.mp4")
    assert r.status_code == 200
    assert len(r.content) > 1000


def test_demo_analyze_defaults_to_group_mode(client):
    for group, name, mode in [
        ("objects", "dog.jpg", "objects"),
        ("faces", "happy.jpg", "faces"),
        ("sign", "letter_B.jpg", "sign-spell"),
    ]:
        r = client.post(f"/demo/analyze/{group}/{name}")
        assert r.status_code == 200
        assert r.json()["mode"] == mode


def test_demo_analyze_mode_override(client):
    r = client.post("/demo/analyze/sign/letter_B.jpg", params={"mode": "sign", "sign_mode": "gestures"})
    assert r.status_code == 200
    assert r.json()["mode"] == "sign-gestures"
    r = client.post("/demo/analyze/objects/dog.jpg", params={"mode": "bogus"})
    assert r.status_code == 400


def test_demo_analyze_video_group_rejected(client):
    assert client.post("/demo/analyze/video/faces.mp4").status_code == 404


def test_unknown_sample(client):
    assert client.get("/demo/samples/objects/nope.jpg").status_code == 404
    assert client.get("/demo/samples/nogroup/dog.jpg").status_code == 404
    assert client.post("/demo/analyze/objects/nope.jpg").status_code == 404
    assert client.post("/demo/analyze/nogroup/dog.jpg").status_code == 404


def test_old_demo_routes_gone(client):
    assert client.get("/demo/samples/dog.jpg").status_code == 404
    assert client.post("/demo/detect/dog.jpg").status_code == 404


@pytest.mark.parametrize(
    "name", ["../main.py", "..%2Fmain.py", "%2e%2e%2fmain.py", "..%2F..%2Fpyproject.toml"]
)
def test_path_traversal_rejected(client, name):
    for group in ("objects", "video"):
        r = client.get(f"/demo/samples/{group}/{name}")
        assert r.status_code == 404
        assert b"create_app" not in r.content
    assert client.post(f"/demo/analyze/objects/{name}").status_code == 404


@pytest.mark.parametrize("group", ["..", "..%2F", "%2e%2e", "..%2Fapp"])
def test_path_traversal_in_group_rejected(client, group):
    assert client.get(f"/demo/samples/{group}/main.py").status_code == 404
    assert client.post(f"/demo/analyze/{group}/main.py").status_code == 404


def test_frame_header_reports_truncation(client, jpeg):
    r = client.post("/api/frame", files={"frame": ("f.jpg", jpeg, "image/jpeg")})
    assert r.headers["X-Detections-Truncated"] == "false"
