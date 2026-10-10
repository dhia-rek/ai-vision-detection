import cv2
import numpy as np
import pytest
from app.detector import StubDetector
from app.main import create_app
from fastapi.testclient import TestClient


def make_image(w: int = 120, h: int = 80) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def jpeg_bytes(w: int = 120, h: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", make_image(w, h))
    assert ok
    return buf.tobytes()


@pytest.fixture
def image() -> np.ndarray:
    return make_image()


@pytest.fixture
def jpeg() -> bytes:
    return jpeg_bytes()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(detector=StubDetector()))


@pytest.fixture
def avi_bytes(tmp_path) -> bytes:
    """A tiny 10-frame MJPG .avi (10 fps, 64x48)."""
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    data = path.read_bytes()
    assert data
    return data


@pytest.fixture(scope="session")
def static_site(tmp_path_factory):
    """Base URL of the built static site (``scripts/build_pages_artifact.py`` layout), served locally."""
    from web_support import build_and_serve

    server, url = build_and_serve(tmp_path_factory.mktemp("pages") / "vision")
    yield url
    server.shutdown()


@pytest.fixture(scope="session")
def playwright_session():
    """A started Playwright driver shared by browser tests."""
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        yield playwright
