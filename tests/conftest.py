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
