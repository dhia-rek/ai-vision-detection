"""Real-model tests: check the labels are actually right on the bundled fixtures.

Models download on first run (app.zoo.ensure_model); a failed download skips the test
locally but fails in CI.
Each detector is built once per session.
"""

import os
import urllib.error
from pathlib import Path

import cv2
import pytest
from app.main import create_app
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "backend" / "samples"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_or_skip(key: str):
    from app.main import build_detector

    try:
        return build_detector(key)
    except urllib.error.URLError as exc:
        # Only a failed download skips, and never in CI: a missing system library
        # (e.g. libEGL for MediaPipe) or a checksum mismatch must fail loudly.
        if os.getenv("CI"):
            raise
        pytest.skip(f"could not download {key} model: {exc}")


@pytest.fixture(scope="session", autouse=True)
def _real_detector_env():
    mp = pytest.MonkeyPatch()
    mp.delenv("AVD_DETECTOR", raising=False)
    yield
    mp.undo()


@pytest.fixture(scope="session")
def objects_det(_real_detector_env):
    return build_or_skip("objects")


@pytest.fixture(scope="session")
def faces_det(_real_detector_env):
    return build_or_skip("faces")


@pytest.fixture(scope="session")
def spell_det(_real_detector_env):
    det = build_or_skip("sign-spell")
    yield det
    det.recognizer.close()  # close before interpreter shutdown, or MediaPipe logs a TypeError


@pytest.fixture(scope="session")
def gestures_det(_real_detector_env):
    det = build_or_skip("sign-gestures")
    yield det
    det.recognizer.close()  # close before interpreter shutdown, or MediaPipe logs a TypeError


def load(path: Path):
    img = cv2.imread(str(path))
    assert img is not None, path
    return img


# ---- objects ----


def test_objects_dog(objects_det):
    dets = objects_det.detect(load(SAMPLES / "objects" / "dog.jpg"))
    assert "dog" in {d.label for d in dets}


def test_objects_two_cats(objects_det):
    dets = objects_det.detect(load(SAMPLES / "objects" / "cats.jpg"))
    assert sum(d.label == "cat" for d in dets) >= 2


# ---- faces ----


@pytest.mark.parametrize("label", ["happy", "angry", "neutral"])
def test_face_expression(faces_det, label):
    dets = faces_det.detect(load(FIXTURES / "faces" / f"{label}.jpg"))
    assert len(dets) == 1, [d.label for d in dets]
    assert dets[0].label == label
    assert label in dets[0].extra["top3"]
    scores = dets[0].extra["scores"]
    assert set(scores) == {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"}
    assert sum(scores.values()) == pytest.approx(1.0, abs=0.01)
    assert max(scores, key=scores.get) == label
    if label == "happy":
        assert dets[0].confidence > 0.9


@pytest.mark.parametrize("name", ["street.jpg", "livingroom.jpg"])
def test_faces_no_face_images_do_not_crash(faces_det, name):
    dets = faces_det.detect(load(SAMPLES / "objects" / name))
    # No portrait in these scenes: at most a stray tiny false positive; never a crash.
    assert len(dets) <= 1
    for d in dets:
        assert d.label in {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"}


# ---- sign ----


@pytest.mark.parametrize("letter", ["A", "B", "C", "L", "V", "Y"])
def test_sign_spell_letters(spell_det, letter):
    dets = spell_det.detect(load(FIXTURES / "sign" / f"letter_{letter}.jpg"))
    assert len(dets) == 1
    assert dets[0].label == letter, f"got {dets[0].label}, top3={dets[0].extra['top3']}"
    assert dets[0].confidence > 0.5


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gesture_thumbs_up.jpg", "thumbs up"),
        ("gesture_thumbs_up_2.jpg", "thumbs up"),
        ("gesture_i_love_you.jpg", "I love you"),
    ],
)
def test_sign_gestures(gestures_det, name, expected):
    dets = gestures_det.detect(load(FIXTURES / "sign" / name))
    assert dets, "no hand found"
    assert dets[0].label.startswith(expected), (dets[0].label, dets[0].extra)


# ---- video + API end to end ----


@pytest.fixture(scope="module")
def real_client(spell_det, faces_det, _real_detector_env):
    app = create_app()
    # Reuse the session detectors instead of building (and loading) them again.
    # create_app caches built detectors lazily, so patch build_detector for this app.
    import app.main as main

    prebuilt = {"sign-spell": spell_det, "faces": faces_det}
    original = main.build_detector
    main.build_detector = lambda key="objects": prebuilt.get(key) or original(key)
    try:
        yield TestClient(app)
    finally:
        main.build_detector = original


def test_api_faces_end_to_end(real_client):
    data = (FIXTURES / "faces" / "happy.jpg").read_bytes()
    r = real_client.post(
        "/api/analyze", params={"mode": "faces"}, files={"file": ("h.jpg", data, "image/jpeg")}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "faces"
    assert body["detections"][0]["class"] == "happy"
    assert "top3" in body["detections"][0]
    assert body["model_used"] == "yunet+mobilefacenet-fer"
