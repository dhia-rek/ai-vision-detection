import numpy as np
from app.detector import Detection, StubDetector, annotate, decode_image, encode_jpeg


def test_detection_to_dict_schema():
    d = Detection("dog", 0.123456, 1.26, 2.34, 3.45, 4.56)
    assert d.to_dict() == {
        "class": "dog",
        "confidence": 0.1235,
        "bbox": {"x1": 1.3, "y1": 2.3, "x2": 3.5, "y2": 4.6},
    }


def test_stub_detector(image):
    dets = StubDetector().detect(image)
    assert len(dets) == 1
    d = dets[0]
    assert 0 <= d.x1 < d.x2 <= image.shape[1]
    assert 0 <= d.y1 < d.y2 <= image.shape[0]


def test_annotate_shape_and_no_mutation(image):
    before = image.copy()
    out = annotate(image, StubDetector().detect(image))
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert np.array_equal(image, before)
    assert not np.array_equal(out, image)


def test_annotate_no_detections_is_copy(image):
    out = annotate(image, [])
    assert np.array_equal(out, image)
    assert out is not image


def test_encode_decode_roundtrip(image):
    raw = encode_jpeg(image, quality=90)
    assert raw[:2] == b"\xff\xd8"
    assert decode_image(raw).shape == image.shape


def test_decode_garbage_returns_none():
    assert decode_image(b"garbage") is None


def test_detection_extra_and_points_in_dict():
    d = Detection("A", 0.9, 0, 0, 1, 1, points=((1.0, 2.0),), extra={"top3": {"A": 0.9}})
    out = d.to_dict()
    assert out["top3"] == {"A": 0.9}
    assert out["class"] == "A"
