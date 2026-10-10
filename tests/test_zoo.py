import hashlib

import pytest
from app import zoo


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    src = tmp_path / "src.bin"
    src.write_bytes(b"model-bytes")
    spec = zoo.ModelSpec(
        "fake.bin", src.as_uri(), hashlib.sha256(b"model-bytes").hexdigest(), 0.0, "MIT", "test"
    )
    monkeypatch.setitem(zoo.MODELS, "fake.bin", spec)
    monkeypatch.setenv("AVD_MODEL_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(zoo, "_verified", set())
    return spec


def test_download_and_verify(fake_model):
    path = zoo.ensure_model("fake.bin")
    assert path.read_bytes() == b"model-bytes"


def test_corrupted_cache_is_redownloaded(fake_model, tmp_path):
    cached = tmp_path / "cache" / "fake.bin"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"truncated")
    assert zoo.ensure_model("fake.bin").read_bytes() == b"model-bytes"


def test_checksum_mismatch_raises(fake_model, monkeypatch):
    bad = zoo.ModelSpec("fake.bin", fake_model.url, "0" * 64, 0.0, "MIT", "test")
    monkeypatch.setitem(zoo.MODELS, "fake.bin", bad)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        zoo.ensure_model("fake.bin")
    assert not list((zoo.model_dir()).glob("*.part"))
