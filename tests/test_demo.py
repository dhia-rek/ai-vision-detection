import pytest
from app import demo


def test_demo_writes_gif(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AVD_DETECTOR", "stub")
    out = tmp_path / "out"
    assert demo.main(["--out", str(out)]) == 0
    gif = out / "demo.gif"
    assert gif.exists()
    assert gif.read_bytes()[:3] == b"GIF"
    assert (out / "dog.jpg").exists()
    assert "dog.jpg" in capsys.readouterr().out


def test_demo_without_out(monkeypatch):
    monkeypatch.setenv("AVD_DETECTOR", "stub")
    assert demo.main([]) == 0


@pytest.mark.parametrize(
    ("mode", "sample", "gif"),
    [
        ("faces", "happy.jpg", "demo-faces.gif"),
        ("sign", "letter_B.jpg", "demo-sign.gif"),
        ("sign-gestures", "gesture_thumbs_up.jpg", "demo-sign-gestures.gif"),
    ],
)
def test_demo_modes(monkeypatch, tmp_path, capsys, mode, sample, gif):
    monkeypatch.setenv("AVD_DETECTOR", "stub")
    assert demo.main(["--mode", mode, "--out", str(tmp_path)]) == 0
    assert (tmp_path / gif).read_bytes()[:3] == b"GIF"
    assert (tmp_path / sample).exists()
    assert sample in capsys.readouterr().out


def test_demo_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("AVD_DETECTOR", "stub")
    with pytest.raises(SystemExit):
        demo.main(["--mode", "bogus"])
