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


def test_demo_without_out(monkeypatch, tmp_path):
    monkeypatch.setenv("AVD_DETECTOR", "stub")
    assert demo.main([]) == 0
