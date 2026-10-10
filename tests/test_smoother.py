import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from app.sign import LETTERS, LetterSmoother, smooth_letters


def scores(letter: str, confidence: float = 0.9) -> dict[str, float]:
    """Build a complete probability vector with a dominant letter."""
    other = (1 - confidence) / (len(LETTERS) - 1)
    return {c: confidence if c == letter else other for c in LETTERS}


def test_flicker_rejection():
    smoother = LetterSmoother()
    result = [smoother.feed(scores(c, 0.99), i * 0.1)[0] for i, c in enumerate("AABAAA")]
    assert result == ["A"] * 6


def test_sustained_new_letter_switches():
    smoother = LetterSmoother()
    smoother.feed(scores("A"), 0)
    result = [smoother.feed(scores("B"), i * 0.1)[0] for i in range(1, 12)]
    assert result[-1] == "B"


def test_gap_reset_clears_state():
    smoother = LetterSmoother()
    assert smoother.feed(scores("A"), 0)[0] == "A"
    assert smoother.feed(None, 0.7) == (None, 0.0)
    assert smoother.current is None and smoother.ema is None
    assert smoother.feed(scores("B"), 0.8)[0] == "B"


def test_low_smoothed_confidence_returns_no_letter():
    smoother = LetterSmoother()
    letter, confidence = smoother.feed({c: 1 / 26 for c in LETTERS}, 0)
    assert letter is None
    assert confidence == pytest.approx(1 / 26)


def test_python_javascript_parity():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is unavailable")
    js_path = Path(__file__).parents[1] / "backend/static/app.js"
    source = js_path.read_text()
    match = re.search(r"class LetterSmoother\s*\{.*?\n\}", source, re.S)
    assert match
    frames = [scores(c) for c in "AABAAABBBBB"]
    frames[4] = None
    times = [i * 0.1 for i in range(len(frames))]
    script = (
        match.group(0)
        + "\nconst s = new LetterSmoother();\n"
        + f"const frames = {json.dumps(frames)};\n"
        + f"const times = {json.dumps(times)};\n"
        + "console.log(JSON.stringify(frames.map((f,i)=>s.feed(f,times[i]))));"
    )
    actual = json.loads(subprocess.check_output([node, "-e", script], text=True))
    smoother = LetterSmoother()
    expected = [smoother.feed(frame, timestamp) for frame, timestamp in zip(frames, times, strict=True)]
    for observed, wanted in zip(actual, expected, strict=True):
        assert observed[0] == wanted[0]
        assert observed[1] == pytest.approx(wanted[1])


def test_batch_helper_uses_frame_timestamps():
    frames = [(0.0, scores("A")), (0.1, None), (0.8, scores("B"))]
    assert [letter for letter, _ in smooth_letters(frames)] == ["A", None, "B"]
