"""Shared helpers for tests that drive the built static site in headless Chromium."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# MediaPipe logs INFO lines through console.error; they are not failures.
BENIGN_CONSOLE = ("INFO:", "No available adapters")
CHROMIUM_ARGS = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]


def require_or_skip(reason: str) -> None:
    """Skip locally, but fail in CI (``AVD_REQUIRE_BROWSER=1``) so a broken setup cannot hide."""
    if os.environ.get("AVD_REQUIRE_BROWSER") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def build_and_serve(directory: Path):
    """Build the Pages artifact into ``directory`` and serve it; returns ``(server, base_url)``."""
    from build_pages_artifact import build, serve

    try:
        site = build(directory)
    except SystemExit as error:
        require_or_skip(str(error))
    server = serve(site, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def make_fake_camera(sample: Path, out_dir: Path, width: int, height: int, fps: int = 30) -> Path:
    """Loop a sample image as a short Y4M stream that Chromium can use as a fake webcam."""
    if shutil.which("ffmpeg") is None:
        require_or_skip("ffmpeg is needed to make the fake camera stream")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{sample.stem}-{width}x{height}-{fps}fps.y4m"
    if not target.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-loop",
                "1",
                "-i",
                str(sample),
                "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
                f"fps={fps},format=yuv420p",
                "-t",
                "3",
                "-f",
                "yuv4mpegpipe",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
    return target


@contextmanager
def chromium(playwright, camera: Path | None = None, extra_args: tuple[str, ...] = ()):
    """Launch headless Chromium (optionally with a fake camera file); fail clearly if it cannot start."""
    args = [*extra_args]
    if camera is not None:
        args += [*CHROMIUM_ARGS, f"--use-file-for-fake-video-capture={camera}"]
    try:
        browser = playwright.chromium.launch(headless=True, args=args)
    except Exception as error:  # noqa: BLE001 - Playwright raises a generic Error for a missing browser
        require_or_skip(f"Chromium could not start: {error}")
    try:
        yield browser
    finally:
        browser.close()


def collect_errors(page, errors: list[str]) -> None:
    """Record page errors and console errors that are not known-benign noise."""
    page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))

    def on_console(message) -> None:
        if message.type == "error" and not any(text in message.text for text in BENIGN_CONSOLE):
            errors.append(f"console.error: {message.text[:300]}")

    page.on("console", on_console)
    page.on("requestfailed", lambda request: errors.append(f"request failed: {request.url}"))
    page.on(
        "response",
        lambda response: (
            errors.append(f"HTTP {response.status}: {response.url}") if response.status >= 400 else None
        ),
    )
