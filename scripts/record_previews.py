"""Record the looping preview clips shown before the camera starts.

Each clip is the real app running on a fake camera fed with the bundled sample media, captured
from the viewport element. Output: ``backend/static/previews/<mode>.mp4`` plus a ``.jpg`` poster.

    uv run python scripts/record_previews.py --base http://127.0.0.1:8013
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import ViewportSize, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "backend" / "samples"
OUT = ROOT / "backend" / "static" / "previews"
FPS = 10
HOLD = 3.5  # seconds each still stays on the fake camera

# mode -> (URL query, camera stills); each clip records one full cycle of its stills
CLIPS = {
    "objects": (
        "mode=objects",
        ["objects/street.jpg", "objects/bicycles.jpg", "objects/dog.jpg"],
    ),
    "faces": ("mode=faces", ["faces/happy.jpg", "faces/angry.jpg", "faces/neutral.jpg"]),
    "sign": (
        "mode=sign&sign_mode=spell",
        ["sign/letter_B.jpg", "sign/letter_L.jpg", "sign/letter_V.jpg", "sign/letter_Y.jpg"],
    ),
}


def run(command: list[str]) -> None:
    """Run ffmpeg and fail loudly."""
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())


def camera_file(sources: list[str], work: Path, name: str) -> Path:
    """Turn stills (HOLD seconds each) into a 640x480 Y4M stream for Chromium's fake camera.

    Fill-crop anchored near the top, so portrait photos keep the face in frame.
    """
    scale = "scale=640:480:force_original_aspect_ratio=increase,crop=640:480:(iw-640)/2:(ih-480)/5"
    destination = work / f"{name}.y4m"
    inputs: list[str] = []
    for source in sources:
        inputs += ["-loop", "1", "-t", str(HOLD), "-i", str(SAMPLES / source)]
    chains = "".join(f"[{i}:v]{scale},setsar=1,fps={FPS}[v{i}];" for i in range(len(sources)))
    joined = "".join(f"[v{i}]" for i in range(len(sources)))
    graph = f"{chains}{joined}concat=n={len(sources)}:v=1:a=0,format=yuv420p[out]"
    run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-f",
            "yuv4mpegpipe",
            str(destination),
        ]
    )
    return destination


def record(base: str, name: str, query: str, camera: Path, seconds: float, work: Path) -> None:
    """Start the camera in the app, screen-record the page and cut out the viewport."""
    args = [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        f"--use-file-for-fake-video-capture={camera}",
    ]
    size: ViewportSize = {"width": 1280, "height": 860}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=args)
        context = browser.new_context(
            viewport=size, record_video_dir=str(work / name), record_video_size=size
        )
        opened = time.monotonic()
        page = context.new_page()
        page.goto(f"{base}/?{query}&theme=light")
        page.locator("#startBtn").click()
        page.wait_for_function("() => document.body.dataset.camera === 'live'", timeout=60_000)
        page.add_style_tag(
            content=".vp-top { display: none !important; }"
        )  # the page adds its own status pill
        page.wait_for_timeout(4000)  # model download and warm-up
        start = time.monotonic() - opened
        box = page.locator("#viewport").bounding_box()
        page.wait_for_timeout(int(seconds * 1000))
        video = page.video
        context.close()
        source = video.path() if video else None
        browser.close()
    if not source or not box:
        raise RuntimeError(f"{name}: no recording")
    x, y, w, h = (int(box[k]) // 2 * 2 for k in ("x", "y", "width", "height"))
    OUT.mkdir(parents=True, exist_ok=True)
    crop = f"crop={w}:{h}:{x}:{y},scale=trunc(min(iw\\,720)/2)*2:-2"
    run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.2f}",
            "-t",
            str(seconds),
            "-i",
            str(source),
            "-vf",
            f"{crop},fps=24",
            "-c:v",
            "libx264",
            "-profile:v",
            "main",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "28",
            "-movflags",
            "+faststart",
            "-an",
            str(OUT / f"{name}.mp4"),
        ]
    )
    run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "1.5",
            "-i",
            str(OUT / f"{name}.mp4"),
            "-frames:v",
            "1",
            "-q:v",
            "4",
            str(OUT / f"{name}.jpg"),
        ]
    )
    print(f"{name}: {(OUT / f'{name}.mp4').stat().st_size // 1024} KB")


def main() -> None:
    """Record every clip, or only the ones named with --only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8013", help="Base URL of the running app")
    parser.add_argument("--only", nargs="*", choices=sorted(CLIPS), help="Clips to record")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="previews-"))
    try:
        for name in args.only or CLIPS:
            query, sources = CLIPS[name]
            seconds = HOLD * len(sources)
            record(args.base.rstrip("/"), name, query, camera_file(sources, work, name), seconds, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
