"""Measure the static app in headless Chromium: model download size, time to first result and FPS.

Each case opens the built site with a fake 30 fps camera, starts the camera and samples the app's own
FPS counter for a few seconds. Cases: every mode on a desktop viewport and on an emulated phone
(Pixel 7 user agent, viewport and touch). Headless Chromium here has no GPU, so everything runs the
CPU/WASM path; WebGPU devices are faster for objects. Pin the CPU count to emulate a smaller device::

    uv run python scripts/web_bench.py --out .agent-scratch/bench.json
    taskset -c 0,1 uv run python scripts/web_bench.py --label 2-core --out .agent-scratch/bench-2core.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from build_pages_artifact import build, serve  # noqa: E402

SAMPLES = ROOT / "backend" / "samples"
CASES = (
    ("objects", "objects", "", "objects/street.jpg"),
    ("objects-640", "objects", "&objects=640", "objects/street.jpg"),
    ("faces", "faces", "", "faces/happy.jpg"),
    ("sign-letters", "sign", "&sign_mode=spell", "sign/letter_B.jpg"),
    ("sign-gestures", "sign", "&sign_mode=gestures", "sign/gesture_i_love_you.jpg"),
)
DEVICES: dict[str, dict[str, Any]] = {
    "desktop": {"size": (640, 480), "viewport": {"width": 1280, "height": 800}},
    "phone": {"size": (480, 640), "device": "Pixel 7"},
}


def fake_camera(sample: Path, directory: Path, size: tuple[int, int]) -> Path:
    """Loop a sample image as a 30 fps Y4M stream."""
    target = directory / f"{sample.stem}-{size[0]}x{size[1]}.y4m"
    if not target.exists():
        scale = f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,crop={size[0]}:{size[1]}"
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
                f"{scale},fps=30,format=yuv420p",
                "-t",
                "3",
                "-f",
                "yuv4mpegpipe",
                str(target),
            ],
            check=True,
        )
    return target


def run_case(playwright, base: str, site_dir: Path, case, device: str, camera: Path, seconds: float) -> dict:
    """Run one mode on one device class; returns sizes, timings and fps."""
    name, mode, extra, _ = case
    spec = DEVICES[device]
    browser = playwright.chromium.launch(
        args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={camera}",
        ]
    )
    options = {**playwright.devices[spec["device"]]} if "device" in spec else {"viewport": spec["viewport"]}
    context = browser.new_context(permissions=["camera"], **options)
    page = context.new_page()
    transferred: dict[str, Any] = {"bytes": 0, "gzip": 0, "files": {}}

    def on_response(response) -> None:
        try:
            body = response.body()
        except Exception:  # noqa: BLE001 - redirects/aborted requests carry no body
            return
        path = response.url.split("/static/", 1)[-1]
        transferred["bytes"] += len(body)
        transferred["gzip"] += len(gzip.compress(body, 6)) if len(body) < 30_000_000 else len(body) // 2
        transferred["files"][path] = len(body)

    page.on("response", on_response)
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on(
        "console",
        lambda m: errors.append(m.text[:200]) if m.type == "error" and "INFO:" not in m.text else None,
    )

    def first_result_time() -> float:
        start = time.time()
        page.click("#startBtn")
        while time.time() - start < 240:
            if page.inner_text("#fps").strip() not in ("–", "", "-"):
                return time.time() - start
            time.sleep(0.1)
        raise TimeoutError(f"{name}/{device}: no result within 240 s")

    page.goto(f"{base}/?mode={mode}{extra}", wait_until="domcontentloaded")
    cold = first_result_time()
    time.sleep(2.0)  # let in-flight response handlers finish before the snapshot
    shell = {
        path: size
        for path, size in transferred["files"].items()
        if not path.startswith(("http", "blob")) and not path.endswith((".onnx", ".wasm", ".task"))
    }
    cached = page.evaluate(
        """async () => {
          const c = await caches.open('avd-models-v1'), out = {};
          for (const r of await c.keys()) out[r.url] = (await (await c.match(r)).arrayBuffer()).byteLength;
          return out;
        }"""
    )
    model_files = {url.split("/static/", 1)[1]: size for url, size in cached.items()}
    model_gzip = sum(len(gzip.compress((site_dir / "static" / name).read_bytes(), 6)) for name in model_files)
    samples, latencies = [], []
    end = time.time() + seconds
    while time.time() < end:
        text = page.inner_text("#fps").strip()
        if text.replace(".", "", 1).isdigit():
            samples.append(float(text))
        latencies.append(float(page.inner_text("#latency").split()[0]))
        time.sleep(0.5)
    model, backend = page.inner_text("#modelName"), page.inner_text("#deviceText")
    # Second visit in the same context: models come from the Cache API.
    transferred.update(bytes=0, gzip=0, files={})
    page.reload(wait_until="domcontentloaded")
    warm = first_result_time()
    time.sleep(1.5)
    result = {
        "case": name,
        "device": device,
        "model": model,
        "backend": backend,
        "fps": round(statistics.mean(samples), 1) if samples else 0.0,
        "fps_min": min(samples, default=0.0),
        "latency_ms": round(statistics.median(latencies)),
        "first_result_s_cold": round(cold, 1),
        "first_result_s_warm": round(warm, 1),
        "model_download_bytes": sum(model_files.values()),
        "model_download_gzip_bytes": model_gzip,
        "app_shell_bytes": sum(shell.values()),
        "model_files": model_files,
        "warm_network_bytes_for_models": sum(
            size for path, size in transferred["files"].items() if path.endswith((".onnx", ".wasm", ".task"))
        ),
        "errors": errors,
    }
    browser.close()
    return result


def main() -> None:
    """Build the site, run every case and print a Markdown table plus JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0, help="sampling window per case")
    parser.add_argument("--out", type=Path, help="write JSON results here")
    parser.add_argument("--label", default="all-cores")
    parser.add_argument("--only", help="substring filter on case or device")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="avd-bench-"))
    site = build(work / "vision")
    server = serve(site, 0)
    import threading

    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    results = []
    with sync_playwright() as playwright:
        for case in CASES:
            for device, spec in DEVICES.items():
                if args.only and args.only not in f"{case[0]}/{device}":
                    continue
                camera = fake_camera(SAMPLES / case[3], work, spec["size"])
                result = run_case(playwright, base, site, case, device, camera, args.seconds)
                result["label"] = args.label
                results.append(result)
                print(
                    f"{result['case']}/{result['device']}: {result['fps']} fps, {result['latency_ms']} ms, "
                    f"model download {result['model_download_bytes'] / 1e6:.1f} MB",
                    flush=True,
                )
    server.shutdown()
    print("\n| case | device | model | backend | fps | latency | download | cold / warm first result |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for r in results:
        print(
            f"| {r['case']} | {r['device']} | {r['model']} | {r['backend']} | {r['fps']} | "
            f"{r['latency_ms']} ms | "
            f"{r['model_download_bytes'] / 1e6:.1f} / {r['model_download_gzip_bytes'] / 1e6:.1f} MB | "
            f"{r['first_result_s_cold']} s / {r['first_result_s_warm']} s |"
        )
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
