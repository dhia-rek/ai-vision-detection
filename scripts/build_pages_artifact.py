"""Assemble the static site exactly as it is published, and optionally serve it for local checks.

Layout (mountable at any sub-path, e.g. ``/vision/``, because every URL is relative)::

    <out>/index.html
    <out>/static/...     app.css, app.js, js/, models/, vendor/, fonts/, asl/

Usage::

    uv run python scripts/build_pages_artifact.py --out dist/vision
    uv run python scripts/build_pages_artifact.py --out dist/vision --serve 8013
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend" / "static"
REQUIRED = (
    "static/app.js",
    "static/models/asl_letters_web.json",
    "static/models/objects/yolo11n.onnx",
    "static/models/face_detection_yunet_2023mar_dynamic.onnx",
    "static/vendor/ort/ort.webgpu.min.mjs",
    "static/vendor/mediapipe/vision_bundle.mjs",
    "static/vendor/mediapipe/gesture_recognizer.task",
)


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static handler that serves ``.mjs``/``.wasm`` with browser-correct types and caching hints."""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".task": "application/octet-stream",
    }

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - signature of the base class
        """Keep test output quiet."""


def build(out: Path) -> Path:
    """Copy ``backend/static`` into ``out`` using the published layout and return ``out``."""
    if out.exists():
        shutil.rmtree(out)
    (out / "static").mkdir(parents=True)
    shutil.copyfile(SOURCE / "index.html", out / "index.html")
    for entry in SOURCE.iterdir():
        if entry.name in {"index.html", "README.md"} or entry.name.startswith("."):
            continue
        target = out / "static" / entry.name
        shutil.copytree(entry, target) if entry.is_dir() else shutil.copyfile(entry, target)
    # The only root-relative link is the app's "home" link; keep it inside the mount point.
    page = (out / "index.html").read_text(encoding="utf-8").replace('href="/"', 'href="./"')
    (out / "index.html").write_text(page, encoding="utf-8")
    missing = [name for name in REQUIRED if not (out / name).is_file()]
    if missing:
        raise SystemExit(
            "Missing files: " + ", ".join(missing) + "\nRun: uv run python scripts/export_web_models.py"
        )
    return out


def serve(directory: Path, port: int) -> http.server.ThreadingHTTPServer:
    """Create (but do not start) a threaded HTTP server for ``directory`` on localhost."""
    handler = functools.partial(Handler, directory=str(directory))
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


def main() -> None:
    """Build the artifact; with ``--serve`` also serve it until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "vision")
    parser.add_argument("--serve", type=int, metavar="PORT", help="serve the result on 127.0.0.1:PORT")
    args = parser.parse_args()
    out = build(args.out.resolve())
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Built {out} ({size / 1e6:.1f} MB)")
    if args.serve:
        server = serve(out, args.serve)
        print(f"Serving http://127.0.0.1:{args.serve}/ (Ctrl-C to stop)")
        server.serve_forever()


if __name__ == "__main__":
    main()
