"""FastAPI app: image upload + live webcam frame detection with YOLOv8.

Environment variables:
  AVD_DETECTOR       "yolo" (default) or "stub" (no torch, for tests/smoke)
  AVD_WEIGHTS        path or Ultralytics model name (default: repo-root yolov8n.pt)
  AVD_CONF           confidence threshold (default 0.25)
  AVD_MAX_UPLOAD_MB  upload size limit in MB (default 10)
  AVD_MAX_PIXELS     decoded image size limit in megapixels (default 40)
  AVD_CORS_ORIGINS   comma-separated allowed origins (default: none, same-origin only)
"""

from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image

from .detector import (
    Detector,
    StubDetector,
    YoloDetector,
    annotate,
    decode_image,
    encode_jpeg,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
STATIC_DIR = BACKEND_DIR / "static"
SAMPLES_DIR = BACKEND_DIR / "samples"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def default_weights() -> str:
    env = os.getenv("AVD_WEIGHTS")
    if env:
        return env
    local = ROOT_DIR / "yolov8n.pt"
    # Fall back to the model name; Ultralytics downloads it on first run.
    return str(local) if local.exists() else "yolov8n.pt"


def build_detector() -> Detector:
    kind = os.getenv("AVD_DETECTOR", "yolo").lower()
    if kind == "stub":
        return StubDetector()
    if kind != "yolo":
        raise ValueError(f"Unknown AVD_DETECTOR={kind!r} (use 'yolo' or 'stub')")
    return YoloDetector(default_weights(), conf=float(os.getenv("AVD_CONF", "0.25")))


def list_samples() -> list[str]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p.name for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def create_app(detector: Detector | None = None) -> FastAPI:
    app = FastAPI(title="AI Vision Detection API", version="1.0.0")
    max_bytes = int(float(os.getenv("AVD_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
    max_pixels = int(float(os.getenv("AVD_MAX_PIXELS", "40")) * 1_000_000)

    origins = [o.strip() for o in os.getenv("AVD_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"]
        )

    @lru_cache(maxsize=1)
    def get_detector() -> Detector:
        # Lazy: the app starts instantly and the model loads on the first request.
        return detector if detector is not None else build_detector()

    async def read_image(upload: UploadFile):
        if not (upload.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        raw = await upload.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail="Image too large")
        return decode_checked(raw)

    def decode_checked(raw: bytes):
        # Read only the header first so a small file can't decode into a huge bitmap.
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                width, height = probe.size
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode image") from None
        if width * height > max_pixels:
            raise HTTPException(status_code=413, detail="Image dimensions too large")
        img = decode_image(raw)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        return img

    def run(img) -> dict:
        det = get_detector()
        detections = det.detect(img)
        jpeg = encode_jpeg(annotate(img, detections), quality=85)
        return {
            # Inline data URL: uploads and results are never written to disk.
            "result_image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
            "detections": [d.to_dict() for d in detections],
            "total_detections": len(detections),
            "model_used": det.name,
        }

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/detect/")
    async def detect(file: UploadFile = File(...)):
        """Upload an image; get detections and an annotated JPEG (data URL)."""
        return run(await read_image(file))

    @app.post("/detect/frame")
    async def detect_frame(frame: UploadFile = File(...), quality: int = Query(80, ge=10, le=95)):
        """One webcam frame in, annotated JPEG out (count in X-Total-Detections)."""
        img = await read_image(frame)
        detections = get_detector().detect(img)
        return Response(
            content=encode_jpeg(annotate(img, detections), quality=quality),
            media_type="image/jpeg",
            headers={"X-Total-Detections": str(len(detections))},
        )

    @app.get("/demo/samples")
    def demo_samples():
        """Bundled, openly licensed sample images (see backend/samples/CREDITS.md)."""
        return {"samples": list_samples()}

    @app.get("/demo/samples/{name}")
    def demo_sample_image(name: str):
        if name not in list_samples():
            raise HTTPException(status_code=404, detail="Unknown sample")
        return FileResponse(SAMPLES_DIR / name)

    @app.post("/demo/detect/{name}")
    def demo_detect(name: str):
        """Run detection on a bundled sample: no camera or upload needed."""
        if name not in list_samples():
            raise HTTPException(status_code=404, detail="Unknown sample")
        return run(decode_checked((SAMPLES_DIR / name).read_bytes()))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
