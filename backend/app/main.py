"""FastAPI app: the Python reference API for detection, facial expressions and ASL signs.

The live site runs every model in the browser; this app serves the same static page and an
image API used for evaluation, tests and local tools.

Modes (query parameter `mode`):
  objects  YOLO11n, 80 COCO classes
  faces    YuNet face detector + MobileFaceNet expression classifier (7 expressions)
  sign     MediaPipe hand landmarks; `sign_mode=spell` (ASL letters) or `gestures`

Environment variables:
  AVD_DETECTOR       "real" (default) or "stub" (no models, for tests/smoke)
  AVD_WEIGHTS        YOLO weights path (default: downloaded yolo11n.pt, see zoo.py)
  AVD_MODEL_DIR      where downloaded models are cached
  AVD_CONF           YOLO confidence threshold (default 0.25)
  AVD_MAX_UPLOAD_MB  image upload limit in MB (default 10)
  AVD_MAX_PIXELS     decoded image size limit in megapixels (default 40)
  AVD_CORS_ORIGINS   comma-separated allowed origins (default: none, same-origin only)
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .detector import Detector, StubDetector, YoloDetector, annotate, decode_image, encode_jpeg
from .sign import SIGN_MODES

BACKEND_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BACKEND_DIR / "static"

MODES = {
    "objects": "Object detection (YOLO11n, 80 COCO classes)",
    "faces": "Facial expression (angry, disgust, fear, happy, neutral, sad, surprised)",
    "sign": "ASL fingerspelling letters (sign_mode=spell) or common hand gestures (sign_mode=gestures)",
}
RESULT_JPEG_QUALITY = 85  # quality of the annotated image returned by /api/analyze


def mode_key(mode: str, sign_mode: str = "spell") -> str:
    """Validate request modes and combine them into a detector key.

    Args:
        mode: One of ``MODES``.
        sign_mode: "spell" or "gestures"; only used when ``mode`` is "sign".

    Returns:
        "objects", "faces", "sign-spell" or "sign-gestures".

    Raises:
        HTTPException: 400 if either value is invalid.
    """
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode; use one of {sorted(MODES)}")
    if mode == "sign":
        if sign_mode not in SIGN_MODES:
            raise HTTPException(status_code=400, detail="sign_mode must be 'spell' or 'gestures'")
        return f"sign-{sign_mode}"
    return mode


def build_detector(key: str = "objects") -> Detector:
    """Create the real detector for a mode key (objects, faces, sign-spell, sign-gestures).

    Heavy modules are imported lazily so unused modes cost nothing. If ``AVD_DETECTOR``
    is "stub", a ``StubDetector`` is returned for every key.

    Raises:
        ValueError: If ``key`` is not a known mode key.
    """
    if os.getenv("AVD_DETECTOR", "real").lower() == "stub":
        return StubDetector()
    if key == "objects":
        from .zoo import DEFAULT_OBJECT_MODEL, ensure_model

        weights = os.getenv("AVD_WEIGHTS") or ensure_model(DEFAULT_OBJECT_MODEL)
        return YoloDetector(weights, conf=float(os.getenv("AVD_CONF", "0.25")))
    if key == "faces":
        from .faces import ExpressionDetector

        return ExpressionDetector()
    if key.startswith("sign-"):
        from .sign import SignDetector

        return SignDetector(mode=key.removeprefix("sign-"))
    raise ValueError(f"Unknown mode key {key!r}")


def create_app(detector: Detector | None = None) -> FastAPI:
    """Build the FastAPI application and register all routes.

    Upload and pixel limits and CORS origins are read from environment variables at
    call time (see the module docstring).

    Args:
        detector: If given, used for every mode (tests pass a StubDetector); otherwise
            detectors are built lazily per mode.
    """
    app = FastAPI(title="AI Vision Detection API", version="2.0.0")
    max_bytes = int(float(os.getenv("AVD_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
    max_pixels = int(float(os.getenv("AVD_MAX_PIXELS", "40")) * 1_000_000)

    origins = [o.strip() for o in os.getenv("AVD_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"]
        )

    detectors: dict[str, Detector] = {}
    build_lock = threading.Lock()

    def get_detector(key: str) -> Detector:
        """Return the detector for a mode key, building and caching it on first use.

        Lazy so each model downloads/loads on the first request for its mode; a lock
        prevents two concurrent first requests from loading the same model twice.
        """
        if detector is not None:
            return detector
        with build_lock:
            if key not in detectors:
                detectors[key] = build_detector(key)
            return detectors[key]

    def decode_checked(raw: bytes) -> np.ndarray:
        """Decode image bytes into a BGR array, enforcing the pixel limit.

        Reads only the header first so a small file can't decode into a huge bitmap.

        Raises:
            HTTPException: 400 if undecodable, 413 if over ``max_pixels``.
        """
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                width, height = probe.size
        except Exception:  # noqa: BLE001
            # Deliberately broad: PIL raises many parser-specific errors (OSError, SyntaxError,
            # DecompressionBomb*, ...) on hostile input and all must map to a clean 400.
            raise HTTPException(status_code=400, detail="Could not decode image") from None
        if width * height > max_pixels:
            raise HTTPException(status_code=413, detail="Image dimensions too large")
        img = decode_image(raw)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        return img

    async def read_image(upload: UploadFile) -> np.ndarray:
        """Validate and decode an uploaded image.

        Reads at most ``max_bytes + 1`` bytes so oversized uploads are rejected without
        buffering them all.

        Raises:
            HTTPException: 400 if not an image content type or undecodable, 413 if too large.
        """
        if not (upload.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        raw = await upload.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail="Image too large")
        return decode_checked(raw)

    def analyze(img: np.ndarray, key: str, annotate_image: bool = True) -> dict:
        """Run a mode on an image and build the ``/api/analyze`` response body.

        Args:
            img: BGR image.
            key: Detector key from ``mode_key``.
            annotate_image: If False, skip drawing and encoding the annotated JPEG and leave
                ``result_image`` out.

        Returns:
            Dict with the annotated image as a base64 JPEG data URL (unless skipped), the
            detections, their count, the analysed image size in pixels (the coordinate space
            of the boxes), the model name, the device and inference time in milliseconds.
        """
        det = get_detector(key)
        t0 = time.perf_counter()
        detections = det.detect(img)
        ms = (time.perf_counter() - t0) * 1000
        result = {
            "mode": key,
            "detections": [d.to_dict() for d in detections],
            "total_detections": len(detections),
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "model_used": det.name,
            "device": getattr(det, "device", "cpu"),
            "inference_ms": round(ms, 1),
        }
        if annotate_image:
            jpeg = encode_jpeg(annotate(img, detections), quality=RESULT_JPEG_QUALITY)
            # Inline data URL: uploads and results are never written to disk.
            result["result_image"] = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        return result

    # CSS and JS of the web UI (index.html references /static/app.css and /static/app.js).
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        """Serve the single-page web UI."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/api/modes")
    def modes() -> dict:
        """List available modes and the registered models with their licences."""
        from .zoo import MODELS

        return {
            "modes": MODES,
            "models": {
                n: {"size_mb": m.size_mb, "license": m.license, "source": m.source} for n, m in MODELS.items()
            },
        }

    @app.post("/api/analyze")
    async def api_analyze(
        file: UploadFile = File(...),
        mode: str = Query("objects"),
        sign_mode: str = Query("spell"),
        annotate: bool = Query(True),
    ):
        """Image in; JSON detections plus an annotated JPEG (data URL) out.

        Pass ``annotate=false`` to get the JSON only (no ``result_image``), which is faster
        when the caller draws the boxes itself from ``detections``.
        """
        key = mode_key(mode, sign_mode)
        img = await read_image(file)
        return await run_in_threadpool(analyze, img, key, annotate)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
