"""FastAPI app: detection, facial expressions and ASL sign recognition.

Serves images, uploaded videos and live webcam frames.

Modes (query parameter `mode`):
  objects  YOLOv8n, 80 COCO classes
  faces    YuNet face detector + MobileFaceNet expression classifier (7 expressions)
  sign     MediaPipe hand landmarks; `sign_mode=spell` (ASL letters) or `gestures`

Environment variables:
  AVD_DETECTOR       "real" (default) or "stub" (no models, for tests/smoke)
  AVD_WEIGHTS        YOLO weights path (default: downloaded yolov8n.pt, see zoo.py)
  AVD_MODEL_DIR      where downloaded models are cached
  AVD_CONF           YOLO confidence threshold (default 0.25)
  AVD_MAX_UPLOAD_MB  image upload limit in MB (default 10)
  AVD_MAX_VIDEO_MB   video upload limit in MB (default 50)
  AVD_MAX_PIXELS     decoded image size limit in megapixels (default 40)
  AVD_CORS_ORIGINS   comma-separated allowed origins (default: none, same-origin only)
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .detector import Detector, StubDetector, YoloDetector, annotate, decode_image, encode_jpeg
from .sign import SIGN_MODES

BACKEND_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BACKEND_DIR / "static"
SAMPLES_DIR = BACKEND_DIR / "samples"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov"}

MODES = {
    "objects": "Object detection (YOLOv8n, 80 COCO classes)",
    "faces": "Facial expression (angry, disgust, fear, happy, neutral, sad, surprised)",
    "sign": "ASL fingerspelling letters (sign_mode=spell) or common hand gestures (sign_mode=gestures)",
}
SAMPLE_GROUPS = ("objects", "faces", "sign", "video")
MAX_HEADER_DETECTIONS = 25  # keep X-Detections well under common 8 KB header limits
MAX_VIDEO_FRAMES = 600  # hard cap on analysed frames per /api/video request
UPLOAD_CHUNK_BYTES = 1 << 20  # read uploads 1 MiB at a time so size limits fail fast
RESULT_JPEG_QUALITY = 85  # quality of the annotated image returned by /api/analyze
FALLBACK_VIDEO_FPS = 25.0  # used when the container reports no frame rate


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
        from .zoo import ensure_model

        weights = os.getenv("AVD_WEIGHTS") or ensure_model("yolov8n.pt")
        return YoloDetector(weights, conf=float(os.getenv("AVD_CONF", "0.25")))
    if key == "faces":
        from .faces import ExpressionDetector

        return ExpressionDetector()
    if key.startswith("sign-"):
        from .sign import SignDetector

        return SignDetector(mode=key.removeprefix("sign-"))
    raise ValueError(f"Unknown mode key {key!r}")


def list_samples(group: str) -> list[str]:
    """List bundled sample file names for a group, sorted.

    Returns videos for "video" and images otherwise; an unknown group or missing
    folder gives an empty list.
    """
    folder = SAMPLES_DIR / group
    suffixes = VIDEO_SUFFIXES if group == "video" else IMAGE_SUFFIXES
    if group not in SAMPLE_GROUPS or not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in suffixes)


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
    max_video_bytes = int(float(os.getenv("AVD_MAX_VIDEO_MB", "50")) * 1024 * 1024)
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
                ``result_image`` out (clients that draw the boxes themselves).

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

    def analyze_frame(img: np.ndarray, key: str, quality: int) -> Response:
        """Run a mode on a webcam frame and return the annotated JPEG.

        Results travel in ``X-*`` headers. At most ``MAX_HEADER_DETECTIONS`` of the most
        confident detections are listed (``X-Detections-Truncated`` flags when cut) to
        stay under header size limits.

        Args:
            img: BGR frame.
            key: Detector key from ``mode_key``.
            quality: JPEG quality of the response image.
        """
        det = get_detector(key)
        t0 = time.perf_counter()
        detections = det.detect(img)
        ms = (time.perf_counter() - t0) * 1000
        ranked = sorted(detections, key=lambda d: d.confidence, reverse=True)[:MAX_HEADER_DETECTIONS]
        brief = [{"class": d.label, "confidence": round(d.confidence, 3)} for d in ranked]
        return Response(
            content=encode_jpeg(annotate(img, detections), quality=quality),
            media_type="image/jpeg",
            headers={
                "X-Total-Detections": str(len(detections)),
                "X-Detections": json.dumps(brief, ensure_ascii=True),
                "X-Inference-Ms": f"{ms:.1f}",
                "X-Detections-Truncated": str(len(detections) > MAX_HEADER_DETECTIONS).lower(),
            },
        )

    def analyze_video(path: str, key: str, fps: float, max_seconds: float) -> dict:
        """Run a mode on sampled frames of a video file.

        Every ``round(source_fps / fps)``-th frame is analysed (at least every frame),
        stopping after ``max_seconds`` of video or ``MAX_VIDEO_FRAMES`` analysed frames.
        In "sign-spell" mode the most confident hand per frame feeds a ``WordBuilder``
        and the typed text is included.

        Args:
            path: Video file on disk (OpenCV cannot decode from memory).
            key: Detector key from ``mode_key``.
            fps: Target analysed frames per second.
            max_seconds: Maximum video time to analyse.

        Returns:
            Summary dict with per-frame boxes, label counts and timing.

        Raises:
            HTTPException: 400 if the video cannot be decoded (or yields no frames),
                413 if a frame exceeds ``max_pixels``.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not decode video")
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_VIDEO_FPS
            step = max(1, round(src_fps / fps))
            det = get_detector(key)
            builder = None
            if key == "sign-spell":
                from .sign import WordBuilder

                builder = WordBuilder()
            frames: list[dict] = []
            counts: Counter[str] = Counter()
            total_ms, idx = 0.0, 0
            while True:
                ok = cap.grab()
                if not ok or idx / src_fps > max_seconds or len(frames) >= MAX_VIDEO_FRAMES:
                    break
                if idx % step == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    if frame.shape[0] * frame.shape[1] > max_pixels:
                        raise HTTPException(status_code=413, detail="Video frames too large")
                    t = idx / src_fps
                    t0 = time.perf_counter()
                    dets = det.detect(frame)
                    total_ms += (time.perf_counter() - t0) * 1000
                    counts.update(d.label for d in dets)
                    if builder is not None:
                        best = max(dets, key=lambda d: d.confidence, default=None)
                        builder.feed(best.label if best else None, best.confidence if best else 0.0, t)
                    # Per-frame output stays small: no hand keypoints, no top-3.
                    brief = [
                        {"class": d.label, "confidence": round(d.confidence, 4), "bbox": d.to_dict()["bbox"]}
                        for d in dets
                    ]
                    frames.append({"t": round(t, 2), "detections": brief})
                idx += 1
        finally:
            cap.release()
        if not frames:
            raise HTTPException(status_code=400, detail="Could not decode video")
        result = {
            "mode": key,
            "model_used": det.name,
            "processed_frames": len(frames),
            "sample_fps": round(src_fps / step, 2),
            "avg_inference_ms": round(total_ms / len(frames), 1),
            "label_counts": dict(counts.most_common()),
            "frames": frames,
        }
        if builder is not None:
            result["text"] = builder.text.strip()
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
        and what the web UI uses: it draws the boxes itself from ``detections``.
        """
        key = mode_key(mode, sign_mode)
        img = await read_image(file)
        return await run_in_threadpool(analyze, img, key, annotate)

    @app.post("/api/frame")
    async def api_frame(
        frame: UploadFile = File(...),
        mode: str = Query("objects"),
        sign_mode: str = Query("spell"),
        quality: int = Query(80, ge=10, le=95),
    ):
        """One webcam/video frame in, annotated JPEG out (results in X-* headers)."""
        key = mode_key(mode, sign_mode)
        img = await read_image(frame)
        return await run_in_threadpool(analyze_frame, img, key, quality)

    @app.post("/api/video")
    async def api_video(
        file: UploadFile = File(...),
        mode: str = Query("objects"),
        sign_mode: str = Query("spell"),
        fps: float = Query(5.0, gt=0, le=10),
        max_seconds: float = Query(60.0, gt=0, le=120),
    ):
        """Video in; per-frame detections at `fps` samples per second out (JSON).

        OpenCV needs a file to decode video, so the upload is written to a private temp
        file that is deleted as soon as the request finishes.
        """
        key = mode_key(mode, sign_mode)
        if not (file.content_type or "").startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")
        with tempfile.NamedTemporaryFile(suffix=".video") as tmp:
            # Stream to disk in chunks so large uploads never sit in RAM.
            size = 0
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > max_video_bytes:
                    raise HTTPException(status_code=413, detail="Video too large")
                tmp.write(chunk)
            tmp.flush()
            return await run_in_threadpool(analyze_video, tmp.name, key, fps, max_seconds)

    # Backwards-compatible object-detection endpoints.
    @app.post("/detect/")
    async def detect(file: UploadFile = File(...)) -> dict:
        """Legacy alias of ``/api/analyze`` fixed to object detection."""
        return await api_analyze(file, "objects", "spell", True)

    @app.post("/detect/frame")
    async def detect_frame(frame: UploadFile = File(...), quality: int = Query(80, ge=10, le=95)) -> Response:
        """Legacy alias of ``/api/frame`` fixed to object detection."""
        return await api_frame(frame, "objects", "spell", quality)

    @app.get("/demo/samples")
    def demo_samples():
        """Bundled, openly licensed samples per mode (see backend/samples/CREDITS.md)."""
        return {group: list_samples(group) for group in SAMPLE_GROUPS}

    @app.get("/demo/samples/{group}/{name}")
    def demo_sample_file(group: str, name: str) -> FileResponse:
        """Serve a bundled sample file.

        Only names returned by ``list_samples`` are accepted, which also prevents path
        traversal.

        Raises:
            HTTPException: 404 for unknown samples.
        """
        if name not in list_samples(group):
            raise HTTPException(status_code=404, detail="Unknown sample")
        return FileResponse(SAMPLES_DIR / group / name)

    @app.post("/demo/analyze/{group}/{name}")
    async def demo_analyze(
        group: str, name: str, mode: str | None = None, sign_mode: str = "spell", annotate: bool = True
    ):
        """Run a mode on a bundled sample image: no camera or upload needed.

        ``annotate=false`` omits the annotated image, as in ``/api/analyze``.
        """
        if group == "video" or name not in list_samples(group):
            raise HTTPException(status_code=404, detail="Unknown sample")
        key = mode_key(mode or group, sign_mode)
        img = decode_checked((SAMPLES_DIR / group / name).read_bytes())
        return await run_in_threadpool(analyze, img, key, annotate)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
