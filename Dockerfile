# Demo image (also the Hugging Face Spaces Docker image): FastAPI + static UI,
# CPU only. Pretrained models are downloaded at build time (not in the repo).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

# MediaPipe's native library needs EGL/GLES at load time (CPU inference only).
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run containers as uid 1000; keep everything under /app.
RUN useradd -m -u 1000 app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    AVD_MODEL_DIR=/app/models \
    PORT=7860

COPY --chown=app pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project && chown -R app /app

COPY --chown=app LICENSE ./
COPY --chown=app backend/ ./backend/
USER app

WORKDIR /app/backend
# Fetch and checksum-verify all pretrained models (~19 MB) into the image.
RUN python -m app.zoo
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
