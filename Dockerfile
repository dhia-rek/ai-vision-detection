# Demo image (also the Hugging Face Spaces Docker image): FastAPI + static UI,
# YOLOv8n on CPU-only torch, bundled CC0 sample images.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

# Spaces run containers as uid 1000; keep everything under /app.
RUN useradd -m -u 1000 app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=7860

COPY --chown=app pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project && chown -R app /app

COPY --chown=app yolov8n.pt LICENSE ./
COPY --chown=app backend/ ./backend/
USER app

WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
