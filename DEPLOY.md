# Deploying to Hugging Face Spaces

The repo ships a CPU-only `Dockerfile` that works as a [Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker).
No secrets are needed: the app runs YOLO11n on CPU and serves the bundled CC0 sample images.

> **Licence note.** Ultralytics and the YOLO11 weights are AGPL-3.0. A public Space serves
> the model over a network, so the AGPL applies: keep the Space's source public (a Docker Space
> is) and keep the `LICENSE` notice. See the README licence section.

## 1. Create the Space

On huggingface.co: **New Space** → SDK **Docker** → template **Blank** → hardware **CPU basic** (free).

## 2. Space card

The Space's `README.md` needs this YAML header. Prepend it to a copy of this repo's README in the Space
(not here, it would show on GitHub):

```yaml
---
title: AI Vision Detection
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
license: agpl-3.0
short_description: YOLO11 object detection on images and webcam (FastAPI)
---
```

`license: agpl-3.0` because the deployed artefact bundles the AGPL YOLO11 weights.

## 3. Push

```bash
git remote add space https://huggingface.co/spaces/<user>/ai-vision-detection
git push space feat/better-models:main
```

The Docker build installs from `pyproject.toml` / `uv.lock`, copies the backend and downloads
verified model weights during the build. `.dockerignore` excludes `*.pt` files; no YOLO weights
need to be committed or pushed with Git LFS. Push the branch containing the model changes.

## 4. Check

- `https://<user>-ai-vision-detection.hf.space/health` → `{"status":"ok"}`
- Open the Space, click **Upload Image / Demo**, then a sample button (`street`, `dog`, …).
- The webcam mode needs HTTPS (Spaces provide it) and the visitor's permission.

## Local Docker run

```bash
docker build -t ai-vision-detection .
docker run --rm -p 7860:7860 ai-vision-detection
# http://localhost:7860/?mode=demo&sample=street.jpg
```
