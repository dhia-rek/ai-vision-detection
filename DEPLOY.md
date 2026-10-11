# Deploying

The app is static files: every model runs in the visitor's browser, so there is no server to run.

## GitHub Pages (the live site)

[`.github/workflows/pages.yml`](.github/workflows/pages.yml) builds the site with
`scripts/build_pages_artifact.py` and publishes it on every push to `main`. Pull requests only build and check it.

Because `dhia-rek.github.io` uses the custom domain `dhiarekik.me`, this repository's Pages site is served at
**https://dhiarekik.me/ai-vision-detection/**.

One-time setup: **Settings → Pages → Build and deployment → Source: GitHub Actions**. The first deploy runs on the next
push to `main`, or from **Actions → Static site (GitHub Pages) → Run workflow**.

What the build does:

1. Downloads the pinned ONNX Runtime Web and MediaPipe runtimes and checks their hashes
   (`export_web_models.py --only vendor`; they are not committed).
2. Copies `backend/static/` into `index.html` + `static/` with relative URLs only, so the site works under any path.
3. Checks the key files exist and that no URL is absolute.

The camera only works over HTTPS (or `localhost`); GitHub Pages provides HTTPS.

> **Licence note.** YOLO11 and its weights are AGPL-3.0. The site serves the model to visitors, so this repository
> stays public and keeps the `LICENSE` and [`THIRD_PARTY.md`](THIRD_PARTY.md) notices.

## Embedding on dhiarekik.me

The portfolio can link to the page, or embed it from the same origin. The iframe needs `allow="camera"`:

```html
<iframe src="/ai-vision-detection/?embed=1&mode=objects&theme=light"
        title="AI Vision Detection" allow="camera" loading="lazy"
        style="width:100%;height:560px;border:0;border-radius:16px"></iframe>
```

URL options: `mode=objects|faces|sign`, `sign_mode=spell|gestures`, `theme=light|dark`, `embed=1` (hides the intro, story
and page chrome).

## Local preview

```bash
uv run python scripts/export_web_models.py --only vendor
uv run python scripts/build_pages_artifact.py --out dist/site --serve 8013
# open http://127.0.0.1:8013/
```

## Docker (Python API, optional)

The `Dockerfile` packages the FastAPI app: the same static page plus the Python image API (`POST /api/analyze`,
`GET /api/modes`, `GET /health`) used for evaluation and local tools. It is not needed for the live site.

```bash
docker build -t ai-vision-detection .
docker run --rm -p 7860:7860 ai-vision-detection
# http://localhost:7860/  ·  health check: /health
```
