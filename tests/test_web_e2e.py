"""End-to-end: the built static site in headless Chromium with a fake camera, desktop and phone.

Every mode must start from a cold cache, show real detections, load only its own models, make no
call to a server API and produce no console errors. Throughput is measured by ``scripts/web_bench.py``;
here the assertions are about behaviour. Headless Chromium has no GPU, so this is the CPU/WASM path.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import pytest
from web_support import chromium, collect_errors, make_fake_camera

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "backend" / "samples"
pytestmark = [pytest.mark.browser, pytest.mark.slow]

DEVICES = {
    "desktop": {"size": (640, 480), "options": {"viewport": {"width": 1280, "height": 800}}},
    "phone": {"size": (480, 640), "device": "Pixel 7"},
}
YOLO = {"yolo11n.onnx"}
FACE = {
    "face_detection_yunet_2023mar_dynamic.onnx",
    "facial_expression_recognition_mobilefacenet_2022july.onnx",
}
HANDS = {"gesture_recognizer.task", "asl_letters_web.json"}
RUNTIME_WASM = {"ort-wasm-simd-threaded.wasm", "vision_wasm_internal.wasm"}
# (id, query, sample, model-name regex, files that must be cached, DOM check evaluated in the page)
MODES = {
    "objects": (
        "?mode=objects",
        "objects/street.jpg",
        r"yolo11n-(320|640)",
        YOLO,
        "document.querySelectorAll('#objBars li').length >= 1",
    ),
    "faces": (
        "?mode=faces",
        "faces/happy.jpg",
        r"yunet\+mobilefacenet-fer",
        FACE,
        "/happy/i.test(document.querySelector('#faceList').textContent)",
    ),
    "spell": (
        "?mode=sign&sign_mode=spell",
        "sign/letter_B.jpg",
        r"mediapipe-hands\+asl-letters",
        HANDS,
        "document.querySelector('#signMain').textContent.trim() === 'B'",
    ),
    "gestures": (
        "?mode=sign&sign_mode=gestures",
        "sign/gesture_i_love_you.jpg",
        r"mediapipe-gestures",
        HANDS,
        "/I love you/i.test(document.querySelector('#signMain').textContent)",
    ),
}
RUNNING = "() => /^\\d/.test(document.querySelector('#fps').textContent.trim())"


def _context_options(playwright, device: str) -> dict:
    spec = DEVICES[device]
    options = {**playwright.devices[spec["device"]]} if "device" in spec else dict(spec["options"])
    return {**options, "permissions": ["camera"]}


@contextmanager
def session(playwright, tmp_path: Path, device: str, sample: str):
    """Yield ``(page, errors, requests)`` for a fresh context with the sample as the fake webcam."""
    width, height = DEVICES[device]["size"]
    camera = make_fake_camera(SAMPLES / sample, tmp_path, width, height)
    with chromium(playwright, camera) as browser:
        context = browser.new_context(**_context_options(playwright, device))
        context.set_default_timeout(180_000)
        errors: list[str] = []
        requests: list[str] = []
        context.on("request", lambda request: requests.append(request.url))
        page = context.new_page()
        collect_errors(page, errors)
        yield page, errors, requests
        context.close()


def start(page, base: str, query: str) -> None:
    """Open the app, start the camera and wait for the first inference result."""
    page.goto(f"{base}/{query}", wait_until="domcontentloaded")
    page.click("#startBtn")
    page.wait_for_function(RUNNING, timeout=180_000)


def cached_files(page) -> set[str]:
    """File names stored by the model download cache (what this page actually downloaded)."""
    urls = page.evaluate(
        "async () => (await (await caches.open('avd-models-v1')).keys()).map((request) => request.url)"
    )
    return {urlparse(url).path.rsplit("/", 1)[-1] for url in urls}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", MODES)
def test_mode_runs_on_device_without_errors(playwright_session, static_site, tmp_path, mode, device) -> None:
    """Cold start -> model overlay hides -> real detection -> own models only -> no server calls."""
    query, sample, model_pattern, expected_files, detection = MODES[mode]
    with session(playwright_session, tmp_path, device, sample) as (page, errors, requests):
        start(page, static_site, query)
        page.wait_for_function(f"() => {detection}", timeout=120_000)
        assert page.locator("#modelLoad").is_hidden(), "the loading overlay must disappear once running"
        model_name = page.inner_text("#modelName")
        assert re.fullmatch(model_pattern, model_name), model_name
        if mode == "objects" and device == "phone":
            assert model_name == "yolo11n-320", "phones use the 320 px input"
        assert float(page.inner_text("#fps")) > 0

        downloaded = cached_files(page)
        everything = YOLO | FACE | HANDS
        assert downloaded & everything == expected_files, (
            f"{mode} must load only its own models: {downloaded}"
        )
        origin = urlparse(static_site).netloc
        assert all(
            urlparse(url).netloc in ("", origin) or urlparse(url).scheme in ("blob", "data")
            for url in requests
        ), "every request must be same-origin"
        assert not [url for url in requests if urlparse(url).path.startswith(("/api", "/demo"))], (
            "the static app must not call a server API"
        )
        assert any(urlparse(url).path.endswith(".js") for url in requests)
        assert not errors, errors


def test_objects_input_size_can_be_forced(playwright_session, static_site, tmp_path) -> None:
    """``?objects=640`` selects the large input on a desktop viewport."""
    with session(playwright_session, tmp_path, "desktop", "objects/street.jpg") as (page, errors, _):
        start(page, static_site, "?mode=objects&objects=640")
        assert page.inner_text("#modelName") == "yolo11n-640"
        assert not errors, errors


def test_second_visit_uses_the_download_cache(playwright_session, static_site, tmp_path) -> None:
    """Models and runtimes download once; a reload in the same profile does not fetch them again."""
    with session(playwright_session, tmp_path, "desktop", "faces/happy.jpg") as (page, errors, requests):
        start(page, static_site, "?mode=faces")
        heavy = (".onnx", ".wasm")
        first_visit = [url for url in requests if urlparse(url).path.endswith(heavy)]
        assert len(first_visit) >= 3, (
            f"the first visit should download the face models and ORT: {first_visit}"
        )
        requests.clear()
        page.reload(wait_until="domcontentloaded")
        page.click("#startBtn")
        page.wait_for_function(RUNNING, timeout=180_000)
        page.wait_for_function("() => /happy/i.test(document.querySelector('#faceList').textContent)")
        assert [url for url in requests if urlparse(url).path.endswith(heavy)] == []
        assert not errors, errors


def test_switching_modes_while_live(playwright_session, static_site, tmp_path) -> None:
    """Objects -> expressions -> sign with the camera on: each switch loads its model and keeps working."""
    with session(playwright_session, tmp_path, "desktop", "faces/happy.jpg") as (page, errors, _):
        start(page, static_site, "?mode=objects")
        for mode, name in (("faces", "yunet+mobilefacenet-fer"), ("sign", "mediapipe-hands+asl-letters")):
            page.click(f'#modeSwitch [data-mode="{mode}"]')
            page.wait_for_function(
                "(name) => document.querySelector('#modelName').textContent.trim() === name",
                arg=name,
                timeout=180_000,
            )
            assert page.locator("#modelLoad").is_hidden()
        page.click('#modeSwitch [data-mode="faces"]')
        page.wait_for_function(
            "() => /happy/i.test(document.querySelector('#faceList').textContent)", timeout=180_000
        )
        assert not errors, errors


def test_embed_mode_loads_cleanly(playwright_session, static_site, tmp_path) -> None:
    """The iframe flavour (?embed=1) boots without errors and hides the outer chrome flag."""
    with session(playwright_session, tmp_path, "phone", "objects/street.jpg") as (page, errors, _):
        page.goto(f"{static_site}/?embed=1&mode=objects&theme=dark", wait_until="networkidle")
        assert page.evaluate("document.documentElement.dataset.embed") == "true"
        assert page.evaluate("document.documentElement.dataset.theme") == "dark"
        assert page.locator("#startBtn").is_visible()
        assert not errors, errors
