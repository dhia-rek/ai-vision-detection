"""Capture and audit live-camera UI scenarios using Chromium's fake camera."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from playwright.sync_api import Browser, Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "backend" / "samples"
FAKE_CAMERA_DIR = ROOT / ".agent-scratch" / "fake-cam"
SCENES = (
    ("objects", "objects", None, "objects/street.jpg"),
    ("faces", "faces", None, "faces/happy.jpg"),
    ("sign-spell", "sign", "spell", "sign/letter_B.jpg"),
    ("sign-gestures", "sign", "gestures", "sign/gesture_i_love_you.jpg"),
)


@dataclass
class ShotResult:
    """Store audit findings for one captured UI scenario."""

    name: str
    problems: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for screenshot capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8013", help="Base URL of the static UI")
    parser.add_argument("--out", default=".agent-scratch/ui-v2-shots", help="Screenshot output directory")
    parser.add_argument("--only", help="Capture shot names containing this substring")
    return parser.parse_args()


def selected(name: str, only: str | None) -> bool:
    """Return whether a shot name passes the optional substring filter."""
    return only is None or only in name


def make_url(base: str, params: dict[str, str]) -> str:
    """Build a page URL with encoded query parameters."""
    return f"{base.rstrip('/')}/?{urlencode(params)}"


def make_camera_file(source: Path, width: int, height: int, name: str) -> Path:
    """Generate a short looping Y4M camera stream at the requested dimensions."""
    FAKE_CAMERA_DIR.mkdir(parents=True, exist_ok=True)
    destination = FAKE_CAMERA_DIR / f"{name}-{width}x{height}.y4m"
    if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
        return destination
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},fps=10,format=yuv420p"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-loop",
        "1",
        "-i",
        str(source),
        "-vf",
        video_filter,
        "-t",
        "3",
        "-f",
        "yuv4mpegpipe",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg could not create {destination.name}: {completed.stderr.strip()}")
    return destination


def attach_audit(page: Page, result: ShotResult) -> None:
    """Collect console, page, and HTTP failures for one page load."""
    page.on(
        "console",
        lambda message: (
            result.problems.append(f"console {message.type}: {message.text}")
            if message.type in ("error", "warning")
            else None
        ),
    )
    page.on("pageerror", lambda error: result.problems.append(f"pageerror: {error}"))
    page.on(
        "requestfailed",
        lambda request: result.problems.append(
            f"request failed: {request.method} {request.url}: {request.failure}"
        ),
    )
    page.on(
        "response",
        lambda response: (
            result.problems.append(f"HTTP {response.status}: {response.url}")
            if response.status >= 400
            else None
        ),
    )


def accessibility_problems(page: Page) -> list[str]:
    """Run compact axe-style structural checks in the rendered document."""
    return page.evaluate("""() => {
      const problems = [];
      const ids = [...document.querySelectorAll('[id]')].map(el => el.id);
      if (!document.documentElement.lang?.trim()) problems.push('html is missing lang');
      if (document.querySelectorAll('h1').length !== 1) problems.push('expected exactly one h1');
      if (new Set(ids).size !== ids.length) problems.push('duplicate element ids');
      for (const img of document.images) {
        if (!img.hasAttribute('alt')) problems.push(`image missing alt: ${img.src}`);
      }
      for (const button of document.querySelectorAll('button')) {
        if (!button.getAttribute('aria-label')?.trim() && !button.innerText.trim())
          problems.push(`button has no accessible name: ${button.outerHTML.slice(0, 120)}`);
      }
      for (const control of document.querySelectorAll('input, select, textarea')) {
        if (control.type === 'hidden' || control.getAttribute('aria-hidden') === 'true') continue;
        const id = control.id;
        const labelled = control.getAttribute('aria-label')?.trim() ||
          control.getAttribute('aria-labelledby')?.trim() ||
          (id && document.querySelector(`label[for="${CSS.escape(id)}"]`));
        if (!labelled) problems.push(`form control has no label: ${control.outerHTML.slice(0, 120)}`);
      }
      return problems;
    }""")


def check_layout(page: Page, result: ShotResult, *, embed: bool = False) -> None:
    """Check for horizontal page overflow and embed viewport vertical scrollbars."""
    dimensions = page.evaluate("""() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      scrollHeight: document.documentElement.scrollHeight,
      clientHeight: document.documentElement.clientHeight
    })""")
    if dimensions["scrollWidth"] > dimensions["clientWidth"]:
        result.problems.append(
            "horizontal overflow: document width "
            f"{dimensions['scrollWidth']} > viewport {dimensions['clientWidth']}"
        )
    if embed and dimensions["scrollHeight"] > dimensions["clientHeight"]:
        result.problems.append(
            "embed vertical overflow: document height "
            f"{dimensions['scrollHeight']} > viewport {dimensions['clientHeight']}"
        )
    result.problems.extend(accessibility_problems(page))


def run_shot(
    playwright: Any,
    base: str,
    out: Path,
    name: str,
    *,
    mode: str = "objects",
    sign_mode: str | None = None,
    theme: str = "light",
    width: int = 1440,
    height: int = 900,
    camera: Path | None = None,
    camera_error: str | None = None,
    embed: bool = False,
    results_wait: bool = False,
    try_success: bool = False,
    filters: bool = False,
    only: str | None = None,
    start: bool = True,
) -> ShotResult | None:
    """Launch an isolated Chromium camera and capture one audited scenario."""
    if not selected(name, only):
        return None
    result = ShotResult(name)
    launch_args = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
    if camera:
        launch_args.append(f"--use-file-for-fake-video-capture={camera}")
    browser: Browser = playwright.chromium.launch(headless=True, args=launch_args)
    context = browser.new_context(viewport={"width": width, "height": height}, permissions=["camera"])
    if camera_error:
        context.add_init_script(f"""(() => {{
          const media = navigator.mediaDevices || {{}};
          Object.defineProperty(navigator, 'mediaDevices', {{ configurable: true, value: {{
            ...media,
            getUserMedia: () => Promise.reject(Object.assign(
              new Error('Camera unavailable'), {{ name: '{camera_error}' }}))
          }} }});
        }})();""")
    page = context.new_page()
    api_requests: list[str] = []
    page.on("request", lambda request: api_requests.append(request.url) if "/api/" in request.url else None)
    attach_audit(page, result)
    params = {"mode": mode, "theme": theme}
    if sign_mode:
        params["sign_mode"] = sign_mode
    if embed:
        params["embed"] = "1"
    try:
        page.goto(make_url(base, params), wait_until="domcontentloaded", timeout=60_000)
        if start:
            page.locator("#startBtn").click(timeout=20_000)
        if camera_error:
            expected = "blocked" if camera_error == "NotAllowedError" else "nocamera"
            page.wait_for_function(
                "expected => document.body.dataset.camera === expected",
                arg=expected,
                timeout=30_000,
            )
        elif camera:
            page.wait_for_function("() => document.body.dataset.camera === 'live'", timeout=60_000)
            if results_wait:
                result_checks = {
                    # The FPS/model counters update even when the current frame has no detections.
                    # This proves inference completed without requiring a particular prediction.
                    "objects": "document.querySelector('#modelName')?.textContent !== '–' && document.querySelector('#latency')?.textContent !== '–'",
                    "faces": "document.querySelector('#modelName')?.textContent !== '–' && document.querySelector('#latency')?.textContent !== '–'",
                    "sign": "document.querySelector('#modelName')?.textContent !== '–' && document.querySelector('#latency')?.textContent !== '–'",
                }
                page.wait_for_function(
                    f"() => {result_checks[mode]}",
                    timeout=90_000,
                )
                if api_requests:
                    result.problems.append(f"browser attempted server API request(s): {api_requests}")
            if filters:
                page.locator("#minConf").evaluate(
                    "element => { element.value = '65'; "
                    "element.dispatchEvent(new Event('input', { bubbles: true })); }"
                )
                chips = page.locator("#classChips .chip:not(.chip-all)")
                chips.first.wait_for(timeout=30_000)
                chips.first.click()
            if try_success:
                page.locator("#tryCard[data-state='success']").wait_for(timeout=90_000)
            if mode == "sign":
                chart_image_size = page.locator("#chartGrid img").first.evaluate(
                    "image => ({ width: image.getBoundingClientRect().width, "
                    "height: image.getBoundingClientRect().height })"
                )
                if not chart_image_size["width"] or not chart_image_size["height"]:
                    result.problems.append("ASL chart illustrations have zero rendered size")
        if embed:
            page.wait_for_timeout(500)
        else:
            page.wait_for_timeout(350)
        if api_requests:
            result.problems.append(f"camera-only UI issued server API request(s): {api_requests}")
        check_layout(page, result, embed=embed)
        # Full-page captures stitch the page, which draws fixed elements (skip link, toasts) in odd places.
        hidden = ".skip-link{display:none!important}" + (
            ".toasts{display:none!important}" if width < 700 else ""
        )
        page.add_style_tag(content=hidden)
        page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    except Exception as error:
        result.problems.append(f"scenario failed: {error}")
    finally:
        context.close()
        browser.close()
    return result


def plan_shots(playwright: Any, args: argparse.Namespace, out: Path) -> list[ShotResult]:
    """Build and run the requested live, empty, error, and embed screenshot matrix."""
    results: list[ShotResult] = []
    dimensions = (("desktop", 1440, 900, 640, 480), ("phone", 390, 844, 480, 640))
    for scene, mode, sign_mode, sample in SCENES:
        source = SAMPLES / sample
        for device, width, height, cam_w, cam_h in dimensions:
            names = (f"{scene}-{device}-light", f"{scene}-{device}-dark")
            if not any(selected(name, args.only) for name in names):
                continue
            camera = make_camera_file(source, cam_w, cam_h, f"{scene}-{device}")
            for theme in ("light", "dark"):
                name = f"{scene}-{device}-{theme}"
                shot = run_shot(
                    playwright,
                    args.base,
                    out,
                    name,
                    mode=mode,
                    sign_mode=sign_mode,
                    theme=theme,
                    width=width,
                    height=height,
                    camera=camera,
                    results_wait=True,
                    try_success=(scene == "sign-spell"),
                    only=args.only,
                )
                if shot:
                    results.append(shot)
    for state, error in (("empty", None), ("blocked", "NotAllowedError"), ("no-camera", "NotFoundError")):
        modes = (("objects", None),) if state != "empty" else (("objects", None),)
        for mode, sign_mode in modes:
            for device, width, height, _, _ in dimensions:
                if state == "no-camera" and device != "desktop":
                    continue
                themes = ("light", "dark") if state != "no-camera" else ("light",)
                for theme in themes:
                    name = f"{state}-{mode}-{device}-{theme}"
                    shot = run_shot(
                        playwright,
                        args.base,
                        out,
                        name,
                        mode=mode,
                        sign_mode=sign_mode,
                        theme=theme,
                        width=width,
                        height=height,
                        camera_error=error,
                        only=args.only,
                        start=state != "empty",
                    )
                    if shot:
                        results.append(shot)
    filtered = "objects-filtered-desktop-light"
    if selected(filtered, args.only):
        shot = run_shot(
            playwright,
            args.base,
            out,
            filtered,
            camera=make_camera_file(SAMPLES / "objects/street.jpg", 640, 480, "objects-filtered"),
            results_wait=True,
            filters=True,
            only=args.only,
        )
        if shot:
            results.append(shot)
    embed_cases = (("desktop", 900, 560), ("phone", 390, 700))
    if any(selected(f"embed-objects-{device}-light", args.only) for device, _, _ in embed_cases):
        embed_camera = make_camera_file(SAMPLES / "objects/street.jpg", 640, 480, "embed-objects")
    else:
        embed_camera = None
    for device, width, height in embed_cases:
        name = f"embed-objects-{device}-light"
        if embed_camera is None:
            continue
        shot = run_shot(
            playwright,
            args.base,
            out,
            name,
            theme="light",
            width=width,
            height=height,
            camera=embed_camera,
            embed=True,
            results_wait=True,
            filters=True,
            only=args.only,
        )
        if shot:
            results.append(shot)
    return results


def print_summary(results: list[ShotResult]) -> bool:
    """Print a compact outcome table and return whether every check passed."""
    name_width = max((len(result.name) for result in results), default=4)
    print(f"{'SHOT':<{name_width}}  RESULT  PROBLEMS")
    print(f"{'-' * name_width}  ------  --------")
    for result in results:
        status = "FAIL" if result.problems else "OK"
        print(f"{result.name:<{name_width}}  {status:<6}  {'; '.join(result.problems) or '-'}")
    print(f"Summary: {len(results)} shots, {sum(bool(item.problems) for item in results)} failed")
    return all(not result.problems for result in results)


def main() -> int:
    """Generate fake-camera media, capture scenarios, and report audit failures."""
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        results = plan_shots(playwright, args, out)
    return 0 if print_summary(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
