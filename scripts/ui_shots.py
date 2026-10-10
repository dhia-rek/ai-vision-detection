"""Capture UI screenshots and report browser-side verification problems."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

SCENES = (
    ("objects", "objects", None, "street.jpg"),
    ("faces", "faces", None, "happy.jpg"),
    ("sign-spell", "sign", "spell", "letter_Y.jpg"),
    ("sign-gestures", "sign", "gestures", "gesture_i_love_you.jpg"),
)


class ShotResult:
    """Store a screenshot name and the problems found while loading its page."""

    def __init__(self, name: str) -> None:
        """Initialize an empty result for a named shot."""
        self.name = name
        self.problems: list[str] = []


def parse_args() -> argparse.Namespace:
    """Parse command-line options for screenshot capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8000", help="Base URL of the UI")
    parser.add_argument("--out", default=".agent-scratch/ui-shots/after", help="Screenshot output directory")
    parser.add_argument("--only", help="Capture shot names containing this substring")
    parser.add_argument("--all-themes", action="store_true", help="Capture extra shots in both themes")
    parser.add_argument("--headed", action="store_true", help="Show browser windows")
    return parser.parse_args()


def selected(name: str, only: str | None) -> bool:
    """Return whether a shot name passes the optional substring filter."""
    return only is None or only in name


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


def make_url(base: str, params: dict[str, str]) -> str:
    """Build a page URL with encoded query parameters."""
    return f"{base.rstrip('/')}/?{urlencode(params)}"


def check_phone_width(page: Page, result: ShotResult) -> None:
    """Record a failure if the document overflows a phone viewport horizontally."""
    widths = page.evaluate("() => ({scroll: document.documentElement.scrollWidth, inner: window.innerWidth})")
    if widths["scroll"] > widths["inner"]:
        result.problems.append(
            f"horizontal overflow: document width {widths['scroll']} > viewport {widths['inner']}"
        )


def capture(
    browser: Browser,
    base: str,
    out: Path,
    name: str,
    params: dict[str, str],
    viewport_name: str,
    theme: str,
    only: str | None,
    *,
    mobile: bool = False,
    state: str | None = None,
    word: bool = False,
) -> ShotResult | None:
    """Load one UI scenario, perform its checks, and save a full-page screenshot."""
    if not selected(name, only):
        return None
    result = ShotResult(name)
    viewport = {"width": 390, "height": 844} if mobile else {"width": 1440, "height": 900}
    context = browser.new_context(
        viewport=viewport,
        device_scale_factor=2 if mobile else 1,
        is_mobile=mobile,
        has_touch=mobile,
    )
    page = context.new_page()
    attach_audit(page, result)
    params = {**params, "theme": theme}
    try:
        page.goto(make_url(base, params), wait_until="domcontentloaded", timeout=60_000)
        if state:
            try:
                page.wait_for_selector(f"body[data-state='{state}']", timeout=60_000)
            except Exception as error:
                result.problems.append(f"timed out waiting for state {state}: {error}")
        if word:
            try:
                page.wait_for_function(
                    "() => Boolean(document.querySelector('#wordText')?.textContent?.trim())",
                    timeout=45_000,
                )
            except Exception as error:
                result.problems.append(f"WARNING: #wordText remained empty for 45 seconds: {error}")
        page.wait_for_timeout(700)
        if mobile:
            try:
                check_phone_width(page, result)
            except Exception as error:
                result.problems.append(f"phone width check failed: {error}")
        try:
            page.screenshot(path=str(out / f"{name}.png"), full_page=True)
        except Exception as error:
            result.problems.append(f"screenshot failed: {error}")
    except Exception as error:
        result.problems.append(f"page load failed: {error}")
    finally:
        context.close()
    return result


def capture_matrix(browser: Browser, args: argparse.Namespace, out: Path, results: list[ShotResult]) -> None:
    """Capture the image scene matrix across desktop/phone and dark/light themes."""
    for scene, mode, sign_mode, sample in SCENES:
        for viewport in ("desktop", "phone"):
            for theme in ("dark", "light"):
                name = f"{scene}-{viewport}-{theme}"
                params = {"mode": mode, "source": "image", "sample": sample}
                if sign_mode:
                    params["sign_mode"] = sign_mode
                result = capture(
                    browser,
                    args.base,
                    out,
                    name,
                    params,
                    viewport,
                    theme,
                    args.only,
                    mobile=viewport == "phone",
                    state="done",
                )
                if result:
                    results.append(result)


def capture_extras(browser: Browser, args: argparse.Namespace, out: Path, results: list[ShotResult]) -> None:
    """Capture video and idle-webcam scenarios in the requested theme set."""
    themes = ("dark", "light") if args.all_themes else ("dark",)
    extras = (
        (
            "sign-spell-video",
            {
                "mode": "sign",
                "sign_mode": "spell",
                "source": "video",
                "sample": "fingerspelling_cab.mp4",
            },
            "running",
            True,
        ),
        ("faces-video", {"mode": "faces", "source": "video", "sample": "faces.mp4"}, "running", False),
        ("webcam-idle", {"mode": "objects", "source": "webcam"}, None, False),
    )
    for scene, params, state, word in extras:
        for viewport in ("desktop", "phone"):
            for theme in themes:
                name = f"{scene}-{viewport}-{theme}"
                result = capture(
                    browser,
                    args.base,
                    out,
                    name,
                    params,
                    viewport,
                    theme,
                    args.only,
                    mobile=viewport == "phone",
                    state=state,
                    word=word,
                )
                if result:
                    results.append(result)


def capture_webcam_smoke(
    playwright: Any, args: argparse.Namespace, out: Path, results: list[ShotResult]
) -> None:
    """Launch fake camera media and capture the desktop live-webcam state."""
    name = "webcam-live-desktop-dark"
    if not selected(name, args.only):
        return
    result = ShotResult(name)
    browser = playwright.chromium.launch(
        headless=not args.headed,
        args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
    )
    context: BrowserContext = browser.new_context(
        viewport={"width": 1440, "height": 900}, permissions=["camera"]
    )
    page = context.new_page()
    attach_audit(page, result)
    try:
        page.goto(
            make_url(args.base, {"mode": "objects", "source": "webcam", "theme": "dark"}),
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        page.locator("#startBtn").click(timeout=15_000)
        page.wait_for_selector("body[data-state='running']", timeout=60_000)
        page.wait_for_function(
            "() => /\\d/.test(document.querySelector('#fps')?.textContent ?? '')",
            timeout=60_000,
        )
        page.wait_for_timeout(700)
        page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    except Exception as error:
        result.problems.append(f"webcam smoke check failed: {error}")
    finally:
        context.close()
        browser.close()
    results.append(result)


def print_summary(results: list[ShotResult]) -> bool:
    """Print a table of shot outcomes and return whether all checks passed."""
    name_width = max((len(result.name) for result in results), default=4)
    print(f"{'SHOT':<{name_width}}  RESULT  PROBLEMS")
    print(f"{'-' * name_width}  ------  --------")
    for result in results:
        status = "FAIL" if result.problems else "OK"
        print(f"{result.name:<{name_width}}  {status:<6}  {'; '.join(result.problems) or '-'}")
    return all(not result.problems for result in results)


def main() -> int:
    """Run selected screenshot scenarios and return a process exit status."""
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: list[ShotResult] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        try:
            capture_matrix(browser, args, out, results)
            capture_extras(browser, args, out, results)
        finally:
            browser.close()
        capture_webcam_smoke(playwright, args, out, results)
    return 0 if print_summary(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
