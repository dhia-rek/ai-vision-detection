import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

STATIC_DIR = Path(__file__).parents[1] / "backend" / "static"
INDEX_PATH = STATIC_DIR / "index.html"
JS_PATH = STATIC_DIR / "app.js"
CSS_PATH = STATIC_DIR / "app.css"
EMOJI_RANGES = ((0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x1F000, 0x1F2FF), (0xFE0F, 0xFE0F))


class StaticHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.stack = []
        self.visible_text = []
        self.buttons = []
        self.hidden_text_depth = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        ancestors = list(self.stack)
        self.elements.append((tag, attributes, ancestors))
        if tag in {"script", "style"}:
            self.hidden_text_depth += 1
        text_parts = []
        if tag == "button":
            self.buttons.append((attributes, text_parts))
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            self.stack.append((tag, attributes, text_parts))

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden_text_depth:
            self.hidden_text_depth -= 1
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if not self.hidden_text_depth:
            self.visible_text.append(data)
            for entry in self.stack:
                if entry[0] == "button":
                    entry[2].append(data)


def parse_html(source):
    parser = StaticHtmlParser()
    parser.feed(source)
    parser.close()
    return parser


def test_static_files_are_served_and_traversal_does_not_expose_python(client):
    responses = {
        "/": ("text/html", "<!doctype html"),
        "/static/app.js": ("javascript", "use strict"),
        "/static/app.css": ("text/css", ""),
    }
    for path, (content_type, marker) in responses.items():
        response = client.get(path)
        assert response.status_code == 200
        assert content_type in response.headers["content-type"]
        if marker:
            assert marker.lower() in response.text.lower()

    for path in ("/static/../app/main.py", "/static/%2e%2e/app/main.py"):
        response = client.get(path)
        is_python_source = b"from fastapi" in response.content or b"def create_app" in response.content
        assert response.status_code != 200 or not is_python_source


def test_javascript_dom_lookups_exist_once_in_html():
    html = INDEX_PATH.read_text()
    js = JS_PATH.read_text()
    requested_ids = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", js))
    requested_ids.update(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", js))
    html_ids = re.findall(r'\bid\s*=\s*["\']([^"\']+)["\']', html)
    counts = Counter(html_ids)
    assert requested_ids <= counts.keys()
    assert all(counts[element_id] == 1 for element_id in requested_ids)


def test_svg_symbols_and_references_match_in_both_directions():
    html = INDEX_PATH.read_text()
    js = JS_PATH.read_text()
    symbols = set(re.findall(r'<symbol\b[^>]*\bid=["\']i-([A-Za-z0-9_-]+)["\']', html))
    references = set(re.findall(r'href=["\']#i-([A-Za-z0-9_-]+)["\']', html))
    references.update(re.findall(r"svgUse\(['\"]([A-Za-z0-9_-]+)['\"]\)", js))
    references.update(re.findall(r"['\"]#i-([A-Za-z0-9_-]+)['\"]", js))
    references.update(re.findall(r"icon:\s*['\"]([A-Za-z0-9_-]+)['\"]", js))
    assert references <= symbols
    assert symbols <= references


def test_html_accessibility_basics():
    parser = parse_html(INDEX_PATH.read_text())
    elements = parser.elements
    html_tags = [attrs for tag, attrs, _ in elements if tag == "html"]
    assert len(html_tags) == 1 and html_tags[0].get("lang", "").strip()

    for tag, attrs, ancestors in elements:
        if tag == "button":
            has_visible_text = any(
                button_attrs is attrs and any(part.strip() for part in parts)
                for button_attrs, parts in parser.buttons
            )
            assert attrs.get("aria-label", "").strip() or has_visible_text
        if tag == "img":
            assert "alt" in attrs
        if tag == "input":
            assert attrs.get("aria-label", "").strip() or attrs.get("aria-hidden") == "true"
        if tag == "svg" and any(parent_tag == "button" for parent_tag, _, _ in ancestors):
            assert attrs.get("aria-hidden") == "true"
        if attrs.get("role") == "radio":
            groups = [
                parent_attrs
                for parent_tag, parent_attrs, _ in ancestors
                if parent_attrs.get("role") == "radiogroup"
            ]
            assert groups and any(group.get("aria-label", "").strip() for group in groups)

    ids = [attrs["id"] for _, attrs, _ in elements if "id" in attrs]
    assert len(ids) == len(set(ids))
    assert sum(tag == "h1" for tag, _, _ in elements) == 1


def test_static_content_rules():
    contents = {path.name: path.read_text() for path in (INDEX_PATH, JS_PATH, CSS_PATH)}
    for name, content in contents.items():
        bad_chars = [
            char for char in content if any(start <= ord(char) <= end for start, end in EMOJI_RANGES)
        ]
        assert not bad_chars, f"{name} contains pictographic characters: {bad_chars!r}"

    visible_text = " ".join(parse_html(contents["index.html"]).visible_text)
    assert re.search(r"\bdemo\b", visible_text, flags=re.IGNORECASE) is None
    css = contents["app.css"]
    assert re.search(r"@import\b", css, flags=re.IGNORECASE) is None
    assert re.search(r"https?://", css, flags=re.IGNORECASE) is None

    for name in ("app.css", "app.js"):
        content = contents[name]
        assert re.search(r"src\s*=\s*[\"']?(?:https?:)?//", content, flags=re.IGNORECASE) is None
        assert re.search(r"url\(\s*[\"']?(?:https?:)?//", content, flags=re.IGNORECASE) is None

    allowed_urls = {"https://github.com/dhia-rek/ai-vision-detection", "https://lucide.dev"}
    all_urls = re.findall(r"https?://[^\s\"'<>),]+", "\n".join(contents.values()))
    external_urls = [url for url in all_urls if "w3.org/2000/svg" not in url]
    assert set(external_urls) <= allowed_urls


def test_analyze_without_annotation_returns_ui_contract(client, jpeg):
    response = client.post(
        "/api/analyze?mode=faces&annotate=false", files={"file": ("frame.jpg", jpeg, "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert {"detections", "image_size", "model_used", "device", "inference_ms"} <= body.keys()
    assert body["image_size"] == {"width": 120, "height": 80}
    assert isinstance(body["image_size"]["width"], int)
    assert isinstance(body["image_size"]["height"], int)
    assert "result_image" not in body


def test_analyze_with_default_annotation_keeps_result_image(client, jpeg):
    response = client.post("/api/analyze?mode=faces", files={"file": ("frame.jpg", jpeg, "image/jpeg")})
    assert response.status_code == 200
    assert "result_image" in response.json()


def test_javascript_uses_client_side_overlay_and_large_palette():
    js = JS_PATH.read_text()
    assert "annotate=false" in js
    assert "result_image" not in js
    assert "X-Detections" not in js
    palette = re.search(r"\bPALETTE\s*=\s*\[(.*?)\]", js, flags=re.DOTALL)
    assert palette
    assert len(re.findall(r"#[0-9a-fA-F]{6}", palette.group(1))) >= 8
