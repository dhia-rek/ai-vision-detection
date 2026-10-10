"""Offline demo: run the detector on the bundled samples, no camera or upload.

uv run python -m app.demo                       # objects (YOLO11n on CPU), prints a table
uv run python -m app.demo --mode faces          # facial expressions on the face samples
uv run python -m app.demo --mode sign           # ASL letters (sign-gestures for gestures)
uv run python -m app.demo --out ../docs/demo    # also save annotated images + demo.gif
AVD_DETECTOR=stub uv run python -m app.demo     # smoke test without models
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2

from .detector import annotate
from .main import SAMPLES_DIR, build_detector, list_samples

# CLI --mode value -> sample folder name; both sign modes use the same sign samples.
GROUPS = {"objects": "objects", "faces": "faces", "sign": "sign", "sign-gestures": "sign"}


def save_gif(frames: list, path: Path, width: int = 640) -> None:
    """Write BGR frames as a looping animated GIF (1.5 s per frame).

    Args:
        frames: Non-empty list of BGR images (numpy arrays, as from OpenCV).
        path: Destination ``.gif`` file.
        width: Every frame is resized to this width, keeping aspect ratio, to keep the
            file small.
    """
    from PIL import Image  # only needed when saving a GIF

    imgs = []
    for f in frames:
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, int(h * width / w)))
        imgs.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=1500, loop=0, optimize=True)


def main(argv: list[str] | None = None) -> int:
    """Run the detector for one mode over the bundled samples and print a table.

    Args:
        argv: Command-line arguments; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 if there are no samples for the mode.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(GROUPS), default="objects")
    ap.add_argument("--out", type=Path, help="directory for annotated images and demo.gif")
    args = ap.parse_args(argv)

    group = GROUPS[args.mode]
    samples = list_samples(group)
    if not samples:
        print(f"No samples found in {SAMPLES_DIR / group}")
        return 1

    detector = build_detector("sign-spell" if args.mode == "sign" else args.mode)
    frames = []
    for name in samples:
        img = cv2.imread(str(SAMPLES_DIR / group / name))
        if img is None:
            raise FileNotFoundError(f"Could not read sample {name}")
        t0 = time.perf_counter()
        dets = detector.detect(img)
        ms = (time.perf_counter() - t0) * 1000
        counts = Counter(d.label for d in dets)
        summary = ", ".join(f"{n}x {label}" for label, n in counts.most_common()) or "nothing"
        if args.mode != "objects":
            summary = ", ".join(f"{d.label} {d.confidence:.2f}" for d in dets) or "nothing"
        print(f"{name:16} {ms:7.0f} ms  {summary}")
        frames.append(annotate(img, dets))
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out / name), frames[-1])

    if args.out:
        save_gif(frames, args.out / ("demo.gif" if args.mode == "objects" else f"demo-{args.mode}.gif"))
        print(f"Saved annotated images and demo.gif to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
