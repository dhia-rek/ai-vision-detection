"""Export the trained ASL MLP as compact, browser-readable JSON.

Format v1 is documented in ``scripts/README.md``. Only numeric weights and feature
normalisation constants are exported; there are no training samples or images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
DEFAULT_MODEL = ROOT / "backend/app/models/asl_letters.npz"
DEFAULT_OUTPUT = ROOT / "backend/app/models/asl_letters_web.json"


def _numbers(array: np.ndarray) -> list[float]:
    """Encode float32 values to eight significant decimal digits for a small JSON."""
    return [float(f"{float(value):.8g}") for value in np.asarray(array).flat]


def export(model_path: Path, output_path: Path) -> dict:
    """Export a feature-versioned MLP and return the JSON document."""
    from app.sign_features import FEATURE_VERSION, N_FEATURES

    with np.load(model_path, allow_pickle=False) as arrays:
        version = int(arrays["feature_version"])
        if version != FEATURE_VERSION:
            raise ValueError(f"Model feature version {version} does not match exporter {FEATURE_VERSION}")
        mean, std = arrays["mean"], arrays["std"]
        if mean.shape != (N_FEATURES,) or std.shape != (N_FEATURES,):
            raise ValueError("Invalid feature-normalisation dimensions")
        classes = [str(c) for c in arrays["classes"]]
        layers = []
        width = N_FEATURES
        i = 1
        while f"W{i}" in arrays:
            weights, bias = arrays[f"W{i}"], arrays[f"b{i}"]
            if weights.ndim != 2 or weights.shape[0] != width or bias.shape != (weights.shape[1],):
                raise ValueError(f"Invalid dimensions for layer {i}")
            layers.append(
                {
                    "input_size": width,
                    "output_size": int(weights.shape[1]),
                    "weights": _numbers(weights),
                    "bias": _numbers(bias),
                    "activation": "linear" if f"W{i + 1}" not in arrays else "relu",
                }
            )
            width = int(weights.shape[1])
            i += 1
        if not layers or width != len(classes):
            raise ValueError("Output layer does not match classes")
        document = {
            "format": "dhia-asl-mlp",
            "format_version": 1,
            "feature_version": version,
            "input_size": N_FEATURES,
            "source_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "classes": classes,
            "mean": _numbers(mean),
            "std": _numbers(std),
            "layers": layers,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    return document


def main() -> None:
    """Read CLI paths and write the browser artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.model, args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
