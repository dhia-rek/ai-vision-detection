"""Export trained ASL letter weights to the browser's versioned JSON format.

This intentionally handles only the ASL model. Object and face exports have
separate owners/commands and can be added to the static build independently.

Usage:
    python scripts/export_web_models.py
    python scripts/export_web_models.py --asl-in model.npz --asl-out web/asl.json

The schema stores dense matrices row-major (`weights[input * output_size + output]`)
and describes every layer's dimensions and activation. New classifier depths and
hidden widths therefore require only a new JSON file, not a browser code change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURE_VERSION = 1


def export_asl(source: Path, destination: Path) -> dict:
    """Convert an inference-compatible training NPZ into compact web JSON."""
    with np.load(source, allow_pickle=False) as model:
        version = int(model["feature_version"])
        if version != FEATURE_VERSION:
            raise ValueError(f"Expected feature_version {FEATURE_VERSION}, got {version}")
        classes = [str(label) for label in model["classes"]]
        mean = np.asarray(model["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(model["std"], dtype=np.float32).reshape(-1)
        if len(mean) != 70 or len(std) != 70 or np.any(std == 0):
            raise ValueError("Expected nonzero scaler vectors of length 70")
        layers = []
        i = 1
        while f"W{i}" in model:
            weights = np.asarray(model[f"W{i}"], dtype=np.float32)
            bias = np.asarray(model[f"b{i}"], dtype=np.float32).reshape(-1)
            if weights.ndim != 2 or weights.shape[1] != len(bias):
                raise ValueError(f"Invalid W{i}/b{i} dimensions")
            layers.append({
                "input_size": int(weights.shape[0]),
                "output_size": int(weights.shape[1]),
                "weights": weights.ravel(order="C").tolist(),
                "bias": bias.tolist(),
                "activation": "relu" if f"W{i + 1}" in model else "linear",
            })
            i += 1
    if not layers:
        raise ValueError("No W1/b1 classifier layer found")
    if layers[0]["input_size"] != 70 or layers[-1]["output_size"] != len(classes):
        raise ValueError("Layer dimensions do not match 70 features and class labels")
    for left, right in zip(layers, layers[1:]):
        if left["output_size"] != right["input_size"]:
            raise ValueError("Adjacent classifier layers have incompatible dimensions")
    payload = {
        "format": "dhia-asl-mlp",
        "format_version": 1,
        "feature_version": FEATURE_VERSION,
        "input_size": 70,
        "classes": classes,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "layers": layers,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asl-in", type=Path, default=ROOT / "backend/app/models/asl_letters.npz")
    parser.add_argument("--asl-out", type=Path, default=ROOT / "backend/static/models/asl_letters.json")
    args = parser.parse_args()
    export_asl(args.asl_in, args.asl_out)
    print(f"Exported ASL JSON: {args.asl_out} ({args.asl_out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
