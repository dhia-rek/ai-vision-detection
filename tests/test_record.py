import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from asl_extract import from_recorded


def test_recorded_json_converts_to_training_npz(tmp_path):
    source = tmp_path / "take-1.json"
    landmarks = [[i / 100, i / 200, -i / 1000] for i in range(21)]
    source.write_text(
        json.dumps(
            {
                "format": "avd-landmarks",
                "version": 1,
                "samples": [
                    {"letter": "B", "landmarks": landmarks, "handedness": "Right", "image_size": [640, 480]},
                    {"letter": "?", "landmarks": landmarks, "handedness": "Left", "image_size": [640, 480]},
                ],
            }
        )
    )
    output = tmp_path / "out.npz"
    from_recorded(str(source), str(output))
    with np.load(output) as data:
        assert data["P"].shape == (1, 21, 3)
        np.testing.assert_allclose(data["P"][0], landmarks)
        np.testing.assert_allclose(data["size"][0], [640, 480])
        assert data["y"].tolist() == ["B"]
        assert data["grp"].tolist() == ["take-1_B"]
        assert data["hand"].tolist() == ["Right"]
