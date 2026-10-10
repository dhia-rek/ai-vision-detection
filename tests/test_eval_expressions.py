import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest


def test_incomplete_expression_archive_is_rejected(tmp_path):
    script = Path(__file__).parents[1] / "scripts/eval_expressions.py"
    spec = importlib.util.spec_from_file_location("eval_expressions_test_module", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    archive = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Raw/Angry/Genuine/example.jpg", b"not decoded by this test")
    with pytest.raises(ValueError, match="all seven"):
        module._extract_images(archive, tmp_path / "images", 10)
