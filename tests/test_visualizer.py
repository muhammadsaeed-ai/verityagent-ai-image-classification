"""Offline tests for saved prediction charts."""

from pathlib import Path

import pytest
from PIL import Image

from classifier import Prediction
from visualizer import save_visualization


def test_visualization_saves_output_and_creates_parent_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jpg"
    Image.new("RGB", (100, 70), (200, 120, 80)).save(input_path)
    predictions = [
        Prediction(1, 5, "sample category A", 0.7132),
        Prediction(2, 2, "sample category B", 0.1934),
        Prediction(3, 9, "sample category C", 0.0710),
    ]
    output_path = tmp_path / "reports" / "run-001.png"

    result = save_visualization(input_path, predictions, output_path)

    assert result == output_path
    assert result.is_file()
    assert result.stat().st_size > 1_000
    with Image.open(result) as saved_image:
        assert saved_image.format == "PNG"
        assert saved_image.width > 0 and saved_image.height > 0


def test_visualizer_rejects_empty_predictions(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jpg"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(input_path)

    with pytest.raises(ValueError, match="At least one prediction"):
        save_visualization(input_path, [], tmp_path / "out.png")


def test_visualizer_adds_png_extension_when_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jpg"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(input_path)
    predictions = [Prediction(1, 0, "example", 1.0)]

    result = save_visualization(input_path, predictions, tmp_path / "out")

    assert result == tmp_path / "out.png"
    assert result.is_file()
