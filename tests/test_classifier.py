"""Offline unit tests for the classifier and JSON export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights

from classifier import (
    ImageClassifier,
    ImageLoadError,
    InferenceError,
    InvalidTopKError,
    export_predictions_json,
)


class FixedLogitsModel(nn.Module):
    """Small deterministic model used to test inference without downloading weights."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor([logits], dtype=torch.float32))
        self.seen_device: torch.device | None = None
        self.inference_mode_seen = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.seen_device = batch.device
        self.inference_mode_seen = torch.is_inference_mode_enabled()
        return self.fixed_logits.to(batch.device)


def make_image(path: Path) -> Path:
    Image.new("RGB", (20, 12), color=(60, 130, 200)).save(path)
    return path


def make_classifier(logits: list[float] | None = None) -> tuple[ImageClassifier, FixedLogitsModel]:
    model = FixedLogitsModel(logits or [0.2, 2.4, 1.3, -0.1, 0.8, -1.5])
    classifier = ImageClassifier(
        device="cpu",
        model=model,
        preprocess=lambda _image: torch.zeros((3, 8, 8), dtype=torch.float32),
        categories=("class zero", "class one", "class two", "class three", "class four", "class five"),
    )
    return classifier, model


def test_selected_weights_supply_matching_categories_and_preprocessing() -> None:
    weights = ResNet50_Weights.IMAGENET1K_V1
    classifier = ImageClassifier(device="cpu", weights=weights, model=nn.Identity())

    assert classifier.weights is weights
    assert classifier.categories == tuple(weights.meta["categories"])
    assert len(classifier.categories) == 1000
    expected_transform = weights.transforms()
    assert type(classifier.preprocess) is type(expected_transform)
    assert classifier.preprocess.resize_size == expected_transform.resize_size
    assert classifier.preprocess.crop_size == expected_transform.crop_size


def test_predict_returns_ranked_probabilities_and_uses_inference_mode(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "sample.png")
    classifier, model = make_classifier()

    predictions = classifier.predict(image_path, top_k=3)

    assert [item.class_name for item in predictions] == ["class one", "class two", "class four"]
    assert [item.rank for item in predictions] == [1, 2, 3]
    assert all(0.0 <= item.probability <= 1.0 for item in predictions)
    assert sum(item.probability for item in predictions) < 1.0
    assert model.inference_mode_seen is True
    assert model.seen_device == classifier.device
    assert [item.probability_percent for item in predictions] == pytest.approx(
        [item.probability * 100 for item in predictions]
    )


def test_predict_defaults_to_five_and_respects_available_class_count(tmp_path: Path) -> None:
    classifier, _ = make_classifier()
    image_path = make_image(tmp_path / "sample.png")

    assert len(classifier.predict(image_path)) == 5
    with pytest.raises(InvalidTopKError, match="between 1 and 6"):
        classifier.predict(image_path, top_k=7)

    assert len(classifier.predict(image_path, top_k=4)) == 4


@pytest.mark.parametrize("top_k", [0, -1, 7, 1.5, True])
def test_invalid_top_k_is_rejected(tmp_path: Path, top_k: object) -> None:
    classifier, _ = make_classifier()
    image_path = make_image(tmp_path / "sample.png")

    with pytest.raises(InvalidTopKError):
        classifier.predict(image_path, top_k=top_k)  # type: ignore[arg-type]


def test_missing_image_is_reported_without_inference(tmp_path: Path) -> None:
    classifier, model = make_classifier()

    with pytest.raises(ImageLoadError, match="does not exist"):
        classifier.predict(tmp_path / "missing.png", top_k=2)

    assert model.seen_device is None


def test_invalid_image_is_reported(tmp_path: Path) -> None:
    invalid_path = tmp_path / "not-an-image.png"
    invalid_path.write_text("not an image", encoding="utf-8")
    classifier, _ = make_classifier()

    with pytest.raises(ImageLoadError, match="valid image"):
        classifier.predict(invalid_path, top_k=2)


def test_model_output_class_count_must_match_labels(tmp_path: Path) -> None:
    classifier, _ = make_classifier([0.1, 0.3, 0.9])
    image_path = make_image(tmp_path / "sample.png")

    with pytest.raises(InferenceError, match="category labels"):
        classifier.predict(image_path, top_k=2)


def test_json_export_creates_parent_and_serializes_predictions(tmp_path: Path) -> None:
    classifier, _ = make_classifier()
    image_path = make_image(tmp_path / "sample.png")
    predictions = classifier.predict(image_path, top_k=2)
    export_path = tmp_path / "nested" / "reports" / "predictions.json"

    result_path = export_predictions_json(
        predictions, export_path, weights=ResNet50_Weights.IMAGENET1K_V1
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert result_path == export_path
    assert payload["model"] == "torchvision.resnet50"
    assert payload["weights"] == "IMAGENET1K_V1"
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][0]["class_name"] == predictions[0].class_name
    assert payload["predictions"][0]["probability_percent"] == pytest.approx(
        predictions[0].probability * 100
    )


def test_inference_failure_is_wrapped_with_a_clear_error(tmp_path: Path) -> None:
    class BrokenModel(nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("deliberate test failure")

    classifier = ImageClassifier(
        device="cpu",
        model=BrokenModel(),
        preprocess=lambda _image: torch.zeros((3, 8, 8), dtype=torch.float32),
        categories=("class zero", "class one"),
    )
    image_path = make_image(tmp_path / "sample.png")

    with pytest.raises(InferenceError, match="deliberate test failure"):
        classifier.predict(image_path, top_k=2)
