"""Image classification with Torchvision's pretrained ResNet-50 weights."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


class ClassificationError(RuntimeError):
    """Base exception for image classification failures."""


class ImageLoadError(ClassificationError):
    """Raised when an image cannot be opened or decoded."""


class InvalidTopKError(ValueError):
    """Raised when top-k is outside the available class range."""


class InferenceError(ClassificationError):
    """Raised when model inference produces an error or invalid output."""


@dataclass(frozen=True, slots=True)
class Prediction:
    """One predicted ImageNet category and its probability in the [0, 1] range."""

    rank: int
    class_index: int
    class_name: str
    probability: float

    @property
    def probability_percent(self) -> float:
        """Probability expressed as a percentage."""
        return self.probability * 100.0

    def to_dict(self) -> dict[str, int | str | float]:
        """Return a JSON-serializable representation, including percentage."""
        data = asdict(self)
        data["probability_percent"] = self.probability_percent
        return data


def select_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve a device request, using CUDA automatically when available.

    Args:
        requested: "auto", "cpu", "cuda", or a torch.device.

    Raises:
        ValueError: If the device name is unknown.
        ClassificationError: If CUDA was requested but is unavailable.
    """
    if isinstance(requested, torch.device):
        device = requested
    elif requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested in {"cpu", "cuda"}:
        device = torch.device(requested)
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ClassificationError(
            "CUDA was requested, but CUDA is not available in this PyTorch environment. "
            "Use --device cpu or --device auto."
        )
    return device


def load_pretrained_model(
    weights: ResNet50_Weights = ResNet50_Weights.DEFAULT,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load ResNet-50 using the selected Torchvision weights and move it to device.

    The first call may download the pretrained weights from the official
    Torchvision model-hosting location if they are not cached locally.
    """
    model = resnet50(weights=weights)
    model = model.to(torch.device(device))
    model.eval()
    return model


def validate_image_path(image_path: str | Path) -> Path:
    """Verify that a path exists and contains a decodable image.

    This helper lets a CLI report invalid input before attempting a potentially
    large pretrained-weight download.
    """
    path = Path(image_path).expanduser()
    if not path.exists():
        raise ImageLoadError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ImageLoadError(f"Image path is not a file: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Could not read a valid image at '{path}': {exc}") from exc
    return path


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """Open an image using Pillow and return a detached RGB image."""
    path = validate_image_path(image_path)
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Could not decode image '{path}': {exc}") from exc


class ImageClassifier:
    """Predict ImageNet labels with ResNet-50.

    Args:
        device: "auto" selects CUDA if available and CPU otherwise.
        weights: The exact Torchvision weight enum used for both transforms
            and ImageNet category metadata.
        model: Optional injected model, mainly useful for offline tests.
        preprocess: Optional transform override, mainly useful for tests.
        categories: Optional labels override; production defaults to the labels
            stored in weights.meta["categories"].

    Injecting a model skips model downloading. If a model is injected, callers
    should inject matching categories and preprocessing for meaningful results.
    """

    def __init__(
        self,
        device: str | torch.device = "auto",
        weights: ResNet50_Weights = ResNet50_Weights.DEFAULT,
        model: nn.Module | None = None,
        preprocess: Callable[[Image.Image], Tensor] | None = None,
        categories: Sequence[str] | None = None,
    ) -> None:
        self.device = select_device(device)
        self.weights = weights
        self.model = model if model is not None else load_pretrained_model(weights, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # These are deliberately taken from the same selected weight enum.
        self.preprocess = preprocess if preprocess is not None else weights.transforms()
        metadata_categories = weights.meta.get("categories", [])
        self.categories = tuple(categories if categories is not None else metadata_categories)
        if not self.categories:
            raise ValueError("No category labels were supplied by the selected weights metadata.")

    def predict(self, image_path: str | Path, top_k: int = 5) -> list[Prediction]:
        """Classify an image and return up to top_k ranked predictions.

        Raises:
            ImageLoadError: If the input image is missing or invalid.
            InvalidTopKError: If top_k is not an integer in [1, class_count].
            InferenceError: If model inference fails or output is inconsistent.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise InvalidTopKError("top_k must be an integer.")
        if not 1 <= top_k <= len(self.categories):
            raise InvalidTopKError(
                f"top_k must be between 1 and {len(self.categories)}; received {top_k}."
            )

        image = load_rgb_image(image_path)
        try:
            input_tensor = self.preprocess(image)
            if not isinstance(input_tensor, Tensor):
                raise TypeError("The preprocessing transform must return a torch.Tensor.")
            if input_tensor.ndim != 3:
                raise ValueError(
                    "Preprocessing must return a 3D (channels, height, width) tensor; "
                    f"received shape {tuple(input_tensor.shape)}."
                )
            batch = input_tensor.unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = self.model(batch)
                if not isinstance(logits, Tensor):
                    raise TypeError("The model must return a torch.Tensor of logits.")
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise ValueError(
                        "Expected model logits with shape (1, class_count); "
                        f"received {tuple(logits.shape)}."
                    )
                if logits.shape[1] != len(self.categories):
                    raise ValueError(
                        f"Model returned {logits.shape[1]} classes, but metadata contains "
                        f"{len(self.categories)} category labels. Ensure model weights and "
                        "category metadata match."
                    )
                probabilities = torch.softmax(logits[0], dim=0)
                scores, indexes = torch.topk(probabilities, k=top_k, largest=True, sorted=True)

            return [
                Prediction(
                    rank=rank,
                    class_index=int(class_index.item()),
                    class_name=self.categories[int(class_index.item())],
                    probability=float(score.item()),
                )
                for rank, (score, class_index) in enumerate(zip(scores, indexes), start=1)
            ]
        except ClassificationError:
            raise
        except Exception as exc:
            raise InferenceError(f"Image inference failed: {exc}") from exc


def export_predictions_json(
    predictions: Sequence[Prediction],
    output_path: str | Path,
    weights: ResNet50_Weights = ResNet50_Weights.DEFAULT,
) -> Path:
    """Write predictions to JSON and return the resulting path."""
    path = Path(output_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "torchvision.resnet50",
            "weights": weights.name,
            "predictions": [prediction.to_dict() for prediction in predictions],
        }
        # Write using a temporary sibling file so interrupted writes are less
        # likely to leave a partially written export behind.
        temporary_path = path.with_name(f"{path.name}.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary_path.replace(path)
    except OSError as exc:
        raise ClassificationError(f"Could not write JSON results to '{path}': {exc}") from exc
    return path
