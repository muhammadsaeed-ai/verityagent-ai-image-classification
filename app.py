"""Command-line interface for VerityAgent AI Image Classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torchvision.models import ResNet50_Weights

from classifier import ImageClassifier, export_predictions_json, validate_image_path
from config import APP_NAME, DEFAULT_DEVICE, DEFAULT_OUTPUT_PATH, DEFAULT_TOP_K
from visualizer import save_visualization


def build_parser() -> argparse.ArgumentParser:
    """Create the application's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify an image using pretrained Torchvision ResNet-50 and save "
            "a visual top-k report."
        ),
        prog="verityagent-ai-image-classification",
    )
    parser.add_argument("image", type=Path, help="Path to the image to classify")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Visualization output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON prediction export",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of ranked predictions to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=DEFAULT_DEVICE,
        help="Inference device; auto selects CUDA when available (default: auto)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate before model loading so missing/invalid paths don't trigger a
    # large weight download.
    available_classes = len(ResNet50_Weights.DEFAULT.meta["categories"])
    if not 1 <= args.top_k <= available_classes:
        print(
            f"Error: --top-k must be between 1 and {available_classes}; received {args.top_k}.",
            file=sys.stderr,
        )
        return 2

    try:
        image_path = validate_image_path(args.image)
        classifier = ImageClassifier(device=args.device)
        predictions = classifier.predict(image_path, top_k=args.top_k)
        visualization_path = save_visualization(image_path, predictions, args.output)
        json_path = (
            export_predictions_json(predictions, args.json_output, weights=classifier.weights)
            if args.json_output is not None
            else None
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"{APP_NAME}")
    print(f"Input: {image_path}")
    print(f"Device: {classifier.device}")
    print(f"\nTop {len(predictions)} predictions:")
    for prediction in predictions:
        print(
            f"{prediction.rank}. {prediction.class_name} — "
            f"{prediction.probability_percent:.2f}%"
        )
    print(f"\nVisualization saved to: {visualization_path}")
    if json_path is not None:
        print(f"JSON results saved to: {json_path}")
    print("\nNote: probabilities are model scores, not calibrated guarantees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
