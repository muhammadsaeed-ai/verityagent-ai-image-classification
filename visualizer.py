"""Matplotlib visualization helpers for image classification predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from PIL import Image

from classifier import Prediction, load_rgb_image
from config import ATTRIBUTION


def save_visualization(
    image_path: str | Path,
    predictions: Sequence[Prediction],
    output_path: str | Path,
    *,
    title: str = "Image Classification Results",
    dpi: int = 160,
) -> Path:
    """Save a visualization with the input image and horizontal score chart.

    The chart displays model probabilities as percentages. The output directory
    is created when needed. A .png, .jpg/.jpeg, .pdf, or other Matplotlib-supported
    extension can be selected through output_path.
    """
    if not predictions:
        raise ValueError("At least one prediction is required to build a visualization.")
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    input_image: Image.Image = load_rgb_image(image_path)
    path = Path(output_path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".png")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure_height = max(4.8, 1.05 * len(predictions) + 2.0)
        fig, (image_axis, chart_axis) = plt.subplots(
            1,
            2,
            figsize=(12.5, figure_height),
            gridspec_kw={"width_ratios": [1.0, 1.35]},
            constrained_layout=True,
        )
        try:
            image_axis.imshow(input_image)
            image_axis.set_title("Input image", fontsize=13, weight="semibold", pad=12)
            image_axis.axis("off")

            names = [prediction.class_name for prediction in predictions]
            percentages = [prediction.probability_percent for prediction in predictions]
            positions = list(range(len(predictions)))
            bars = chart_axis.barh(positions, percentages, height=0.66)
            chart_axis.set_yticks(positions, labels=names)
            chart_axis.invert_yaxis()
            chart_axis.set_xlim(0, 105)
            chart_axis.set_xlabel("Model probability (%)", labelpad=8)
            chart_axis.set_title("Top predictions", fontsize=13, weight="semibold", pad=12)
            chart_axis.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
            chart_axis.set_axisbelow(True)
            chart_axis.spines["top"].set_visible(False)
            chart_axis.spines["right"].set_visible(False)

            for bar, percentage in zip(bars, percentages):
                chart_axis.text(
                    min(percentage + 1.2, 101.5),
                    bar.get_y() + bar.get_height() / 2,
                    f"{percentage:.2f}%",
                    va="center",
                    ha="left",
                    fontsize=9,
                )

            fig.suptitle(title, fontsize=16, weight="bold")
            fig.text(
                0.995,
                0.004,
                ATTRIBUTION,
                ha="right",
                va="bottom",
                fontsize=8,
                color="dimgray",
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Could not save visualization to '{path}': {exc}") from exc

    return path
