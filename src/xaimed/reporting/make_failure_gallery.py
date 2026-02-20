"""Failure gallery generation utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from xaimed.eval.error_analysis import collect_misclassified_indices, select_failure_gallery_indices
from xaimed.utils.viz import save_image_grid


@dataclass
class FailureGalleryArtifacts:
    """Filesystem outputs for failure-gallery analysis."""

    csv_path: Path
    high_conf_wrong_grid_path: Path
    low_conf_correct_grid_path: Path


def _write_failure_gallery_csv(
    path: Path,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    high_conf_wrong: list[int],
    low_conf_correct: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "rank", "index", "target", "prediction", "confidence"],
        )
        writer.writeheader()

        for rank, sample_idx in enumerate(high_conf_wrong, start=1):
            writer.writerow(
                {
                    "group": "high_confidence_wrong",
                    "rank": rank,
                    "index": sample_idx,
                    "target": int(targets[sample_idx].item()),
                    "prediction": int(predictions[sample_idx].item()),
                    "confidence": f"{float(confidences[sample_idx].item()):.6f}",
                }
            )

        for rank, sample_idx in enumerate(low_conf_correct, start=1):
            writer.writerow(
                {
                    "group": "low_confidence_correct",
                    "rank": rank,
                    "index": sample_idx,
                    "target": int(targets[sample_idx].item()),
                    "prediction": int(predictions[sample_idx].item()),
                    "confidence": f"{float(confidences[sample_idx].item()):.6f}",
                }
            )


def build_failure_gallery(
    output_dir: str | Path,
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    top_k: int = 16,
) -> FailureGalleryArtifacts:
    """Create CSV and image-grid artifacts for failure analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mismatched_indices = collect_misclassified_indices(targets, predictions)

    high_conf_wrong, low_conf_correct = select_failure_gallery_indices(
        targets=targets,
        predictions=predictions,
        confidences=confidences,
        top_k=top_k,
    )

    csv_path = output_dir / "failure_gallery_selection.csv"
    high_conf_wrong_grid_path = output_dir / "high_confidence_wrongs_grid.png"
    low_conf_correct_grid_path = output_dir / "low_confidence_corrects_grid.png"

    if set(high_conf_wrong) - set(mismatched_indices):
        raise ValueError("High-confidence wrong selections must be misclassified samples.")

    _write_failure_gallery_csv(
        csv_path,
        targets=targets,
        predictions=predictions,
        confidences=confidences,
        high_conf_wrong=high_conf_wrong,
        low_conf_correct=low_conf_correct,
    )

    save_image_grid(images, high_conf_wrong, high_conf_wrong_grid_path, title="High-confidence wrong")
    save_image_grid(images, low_conf_correct, low_conf_correct_grid_path, title="Low-confidence correct")

    return FailureGalleryArtifacts(
        csv_path=csv_path,
        high_conf_wrong_grid_path=high_conf_wrong_grid_path,
        low_conf_correct_grid_path=low_conf_correct_grid_path,
    )
