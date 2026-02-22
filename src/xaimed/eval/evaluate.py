"""Model evaluation utilities and confusion matrix reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from xaimed.eval.calibration import expected_calibration_error
from xaimed.models.factory import build_model
from xaimed.seed import set_global_seed
from xaimed.reporting.make_failure_gallery import FailureGalleryArtifacts, build_failure_gallery
from xaimed.train.loops import _prepare_batch
from xaimed.train.train import build_dataloaders_from_config
from xaimed.utils.metrics import (
    accuracy_from_predictions,
    confusion_matrix,
    macro_f1_from_confusion_matrix,
)
from xaimed.utils.viz import save_confusion_matrix_plot, save_metric_history_plot


@dataclass
class EvalResult:
    """Filesystem outputs and aggregate metrics for an evaluation run."""

    metrics: dict[str, float | int | str]
    metrics_path: Path
    confusion_matrix_path: Path
    training_curves_path: Path
    failure_gallery: FailureGalleryArtifacts


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    all_targets: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    all_confidences: list[torch.Tensor] = []
    all_images: list[torch.Tensor] = []

    for batch in dataloader:
        inputs, targets = _prepare_batch(batch, device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        confidences, preds = probabilities.max(dim=1)

        all_targets.append(targets.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_confidences.append(confidences.detach().cpu())
        all_images.append(inputs.detach().cpu())

    if not all_targets:
        empty_long = torch.empty(0, dtype=torch.long)
        empty_float = torch.empty(0, dtype=torch.float32)
        empty_images = torch.empty(0, 3, 1, 1, dtype=torch.float32)
        return empty_long, empty_long, empty_float, empty_images

    return (
        torch.cat(all_targets, dim=0),
        torch.cat(all_preds, dim=0),
        torch.cat(all_confidences, dim=0),
        torch.cat(all_images, dim=0),
    )


def run_evaluation(config: dict[str, Any]) -> EvalResult:
    """Evaluate a trained checkpoint and generate metrics + split-wide training charts."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    device = torch.device(str(eval_cfg.get("device", train_cfg.get("device", "cpu"))))
    num_classes = int(model_cfg.get("num_classes", 2))

    dataloaders = build_dataloaders_from_config(config)
    split = str(eval_cfg.get("split", "val"))
    if split not in dataloaders:
        available = ", ".join(sorted(dataloaders.keys()))
        raise ValueError(f"Unknown eval split '{split}'. Available splits: {available}")

    model = build_model(
        name=str(model_cfg.get("name", "resnet18")),
        num_classes=num_classes,
        in_channels=int(model_cfg.get("in_channels", 3)),
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    default_checkpoint = Path(str(train_cfg.get("checkpoint_dir", "artifacts/checkpoints"))) / "best.pt"
    checkpoint_path = Path(str(eval_cfg.get("checkpoint_path", default_checkpoint)))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    history = checkpoint.get("history", {})

    targets, predictions, confidences, images = _collect_predictions(model, dataloaders[split], device)
    if targets.numel() == 0:
        raise ValueError(f"No samples found in split '{split}'.")

    matrix = confusion_matrix(targets, predictions, num_classes)
    metrics: dict[str, float | int | str] = {
        "split": split,
        "num_samples": int(targets.numel()),
        "accuracy": accuracy_from_predictions(targets, predictions),
        "macro_f1": macro_f1_from_confusion_matrix(matrix),
        "ece": expected_calibration_error(confidences, predictions == targets),
    }

    output_dir = Path(str(eval_cfg.get("output_dir", "artifacts/eval")))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    confusion_matrix_path = output_dir / "confusion_matrix.png"

    training_curves_path = output_dir / "training_curves.png"
    save_metric_history_plot(history, training_curves_path, split_name="train/val")

    failure_gallery = build_failure_gallery(
        output_dir=output_dir,
        images=images,
        targets=targets,
        predictions=predictions,
        confidences=confidences,
        top_k=int(eval_cfg.get("failure_gallery_top_k", 16)),
    )

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix_plot(matrix, confusion_matrix_path)

    return EvalResult(
        metrics=metrics,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        training_curves_path=training_curves_path,
        failure_gallery=failure_gallery,
    )
