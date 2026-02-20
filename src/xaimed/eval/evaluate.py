"""Model evaluation utilities and confusion matrix reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader

from xaimed.models.factory import build_model
from xaimed.train.loops import _prepare_batch
from xaimed.train.train import build_dataloaders_from_config


@dataclass
class EvalResult:
    """Filesystem outputs and aggregate metrics for an evaluation run."""

    metrics: dict[str, float | int | str]
    metrics_path: Path
    confusion_matrix_path: Path


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    all_targets: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []

    for batch in dataloader:
        inputs, targets = _prepare_batch(batch, device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)

        all_targets.append(targets.detach().cpu())
        all_preds.append(preds.detach().cpu())

    if not all_targets:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    return torch.cat(all_targets, dim=0), torch.cat(all_preds, dim=0)


def _compute_confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for target, pred in zip(targets.tolist(), predictions.tolist()):
        matrix[target, pred] += 1

    return matrix


def _macro_f1_from_confusion_matrix(confusion_matrix: torch.Tensor) -> float:
    eps = 1e-12
    tp = confusion_matrix.diag().float()
    fp = confusion_matrix.sum(dim=0).float() - tp
    fn = confusion_matrix.sum(dim=1).float() - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_per_class = (2 * precision * recall) / (precision + recall + eps)
    return float(f1_per_class.mean().item())


def _save_confusion_matrix_plot(confusion_matrix: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = confusion_matrix.shape[0]
    cell_size = 64
    image_size = max(1, num_classes) * cell_size
    image = Image.new("RGB", (image_size, image_size), color="white")
    draw = ImageDraw.Draw(image)

    max_value = int(confusion_matrix.max().item()) if confusion_matrix.numel() else 1
    max_value = max(max_value, 1)

    for row in range(num_classes):
        for col in range(num_classes):
            value = int(confusion_matrix[row, col].item())
            intensity = int(255 * (1 - (value / max_value)))
            fill = (intensity, intensity, 255)

            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline="black")

            text = str(value)
            text_box = draw.textbbox((0, 0), text)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            text_x = x0 + (cell_size - text_w) / 2
            text_y = y0 + (cell_size - text_h) / 2
            draw.text((text_x, text_y), text, fill="black")

    image.save(out_path)


def run_evaluation(config: dict[str, Any]) -> EvalResult:
    """Evaluate a trained checkpoint and generate metrics + confusion matrix plot."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})

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

    checkpoint_path = Path(str(eval_cfg.get("checkpoint_path", Path(str(train_cfg.get("checkpoint_dir", "artifacts/checkpoints"))) / "best.pt")))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    targets, predictions = _collect_predictions(model, dataloaders[split], device)
    if targets.numel() == 0:
        raise ValueError(f"No samples found in split '{split}'.")

    confusion_matrix = _compute_confusion_matrix(targets, predictions, num_classes)
    accuracy = float((targets == predictions).float().mean().item())
    macro_f1 = _macro_f1_from_confusion_matrix(confusion_matrix)

    metrics: dict[str, float | int | str] = {
        "split": split,
        "num_samples": int(targets.numel()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

    output_dir = Path(str(eval_cfg.get("output_dir", "artifacts/eval")))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    confusion_matrix_path = output_dir / "confusion_matrix.png"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_confusion_matrix_plot(confusion_matrix, confusion_matrix_path)

    return EvalResult(
        metrics=metrics,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
    )
