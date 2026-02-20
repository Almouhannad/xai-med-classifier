"""Training entry points and checkpoint orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from xaimed.data.medmnist import build_medmnist_dataloaders
from xaimed.models.factory import build_model
from xaimed.train.loops import format_epoch_metrics, train_one_epoch, validate_one_epoch


@dataclass
class TrainResult:
    """Result metadata for a training run."""

    best_val_loss: float
    best_checkpoint_path: Path
    last_checkpoint_path: Path


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Persist a checkpoint payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _build_fake_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    data_cfg = config.get("data", {})
    train_samples = int(data_cfg.get("train_samples", 64))
    val_samples = int(data_cfg.get("val_samples", 32))
    image_size = int(data_cfg.get("image_size", 64))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))

    num_classes = int(config.get("model", {}).get("num_classes", 2))
    transform = Compose(
        [
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    return {
        "train": DataLoader(
            FakeData(
                size=train_samples,
                image_size=(3, image_size, image_size),
                num_classes=num_classes,
                transform=transform,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            FakeData(
                size=val_samples,
                image_size=(3, image_size, image_size),
                num_classes=num_classes,
                transform=transform,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def build_dataloaders_from_config(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Build train/val dataloaders from config options."""
    data_cfg = config.get("data", {})
    if data_cfg.get("use_fake_data", False):
        return _build_fake_dataloaders(config)

    dataloaders = build_medmnist_dataloaders(
        dataset_name=str(data_cfg.get("dataset", "pathmnist")),
        data_dir=str(data_cfg.get("data_dir", "data")),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        image_size=int(data_cfg.get("image_size", 224)),
    )
    return {"train": dataloaders["train"], "val": dataloaders["val"]}


def run_training(config: dict[str, Any]) -> TrainResult:
    """Train a model using modular epoch loops and save checkpoints."""
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})

    device = torch.device(str(train_cfg.get("device", "cpu")))
    dataloaders = build_dataloaders_from_config(config)

    model = build_model(
        name=str(model_cfg.get("name", "resnet18")),
        num_classes=int(model_cfg.get("num_classes", 2)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        pretrained=bool(model_cfg.get("pretrained", False)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 1))
    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "artifacts/checkpoints")))
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, dataloaders["train"], optimizer, criterion, device)
        val_metrics = validate_one_epoch(model, dataloaders["val"], criterion, device)

        print(format_epoch_metrics(epoch, train_metrics, val_metrics))

        payload = _checkpoint_payload(model, optimizer, epoch, train_metrics, val_metrics)
        save_checkpoint(last_checkpoint_path, payload)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(best_checkpoint_path, payload)

    return TrainResult(
        best_val_loss=best_val_loss,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )
