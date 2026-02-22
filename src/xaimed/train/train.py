"""Training entry points and checkpoint orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from xaimed.data.medmnist import build_medmnist_dataloaders
from xaimed.models.factory import build_model
from xaimed.seed import set_global_seed
from xaimed.train.loops import format_epoch_metrics, train_one_epoch, validate_one_epoch


@dataclass
class TrainResult:
    """Result metadata for a training run."""

    best_val_loss: float
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_epoch: int
    best_score: float
    best_monitor: str
    best_mode: str
    epochs_ran: int
    early_stopped: bool


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    history: dict[str, list[float]],
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": history,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


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


def _build_optimizer(model: nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            nesterov=bool(train_cfg.get("nesterov", False)),
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported: adam, adamw, sgd")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = str(train_cfg.get("lr_scheduler", "none")).lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("lr_step_size", 5)),
            gamma=float(train_cfg.get("lr_gamma", 0.1)),
        )

    if scheduler_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("lr_t_max", int(train_cfg.get("epochs", 1)))),
            eta_min=float(train_cfg.get("lr_eta_min", 0.0)),
        )

    if scheduler_name == "reducelronplateau":
        plateau_mode = str(train_cfg.get("lr_plateau_mode", "min"))
        if plateau_mode not in {"min", "max"}:
            raise ValueError("lr_plateau_mode must be either 'min' or 'max'")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cast(Literal["min", "max"], plateau_mode),
            factor=float(train_cfg.get("lr_plateau_factor", 0.1)),
            patience=int(train_cfg.get("lr_plateau_patience", 2)),
            min_lr=float(train_cfg.get("lr_plateau_min_lr", 0.0)),
        )

    raise ValueError(
        "Unsupported lr_scheduler "
        f"'{scheduler_name}'. Supported: none, StepLR, CosineAnnealingLR, ReduceLROnPlateau"
    )


def _is_improved(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "max":
        return current > (best + min_delta)
    return current < (best - min_delta)


def run_training(config: dict[str, Any]) -> TrainResult:
    """Train a model using modular epoch loops and save checkpoints."""
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

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
    optimizer = _build_optimizer(model, train_cfg)
    scheduler = _build_scheduler(optimizer, train_cfg)

    epochs = int(train_cfg.get("epochs", 1))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))

    early_stopping_enabled = bool(train_cfg.get("early_stopping", False))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 3))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    early_stopping_monitor = str(train_cfg.get("early_stopping_monitor", "val_loss"))
    early_stopping_mode = str(train_cfg.get("early_stopping_mode", "min"))

    lr_plateau_monitor = str(train_cfg.get("lr_plateau_monitor", "val_loss"))

    if early_stopping_monitor not in {"val_loss", "val_accuracy"}:
        raise ValueError("early_stopping_monitor must be either 'val_loss' or 'val_accuracy'")
    if early_stopping_mode not in {"min", "max"}:
        raise ValueError("early_stopping_mode must be either 'min' or 'max'")
    if lr_plateau_monitor not in {"val_loss", "val_accuracy"}:
        raise ValueError("lr_plateau_monitor must be either 'val_loss' or 'val_accuracy'")

    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "artifacts/checkpoints")))
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"

    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False

    best_score = float("inf") if early_stopping_mode == "min" else float("-inf")
    best_val_loss = float("inf")

    epochs_ran = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            criterion,
            device,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = validate_one_epoch(model, dataloaders["val"], criterion, device)

        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_accuracy"].append(float(train_metrics["accuracy"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        print(format_epoch_metrics(epoch, train_metrics, val_metrics))

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                plateau_metric = float(val_metrics["loss"] if lr_plateau_monitor == "val_loss" else val_metrics["accuracy"])
                scheduler.step(plateau_metric)
            else:
                scheduler.step()

        payload = _checkpoint_payload(model, optimizer, epoch, train_metrics, val_metrics, history, scheduler=scheduler)
        save_checkpoint(last_checkpoint_path, payload)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

        score = float(val_metrics["loss"] if early_stopping_monitor == "val_loss" else val_metrics["accuracy"])
        if _is_improved(score, best_score, early_stopping_mode, early_stopping_min_delta):
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_checkpoint_path, payload)
        else:
            epochs_without_improvement += 1

        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            early_stopped = True
            break

    return TrainResult(
        best_val_loss=best_val_loss,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        best_epoch=best_epoch,
        best_score=best_score,
        best_monitor=early_stopping_monitor,
        best_mode=early_stopping_mode,
        epochs_ran=epochs_ran,
        early_stopped=early_stopped,
    )
