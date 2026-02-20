"""Reusable training and validation loops."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def _prepare_batch(batch: object, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Expected dataloader batch to contain (inputs, targets).")

    inputs = batch[0].to(device)
    targets = batch[1].to(device)

    if targets.ndim > 1:
        targets = targets.squeeze(-1)

    return inputs, targets.long()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch and return aggregate metrics."""
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    for batch in dataloader:
        inputs, targets = _prepare_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        running_acc += _accuracy_from_logits(logits, targets)
        num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": running_loss / num_batches,
        "accuracy": running_acc / num_batches,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation epoch and return aggregate metrics."""
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    for batch in dataloader:
        inputs, targets = _prepare_batch(batch, device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        running_loss += float(loss.item())
        running_acc += _accuracy_from_logits(logits, targets)
        num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": running_loss / num_batches,
        "accuracy": running_acc / num_batches,
    }


def format_epoch_metrics(epoch: int, train_metrics: Mapping[str, float], val_metrics: Mapping[str, float]) -> str:
    """Format metrics for human-readable training logs."""
    return (
        f"Epoch {epoch:03d} | "
        f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
    )
