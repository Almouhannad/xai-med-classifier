"""Metric helpers shared by training and evaluation flows."""

from __future__ import annotations

import torch


def accuracy_from_predictions(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    """Compute classification accuracy."""
    if targets.numel() == 0:
        return 0.0
    return float((targets == predictions).float().mean().item())


def confusion_matrix(targets: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build a confusion matrix with rows=true class and cols=predicted class."""
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, pred in zip(targets.tolist(), predictions.tolist()):
        matrix[target, pred] += 1
    return matrix


def macro_f1_from_confusion_matrix(matrix: torch.Tensor) -> float:
    """Compute macro-F1 from a confusion matrix."""
    eps = 1e-12
    tp = matrix.diag().float()
    fp = matrix.sum(dim=0).float() - tp
    fn = matrix.sum(dim=1).float() - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return float(f1.mean().item())
