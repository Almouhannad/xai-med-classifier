"""Classifier head helpers."""

from __future__ import annotations

from torch import nn


def build_classification_head(in_features: int, num_classes: int, dropout: float = 0.0) -> nn.Module:
    """Create a linear classifier head with optional dropout."""
    if dropout > 0:
        return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)
