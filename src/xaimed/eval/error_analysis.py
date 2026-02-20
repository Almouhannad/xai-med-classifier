"""Error analysis helpers for misclassified samples."""

from __future__ import annotations

import torch


def collect_misclassified_indices(targets: torch.Tensor, predictions: torch.Tensor) -> list[int]:
    """Return indices where prediction does not match target."""
    mismatches = (targets != predictions).nonzero(as_tuple=False).flatten()
    return [int(i) for i in mismatches.tolist()]
