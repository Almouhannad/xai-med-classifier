"""Calibration helpers for confidence analysis."""

from __future__ import annotations

import torch


def expected_calibration_error(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    num_bins: int = 10,
) -> float:
    """Compute ECE with equal-width confidence bins."""
    if confidences.numel() == 0:
        return 0.0

    ece = 0.0
    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences >= start) & (confidences < end)
        if not in_bin.any():
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = correctness[in_bin].float().mean()
        ece += float(torch.abs(bin_conf - bin_acc) * in_bin.float().mean())
    return ece
