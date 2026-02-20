"""Error analysis helpers for confidence-based sample selection."""

from __future__ import annotations

import torch


def collect_misclassified_indices(targets: torch.Tensor, predictions: torch.Tensor) -> list[int]:
    """Return indices where prediction does not match target."""
    mismatches = (targets != predictions).nonzero(as_tuple=False).flatten()
    return [int(i) for i in mismatches.tolist()]


def select_failure_gallery_indices(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    top_k: int = 16,
) -> tuple[list[int], list[int]]:
    """Select high-confidence wrong and low-confidence correct samples.

    Returns
    -------
    tuple[list[int], list[int]]
        First list contains indices of misclassified samples sorted by confidence
        descending (most confidently wrong first). Second list contains indices
        of correct samples sorted by confidence ascending (least confident
        correct first).
    """
    if targets.shape != predictions.shape or targets.shape != confidences.shape:
        raise ValueError("targets, predictions, and confidences must have the same shape")

    incorrect_mask = predictions != targets
    correct_mask = ~incorrect_mask

    incorrect_indices = incorrect_mask.nonzero(as_tuple=False).flatten()
    correct_indices = correct_mask.nonzero(as_tuple=False).flatten()

    if incorrect_indices.numel() > 0:
        wrong_scores = confidences[incorrect_indices]
        wrong_order = torch.argsort(wrong_scores, descending=True)
        high_conf_wrong = incorrect_indices[wrong_order][:top_k]
        high_conf_wrong_list = [int(index) for index in high_conf_wrong.tolist()]
    else:
        high_conf_wrong_list = []

    if correct_indices.numel() > 0:
        correct_scores = confidences[correct_indices]
        correct_order = torch.argsort(correct_scores, descending=False)
        low_conf_correct = correct_indices[correct_order][:top_k]
        low_conf_correct_list = [int(index) for index in low_conf_correct.tolist()]
    else:
        low_conf_correct_list = []

    return high_conf_wrong_list, low_conf_correct_list
