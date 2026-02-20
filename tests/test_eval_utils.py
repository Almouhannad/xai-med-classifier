from __future__ import annotations

import torch

from xaimed.eval.error_analysis import select_failure_gallery_indices
from xaimed.utils.metrics import (
    accuracy_from_predictions,
    confusion_matrix,
    macro_f1_from_confusion_matrix,
)


def test_metrics_helpers_compute_expected_shapes_and_ranges():
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    preds = torch.tensor([0, 1, 0, 0], dtype=torch.long)

    matrix = confusion_matrix(targets, preds, num_classes=2)

    assert matrix.shape == (2, 2)
    assert matrix.sum().item() == 4
    assert 0.0 <= accuracy_from_predictions(targets, preds) <= 1.0
    assert 0.0 <= macro_f1_from_confusion_matrix(matrix) <= 1.0


def test_select_failure_gallery_indices_sorts_by_confidence():
    targets = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
    preds = torch.tensor([1, 1, 0, 0, 1], dtype=torch.long)
    conf = torch.tensor([0.95, 0.22, 0.87, 0.15, 0.35], dtype=torch.float32)

    high_conf_wrong, low_conf_correct = select_failure_gallery_indices(targets, preds, conf, top_k=2)

    assert high_conf_wrong == [0, 2]
    assert low_conf_correct == [3, 1]
