from __future__ import annotations

import torch

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
