from __future__ import annotations

import csv

import torch

from xaimed.reporting.make_failure_gallery import build_failure_gallery


def test_build_failure_gallery_writes_csv_and_grids(tmp_path):
    images = torch.rand(5, 3, 8, 8)
    targets = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    preds = torch.tensor([1, 1, 0, 0, 1], dtype=torch.long)
    conf = torch.tensor([0.90, 0.10, 0.40, 0.80, 0.20], dtype=torch.float32)

    artifacts = build_failure_gallery(tmp_path, images, targets, preds, conf, top_k=2)

    assert artifacts.csv_path.exists()
    assert artifacts.high_conf_wrong_grid_path.exists()
    assert artifacts.low_conf_correct_grid_path.exists()

    with artifacts.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[0]["group"] == "high_confidence_wrong"
    assert rows[0]["index"] == "0"
    assert rows[2]["group"] == "low_confidence_correct"
