from __future__ import annotations

import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xaimed.eval.evaluate import run_evaluation


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(3 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.flatten(x))


def test_run_evaluation_writes_metrics_and_confusion_matrix(tmp_path, monkeypatch):
    model = TinyModel()
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    features = torch.randn(6, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=2)

    monkeypatch.setattr("xaimed.eval.evaluate.build_model", lambda **_: TinyModel())
    monkeypatch.setattr("xaimed.eval.evaluate.build_dataloaders_from_config", lambda _cfg: {"val": loader})

    config = {
        "model": {"name": "tiny", "num_classes": 2, "in_channels": 3},
        "eval": {
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(tmp_path / "eval"),
            "split": "val",
            "device": "cpu",
        },
    }

    result = run_evaluation(config)

    assert result.metrics_path.exists()
    assert result.confusion_matrix_path.exists()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["num_samples"] == 6
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
