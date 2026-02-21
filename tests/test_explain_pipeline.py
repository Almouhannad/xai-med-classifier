from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xaimed.xai.explain import run_explain


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_run_explain_writes_gradcam_overlays(tmp_path, monkeypatch):
    model = TinyConvNet()
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    images = torch.randn(5, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    monkeypatch.setattr("xaimed.xai.explain.build_model", lambda **_: TinyConvNet())
    monkeypatch.setattr("xaimed.xai.explain.build_dataloaders_from_config", lambda _cfg: {"val": loader})

    config = {
        "model": {"name": "tiny", "num_classes": 2, "in_channels": 3},
        "train": {"checkpoint_dir": str(tmp_path)},
        "explain": {
            "split": "val",
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(tmp_path / "explain"),
            "max_samples": 3,
            "device": "cpu",
        },
    }

    result = run_explain(config)

    assert result.output_dir.exists()
    assert len(result.overlay_paths) == 3
    assert all(path.exists() for path in result.overlay_paths)
