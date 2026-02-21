from __future__ import annotations

import torch
from torch import nn

from xaimed.xai.gradcam import gradcam


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 6, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def test_gradcam_returns_normalized_maps_and_predictions():
    model = TinyConvNet()
    images = torch.randn(3, 3, 16, 16)

    cams, preds = gradcam(model, images)

    assert cams.shape == (3, 16, 16)
    assert preds.shape == (3,)
    assert float(cams.min()) >= 0.0
    assert float(cams.max()) <= 1.0


def test_gradcam_accepts_target_classes():
    model = TinyConvNet()
    images = torch.randn(2, 3, 16, 16)

    cams, _ = gradcam(model, images, target_classes=[0, 1])

    assert cams.shape == (2, 16, 16)
