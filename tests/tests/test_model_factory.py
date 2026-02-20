from __future__ import annotations

import pytest
import torch
from torchvision import models

from xaimed.models.factory import ModelFactoryError, _adapt_input_channels, build_model

@pytest.mark.parametrize("model_name", ["resnet18", "resnet50"])
def test_build_model_resnet_output_shape(model_name):
    model = build_model(model_name, num_classes=7, pretrained=False)
    batch = torch.randn(4, 3, 224, 224)

    output = model(batch)

    assert output.shape == (4, 7)


def test_build_model_supports_custom_input_channels():
    model = build_model("resnet18", num_classes=5, in_channels=1, pretrained=False)
    batch = torch.randn(2, 1, 224, 224)

    output = model(batch)

    assert output.shape == (2, 5)

def test_adapt_input_channels_preserves_conv1_information_for_grayscale():
    model = models.resnet18(weights=None)
    original_weight = model.conv1.weight.detach().clone()

    _adapt_input_channels(model, in_channels=1)

    expected = original_weight.mean(dim=1, keepdim=True)
    assert torch.allclose(model.conv1.weight, expected)

def test_adapt_input_channels_repeats_conv1_weights_for_extra_channels():
    model = models.resnet18(weights=None)
    original_weight = model.conv1.weight.detach().clone()

    _adapt_input_channels(model, in_channels=5)

    expected = torch.cat([original_weight, original_weight[:, :2, :, :]], dim=1)
    assert torch.allclose(model.conv1.weight, expected)


def test_build_model_rejects_unknown_model_name():
    with pytest.raises(ModelFactoryError):
        build_model("Messi", num_classes=3)