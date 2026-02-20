from __future__ import annotations

import pytest
import torch
from torchvision import models

from xaimed.models.factory import (
    ModelFactoryError,
    _adapt_input_channels,
    _resolve_resnet_weights,
    build_model,
)

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

    expected = torch.cat([original_weight, original_weight[:, :2, :, :]], dim=1) * (3.0 / 5.0)
    assert torch.allclose(model.conv1.weight, expected)

def test_build_model_rejects_unknown_model_name():
    with pytest.raises(ModelFactoryError):
        build_model("vit_b16", num_classes=3)

def test_adapt_input_channels_scales_two_channel_mapping():
    model = models.resnet18(weights=None)
    original_weight = model.conv1.weight.detach().clone()

    _adapt_input_channels(model, in_channels=2)

    expected = original_weight[:, :2, :, :] * (3.0 / 2.0)
    assert torch.allclose(model.conv1.weight, expected)

def test_resolve_resnet_weights_uses_correct_enum_name():
    resolved = _resolve_resnet_weights("resnet18", pretrained=True, weights=None)
    assert resolved == models.ResNet18_Weights.DEFAULT

def test_build_model_rejects_pretrained_with_explicit_weights():
    with pytest.raises(ModelFactoryError):
        build_model("resnet18", num_classes=2, pretrained=True, weights="DEFAULT")

@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_build_model_rejects_invalid_dropout(dropout):
    with pytest.raises(ModelFactoryError):
        build_model("resnet18", num_classes=2, dropout=dropout)