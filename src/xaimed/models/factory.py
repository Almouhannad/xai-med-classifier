"""Model factory helpers.

Currently supports ResNet backbones from torchvision.
"""

from __future__ import annotations

from typing import Callable, cast

import torch.nn as nn
from torchvision import models


class ModelFactoryError(ValueError):
    """Raised when an unsupported model configuration is requested."""


_RESNET_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}


def _resolve_resnet_weights(model_name: str, pretrained: bool):
    """Return the torchvision weights enum value for a ResNet model."""
    if not pretrained:
        return None

    weights_attr = f"{model_name.upper()}_Weights"
    weights_enum = getattr(models, weights_attr)
    return weights_enum.DEFAULT


def _adapt_input_channels(model: nn.Module, in_channels: int) -> None:
    """Patch the first convolution for non-RGB input channels."""
    if in_channels <= 0:
        raise ModelFactoryError("in_channels must be greater than 0.")

    if in_channels == 3:
        return

    if not hasattr(model, "conv1"):
        raise ModelFactoryError("Cannot adapt input channels: model has no conv1 layer.")

    conv1 = getattr(model, "conv1")
    if not isinstance(conv1, nn.Conv2d):
        raise ModelFactoryError("Cannot adapt input channels: conv1 is not a Conv2d layer.")

    model.conv1 = nn.Conv2d(
        in_channels,
        conv1.out_channels,
        kernel_size=cast(tuple[int, int], conv1.kernel_size),
        stride=cast(tuple[int, int], conv1.stride),
        padding=cast(str | tuple[int, int], conv1.padding),
        bias=conv1.bias is not None,
    )


def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> None:
    """Replace final fully-connected layer with a task-specific classifier head."""
    if num_classes <= 0:
        raise ModelFactoryError("num_classes must be greater than 0.")

    if not hasattr(model, "fc"):
        raise ModelFactoryError("Cannot replace classifier: model has no fc layer.")

    fc_layer = getattr(model, "fc")
    if not isinstance(fc_layer, nn.Linear):
        raise ModelFactoryError("Cannot replace classifier: fc is not a Linear layer.")

    in_features = fc_layer.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)


def build_model(
    name: str,
    num_classes: int,
    *,
    in_channels: int = 3,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a classifier model.

    Args:
        name: Backbone name (e.g., ``resnet18``).
        num_classes: Number of output classes.
        in_channels: Input image channels.
        pretrained: Load torchvision pretrained weights when True.
        dropout: Optional dropout before the final linear layer.
    """
    model_name = name.lower()
    if model_name not in _RESNET_BUILDERS:
        supported = ", ".join(sorted(_RESNET_BUILDERS))
        raise ModelFactoryError(f"Unsupported model '{name}'. Supported models: {supported}.")

    builder = _RESNET_BUILDERS[model_name]
    model = builder(weights=_resolve_resnet_weights(model_name, pretrained))

    _adapt_input_channels(model, in_channels)
    _replace_classifier(model, num_classes, dropout)

    return model