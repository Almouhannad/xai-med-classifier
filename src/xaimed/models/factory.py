"""Model factory helpers.

Currently supports ResNet backbones from torchvision.
"""

from __future__ import annotations

import math
from typing import Any, Callable, cast

import torch
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

_RESNET_WEIGHTS: dict[str, Any] = {
    "resnet18": models.ResNet18_Weights,
    "resnet34": models.ResNet34_Weights,
    "resnet50": models.ResNet50_Weights,
    "resnet101": models.ResNet101_Weights,
}

def _resolve_resnet_weights(model_name: str, pretrained: bool, weights: str | Any | None):
    """Return the torchvision weights enum value for a ResNet model."""
    if pretrained and weights is not None:
        raise ModelFactoryError("Use either `pretrained` or `weights`, not both.")

    weights_enum = _RESNET_WEIGHTS[model_name]

    if weights is None:
        return weights_enum.DEFAULT if pretrained else None

    if isinstance(weights, str):
        if weights == "DEFAULT":
            return weights_enum.DEFAULT

        try:
            return weights_enum[weights]
        except KeyError as exc:
            raise ModelFactoryError(
                f"Unsupported weights '{weights}' for '{model_name}'."
            ) from exc

    return weights

def _remap_conv1_weights(old_weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Project RGB conv1 weights to a requested input-channel size.

    The mapping preserves pretrained information and keeps activation scale stable
    when input channels differ from RGB.
    """
    old_in_channels = old_weight.shape[1]

    if in_channels == 1:
        return old_weight.mean(dim=1, keepdim=True)

    repeats = math.ceil(in_channels / old_in_channels)
    expanded = old_weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]

    # Scale to keep expected conv1 activation magnitude close to the RGB case.
    expanded *= old_in_channels / float(in_channels)
    return expanded


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

    replacement = nn.Conv2d(
        in_channels,
        conv1.out_channels,
        kernel_size=cast(tuple[int, int], conv1.kernel_size),
        stride=cast(tuple[int, int], conv1.stride),
        padding=cast(str | tuple[int, int], conv1.padding),
        dilation=cast(tuple[int, int], conv1.dilation),
        groups=conv1.groups,
        bias=conv1.bias is not None,
        padding_mode=conv1.padding_mode,
        device=conv1.weight.device,
        dtype=conv1.weight.dtype,
    )

    with torch.no_grad():
        replacement.weight.copy_(_remap_conv1_weights(conv1.weight, in_channels))
        if conv1.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(conv1.bias)

    model.conv1 = replacement

def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> None:
    """Replace final fully-connected layer with a task-specific classifier head."""
    if num_classes <= 0:
        raise ModelFactoryError("num_classes must be greater than 0.")

    if not 0.0 <= dropout < 1.0:
        raise ModelFactoryError("dropout must be in [0.0, 1.0).")

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
    weights: str | Any | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a classifier model.

    Args:
        name: Backbone name (e.g., ``resnet18``).
        num_classes: Number of output classes.
        in_channels: Input image channels.
        pretrained: If True, use torchvision ``DEFAULT`` weights.
        weights: Optional explicit torchvision weights enum value or enum-name string.
        dropout: Optional dropout before the final linear layer.
    """
    model_name = name.lower()
    if model_name not in _RESNET_BUILDERS:
        supported = ", ".join(sorted(_RESNET_BUILDERS))
        raise ModelFactoryError(f"Unsupported model '{name}'. Supported models: {supported}.")

    builder = _RESNET_BUILDERS[model_name]
    resolved_weights = _resolve_resnet_weights(model_name, pretrained, weights)
    model = builder(weights=resolved_weights)

    _adapt_input_channels(model, in_channels)
    _replace_classifier(model, num_classes, dropout)

    return model