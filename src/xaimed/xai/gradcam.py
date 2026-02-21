"""Grad-CAM implementation for CNN-style models."""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _get_module_by_name(model: nn.Module, dotted_name: str) -> nn.Module:
    """Resolve a module by dotted path, e.g. 'layer3.1.conv2'."""
    current: nn.Module = model
    for part in dotted_name.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


def _find_last_conv_layer(model: nn.Module) -> nn.Module:
    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("Grad-CAM requires a model with at least one Conv2d layer.")
    return conv_layers[-1]


def _resolve_target_layer(model: nn.Module, target_layer: nn.Module | str | None) -> nn.Module:
    if target_layer is None:
        return _find_last_conv_layer(model)
    if isinstance(target_layer, str):
        layer = _get_module_by_name(model, target_layer)
        if not isinstance(layer, nn.Module):
            raise ValueError(f"Resolved target layer '{target_layer}' is not an nn.Module.")
        return layer
    return target_layer


def _normalize_cams(cams: torch.Tensor) -> torch.Tensor:
    cams = cams.detach()
    flat = cams.view(cams.shape[0], -1)
    mins = flat.min(dim=1).values.view(-1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1)
    denom = (maxs - mins).clamp_min(1e-8)
    return (cams - mins) / denom


def _resolve_target_classes(
    logits: torch.Tensor,
    target_classes: Iterable[int] | torch.Tensor | None,
) -> torch.Tensor:
    if target_classes is None:
        return logits.argmax(dim=1)

    if isinstance(target_classes, torch.Tensor):
        classes = target_classes.to(device=logits.device, dtype=torch.long)
    else:
        classes = torch.tensor(list(target_classes), device=logits.device, dtype=torch.long)

    classes = classes.view(-1)
    if classes.numel() != logits.shape[0]:
        raise ValueError("target_classes must provide exactly one class id per input sample.")

    return classes


def gradcam(
    model: nn.Module,
    images: torch.Tensor,
    *,
    target_classes: Iterable[int] | torch.Tensor | None = None,
    target_layer: nn.Module | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Grad-CAM heatmaps for a batch of images.

    Args:
        model: CNN classifier returning logits of shape (N, C).
        images: Input tensor of shape (N, C, H, W).
        target_classes: Optional class ids (one per sample). If None, uses predicted class.
        target_layer: Optional layer to use for CAM. Can be:
            - None (auto: last Conv2d)
            - nn.Module
            - dotted module path string, e.g. "layer2", "layer3.1.conv2"

    Returns:
        (cams, predicted_classes)
        - cams: (N, H, W) normalized to [0, 1]
        - predicted_classes: (N,)
    """
    if images.ndim != 4:
        raise ValueError("images must be a 4D tensor with shape (N, C, H, W).")

    layer = _resolve_target_layer(model, target_layer)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def _forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def _backward_hook(_module: nn.Module, _grad_input: Any, grad_output: Any) -> None:
        grad_tensor = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        gradients.append(grad_tensor)

    was_training = model.training
    model.eval()

    forward_handle = layer.register_forward_hook(_forward_hook)
    backward_handle = layer.register_full_backward_hook(_backward_hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(images)
            predicted = logits.argmax(dim=1).detach()
            classes = _resolve_target_classes(logits, target_classes)

            score = logits.gather(1, classes.view(-1, 1)).sum()
            score.backward()

        if not activations or not gradients:
            raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

        acts = activations[-1]
        grads = gradients[-1]

        # Channel weights = global average pooled gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted sum over channels + ReLU (standard Grad-CAM)
        cams = torch.relu((weights * acts).sum(dim=1))

        # Upsample to input resolution
        cams = F.interpolate(
            cams.unsqueeze(1),
            size=images.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Normalize per-image to [0,1]
        cams = _normalize_cams(cams)

    finally:
        forward_handle.remove()
        backward_handle.remove()
        if was_training:
            model.train()

    return cams, predicted