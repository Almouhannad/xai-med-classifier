"""Explanation workflow using Grad-CAM overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

from xaimed.models.factory import build_model
from xaimed.seed import set_global_seed
from xaimed.train.loops import _prepare_batch
from xaimed.train.train import build_dataloaders_from_config
from xaimed.xai.gradcam import gradcam


@dataclass
class ExplainResult:
    """Filesystem outputs for an explanation run."""

    output_dir: Path
    overlay_paths: list[Path]


def _tensor_to_rgb_uint8(image: torch.Tensor) -> torch.Tensor:
    """Convert CHW tensor to displayable RGB uint8 (per-image min-max normalized)."""
    if image.ndim != 3:
        raise ValueError("Expected CHW image tensor.")

    image = image.detach().cpu().float()
    channels = image.shape[0]

    if channels == 1:
        image = image.repeat(3, 1, 1)
    elif channels < 3:
        image = image.repeat(3 // channels + 1, 1, 1)[:3]
    elif channels > 3:
        image = image[:3]

    image = image - image.min()
    max_val = image.max()
    if max_val > 0:
        image = image / max_val
    return (image * 255).clamp(0, 255).byte()


def _cam_to_gray_uint8(cam: torch.Tensor) -> torch.Tensor:
    """Convert HxW CAM in [0,1] to uint8 grayscale."""
    cam = cam.detach().cpu().float().clamp(0, 1)
    return (cam * 255).byte()


def _cam_to_heatmap_rgb_uint8(cam: torch.Tensor) -> torch.Tensor:
    """Create a high-contrast pseudo-color heatmap (CHW uint8) from CAM in [0,1].

    This avoids red-on-pink-only overlays that are hard to read on H&E images.
    """
    x = cam.detach().cpu().float().clamp(0, 1)

    # Jet-like piecewise mapping (blue -> cyan -> yellow -> red)
    r = (1.5 * x - 0.5).clamp(0, 1)
    g = (1.5 - (2.0 * x - 1.0).abs() * 1.5).clamp(0, 1)
    b = (1.5 * (1.0 - x) - 0.5).clamp(0, 1)

    heat = torch.stack([r, g, b], dim=0)
    return (heat * 255).byte()


def _overlay_heatmap_on_image(
    image: torch.Tensor,
    cam: torch.Tensor,
    alpha: float = 0.55,
) -> Image.Image:
    """Blend pseudo-color CAM heatmap over the image."""
    base = _tensor_to_rgb_uint8(image).float()
    heat = _cam_to_heatmap_rgb_uint8(cam).float()

    blended = ((1.0 - alpha) * base + alpha * heat).clamp(0, 255).byte()
    return Image.fromarray(blended.permute(1, 2, 0).numpy(), mode="RGB")


def _raw_cam_image(cam: torch.Tensor) -> Image.Image:
    gray = _cam_to_gray_uint8(cam)
    return Image.fromarray(gray.numpy(), mode="L")


def _heatmap_image(cam: torch.Tensor) -> Image.Image:
    heat = _cam_to_heatmap_rgb_uint8(cam)
    return Image.fromarray(heat.permute(1, 2, 0).numpy(), mode="RGB")


def _base_image(image: torch.Tensor) -> Image.Image:
    base = _tensor_to_rgb_uint8(image)
    return Image.fromarray(base.permute(1, 2, 0).numpy(), mode="RGB")


def _make_panel(image: torch.Tensor, cam: torch.Tensor, alpha: float = 0.55) -> Image.Image:
    """Create side-by-side panel: original | heatmap | overlay."""
    img = _base_image(image)
    heat = _heatmap_image(cam)
    overlay = _overlay_heatmap_on_image(image, cam, alpha=alpha)

    w, h = img.size
    panel = Image.new("RGB", (w * 3, h), color=(255, 255, 255))
    panel.paste(img, (0, 0))
    panel.paste(heat, (w, 0))
    panel.paste(overlay, (2 * w, 0))
    return panel


def _resolve_device(explain_cfg: dict[str, Any], eval_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> torch.device:
    requested = str(explain_cfg.get("device", eval_cfg.get("device", train_cfg.get("device", "cpu"))))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def run_explain(config: dict[str, Any]) -> ExplainResult:
    """Generate Grad-CAM overlays for a small sample from a configured split."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    explain_cfg = config.get("explain", {})

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    device = _resolve_device(explain_cfg, eval_cfg, train_cfg)

    model = build_model(
        name=str(model_cfg.get("name", "resnet18")),
        num_classes=int(model_cfg.get("num_classes", 2)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    default_checkpoint = Path(str(train_cfg.get("checkpoint_dir", "artifacts/checkpoints"))) / "best.pt"
    checkpoint_path = Path(str(explain_cfg.get("checkpoint_path", eval_cfg.get("checkpoint_path", default_checkpoint))))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    split = str(explain_cfg.get("split", "val"))
    dataloaders: dict[str, DataLoader] = build_dataloaders_from_config(config)
    if split not in dataloaders:
        available = ", ".join(sorted(dataloaders.keys()))
        raise ValueError(f"Unknown explain split '{split}'. Available splits: {available}")

    max_samples = int(explain_cfg.get("max_samples", 8))
    images_batches: list[torch.Tensor] = []
    labels_batches: list[torch.Tensor] = []
    total = 0

    for batch in dataloaders[split]:
        inputs, labels = _prepare_batch(batch, device)

        # PathMNIST loaders sometimes produce labels with shape (N, 1); flatten safely.
        labels = labels.view(-1).long()

        images_batches.append(inputs)
        labels_batches.append(labels)
        total += inputs.shape[0]
        if total >= max_samples:
            break

    if not images_batches:
        raise ValueError(f"No samples found in split '{split}'.")

    images = torch.cat(images_batches, dim=0)[:max_samples]
    labels = torch.cat(labels_batches, dim=0)[:max_samples]

    # If target_layer is omitted, gradcam() auto-selects the last Conv2d layer.
    # This keeps explain robust for tiny/custom models used in tests and downstream tasks.
    target_layer = explain_cfg.get("target_layer")
    overlay_alpha = float(explain_cfg.get("overlay_alpha", 0.55))
    save_raw_cam = bool(explain_cfg.get("save_raw_cam", True))
    save_panel = bool(explain_cfg.get("save_panel", True))

    cams, preds = gradcam(
        model,
        images,
        target_classes=labels,   # explain true-class score
        target_layer=target_layer,
    )

    output_dir = Path(str(explain_cfg.get("output_dir", "artifacts/explain")))
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_paths: list[Path] = []
    for idx in range(images.shape[0]):
        image_i = images[idx].detach().cpu()
        cam_i = cams[idx].detach().cpu()

        stem = f"sample_{idx:03d}_t{int(labels[idx].item())}_p{int(preds[idx].item())}"

        overlay = _overlay_heatmap_on_image(image_i, cam_i, alpha=overlay_alpha)
        overlay_path = output_dir / f"{stem}_gradcam.png"
        overlay.save(overlay_path)
        overlay_paths.append(overlay_path)

        if save_raw_cam:
            raw_cam = _raw_cam_image(cam_i)
            raw_cam.save(output_dir / f"{stem}_gradcam_raw.png")

        if save_panel:
            panel = _make_panel(image_i, cam_i, alpha=overlay_alpha)
            panel.save(output_dir / f"{stem}_gradcam_panel.png")

    return ExplainResult(output_dir=output_dir, overlay_paths=overlay_paths)