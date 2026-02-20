"""Visualization helpers for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
import torch


def save_confusion_matrix_plot(matrix: torch.Tensor, out_path: Path, title: str = "Confusion Matrix") -> None:
    """Save an annotated confusion matrix heatmap-like image without matplotlib dependency."""
    del title  # reserved for future style customization
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = matrix.shape[0]
    cell_size = 64
    image_size = max(1, num_classes) * cell_size
    image = Image.new("RGB", (image_size, image_size), color="white")
    draw = ImageDraw.Draw(image)

    max_value = int(matrix.max().item()) if matrix.numel() else 1
    max_value = max(max_value, 1)

    for row in range(num_classes):
        for col in range(num_classes):
            value = int(matrix[row, col].item())
            intensity = int(255 * (1 - (value / max_value)))
            fill = (intensity, intensity, 255)

            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline="black")

            text = str(value)
            text_box = draw.textbbox((0, 0), text)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            draw.text(
                (x0 + (cell_size - text_w) / 2, y0 + (cell_size - text_h) / 2),
                text,
                fill="black",
            )

    image.save(out_path)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3:
        raise ValueError("Expected CHW image tensor")

    channels = image.shape[0]
    if channels not in (1, 3):
        image = image[:3] if channels > 3 else image.repeat(3 // channels + 1, 1, 1)[:3]

    image = image.detach().cpu().float()
    image = image - image.min()
    max_val = image.max()
    if max_val > 0:
        image = image / max_val
    image = (image * 255).clamp(0, 255).byte()

    if image.shape[0] == 1:
        array = image[0].numpy()
        return Image.fromarray(array, mode="L").convert("RGB")

    array = image.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def save_image_grid(
    images: torch.Tensor,
    indices: list[int],
    out_path: Path,
    title: str,
    cols: int = 4,
    cell_size: int = 128,
) -> None:
    """Save a simple tiled grid image for selected sample indices."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not indices:
        canvas = Image.new("RGB", (cols * cell_size, cell_size), color="white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{title}: no samples", fill="black")
        canvas.save(out_path)
        return

    rows = (len(indices) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_size, rows * cell_size + 24), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill="black")

    for tile_idx, sample_idx in enumerate(indices):
        row = tile_idx // cols
        col = tile_idx % cols
        pil_image = _tensor_to_pil(images[sample_idx]).resize((cell_size, cell_size))

        x = col * cell_size
        y = row * cell_size + 24
        canvas.paste(pil_image, (x, y))

        draw.rectangle((x, y, x + cell_size, y + cell_size), outline="black")
        draw.text((x + 4, y + 4), f"#{sample_idx}", fill="yellow")

    canvas.save(out_path)
