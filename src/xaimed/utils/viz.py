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
