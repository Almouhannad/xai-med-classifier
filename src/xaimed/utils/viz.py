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


def save_metric_history_plot(
    history: dict[str, list[float]],
    out_path: Path,
    split_name: str,
) -> None:
    """Save a compact per-split training history chart with loss and accuracy trends."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_accuracy = history.get("train_accuracy", [])
    val_accuracy = history.get("val_accuracy", [])

    epochs = max(len(train_loss), len(val_loss), len(train_accuracy), len(val_accuracy))
    chart_width = max(480, 72 * max(1, epochs))
    panel_height = 220
    margin_left = 52
    margin_right = 24
    margin_top = 24
    margin_bottom = 34
    panel_gap = 24

    width = chart_width
    height = panel_height * 2 + panel_gap + margin_top * 2 + margin_bottom * 2
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    def draw_panel(y_offset: int, title: str, train: list[float], val: list[float], y_min: float, y_max: float) -> None:
        x0 = margin_left
        y0 = y_offset + margin_top
        x1 = width - margin_right
        y1 = y_offset + panel_height - margin_bottom
        draw.rectangle((x0, y0, x1, y1), outline="black", width=1)
        draw.text((x0, y_offset + 2), title, fill="black")

        if epochs <= 0:
            draw.text((x0 + 8, y0 + 8), "No history data", fill="black")
            return

        draw.text((8, y0 - 6), f"{y_max:.3f}", fill="black")
        draw.text((8, y1 - 10), f"{y_min:.3f}", fill="black")

        def project(idx: int, value: float) -> tuple[float, float]:
            x = x0 if epochs == 1 else x0 + (idx / (epochs - 1)) * (x1 - x0)
            denom = max(y_max - y_min, 1e-8)
            y = y1 - ((value - y_min) / denom) * (y1 - y0)
            return x, y

        def draw_series(values: list[float], color: tuple[int, int, int]) -> None:
            if not values:
                return
            points = [project(i, float(values[i])) for i in range(len(values))]
            if len(points) == 1:
                px, py = points[0]
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
                return
            draw.line(points, fill=color, width=2)
            for px, py in points:
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)

        draw_series(train, (0, 102, 204))
        draw_series(val, (204, 102, 0))

        draw.rectangle((x0, y1 + 8, x0 + 14, y1 + 20), fill=(0, 102, 204))
        draw.text((x0 + 20, y1 + 8), "train", fill="black")
        draw.rectangle((x0 + 96, y1 + 8, x0 + 110, y1 + 20), fill=(204, 102, 0))
        draw.text((x0 + 116, y1 + 8), "val", fill="black")

    all_loss = [float(v) for v in (train_loss + val_loss)]
    loss_min = min(all_loss) if all_loss else 0.0
    loss_max = max(all_loss) if all_loss else 1.0
    if abs(loss_max - loss_min) < 1e-8:
        loss_max = loss_min + 1.0

    all_acc = [float(v) for v in (train_accuracy + val_accuracy)]
    acc_min = min(all_acc) if all_acc else 0.0
    acc_max = max(all_acc) if all_acc else 1.0
    acc_min = min(acc_min, 0.0)
    acc_max = max(acc_max, 1.0)

    header_height = 32

    draw.text((margin_left, 6), f"Training Curves ({split_name})", fill="black")

    draw_panel(header_height, "Loss", train_loss, val_loss, loss_min, loss_max)
    draw_panel(header_height + panel_height + panel_gap,
            "Accuracy", train_accuracy, val_accuracy, acc_min, acc_max)
    image.save(out_path)
