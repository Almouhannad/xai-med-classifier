"""MedMNIST dataset loading and transform helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

VALID_SPLITS = ("train", "val", "test")


class DataModuleError(RuntimeError):
    """Raised for invalid data module usage or missing dependencies."""


def _load_medmnist_class(dataset_name: str):
    """Resolve a MedMNIST dataset class from its registry entry."""
    try:
        from medmnist import INFO
        import medmnist
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
        raise DataModuleError(
            "medmnist is required for dataset loading. Install it with `pip install medmnist`."
        ) from exc

    dataset_key = dataset_name.lower()
    if dataset_key not in INFO:
        available = ", ".join(sorted(INFO))
        raise DataModuleError(
            f"Unknown MedMNIST dataset '{dataset_name}'. Available datasets: {available}."
        )

    python_class_name = INFO[dataset_key]["python_class"]
    return getattr(medmnist, python_class_name)


def build_transforms(image_size: int = 224):
    """Build a default transform pipeline for MedMNIST images."""
    try:
        from torchvision import transforms
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
        raise DataModuleError(
            "torchvision is required for image transforms. Install it with `pip install torchvision`."
        ) from exc

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def load_medmnist_dataset(
    dataset_name: str,
    split: str,
    data_dir: str | Path,
    download: bool = False,
    transform: Any | None = None,
):
    """Load a MedMNIST dataset split."""
    if split not in VALID_SPLITS:
        allowed = ", ".join(VALID_SPLITS)
        raise DataModuleError(f"Invalid split '{split}'. Expected one of: {allowed}.")

    dataset_class = _load_medmnist_class(dataset_name)
    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    return dataset_class(
        split=split,
        root=str(data_root),
        download=download,
        transform=transform,
        as_rgb=True,
    )


def build_medmnist_dataloaders(
    dataset_name: str,
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
):
    """Build train/val/test dataloaders for MedMNIST."""
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
        raise DataModuleError(
            "torch is required to build dataloaders. Install it with `pip install torch`."
        ) from exc

    transform = build_transforms(image_size=image_size)

    train_dataset = load_medmnist_dataset(
        dataset_name=dataset_name,
        split="train",
        data_dir=data_dir,
        download=True,
        transform=transform,
    )
    val_dataset = load_medmnist_dataset(
        dataset_name=dataset_name,
        split="val",
        data_dir=data_dir,
        download=True,
        transform=transform,
    )
    test_dataset = load_medmnist_dataset(
        dataset_name=dataset_name,
        split="test",
        data_dir=data_dir,
        download=True,
        transform=transform,
    )

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def download_medmnist(dataset_name: str, data_dir: str | Path) -> dict[str, int]:
    """Download all MedMNIST splits and return split sizes."""
    counts: dict[str, int] = {}
    for split in VALID_SPLITS:
        dataset = load_medmnist_dataset(
            dataset_name=dataset_name,
            split=split,
            data_dir=data_dir,
            download=True,
            transform=None,
        )
        counts[split] = len(dataset)
    return counts