from __future__ import annotations

import sys
from types import ModuleType

from xaimed.data.medmnist import (
    DataModuleError,
    build_medmnist_dataloaders,
    download_medmnist,
    load_medmnist_dataset,
    resolve_data_dir,
)

import pytest


class FakeDataset:
    def __init__(self, split, root, download, transform, as_rgb):
        self.split = split
        self.root = root
        self.download = download
        self.transform = transform
        self.as_rgb = as_rgb
        self._items = {
            "train": [0, 1, 2, 3],
            "val": [0, 1],
            "test": [0, 1, 2],
        }[split]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        value = self._items[idx]
        if self.transform is not None:
            value = self.transform(value)
        return value, idx


class FakeDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __iter__(self):
        batch = [self.dataset[i] for i in range(min(self.batch_size, len(self.dataset)))]
        yield batch


def _install_fake_dependencies(monkeypatch):
    fake_medmnist = ModuleType("medmnist")
    fake_medmnist.INFO = {"pathmnist": {"python_class": "PathMNIST"}}
    fake_medmnist.PathMNIST = FakeDataset

    fake_torch = ModuleType("torch")
    fake_torch_utils = ModuleType("torch.utils")
    fake_torch_utils_data = ModuleType("torch.utils.data")
    fake_torch_utils_data.DataLoader = FakeDataLoader

    fake_torchvision = ModuleType("torchvision")
    fake_transforms = ModuleType("torchvision.transforms")

    class _Identity:
        def __call__(self, value):
            return value

    fake_transforms.Compose = lambda _steps: _Identity()
    fake_transforms.Resize = lambda size: ("resize", size)
    fake_transforms.ToTensor = lambda: "tensor"
    fake_transforms.Normalize = lambda mean, std: ("norm", mean, std)
    fake_torchvision.transforms = fake_transforms

    monkeypatch.setitem(sys.modules, "medmnist", fake_medmnist)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_torch_utils_data)
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", fake_transforms)


def test_download_medmnist_downloads_all_splits(monkeypatch, tmp_path):
    _install_fake_dependencies(monkeypatch)

    counts = download_medmnist(dataset_name="pathmnist", data_dir=tmp_path)

    assert counts == {"train": 4, "val": 2, "test": 3}


def test_build_medmnist_dataloaders_smoke(monkeypatch, tmp_path):
    _install_fake_dependencies(monkeypatch)

    dataloaders = build_medmnist_dataloaders(
        dataset_name="pathmnist",
        data_dir=tmp_path,
        batch_size=2,
    )

    batch = next(iter(dataloaders["train"]))

    assert set(dataloaders) == {"train", "val", "test"}
    assert len(batch) == 2


def test_load_medmnist_dataset_rejects_invalid_split(monkeypatch, tmp_path):
    _install_fake_dependencies(monkeypatch)

    with pytest.raises(DataModuleError):
        load_medmnist_dataset(
            dataset_name="pathmnist",
            split="training",
            data_dir=tmp_path,
        )

def test_resolve_data_dir_relative_anchors_to_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = resolve_data_dir("data")

    assert resolved.name == "data"
    assert (resolved.parent / "pyproject.toml").exists()


def test_resolve_data_dir_absolute_kept_as_is(tmp_path):
    resolved = resolve_data_dir(tmp_path)

    assert resolved == tmp_path


def test_load_medmnist_dataset_rejects_corrupted_archive(monkeypatch, tmp_path):
    _install_fake_dependencies(monkeypatch)
    (tmp_path / "pathmnist.npz").write_bytes(b"not-a-zip")

    with pytest.raises(DataModuleError, match="Corrupted MedMNIST archive"):
        load_medmnist_dataset(
            dataset_name="pathmnist",
            split="train",
            data_dir=tmp_path,
            download=False,
        )
