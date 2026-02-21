from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xaimed.train.loops import train_one_epoch, validate_one_epoch
from xaimed.train.train import run_training


def test_train_and_validate_loops_return_metrics():
    features = torch.randn(8, 3, 8, 8)
    labels = torch.randint(low=0, high=2, size=(8,))
    loader = DataLoader(TensorDataset(features, labels), batch_size=4)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_metrics = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    val_metrics = validate_one_epoch(model, loader, criterion, torch.device("cpu"))

    assert set(train_metrics) == {"loss", "accuracy"}
    assert set(val_metrics) == {"loss", "accuracy"}
    assert train_metrics["loss"] >= 0.0
    assert 0.0 <= val_metrics["accuracy"] <= 1.0


def test_run_training_saves_best_and_last_checkpoint(tmp_path):
    config = {
        "model": {"name": "resnet18", "num_classes": 2},
        "train": {
            "epochs": 1,
            "lr": 0.001,
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "ckpts"),
        },
        "data": {
            "use_fake_data": True,
            "image_size": 32,
            "batch_size": 4,
            "train_samples": 8,
            "val_samples": 4,
            "num_workers": 0,
        },
    }

    result = run_training(config)

    assert result.best_checkpoint_path.exists()
    assert result.last_checkpoint_path.exists()
    saved = torch.load(result.last_checkpoint_path, map_location="cpu")
    assert "model_state_dict" in saved
    assert saved["epoch"] == 1
    assert saved["history"] == {
        "train_loss": [saved["train_metrics"]["loss"]],
        "train_accuracy": [saved["train_metrics"]["accuracy"]],
        "val_loss": [saved["val_metrics"]["loss"]],
        "val_accuracy": [saved["val_metrics"]["accuracy"]],
    }
