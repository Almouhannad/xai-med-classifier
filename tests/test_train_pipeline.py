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

    train_metrics = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"), grad_clip_norm=1.0)
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
            "optimizer": "adamw",
            "lr_scheduler": "none",
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
    assert result.best_epoch == 1
    assert result.best_monitor == "val_loss"
    assert result.best_mode == "min"
    assert result.best_score == result.best_val_loss
    assert result.epochs_ran == 1
    assert result.early_stopped is False

    saved = torch.load(result.last_checkpoint_path, map_location="cpu")
    assert "model_state_dict" in saved
    assert saved["epoch"] == 1
    assert saved["history"] == {
        "train_loss": [saved["train_metrics"]["loss"]],
        "train_accuracy": [saved["train_metrics"]["accuracy"]],
        "val_loss": [saved["val_metrics"]["loss"]],
        "val_accuracy": [saved["val_metrics"]["accuracy"]],
        "learning_rate": [config["train"]["lr"]],
    }


def test_run_training_early_stopping_stops_before_max_epochs(tmp_path, monkeypatch):
    config = {
        "model": {"name": "resnet18", "num_classes": 2},
        "train": {
            "epochs": 5,
            "lr": 0.001,
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "early_stopping": True,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
            "early_stopping_monitor": "val_loss",
            "early_stopping_mode": "min",
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

    monkeypatch.setattr("xaimed.train.train.train_one_epoch", lambda *args, **kwargs: {"loss": 0.5, "accuracy": 0.7})

    val_losses = iter([1.0, 1.2, 1.3, 1.4, 1.5])
    monkeypatch.setattr(
        "xaimed.train.train.validate_one_epoch",
        lambda *args, **kwargs: {"loss": next(val_losses), "accuracy": 0.6},
    )

    result = run_training(config)

    assert result.early_stopped is True
    assert result.epochs_ran == 2
    assert result.best_epoch == 1


def test_run_training_zero_epochs_returns_without_crashing(tmp_path):
    config = {
        "model": {"name": "resnet18", "num_classes": 2},
        "train": {
            "epochs": 0,
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

    assert result.epochs_ran == 0
    assert result.best_epoch == 0
    assert result.last_checkpoint_path.exists() is False
    assert result.best_checkpoint_path.exists() is False
