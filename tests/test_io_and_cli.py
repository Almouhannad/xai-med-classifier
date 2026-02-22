from __future__ import annotations

import pytest

from xaimed.cli import main
from xaimed.utils.io import ConfigError, load_yaml_config


def test_load_yaml_config_reads_mapping(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("seed: 123\nmodel:\n  name: resnet18\n", encoding="utf-8")

    data = load_yaml_config(cfg)

    assert data == {"seed": 123, "model": {"name": "resnet18"}}


def test_load_yaml_config_rejects_non_mapping(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_yaml_config(cfg)


def test_cli_main_with_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("seed: 42\n", encoding="utf-8")

    result = main(["--config", str(cfg)])

    assert result == 0

def test_cli_download_data_command(monkeypatch, capsys):
    called = {}

    def _fake_download(dataset_name, data_dir):
        called["dataset_name"] = dataset_name
        called["data_dir"] = data_dir
        return {"train": 1, "val": 1, "test": 1}

    monkeypatch.setattr("xaimed.data.medmnist.download_medmnist", _fake_download)

    result = main(["download-data", "--dataset", "pathmnist", "--data-dir", "tmp-data"])

    out = capsys.readouterr().out
    assert result == 0
    assert called == {"dataset_name": "pathmnist", "data_dir": "tmp-data"}
    assert "Downloaded 'pathmnist' into tmp-data" in out

def test_cli_train_command(monkeypatch, capsys):
    class _Result:
        best_checkpoint_path = "best.pt"
        last_checkpoint_path = "last.pt"
        best_epoch = 1
        best_score = 0.123456
        best_monitor = "val_loss"
        best_mode = "min"
        epochs_ran = 1
        early_stopped = False

    def _fake_run_training(config):
        assert config == {"train": {"epochs": 1, "early_stopping": False}}
        return _Result()

    monkeypatch.setattr("xaimed.train.run_training", _fake_run_training)

    result = main(["--config", "tests/fixtures/train_config.yaml", "train"])

    out = capsys.readouterr().out
    assert result == 0
    assert "Best checkpoint: best.pt" in out
    assert "Last checkpoint: last.pt" in out
    assert "Best epoch: 1" in out
    assert "Best score (val_loss, min): 0.123456" in out
    assert "Epochs ran: 1" in out
    assert "Early stopped: False" in out

def test_cli_eval_command(monkeypatch, capsys):
    class _FailureGallery:
        csv_path = "failure_gallery_selection.csv"
        high_conf_wrong_grid_path = "high_confidence_wrongs_grid.png"
        low_conf_correct_grid_path = "low_confidence_corrects_grid.png"

    class _EvalResult:
        metrics_path = "metrics.json"
        confusion_matrix_path = "cm.png"
        training_curves_path = "training_curves.png"
        metrics = {"accuracy": 0.5, "macro_f1": 0.4}
        failure_gallery = _FailureGallery()

    def _fake_run_evaluation(config):
        assert config == {"eval": {"split": "val"}}
        return _EvalResult()

    monkeypatch.setattr("xaimed.eval.run_evaluation", _fake_run_evaluation)

    result = main(["--config", "tests/fixtures/eval_config.yaml", "eval"])

    out = capsys.readouterr().out
    assert result == 0
    assert "Metrics saved: metrics.json" in out
    assert "Confusion matrix saved: cm.png" in out
    assert "Accuracy: 0.5000" in out
    assert "Macro F1: 0.4000" in out
    assert "Failure gallery CSV saved: failure_gallery_selection.csv" in out
    assert "Training curves saved: training_curves.png" in out


def test_cli_report_command(monkeypatch, capsys):
    called = {}

    def _fake_build_report(output_dir):
        called["output_dir"] = str(output_dir)
        return "artifacts/report/README.md"

    monkeypatch.setattr("xaimed.reporting.build_report", _fake_build_report)

    result = main(["report"])

    out = capsys.readouterr().out
    assert result == 0
    assert called["output_dir"] == "artifacts/report"
    assert "Report scaffold saved: artifacts/report/README.md" in out

def test_cli_explain_command(monkeypatch, capsys):
    class _ExplainResult:
        output_dir = "artifacts/explain/test"
        overlay_paths = [
            "artifacts/explain/test/sample_000_t0_p0_gradcam.png",
            "artifacts/explain/test/sample_001_t1_p0_gradcam.png",
        ]

    def _fake_run_explain(config):
        assert config == {"explain": {"max_samples": 2}}
        return _ExplainResult()

    monkeypatch.setattr("xaimed.xai.explain.run_explain", _fake_run_explain)

    result = main(["--config", "tests/fixtures/explain_config.yaml", "explain"])

    out = capsys.readouterr().out
    assert result == 0
    assert "Explain artifacts directory: artifacts/explain/test" in out
    assert "Grad-CAM overlay saved: artifacts/explain/test/sample_000_t0_p0_gradcam.png" in out