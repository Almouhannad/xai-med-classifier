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