"""Tests for celldform.config — typed config loader."""

import os
from pathlib import Path

import pytest
import yaml

from celldform.config import CelldformConfig, load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_minimal(path: Path, overrides: dict = None) -> None:
    """Write a minimal valid config YAML to *path*, with optional overrides."""
    base = {
        "data": {"frames_dir": "data/frames", "masks_dir": "data/masks"},
        "unet": {"in_channels": 1, "base_features": 32},
        "training": {"epochs": 5, "batch_size": 4, "lr": 0.001},
    }
    if overrides:
        for section, values in overrides.items():
            if isinstance(values, dict) and section in base:
                base[section].update(values)
            else:
                base[section] = values
    with open(path, "w") as fh:
        yaml.dump(base, fh)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def test_returns_celldform_config(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        _write_minimal(p)
        conf = load(p)
        assert isinstance(conf, CelldformConfig)

    def test_data_fields_populated(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        _write_minimal(p)
        conf = load(p)
        assert conf.data.frames_dir == "data/frames"
        assert conf.data.masks_dir == "data/masks"

    def test_optional_fields_get_defaults(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        _write_minimal(p)
        conf = load(p)
        assert conf.training.optimizer == "adamw"
        assert conf.training.scheduler == "reduce_on_plateau"
        assert conf.training.seed == 42
        assert conf.data.val_split == 0.15
        assert conf.data.test_split == 0.10
        assert conf.resume.checkpoint is None
        assert conf.device is None

    def test_resume_block_parsed(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        ckpt = tmp_path / "unet_best.pt"
        ckpt.touch()
        _write_minimal(p, {"resume": {"checkpoint": str(ckpt), "load_optimizer": False}})
        conf = load(p)
        assert conf.resume.checkpoint == str(ckpt)
        assert conf.resume.load_optimizer is False
        assert conf.resume.load_scheduler is True  # default

    def test_env_var_expansion_in_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_DATA", str(tmp_path))
        p = tmp_path / "cfg.yaml"
        raw = {
            "data": {"frames_dir": "$MY_DATA/frames", "masks_dir": "$MY_DATA/masks"},
            "unet": {},
            "training": {"epochs": 1, "batch_size": 1, "lr": 0.001},
        }
        with open(p, "w") as fh:
            yaml.dump(raw, fh)
        conf = load(p)
        assert conf.data.frames_dir == str(tmp_path / "frames")

    def test_default_yaml_loads_cleanly(self):
        conf = load("configs/default.yaml")
        assert isinstance(conf, CelldformConfig)
        assert conf.unet.in_channels == 1
        assert conf.biomechanics.metric == "aspect_ratio"


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_missing_data_section_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        with open(p, "w") as fh:
            yaml.dump({"unet": {}, "training": {"epochs": 1, "batch_size": 1, "lr": 0.001}}, fh)
        with pytest.raises(ValueError, match="data"):
            load(p)

    def test_missing_frames_dir_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        with open(p, "w") as fh:
            yaml.dump({
                "data": {"masks_dir": "data/masks"},
                "unet": {},
                "training": {"epochs": 1, "batch_size": 1, "lr": 0.001},
            }, fh)
        with pytest.raises(ValueError, match="frames_dir"):
            load(p)

    def test_invalid_optimizer_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        _write_minimal(p, {"training": {"optimizer": "rmsprop"}})
        with pytest.raises(ValueError, match="optimizer"):
            load(p)

    def test_invalid_scheduler_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        _write_minimal(p, {"training": {"scheduler": "step_lr"}})
        with pytest.raises(ValueError, match="scheduler"):
            load(p)

    def test_split_sum_exceeds_one_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        _write_minimal(p, {"data": {"val_split": 0.7, "test_split": 0.4}})
        with pytest.raises(ValueError, match="val_split"):
            load(p)

    def test_nonexistent_resume_checkpoint_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        _write_minimal(p, {"resume": {"checkpoint": "/nonexistent/path.pt"}})
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            load(p)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "no_such_file.yaml")
