"""Analysis settings (celldform.analysis_config) and the shipped configs/analysis/*.yaml."""
import pytest

from celldform import analysis_config as ac


@pytest.mark.parametrize("section", sorted(ac.SECTIONS))
def test_shipped_files_equal_the_defaults(section):
    """The YAML files hold the values the reported results were made with -- the code defaults."""
    assert ac.as_dict(ac.load(section)) == ac.as_dict(ac.SECTIONS[section]())


def test_override_and_nested_values(tmp_path):
    f = tmp_path / "cnn.yaml"
    f.write_text("epochs: 12\ntrainable_blocks: [layer3, layer4]\naugment: {rot90: false}\n")
    cfg = ac.load("cnn", f)
    assert cfg.epochs == 12 and cfg.trainable_blocks == ("layer3", "layer4")
    assert cfg.augment.rot90 is False and cfg.augment.flips is True      # untouched keys keep their default
    assert cfg.lr == ac.CNNConfig().lr


def test_misspelt_key_is_an_error(tmp_path):
    f = tmp_path / "features.yaml"
    f.write_text("frame_checks: {max_trap_dst: 40}\n")
    with pytest.raises(KeyError, match="max_trap_dst"):
        ac.load("features", f)


def test_missing_explicit_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ac.load("regression", tmp_path / "nope.yaml")


def test_dump_round_trips(tmp_path):
    cfg = ac.load("classifiers")
    ac.dump(cfg, tmp_path / "used.yaml")
    assert ac.as_dict(ac.load("classifiers", tmp_path / "used.yaml")) == ac.as_dict(cfg)
