"""The files the app needs from the user: each check, the bundle that carries them, and every way a bundle can be bad."""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from celldform.demo import framework as fw
from celldform.demo import models as M
from tests.test_demo_session import fake_framework

SETTINGS = {"preprocessing": {"median_kernel": 3, "clahe_clip_limit": 2.0, "clahe_tile_grid": [8, 8], "morph_kernel_size": 3,
                              "target_size": [384, 384], "crop_size": 384, "crop_center": [579, 510]}}


def write_settings(path, **changes):
    cfg = json.loads(json.dumps(SETTINGS))
    for k, v in changes.items():
        if v is None:
            cfg["preprocessing"].pop(k, None)
        else:
            cfg["preprocessing"][k] = v
    Path(path).write_text(yaml.safe_dump(cfg))
    return str(path)


def tiny_unet(path, out=3, wrapped=False):
    from celldform.segmentation.unet import UNet
    m = UNet(1, 4, out)
    if wrapped:
        torch.save({"epoch": 9, "val_dsc": 0.9, "model_state_dict": m.state_dict(),
                    "optimizer_state_dict": {"state": {i: {"m": torch.zeros(1000)} for i in range(50)}}}, path)
    else:
        m.save(path)
    return str(path)


def good_files(tmp, wrapped=False):
    return (tiny_unet(tmp / "w.pt", wrapped=wrapped), write_settings(tmp / "c.yaml"), _save_framework(tmp / "f.joblib"))


def _save_framework(path):
    fw.save(fake_framework(), path)
    return str(path)


# ── the U-Net weights ───────────────────────────────────────────────────────────────────────────────────────────

def test_a_good_unet_file(tmp_path):
    c = M.check_unet(tiny_unet(tmp_path / "w.pt"))
    assert c.ok and "3 classes, base width 4" in c.message and len(c.sha256) == 64 and c.size > 0 and c.short_hash == c.sha256[:12]
    assert c.detail["base_features"] == 4


def test_unet_problems_are_explained(tmp_path):
    assert M.check_unet(None).message == "not loaded"
    assert "not found" in M.check_unet(tmp_path / "nope.pt").message
    (tmp_path / "empty.pt").write_bytes(b"")
    assert "is empty" in M.check_unet(tmp_path / "empty.pt").message
    (tmp_path / "a.txt").write_text("x")
    assert "is not a .pt/.pth file" in M.check_unet(tmp_path / "a.txt").message
    (tmp_path / "junk.pt").write_bytes(b"not a checkpoint")
    assert not M.check_unet(tmp_path / "junk.pt").ok
    torch.save({"w": torch.zeros(2)}, tmp_path / "other.pt")
    assert "not a celldform U-Net" in M.check_unet(tmp_path / "other.pt").message
    assert "binary" in M.check_unet(tiny_unet(tmp_path / "b.pt", out=1)).message


# ── the settings file ───────────────────────────────────────────────────────────────────────────────────────────

def test_good_settings(tmp_path):
    c = M.check_unet_settings(write_settings(tmp_path / "c.yaml"))
    assert c.ok and "384 px crop at (579, 510)" in c.message and c.detail["target_size"] == [384, 384]


def test_a_full_training_config_is_accepted_but_only_the_needed_keys_are_read(tmp_path):
    cfg = {**SETTINGS, "data": {"frames_dir": "data/secret/frames"}, "checkpoint_dir": "checkpoints/x", "split": {"seed": 42}}
    (tmp_path / "full.yaml").write_text(yaml.safe_dump(cfg))
    pp = M.load_unet_settings(tmp_path / "full.yaml")
    assert set(pp) == {"median_kernel", "clahe_clip_limit", "clahe_tile_grid", "morph_kernel_size", "target_size", "crop_size", "crop_center"}


@pytest.mark.parametrize("changes, text", [
    ({"median_kernel": 4}, "odd whole number"), ({"median_kernel": "3"}, "odd whole number"),
    ({"clahe_clip_limit": 0}, "positive"), ({"clahe_clip_limit": None}, "lacks clahe_clip_limit"),
    ({"clahe_tile_grid": [8]}, "two numbers"), ({"clahe_tile_grid": [0, 8]}, "positive"),
    ({"morph_kernel_size": -1}, "whole number"), ({"morph_kernel_size": None}, "lacks morph_kernel_size"),
    ({"target_size": [384, 380]}, "multiples of 16"), ({"target_size": "big"}, "two numbers"),
    ({"crop_size": 2000}, "up to 922"), ({"crop_size": 8}, "up to 922"),
    ({"crop_center": [5000, 10]}, "outside a 1228x922 frame"), ({"crop_center": [1]}, "two numbers"),
])
def test_settings_mistakes_are_named(tmp_path, changes, text):
    c = M.check_unet_settings(write_settings(tmp_path / "c.yaml", **changes))
    assert not c.ok and text in c.message


def test_settings_that_are_not_settings(tmp_path):
    (tmp_path / "a.yaml").write_text("just: a mapping")
    assert "no 'preprocessing' section" in M.check_unet_settings(tmp_path / "a.yaml").message
    (tmp_path / "b.yaml").write_text("- a\n- list")
    assert "no 'preprocessing' section" in M.check_unet_settings(tmp_path / "b.yaml").message
    (tmp_path / "c.yaml").write_text("a: [unclosed")
    assert "not valid YAML" in M.check_unet_settings(tmp_path / "c.yaml").message
    (tmp_path / "d.yaml").write_text("x: " + "y" * (M.MAX_YAML_BYTES + 10))
    assert "too large" in M.check_unet_settings(tmp_path / "d.yaml").message
    (tmp_path / "e.yaml").write_text("!!python/object/apply:os.system ['echo no']")      # yaml code execution is not possible
    assert not M.check_unet_settings(tmp_path / "e.yaml").ok


def test_no_crop_is_allowed(tmp_path):
    c = M.check_unet_settings(write_settings(tmp_path / "c.yaml", crop_size=None, crop_center=None))
    assert c.ok and "no crop" in c.message


def test_the_pair_is_tried_together_on_this_machine(tmp_path):
    c = M.smoke_test_unet(tiny_unet(tmp_path / "w.pt"), write_settings(tmp_path / "c.yaml"), device="cpu")
    assert c.ok and "runs on cpu" in c.message and c.detail["labels_shape"] == (922, 1228)
    bad = M.smoke_test_unet(tiny_unet(tmp_path / "w2.pt"), write_settings(tmp_path / "c.yaml"), device="not_a_device")
    assert not bad.ok and "do not work together" in bad.message


# ── the classifier ──────────────────────────────────────────────────────────────────────────────────────────────

def test_a_good_classifier(tmp_path):
    f = fake_framework()
    f.info = {"n_high": 9, "n_low": 6, "loocv_correct": 14, "loocv_auc": 0.94}
    fw.save(f, tmp_path / "c.joblib")
    c = M.check_classifier(tmp_path / "c.joblib")
    assert c.ok and "9 high + 6 low HER2 cells" in c.message and "14/15" in c.message and c.detail["loocv_correct"] == 14


def test_a_classifier_with_other_features_is_refused(tmp_path):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 40)
    step = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()).fit(rng.normal(size=(40, 5)), y)
    cell = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()).fit(rng.normal(size=(40, 3)), y)
    fw.save(fw.HER2Framework(step, cell, info={}), tmp_path / "other.joblib")
    c = M.check_classifier(tmp_path / "other.joblib")
    assert not c.ok and "trained on 5 step features and 3 cell features" in c.message and "cannot be used" in c.message


def test_classifier_files_that_are_not_classifiers(tmp_path):
    import joblib
    joblib.dump({"a": 1}, tmp_path / "dict.joblib")
    assert "holds a dict, not a HER2 framework" in M.check_classifier(tmp_path / "dict.joblib").message
    (tmp_path / "junk.joblib").write_bytes(b"junk")
    assert "could not be read" in M.check_classifier(tmp_path / "junk.joblib").message
    assert "not found" in M.check_classifier(tmp_path / "gone.joblib").message


def test_a_classifier_that_would_run_code_is_refused_unopened(tmp_path):
    import joblib
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))
    joblib.dump(Evil(), tmp_path / "evil.joblib")
    c = M.check_classifier(tmp_path / "evil.joblib")
    assert not c.ok and "not part of a HER2 classifier" in c.message and "was not opened" in c.message
    assert not marker.exists()
    with pytest.raises(fw.UnsafeFile):
        fw.load(tmp_path / "evil.joblib")
    # a sklearn object that is not part of a framework is refused too
    from sklearn.ensemble import RandomForestClassifier
    joblib.dump(RandomForestClassifier(n_estimators=2), tmp_path / "rf.joblib")
    assert "not part of a HER2 classifier" in M.check_classifier(tmp_path / "rf.joblib").message


def test_trusted_loading_is_still_possible_for_your_own_files(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    joblib.dump(RandomForestClassifier(n_estimators=2), tmp_path / "rf.joblib")
    with pytest.raises(ValueError, match="not a HER2 framework"):
        fw.load(tmp_path / "rf.joblib", trusted=True)


# ── the reference table ─────────────────────────────────────────────────────────────────────────────────────────

def test_reference_table(tmp_path):
    (tmp_path / "r.csv").write_text("her2,len_par_base,fluctuation\nhigh,30,0.01\nlow,35,0.02\n")
    c = M.check_reference(tmp_path / "r.csv")
    assert c.ok and "2 training cells" in c.message
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    assert "lacks the columns" in M.check_reference(tmp_path / "bad.csv").message
    (tmp_path / "junk.csv").write_bytes(b"\x00\x01\x02")
    assert not M.check_reference(tmp_path / "junk.csv").ok


# ── the bundle ──────────────────────────────────────────────────────────────────────────────────────────────────

def make(tmp_path, **kw):
    w, c, f = good_files(tmp_path, wrapped=True)
    return M.make_bundle(tmp_path / "bundle", unet=w, config=c, classifier=f, **kw)


def test_a_bundle_round_trips_and_every_file_passes_its_check(tmp_path):
    b = make(tmp_path, description="demo models")
    assert sorted(p.name for p in b.iterdir()) == ["celldform-models.json", "classifier.joblib", "unet.pt", "unet_settings.yaml"]
    files = M.open_bundle(b)
    assert set(files) == {"unet", "unet_config", "classifier"}
    assert all(M.check(role, p).ok for role, p in files.items())
    assert M.describe_bundle(b) == "demo models"
    mf = json.loads((b / M.MANIFEST).read_text())
    assert mf["format"] == 1 and mf["files"]["unet"]["sha256"] == M.sha256_file(b / "unet.pt")


def test_a_bundle_holds_weights_only_and_a_minimal_settings_file(tmp_path):
    w, c, f = good_files(tmp_path, wrapped=True)
    full = {**SETTINGS, "unet": {"in_channels": 1, "base_features": 4, "out_channels": 3}, "data": {"frames_dir": "data/private/frames"},
            "checkpoint_dir": "/home/someone/checkpoints/x", "split": {"train_cells": 11}}
    Path(c).write_text(yaml.safe_dump(full))
    b = M.make_bundle(tmp_path / "bundle", unet=w, config=c, classifier=f)
    text = (b / "unet_settings.yaml").read_text()
    assert "private" not in text and "/home/someone" not in text and "split" not in text and "checkpoint_dir" not in text
    assert set(yaml.safe_load(text)) == {"preprocessing", "unet"}
    sd = torch.load(b / "unet.pt", weights_only=True)
    assert "optimizer_state_dict" not in sd and "model_state_dict" not in sd and "enc1.block.0.weight" in sd      # a bare state dict
    assert (b / "unet.pt").stat().st_size < Path(w).stat().st_size                                              # smaller than the training file
    assert M.check_unet(b / "unet.pt").ok


def test_the_reference_table_is_optional_and_checked(tmp_path):
    (tmp_path / "r.csv").write_text("her2,len_par_base,fluctuation\nhigh,30,0.01\n")
    b = make(tmp_path, reference=tmp_path / "r.csv")
    assert "reference" in M.open_bundle(b)
    (tmp_path / "bad.csv").write_text("x\n1\n")
    w, c, f = good_files(tmp_path)
    with pytest.raises(M.BundleError, match="lacks the columns"):
        M.make_bundle(tmp_path / "b2", unet=w, config=c, classifier=f, reference=tmp_path / "bad.csv")


def test_a_bundle_cannot_be_made_with_personal_data_in_it(tmp_path):
    w, c, f = good_files(tmp_path)
    with pytest.raises(M.BundleError, match="description must not contain"):
        M.make_bundle(tmp_path / "b", unet=w, config=c, classifier=f, description="made by someone@corp.io")
    with pytest.raises(M.BundleError, match="description must not contain"):
        M.make_bundle(tmp_path / "b", unet=w, config=c, classifier=f, description=r"from C:\Users\alice\models")
    m = fake_framework()
    m.info = {"owner": "someone@corp.io"}
    fw.save(m, tmp_path / "leaky.joblib")
    with pytest.raises(M.BundleError, match="classifier's own record contains"):
        M.make_bundle(tmp_path / "b3", unet=w, config=c, classifier=tmp_path / "leaky.joblib")


def test_a_changed_file_is_caught_by_its_hash(tmp_path):
    b = make(tmp_path)
    data = bytearray((b / "unet.pt").read_bytes())
    data[len(data) // 2] ^= 0xFF
    (b / "unet.pt").write_bytes(bytes(data))                       # same size, one byte different
    with pytest.raises(M.BundleError, match="does not match its SHA-256.*was not used"):
        M.open_bundle(b)


def test_a_truncated_or_missing_file_is_caught(tmp_path):
    b = make(tmp_path)
    (b / "classifier.joblib").write_bytes(b"short")
    with pytest.raises(M.BundleError, match="wrong size"):
        M.open_bundle(b)
    (b / "classifier.joblib").unlink()
    with pytest.raises(M.BundleError, match="missing"):
        M.open_bundle(b)


def test_manifest_problems(tmp_path):
    b = make(tmp_path)
    mf = json.loads((b / M.MANIFEST).read_text())

    def write(m):
        (b / M.MANIFEST).write_text(json.dumps(m))
    write({**mf, "format": 99}); pytest.raises(M.BundleError, M.open_bundle, b)
    write({**mf, "files": {**mf["files"], "backdoor": mf["files"]["unet"]}})
    with pytest.raises(M.BundleError, match="unknown files: backdoor"):
        M.open_bundle(b)
    short = {k: v for k, v in mf["files"].items() if k != "classifier"}
    write({**mf, "files": short})
    with pytest.raises(M.BundleError, match="lacks: HER2 classifier"):
        M.open_bundle(b)
    evil = json.loads(json.dumps(mf)); evil["files"]["unet"]["name"] = "../outside.pt"
    write(evil)
    with pytest.raises(M.BundleError, match="unsafe file"):
        M.open_bundle(b)
    (b / M.MANIFEST).write_text("not json")
    with pytest.raises(M.BundleError, match="cannot be read"):
        M.open_bundle(b)
    (b / M.MANIFEST).unlink()
    with pytest.raises(M.BundleError, match="not a model bundle"):
        M.open_bundle(b)
    with pytest.raises(M.BundleError, match="neither a folder nor a .zip"):
        M.open_bundle(tmp_path / "nothing")


def test_a_zip_bundle_is_unpacked_to_the_cache_and_verified(tmp_path):
    b = make(tmp_path)
    z = M.pack_zip(b, tmp_path / "models.zip")
    files = M.open_bundle(z, cache_dir=tmp_path / "cache")
    assert all(tmp_path / "cache" in p.parents for p in files.values()) and all(M.check(r, p).ok for r, p in files.items())
    assert M.describe_bundle(z) == "" or isinstance(M.describe_bundle(z), str)
    files2 = M.open_bundle(z, cache_dir=tmp_path / "cache")           # opening again replaces the cache cleanly
    assert files2["unet"].read_bytes() == files["unet"].read_bytes()


def test_a_tampered_zip_is_refused(tmp_path):
    b = make(tmp_path)
    z = tmp_path / "t.zip"
    with zipfile.ZipFile(z, "w") as out:
        for f in b.iterdir():
            data = f.read_bytes()
            if f.name == "classifier.joblib":
                data = data[:-1] + bytes([data[-1] ^ 1])
            out.writestr(f.name, data)
    with pytest.raises(M.BundleError, match="wrong size|SHA-256"):
        M.open_bundle(z, cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("name", ["../escape.txt", "/abs/escape.txt", "sub/dir.txt", "..\\win.txt", "a/../../b"])
def test_a_zip_cannot_write_outside_the_cache(tmp_path, name):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as out:
        out.writestr(name, "pwned")
        out.writestr(M.MANIFEST, "{}")
    with pytest.raises(M.BundleError, match="unsafe entry"):
        M.open_bundle(z, cache_dir=tmp_path / "cache" / "deep")
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path / "cache" / "escape.txt").exists()
    assert not any(p.name in ("escape.txt", "win.txt", "b") for p in tmp_path.rglob("*") if p.is_file() and p != z)


def test_a_zip_with_a_symlink_or_too_many_files_is_refused(tmp_path):
    z = tmp_path / "link.zip"
    with zipfile.ZipFile(z, "w") as out:
        info = zipfile.ZipInfo("unet.pt")
        info.external_attr = 0o120777 << 16
        out.writestr(info, "/etc/passwd")
    with pytest.raises(M.BundleError, match="unsafe entry"):
        M.open_bundle(z, cache_dir=tmp_path / "c1")
    many = tmp_path / "many.zip"
    with zipfile.ZipFile(many, "w") as out:
        for i in range(25):
            out.writestr(f"f{i}.txt", "x")
    with pytest.raises(M.BundleError, match="too many files"):
        M.open_bundle(many, cache_dir=tmp_path / "c2")
    (tmp_path / "bad.zip").write_bytes(b"this is not a zip")
    with pytest.raises(M.BundleError, match="not a valid zip"):
        M.open_bundle(tmp_path / "bad.zip")


def test_a_zip_that_claims_to_be_huge_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MAX_BUNDLE_BYTES", 100)
    z = tmp_path / "big.zip"
    with zipfile.ZipFile(z, "w") as out:
        out.writestr("a.bin", b"x" * 1000)
    with pytest.raises(M.BundleError, match="too large"):
        M.open_bundle(z, cache_dir=tmp_path / "c")


def test_missing_roles_are_listed_by_name():
    assert M.missing_roles({}) == ["U-Net weights", "U-Net settings", "HER2 classifier"]
    ok = M.Check("unet", None, True, "")
    bad = M.Check("unet_config", None, False, "no")
    assert M.missing_roles({"unet": ok, "unet_config": bad}) == ["U-Net settings", "HER2 classifier"]
    assert M.missing_roles({r: M.Check(r, None, True, "") for r in M.REQUIRED}) == []


def test_the_real_files_make_a_working_bundle(tmp_path):
    root = Path(__file__).resolve().parents[1]
    unet = root / "checkpoints/randswap/experiment_0_cv_baseline/cv_baseline/fold_deploy/unet_deploy.pt"
    cfg = unet.parent / "config.yaml"
    clf = root / "checkpoints/classification/nnunet1000_loocv/framework/her2_framework.joblib"
    if not (unet.is_file() and cfg.is_file() and clf.is_file()):
        pytest.skip("the trained files are not on this machine")
    b = M.make_bundle(tmp_path / "bundle", unet=unet, config=cfg, classifier=clf, description="deployment U-Net + 15-cell framework")
    assert (b / "unet.pt").stat().st_size < unet.stat().st_size / 2.5                  # weights only
    files = M.open_bundle(b)
    assert all(M.check(r, p).ok for r, p in files.items())
    assert M.smoke_test_unet(files["unet"], files["unet_config"], device="cpu").ok


def test_the_bundle_command_line(tmp_path, capsys):
    w, c, f = good_files(tmp_path)
    assert M.main(["make", "--unet", w, "--config", c, "--classifier", f, "--out", str(tmp_path / "b"),
                   "--zip", str(tmp_path / "b.zip"), "--description", "demo"]) == 0
    out = capsys.readouterr().out
    assert "unet.pt" in out and "sha256" in out and "bundle written to" in out and (tmp_path / "b.zip").is_file()
    assert M.main(["verify", str(tmp_path / "b")]) == 0
    assert "bundle is usable" in capsys.readouterr().out
    assert M.main(["verify", str(tmp_path / "b.zip")]) == 0
    (tmp_path / "b" / "unet.pt").write_bytes(b"changed")
    assert M.main(["verify", str(tmp_path / "b")]) == 1
    assert "error:" in capsys.readouterr().err
    assert M.main(["make", "--unet", w, "--config", c, "--classifier", f, "--out", str(tmp_path / "c"), "--description", "me@corp.io"]) == 1
    assert "description must not contain" in capsys.readouterr().err
