"""Loading an nnU-Net from its single .pth file: safe reading, the prepared folder, the checks, bundles, the window, the CLI."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from celldform.demo import models as M
from celldform.demo import segment as seg
from celldform.demo.run import protocol_caveats
from tests.test_demo_models import _save_framework, tiny_unet, write_settings
from tests.test_demo_run import DiscSegmenter, make_recording
from tests.test_demo_session import fake_framework

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "checkpoints/randswap/experiment_4_loocv/nnunet/Dataset502_CellTrapLOOCV/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth"
FRAME = sorted((ROOT / "data/frames/03_mask_factory").glob("high_cell2_*.jpg"))[3:4]
real = pytest.mark.skipif(not (REAL.is_file() and FRAME), reason="the trained nnU-Net / factory frame are not on this machine")


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    """The prepared nnU-Net folders and unpacked bundles go in a folder of the test's own."""
    monkeypatch.setenv("CELLDFORM_HOME", str(tmp_path / "home"))


def fake_nn(path, conf="2d", channels=None, labels=None, optimizer=True, **extra):
    """A checkpoint with nnU-Net's structure and no real network: enough to test reading, checking and preparing."""
    ck = {"network_weights": {"w": torch.zeros(4)}, "trainer_name": "nnUNetTrainer", "inference_allowed_mirroring_axes": (0, 1),
          "init_args": {"plans": {"dataset_name": "DatasetX", "configurations": {"2d": {}}}, "configuration": conf, "fold": "all",
                        "dataset_json": {"channel_names": channels if channels is not None else {"0": "brightfield"},
                                         "labels": labels if labels is not None else {"background": 0, "trapped_cell": 1, "decoy": 2}},
                        "device": "cuda"},
          "logging": {"loss": [np.float32(0.5), np.float64(0.25)]}, "current_epoch": np.int64(7)}
    if optimizer:
        ck["optimizer_state"] = {"state": {i: {"m": torch.zeros(5000)} for i in range(40)}}
    ck.update(extra)
    torch.save(ck, path)
    return Path(path)


# ── reading a checkpoint safely ─────────────────────────────────────────────────────────────────────────────────

def test_a_checkpoint_with_numpy_scalars_in_its_log_is_read(tmp_path):
    ck = seg.read_nnunet_checkpoint(fake_nn(tmp_path / "n.pth"))
    assert set(ck) >= {"network_weights", "init_args", "trainer_name"} and ck["init_args"]["configuration"] == "2d"


def test_inspect_reports_what_the_app_needs(tmp_path):
    info = seg.inspect_nnunet(seg.read_nnunet_checkpoint(fake_nn(tmp_path / "n.pth")))
    assert info["configuration"] == "2d" and info["n_classes"] == 3 and info["labels"] == {"background": 0, "trapped_cell": 1, "decoy": 2}
    assert info["trainer"] == "nnUNetTrainer" and tuple(info["mirroring"]) == (0, 1)
    two = seg.inspect_nnunet(seg.read_nnunet_checkpoint(fake_nn(tmp_path / "t.pth", labels={"background": 0, "cell": 1})))
    assert two["n_classes"] == 2


def test_a_file_that_would_run_code_is_refused_and_runs_nothing(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))
    torch.save({"network_weights": {}, "init_args": Evil(), "trainer_name": "x"}, tmp_path / "evil.pth")
    with pytest.raises(ValueError, match="which an nnU-Net checkpoint never contains; the file was not opened"):
        seg.read_nnunet_checkpoint(tmp_path / "evil.pth")
    assert not marker.exists()
    import fractions
    torch.save({"x": fractions.Fraction(1, 3)}, tmp_path / "frac.pth")
    with pytest.raises(ValueError, match="fractions.Fraction.*never contains"):
        seg.read_nnunet_checkpoint(tmp_path / "frac.pth")


@pytest.mark.parametrize("content, text", [
    (b"not a checkpoint", "not a PyTorch checkpoint"),
])
def test_junk_is_refused(tmp_path, content, text):
    (tmp_path / "j.pth").write_bytes(content)
    with pytest.raises(ValueError, match=text):
        seg.read_nnunet_checkpoint(tmp_path / "j.pth")


def test_files_that_are_not_nnunet_checkpoints(tmp_path):
    torch.save([1, 2, 3], tmp_path / "list.pth")
    with pytest.raises(ValueError, match="not an nnU-Net checkpoint"):
        seg.read_nnunet_checkpoint(tmp_path / "list.pth")
    torch.save({"model_state_dict": {}}, tmp_path / "unet.pth")
    with pytest.raises(ValueError, match="lacks network_weights, init_args, trainer_name"):
        seg.read_nnunet_checkpoint(tmp_path / "unet.pth")
    torch.save({"network_weights": {}, "trainer_name": "t", "init_args": {"plans": {}}}, tmp_path / "half.pth")
    with pytest.raises(ValueError, match="training record is incomplete"):
        seg.read_nnunet_checkpoint(tmp_path / "half.pth")
    with pytest.raises(FileNotFoundError):
        seg.read_nnunet_checkpoint(tmp_path / "nope.pth")


@pytest.mark.parametrize("kw, text", [
    ({"conf": "3d_fullres"}, "'3d_fullres' nnU-Net; the app needs a 2d one"),
    ({"channels": {"0": "a", "1": "b"}}, "expects 2 input channels"),
    ({"labels": {"background": 0, "cell": 2}}, "needs 0 = background and 1 = the trapped cell"),
    ({"labels": {"background": 0, "cell": [1, 2]}}, "region-based labels"),
])
def test_checkpoints_the_app_cannot_use(tmp_path, kw, text):
    ck = seg.read_nnunet_checkpoint(fake_nn(tmp_path / "n.pth", **kw))
    with pytest.raises(ValueError, match=text):
        seg.inspect_nnunet(ck, "n.pth")


def test_an_old_pytorch_is_explained(tmp_path, monkeypatch):
    f = fake_nn(tmp_path / "n.pth")
    monkeypatch.delattr(torch.serialization, "safe_globals")
    with pytest.raises(ValueError, match="needs PyTorch 2.5 or newer"):
        seg.read_nnunet_checkpoint(f)


# ── the prepared folder ─────────────────────────────────────────────────────────────────────────────────────────

def test_the_folder_nnunet_loads_from_is_made_without_the_training_state(tmp_path):
    f = fake_nn(tmp_path / "n.pth")
    cache = tmp_path / "cache"
    folder, info = seg.prepare_nnunet_folder(f, cache)
    assert folder.parent == cache / "nnunet" and len(folder.name) == 16 and info["n_classes"] == 3
    assert sorted(p.name for p in folder.iterdir()) == [".complete", "dataset.json", "fold_all", "info.json", "plans.json"]
    assert json.loads((folder / "dataset.json").read_text())["labels"]["trapped_cell"] == 1
    slim = torch.load(folder / "fold_all" / "checkpoint_final.pth", weights_only=False)
    assert set(slim) == {"network_weights", "trainer_name", "init_args", "inference_allowed_mirroring_axes"}
    assert set(slim["init_args"]) == {"plans", "configuration", "dataset_json"}                # no optimizer, log, device
    assert (folder / "fold_all" / "checkpoint_final.pth").stat().st_size < f.stat().st_size / 5


def test_preparing_twice_reuses_the_folder_and_a_changed_file_gets_its_own(tmp_path, monkeypatch):
    f = fake_nn(tmp_path / "n.pth")
    first, _ = seg.prepare_nnunet_folder(f, tmp_path / "c")
    monkeypatch.setattr(seg, "read_nnunet_checkpoint", lambda *a, **k: pytest.fail("must not read it again"))
    again, info = seg.prepare_nnunet_folder(f, tmp_path / "c")
    assert again == first and info["configuration"] == "2d"
    monkeypatch.undo()
    other = fake_nn(tmp_path / "o.pth", labels={"background": 0, "cell": 1})
    folder2, _ = seg.prepare_nnunet_folder(other, tmp_path / "c")
    assert folder2 != first


def test_a_refused_checkpoint_leaves_no_half_folder(tmp_path):
    f = fake_nn(tmp_path / "n.pth", conf="3d_fullres")
    with pytest.raises(ValueError):
        seg.prepare_nnunet_folder(f, tmp_path / "c")
    assert not (tmp_path / "c" / "nnunet").exists() or not any((tmp_path / "c" / "nnunet").iterdir())


# ── the file check in models ────────────────────────────────────────────────────────────────────────────────────

def test_the_nnunet_role_exists_and_is_an_alternative():
    r = M.ROLES["nnunet"]
    assert r.suffixes == (".pth",) and not r.required and "alternative" in r.title
    assert "nnunet" in M.SEGMENTATION and M.BUNDLE_NAMES["nnunet"] == "nnunet.pth"


def test_check_nnunet_good_file(tmp_path):
    c = M.check_nnunet(fake_nn(tmp_path / "n.pth"))
    assert c.ok and "nnU-Net 2d, 3 classes (background, trapped_cell, decoy)" in c.message and len(c.sha256) == 64
    assert c.detail["labels"]["decoy"] == 2 and Path(c.detail["folder"]).is_dir()


def test_check_nnunet_problems_are_explained(tmp_path, monkeypatch):
    assert M.check_nnunet(None).message == "not loaded"
    assert "not found" in M.check_nnunet(tmp_path / "x.pth").message
    (tmp_path / "a.pt").write_bytes(b"x")
    assert "is not a .pth file" in M.check_nnunet(tmp_path / "a.pt").message
    assert "'3d_fullres'" in M.check_nnunet(fake_nn(tmp_path / "d.pth", conf="3d_fullres")).message
    monkeypatch.setattr(seg, "nnunet_available", lambda: False)
    c = M.check_nnunet(fake_nn(tmp_path / "n.pth"))
    assert not c.ok and "need the 'nnunetv2' package" in c.message


def test_missing_roles_with_an_nnunet():
    ok = lambda r: M.Check(r, None, True, "")
    assert M.missing_roles({"nnunet": ok("nnunet"), "classifier": ok("classifier")}) == []
    assert M.missing_roles({"nnunet": ok("nnunet")}) == ["HER2 classifier"]
    assert M.missing_roles({"classifier": ok("classifier")}) == [M.SEGMENTATION_NEEDED]
    assert M.missing_roles({"unet": ok("unet"), "classifier": ok("classifier")}) == ["U-Net settings"]
    assert M.missing_roles({"unet_config": ok("unet_config"), "classifier": ok("classifier")}) == ["U-Net weights"]
    assert M.missing_roles({"nnunet": M.Check("nnunet", None, False, "bad"), "classifier": ok("classifier")}) == [M.SEGMENTATION_NEEDED]
    assert M.missing_roles({r: ok(r) for r in ("unet", "unet_config", "nnunet", "classifier")}) == []


# ── bundles with an nnU-Net ─────────────────────────────────────────────────────────────────────────────────────

def test_a_bundle_can_carry_an_nnunet_alone_or_with_the_unet(tmp_path):
    clf = _save_framework(tmp_path / "f.joblib")
    nn = fake_nn(tmp_path / "n.pth")
    b = M.make_bundle(tmp_path / "b1", nnunet=nn, classifier=clf, description="nnU-Net only")
    assert sorted(p.name for p in b.iterdir()) == ["celldform-models.json", "classifier.joblib", "nnunet.pth"]
    files = M.open_bundle(b)
    assert set(files) == {"nnunet", "classifier"} and M.check_nnunet(files["nnunet"]).ok
    slim = torch.load(b / "nnunet.pth", weights_only=False)
    assert "optimizer_state" not in slim and "logging" not in slim and (b / "nnunet.pth").stat().st_size < nn.stat().st_size / 5
    both = M.make_bundle(tmp_path / "b2", nnunet=nn, unet=tiny_unet(tmp_path / "w.pt"), config=write_settings(tmp_path / "c.yaml"), classifier=clf)
    assert set(M.open_bundle(both)) == {"nnunet", "unet", "unet_config", "classifier"}


def test_bundle_rules_for_the_segmentation_model(tmp_path):
    clf = _save_framework(tmp_path / "f.joblib")
    with pytest.raises(M.BundleError, match="needs a segmentation model"):
        M.make_bundle(tmp_path / "a", classifier=clf)
    with pytest.raises(M.BundleError, match="needs its settings file"):
        M.make_bundle(tmp_path / "b", unet=tiny_unet(tmp_path / "w.pt"), classifier=clf)
    with pytest.raises(M.BundleError, match="needs its settings file"):
        M.make_bundle(tmp_path / "c", config=write_settings(tmp_path / "c.yaml"), classifier=clf)
    b = M.make_bundle(tmp_path / "d", nnunet=fake_nn(tmp_path / "n.pth"), classifier=clf)
    mf = json.loads((b / M.MANIFEST).read_text())
    mf["files"].pop("nnunet")
    (b / M.MANIFEST).write_text(json.dumps(mf))
    with pytest.raises(M.BundleError, match="lacks: a segmentation model"):
        M.open_bundle(b)
    mf["files"].pop("classifier"); (b / M.MANIFEST).write_text(json.dumps(mf))
    with pytest.raises(M.BundleError, match="lacks: HER2 classifier, a segmentation model"):
        M.open_bundle(b)


def test_a_bundle_refuses_an_nnunet_with_personal_data_or_that_cannot_be_used(tmp_path):
    clf = _save_framework(tmp_path / "f.joblib")
    leaky = fake_nn(tmp_path / "l.pth")
    ck = torch.load(leaky, weights_only=False)
    ck["init_args"]["plans"]["dataset_name"] = "/home/someone/data/Dataset"
    torch.save(ck, leaky)
    with pytest.raises(M.BundleError, match="plans contain"):
        M.make_bundle(tmp_path / "x", nnunet=leaky, classifier=clf)
    with pytest.raises(ValueError, match="needs a 2d one"):
        M.make_bundle(tmp_path / "y", nnunet=fake_nn(tmp_path / "d.pth", conf="3d_fullres"), classifier=clf)


def test_the_bundle_command_line_takes_an_nnunet(tmp_path, capsys):
    clf = _save_framework(tmp_path / "f.joblib")
    nn = fake_nn(tmp_path / "n.pth")
    assert M.main(["make", "--nnunet", str(nn), "--classifier", clf, "--out", str(tmp_path / "b")]) == 0
    assert "nnunet.pth" in capsys.readouterr().out
    assert M.main(["verify", str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert "nnU-Net checkpoint" in out and "bundle is usable" in out
    assert M.main(["make", "--classifier", clf, "--out", str(tmp_path / "c")]) == 1
    assert "needs a segmentation model" in capsys.readouterr().err


# ── what the run says about the masks ───────────────────────────────────────────────────────────────────────────

class FakeRec:
    frame_log = None
    fps = 22.0


@pytest.mark.parametrize("name, caveat", [("nnunet", False), ("nnU-Net (model.pth)", False), ("unet", True), ("U-Net (w.pt)", True)])
def test_the_mask_caveat_appears_only_without_the_nnunet(name, caveat):
    text = " ".join(protocol_caveats(FakeRec(), name))
    assert ("features were measured on nnU-Net masks" in text) is caveat


# ── in the window ───────────────────────────────────────────────────────────────────────────────────────────────

qt = pytest.importorskip("celldform.demo.qt")
from celldform.demo.gui import MainWindow  # noqa: E402
from celldform.demo.qt import QtWidgets  # noqa: E402
from tests.test_demo_gui import _patched_run  # noqa: E402
from tests.test_demo_setup import FakeUNet, files, wait  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeNN(DiscSegmenter):
    device = "cpu"
    info = {"configuration": "2d", "n_classes": 3}

    def describe(self):
        return "fake nn"


def patch_nn(monkeypatch, ok=True):
    monkeypatch.setitem(M.CHECKS, "nnunet", lambda p: M.Check("nnunet", Path(p), True, "nnU-Net 2d, 3 classes (background, trapped_cell, decoy)", "ab" * 32, 123))
    monkeypatch.setattr(M, "try_nnunet", lambda p, device=None: (FakeNN(), M.Check("nnunet", Path(p), True, "nnU-Net 2d, 3 classes, runs on cpu (0.1 s incl. loading)", "ab" * 32, 123))
                        if ok else (None, M.Check("nnunet", Path(p), False, "the nnU-Net could not be run: boom")))
    monkeypatch.setattr(seg.UNetSegmenter, "from_file", staticmethod(lambda path, config=None, **kw: FakeUNet()))
    monkeypatch.setattr(seg.NNUNetSegmenter, "from_checkpoint", classmethod(lambda cls, path, **kw: FakeNN()))


def pth(tmp_path):
    p = tmp_path / "model.pth"
    p.write_bytes(b"x")
    return str(p)


def test_the_window_has_an_nnunet_row_and_the_pth_filter(app, tmp_path, monkeypatch):
    w = MainWindow()
    assert "nnunet" in w.rows and "alternative" in w.rows["nnunet"].title.text() and w.rows["nnunet"].mark.text() == "–"
    got = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: got.append(a[3]) or ("", "")))
    w.rows["nnunet"].button.click()
    assert "*.pth" in got[0] and "nnU-Net checkpoint" in got[0]
    assert "either" in w.models_box.findChildren(QtWidgets.QLabel)[0].text().lower().replace("<b>", "").replace("</b>", "")


def test_an_nnunet_and_the_classifier_are_enough(app, tmp_path, monkeypatch):
    patch_nn(monkeypatch)
    w = MainWindow()
    w.open_paths([str(make_recording(tmp_path / "rec", 450))])
    assert not w.btn_analyze.isEnabled() and "segmentation model" in w.btn_analyze.toolTip()
    assert w.load_role("nnunet", pth(tmp_path))
    assert not w.models_ready and "HER2 classifier" in w.btn_analyze.toolTip() and "segmentation" not in w.btn_analyze.toolTip()
    assert w.load_role("classifier", files(tmp_path)["classifier"])
    assert w.models_ready and w.btn_analyze.isEnabled()
    assert w.rows["nnunet"].mark.text() == "✔" and "runs on cpu" in w.rows["nnunet"].detail.text()
    assert w.segmenter_name == "nnU-Net (model.pth)" and isinstance(w.segmenter, FakeNN)
    assert w.seg_box.count() == 1 and not w.seg_box.isEnabled()                       # nothing to choose between


def test_the_unet_pair_alone_still_works(app, tmp_path, monkeypatch):
    patch_nn(monkeypatch)
    w = MainWindow()
    f = files(tmp_path)
    w.load_role("unet", f["unet"]); w.load_role("classifier", f["classifier"])
    assert w.models_ready and w.segmenter_name == "U-Net (unet.pt)" and w.seg_box.count() == 1


def test_with_both_loaded_you_choose_and_the_nnunet_is_the_default(app, tmp_path, monkeypatch):
    patch_nn(monkeypatch)
    w = MainWindow()
    f = files(tmp_path)
    for r in ("unet", "unet_config", "classifier"):
        w.load_role(r, f[r])
    assert w.segmenter_name == "U-Net (unet.pt)"
    w.load_role("nnunet", pth(tmp_path))
    assert w.seg_box.count() == 2 and w.seg_box.isEnabled()
    assert w.seg_box.currentData() == "nnunet" and w.segmenter_name == "nnU-Net (model.pth)"        # the new default
    w.seg_box.setCurrentIndex(w.seg_box.findData("unet"))
    assert w.segmenter_name == "U-Net (unet.pt)" and isinstance(w.segmenter, FakeUNet)
    assert "Analyze will use the U-Net (unet.pt)" in w.statusBar().currentMessage()
    w.seg_box.setCurrentIndex(w.seg_box.findData("nnunet"))
    assert w.segmenter_name == "nnU-Net (model.pth)" and isinstance(w.segmenter, FakeNN)


def test_the_choice_decides_what_analyze_runs_and_the_caveat_follows(app, tmp_path, monkeypatch):
    _patched_run(monkeypatch)
    patch_nn(monkeypatch)
    w = MainWindow(results_dir=tmp_path / "results")
    f = files(tmp_path)
    for r in ("unet", "unet_config", "classifier"):
        w.load_role(r, f[r])
    w.load_role("nnunet", pth(tmp_path))
    w.open_paths([str(make_recording(tmp_path / "rec", 450))])
    w.seg_box.setCurrentIndex(w.seg_box.findData("nnunet"))
    w.btn_analyze.click(); assert wait(lambda: w.worker.isFinished())
    log = w.dash.log.toPlainText()
    assert "segmenter=nnU-Net (model.pth)" in log and "masks come from" not in log      # what the classifier was built on
    assert not w.seg_box.isEnabled() or w._run_state == "idle"
    w.seg_box.setCurrentIndex(w.seg_box.findData("unet"))
    w.btn_analyze.click(); assert wait(lambda: w.worker.isFinished())
    log = w.dash.log.toPlainText()
    assert "segmenter=U-Net (unet.pt)" in log and "masks come from U-Net (unet.pt); the classifier's features were measured on nnU-Net masks" in log


def test_the_choice_is_locked_while_running(app, tmp_path, monkeypatch):
    _patched_run(monkeypatch)
    patch_nn(monkeypatch)
    from tests.test_demo_gui import SlowSegmenter
    w = MainWindow(results_dir=tmp_path / "results")
    f = files(tmp_path)
    for r in ("unet", "unet_config", "classifier"):
        w.load_role(r, f[r])
    w.load_role("nnunet", pth(tmp_path))
    w.segmenter = SlowSegmenter()
    w.open_paths([str(make_recording(tmp_path / "rec", 1200))])
    w.btn_analyze.click()
    assert not w.seg_box.isEnabled() and not w.models_box.isEnabled()
    w.btn_stop.click(); assert wait(lambda: w._run_state == "idle")
    assert w.seg_box.isEnabled()


def test_an_nnunet_that_cannot_run_keeps_analyze_off_and_says_why(app, tmp_path, monkeypatch):
    patch_nn(monkeypatch, ok=False)
    w = MainWindow()
    w.load_role("classifier", files(tmp_path)["classifier"])
    w.load_role("nnunet", pth(tmp_path))
    assert not w.models_ready and w.rows["nnunet"].mark.text() == "✘" and "could not be run: boom" in w.rows["nnunet"].detail.text()
    assert "nnunet" not in w._seg_objects and w.seg_box.count() == 1 and w.seg_box.currentData() is None


def test_a_refused_nnunet_file_is_explained_and_the_unet_stays(app, tmp_path):
    w = MainWindow()
    f = files(tmp_path)
    for r in ("unet", "unet_config", "classifier"):
        w.load_role(r, f[r])
    assert not w.load_role("nnunet", fake_nn(tmp_path / "d.pth", conf="3d_fullres"))
    assert "needs a 2d one" in w.statusBar().currentMessage() and w.models_ready and w.segmenter_name == "U-Net (unet.pt)"
    assert "nnunet" not in w.model_paths


def test_clearing_and_remembering_include_the_nnunet(app, tmp_path, monkeypatch):
    from celldform.demo.settings import Settings
    patch_nn(monkeypatch)
    path = tmp_path / "s.json"
    w = MainWindow(settings=Settings(path))
    w.load_role("nnunet", pth(tmp_path)); w.load_role("classifier", files(tmp_path)["classifier"])
    assert set(json.loads(path.read_text())["models"]) == {"nnunet", "classifier"}
    w2 = MainWindow(settings=Settings(path))
    w2.restore_models()
    assert w2.models_ready and w2.segmenter_name == "nnU-Net (model.pth)"
    w2.clear_models()
    assert not w2.models_ready and w2._seg_objects == {} and w2.seg_box.currentData() is None and Settings(path).model_paths() == {}


def test_a_bundle_with_an_nnunet_loads_it(app, tmp_path, monkeypatch):
    patch_nn(monkeypatch)
    clf = _save_framework(tmp_path / "f.joblib")
    b = M.make_bundle(tmp_path / "b", nnunet=fake_nn(tmp_path / "n.pth"), classifier=clf, description="nnU-Net bundle")
    w = MainWindow()
    assert w.load_bundle(b) and w.models_ready
    assert w.rows["nnunet"].mark.text() == "✔" and w.rows["unet"].mark.text() == "○" and w.rows["classifier"].mark.text() == "✔"
    assert "nnU-Net bundle" in w.statusBar().currentMessage()


# ── the command line ────────────────────────────────────────────────────────────────────────────────────────────

def _cli(monkeypatch, tmp_path, *args):
    import celldform.demo.run as runmod
    got = {}

    def fake_run(recording, out, segmenter, framework, **k):
        got.update(k, segmenter=segmenter)
        from celldform.demo.session import Verdict
        return Verdict("provisional", "r", "low", 0.3, 0.3, 0.3, 2, 0.5)
    monkeypatch.setattr(runmod, "run", fake_run)
    monkeypatch.setattr(runmod, "load_framework", lambda p: fake_framework())
    monkeypatch.setattr(seg.UNetSegmenter, "from_file", staticmethod(lambda p, c=None, **k: FakeUNet()))
    monkeypatch.setattr(seg.NNUNetSegmenter, "from_checkpoint", classmethod(lambda cls, p, **k: FakeNN()))
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), "--out", str(tmp_path / "o"), *args])
    runmod.main()
    return got


def test_the_command_line_takes_an_nnunet_file(tmp_path, monkeypatch):
    got = _cli(monkeypatch, tmp_path, "--nnunet-file", "m.pth", "--classifier", "f.joblib")
    assert got["segmenter_name"] == "nnU-Net (m.pth)" and isinstance(got["segmenter"], FakeNN)


def test_the_command_line_picks_with_use_and_explains_a_missing_model(tmp_path, monkeypatch, capsys):
    both = ("--nnunet-file", "m.pth", "--unet-file", "w.pt", "--unet-config", "c.yaml", "--classifier", "f.joblib")
    assert _cli(monkeypatch, tmp_path, *both)["segmenter_name"] == "nnU-Net (m.pth)"                # the default
    assert _cli(monkeypatch, tmp_path, *both, "--use", "unet")["segmenter_name"] == "U-Net (w.pt)"
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, "--nnunet-file", "m.pth", "--classifier", "f.joblib", "--use", "unet")
    assert "--use unet was asked for but no U-Net was given" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, "--nnunet-file", "m.pth")
    assert "--classifier too" in capsys.readouterr().err


def test_the_command_line_runs_an_nnunet_bundle(tmp_path, monkeypatch):
    b = M.make_bundle(tmp_path / "b", nnunet=fake_nn(tmp_path / "n.pth"), classifier=_save_framework(tmp_path / "f.joblib"))
    got = _cli(monkeypatch, tmp_path, "--bundle", str(b))
    assert got["segmenter_name"] == "nnU-Net (nnunet.pth)"


# ── the real trained nnU-Net ────────────────────────────────────────────────────────────────────────────────────

@real
def test_the_real_checkpoint_is_read_checked_and_prepared(tmp_path):
    c = M.check_nnunet(REAL)
    assert c.ok and "nnU-Net 2d, 3 classes (background, trapped_cell, decoy)" in c.message
    slim = Path(c.detail["folder"]) / "fold_all" / "checkpoint_final.pth"
    assert slim.stat().st_size < REAL.stat().st_size * 0.6                           # without the optimizer state


@real
def test_the_real_nnunet_finds_the_cell_from_its_single_file():
    import cv2
    s = seg.NNUNetSegmenter.from_checkpoint(REAL)
    out = s.segment(cv2.imread(str(FRAME[0]), cv2.IMREAD_GRAYSCALE))
    assert out.labels.shape == (922, 1228) and 300 < out.cell.sum() < 3000 and set(np.unique(out.labels)) <= {0, 1, 2}
    assert "nnU-Net 2d, 3 classes" in s.describe()


@real
def test_the_real_nnunet_works_without_simpleitk_and_the_network_libraries(tmp_path):
    """The Windows installer leaves SimpleITK out (265 MB) and the app blocks the network: the single-file path must not need either."""
    code = r'''
import sys, importlib.abc
class B(importlib.abc.MetaPathFinder):
    def find_spec(self, n, p, t=None):
        if n.split(".")[0] in ("SimpleITK", "imagecodecs", "requests", "urllib3"): raise ImportError("blocked " + n)
sys.meta_path.insert(0, B())
import cv2
from celldform.demo.privacy import block_network; block_network()
from celldform.demo.segment import NNUNetSegmenter
s = NNUNetSegmenter.from_checkpoint(sys.argv[1])
out = s.segment(cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE))
print("OK", int(out.cell.sum()), "SimpleITK is a stand-in:", not hasattr(sys.modules["SimpleITK"], "ReadImage"))
'''
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p),
           "CELLDFORM_HOME": str(tmp_path / "h")}
    r = subprocess.run([sys.executable, "-s", "-c", code, str(REAL), str(FRAME[0])], capture_output=True, text=True, timeout=300, cwd=ROOT, env=env)
    assert r.returncode == 0 and "OK" in r.stdout and "stand-in: True" in r.stdout, r.stderr[-1500:]


@real
def test_a_real_bundle_with_the_nnunet_is_half_the_size_and_loads(tmp_path):
    clf = _save_framework(tmp_path / "f.joblib")
    b = M.make_bundle(tmp_path / "b", nnunet=REAL, classifier=clf)
    assert (b / "nnunet.pth").stat().st_size < REAL.stat().st_size * 0.6
    files_ = M.open_bundle(b)
    segr, chk = M.try_nnunet(files_["nnunet"])
    assert segr is not None and chk.ok and "runs on" in chk.message
