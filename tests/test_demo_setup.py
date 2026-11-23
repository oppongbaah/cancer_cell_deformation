"""First run on a machine with nothing installed but the app: every required file must be loaded before Analyze works."""
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("celldform.demo.qt")
torch = pytest.importorskip("torch")
from celldform.demo.qt import QtWidgets

from celldform.demo import models as M
from celldform.demo import privacy
from celldform.demo.gui import MainWindow
from celldform.demo.segment import UNetSegmenter
from celldform.demo.settings import Settings
from tests.test_demo_gui import _patched_run
from tests.test_demo_models import _save_framework, tiny_unet, write_settings
from tests.test_demo_run import DiscSegmenter, make_recording
from tests.test_demo_session import fake_framework


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait(cond, secs=15.0):
    t = time.time()
    while not cond() and time.time() - t < secs:
        QtWidgets.QApplication.processEvents(); time.sleep(0.005)
    QtWidgets.QApplication.processEvents()
    return cond()


class FakeUNet(DiscSegmenter):
    """Stands in for the real U-Net on the 160x120 test videos (the real one wants 1228x922 frames)."""
    arch = {"in_channels": 1, "base_features": 4, "out_channels": 3}
    device = "cpu"

    def describe(self):
        return "fake"


def files(tmp, name="m"):
    d = tmp / name
    d.mkdir(exist_ok=True)
    return {"unet": tiny_unet(d / "unet.pt"), "unet_config": write_settings(d / "unet_settings.yaml"),
            "classifier": _save_framework(d / "classifier.joblib")}


def load_all(w, f):
    return [w.load_role(r, p) for r, p in f.items()]


# ── nothing is loaded, nothing works ────────────────────────────────────────────────────────────────────────────

def test_a_new_window_has_no_models_even_when_the_repository_has_them(app):
    w = MainWindow()
    assert w.segmenter is None and w.framework is None and not w.models_ready
    assert M.missing_roles(w.model_checks) == ["U-Net weights", "U-Net settings", "HER2 classifier"]
    assert "Load first:" in w.setup_lbl.text() and "U-Net weights" in w.setup_lbl.text()
    assert all(not r.button.text().startswith("Change") for r in w.rows.values())
    assert w.rows["reference"].title.text().endswith("(optional)")


def test_analyze_stays_off_until_every_required_file_is_loaded(app, tmp_path):
    w = MainWindow()
    w.open_paths([str(make_recording(tmp_path / "rec", 450))])
    assert w.loaded and not w.btn_analyze.isEnabled()
    assert "Load the required files first: U-Net weights, U-Net settings, HER2 classifier" in w.btn_analyze.toolTip()
    w.start()
    assert "Load the required files first" in w.statusBar().currentMessage() and w._run_state == "idle"
    f = files(tmp_path)
    w.load_role("unet", f["unet"])                                          # its settings file sits beside it: found too
    assert not w.btn_analyze.isEnabled()
    assert "HER2 classifier" in w.btn_analyze.toolTip() and "U-Net settings" not in w.btn_analyze.toolTip()
    w.load_role("classifier", f["classifier"])
    assert w.models_ready and w.btn_analyze.isEnabled()
    assert w.setup_lbl.text() == "All required files are loaded."
    w.clear_models()
    w.load_role("classifier", f["classifier"]); w.load_role("unet_config", f["unet_config"])
    assert not w.btn_analyze.isEnabled() and "U-Net weights" in w.btn_analyze.toolTip()       # the third file, in any order
    w.load_role("unet", f["unet"])
    assert w.btn_analyze.isEnabled()


def test_browsing_works_without_any_models(app, tmp_path):
    w = MainWindow()
    w.open_paths([str(make_recording(tmp_path / "rec", 450))])
    assert w.player.isEnabled()
    w.player.show_frame(100)
    assert w.player.idx == 100 and not w.models_ready


def test_open_is_not_blocked_by_missing_models(app, tmp_path):
    w = MainWindow()
    assert w.open_btn.isEnabled() and w.models_box.isEnabled()


# ── loading the files ───────────────────────────────────────────────────────────────────────────────────────────

def test_loading_each_file_shows_what_was_found(app, tmp_path):
    w = MainWindow()
    f = files(tmp_path)
    assert all(load_all(w, f))
    assert w.models_ready and w.segmenter is not None and w.framework is not None
    assert "3 classes" in w.rows["unet"].detail.text() and "runs on" in w.rows["unet"].detail.text()
    assert "SHA-256" in w.rows["unet"].detail.text() and "unet.pt" in w.rows["unet"].detail.text()
    assert "384 px crop" in w.rows["unet_config"].detail.text()
    assert "HER2 classifier" in w.rows["classifier"].detail.text()
    assert all(w.rows[r].button.text() == "Change…" for r in M.REQUIRED)
    assert w.segmenter_name == "U-Net (unet.pt)"
    assert w.models_box.styleSheet() == ""                                  # no more warning border


def test_the_settings_file_beside_the_weights_is_found_automatically(app, tmp_path):
    f = files(tmp_path)
    w = MainWindow()
    assert w.load_role("unet", f["unet"])                                   # unet_settings.yaml is beside it
    assert w.model_paths["unet_config"].name == "unet_settings.yaml"
    d = tmp_path / "other"; d.mkdir()
    w2 = MainWindow()
    w2.load_role("unet", tiny_unet(d / "w.pt"))                              # nothing beside it: still asked for
    assert "unet_config" not in w2.model_paths and "U-Net settings" in w2.setup_lbl.text()
    (d / "config.yaml").write_text((tmp_path / "m" / "unet_settings.yaml").read_text())
    w3 = MainWindow()
    w3.load_role("unet", d / "w.pt")
    assert w3.model_paths["unet_config"].name == "config.yaml"


def test_a_bad_file_is_refused_with_the_reason_and_the_good_one_stays(app, tmp_path):
    w = MainWindow()
    f = files(tmp_path)
    load_all(w, f)
    (tmp_path / "junk.pt").write_bytes(b"nope")
    assert not w.load_role("unet", tmp_path / "junk.pt")
    assert "U-Net weights" in w.statusBar().currentMessage() and "not a PyTorch checkpoint" in w.statusBar().currentMessage()
    assert w.model_paths["unet"].name == "unet.pt" and w.models_ready                  # unchanged
    bad_cfg = write_settings(tmp_path / "bad.yaml", target_size=[384, 380])
    assert not w.load_role("unet_config", bad_cfg) and "multiples of 16" in w.statusBar().currentMessage()
    import joblib
    joblib.dump({"a": 1}, tmp_path / "dict.joblib")
    assert not w.load_role("classifier", tmp_path / "dict.joblib") and "not a HER2 framework" in w.statusBar().currentMessage()
    assert w.models_ready


def test_a_bad_first_file_leaves_the_slot_empty(app, tmp_path):
    w = MainWindow()
    assert not w.load_role("unet", tmp_path / "missing.pt")
    assert "unet" not in w.model_checks and "unet" not in w.model_paths and not w.models_ready


def test_a_pair_that_does_not_work_together_keeps_analyze_off(app, tmp_path, monkeypatch):
    f = files(tmp_path)
    w = MainWindow()
    monkeypatch.setattr(M, "try_segmenter", lambda *a, **k: (None, M._fail("unet", a[0], "the U-Net and its settings do not work together: boom")))
    w.load_role("classifier", f["classifier"]); w.load_role("unet", f["unet"])
    assert not w.models_ready and not w.model_checks["unet"].ok
    assert "do not work together" in w.rows["unet"].detail.text() and w.rows["unet"].mark.text() == "✘"


def test_the_reference_table_is_optional(app, tmp_path):
    (tmp_path / "r.csv").write_text("her2,len_par_base,fluctuation\nhigh,30,0.01\nlow,35,0.02\n")
    w = MainWindow(DiscSegmenter(), fake_framework())
    w.open_paths([str(make_recording(tmp_path / "rec", 450))])
    assert w.models_ready and w.dash.state.reference is None
    assert w.load_role("reference", tmp_path / "r.csv")
    assert w.dash.state.reference is not None and len(w.dash.state.reference) == 2
    assert "2 training cells" in w.rows["reference"].detail.text()


def test_clearing_forgets_everything(app, tmp_path):
    st = Settings(tmp_path / "s.json")
    w = MainWindow(settings=st)
    load_all(w, files(tmp_path))
    w.clear_models()
    assert not w.models_ready and w.segmenter is None and w.framework is None and st.model_paths() == {}
    assert "Load first:" in w.setup_lbl.text()


# ── remembered between sessions ─────────────────────────────────────────────────────────────────────────────────

def test_the_files_are_remembered_and_checked_again_next_time(app, tmp_path):
    path = tmp_path / "cfg" / "settings.json"
    f = files(tmp_path)
    w = MainWindow(settings=Settings(path))
    load_all(w, f)
    saved = json.loads(path.read_text())
    assert set(saved["models"]) == {"unet", "unet_config", "classifier"}
    assert saved["models"]["unet"]["path"] == f["unet"] and len(saved["models"]["unet"]["sha256"]) == 64
    assert "state_dict" not in path.read_text()                              # paths and hashes only, never contents
    w2 = MainWindow(settings=Settings(path))
    assert not w2.models_ready
    w2.restore_models()
    assert w2.models_ready and w2.segmenter is not None and w2.framework is not None


def test_a_remembered_file_that_has_gone_is_reported_not_fatal(app, tmp_path):
    path = tmp_path / "settings.json"
    f = files(tmp_path)
    w = MainWindow(settings=Settings(path))
    load_all(w, f)
    os.remove(f["classifier"])
    w2 = MainWindow(settings=Settings(path))
    w2.restore_models()
    assert not w2.models_ready and "file not found: classifier.joblib" in w2.rows["classifier"].detail.text()
    assert w2.rows["unet"].mark.text() == "✔" and w2.rows["classifier"].mark.text() == "✘"
    assert w2.load_role("classifier", _save_framework(tmp_path / "new.joblib")) and w2.models_ready


def test_a_remembered_file_that_was_replaced_by_junk_is_refused(app, tmp_path):
    path = tmp_path / "settings.json"
    f = files(tmp_path)
    load_all(MainWindow(settings=Settings(path)), f)
    (tmp_path / "m" / "classifier.joblib").write_bytes(b"replaced")
    w = MainWindow(settings=Settings(path))
    w.restore_models()
    assert not w.models_ready and not w.model_checks["classifier"].ok


def test_damaged_settings_are_ignored(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{ not json")
    assert Settings(p).model_paths() == {}
    p.write_text(json.dumps({"version": 99, "models": {"unet": {"path": "x"}}}))
    assert Settings(p).model_paths() == {}
    p.write_text(json.dumps({"version": 1, "models": {"unet": {"path": ""}, "x": 5}}))
    assert Settings(p).model_paths() == {}


def test_settings_round_trip_and_never_fail_on_a_read_only_place(tmp_path):
    s = Settings(tmp_path / "a" / "s.json")
    s.set_model("unet", "/m/unet.pt", "ab" * 32)
    s.last_dir = "/videos"; s.results_dir = "/results"
    again = Settings(tmp_path / "a" / "s.json")
    assert again.model_paths() == {"unet": __import__("pathlib").Path("/m/unet.pt")} and again.last_dir == "/videos"
    assert str(again.results_dir) == "/results"
    again.reset()
    assert Settings(tmp_path / "a" / "s.json").model_paths() == {}
    blocked = tmp_path / "file"
    blocked.write_text("x")
    Settings(blocked / "s.json").set_model("unet", "x")                      # parent is a file: saving fails quietly
    Settings(None).set_model("unet", "x")                                    # in memory only


def test_settings_live_in_a_chosen_folder(monkeypatch, tmp_path):
    from celldform.demo import settings as S
    monkeypatch.setenv("CELLDFORM_HOME", str(tmp_path / "portable"))
    assert S.config_dir() == tmp_path / "portable" / "config" and S.default_results_dir() == tmp_path / "portable" / "results"
    monkeypatch.delenv("CELLDFORM_HOME")
    assert "celldform" in str(S.config_dir()).lower() and "celldform" in str(S.default_results_dir()).lower()


# ── bundles ─────────────────────────────────────────────────────────────────────────────────────────────────────

def make_bundle(tmp_path):
    f = files(tmp_path, "src")
    return M.make_bundle(tmp_path / "bundle", unet=f["unet"], config=f["unet_config"], classifier=f["classifier"],
                         description="test models")


def test_a_bundle_loads_everything_at_once(app, tmp_path):
    w = MainWindow()
    assert w.load_bundle(make_bundle(tmp_path))
    assert w.models_ready and "Model bundle loaded and verified: test models" in w.statusBar().currentMessage()
    assert all(w.rows[r].mark.text() == "✔" for r in M.REQUIRED)


def test_a_zip_bundle_loads_too(app, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_user_cache", lambda: tmp_path / "cache")
    z = M.pack_zip(make_bundle(tmp_path), tmp_path / "models.zip")
    w = MainWindow()
    assert w.load_bundle(z) and w.models_ready
    assert tmp_path / "cache" in w.model_paths["unet"].parents


def test_a_tampered_bundle_loads_nothing(app, tmp_path):
    b = make_bundle(tmp_path)
    data = bytearray((b / "classifier.joblib").read_bytes()); data[-5] ^= 0xFF
    (b / "classifier.joblib").write_bytes(bytes(data))
    w = MainWindow()
    assert not w.load_bundle(b)
    assert "SHA-256" in w.statusBar().currentMessage() and w.model_paths == {} and not w.models_ready


def test_not_a_bundle(app, tmp_path):
    w = MainWindow()
    (tmp_path / "empty").mkdir()
    assert not w.load_bundle(tmp_path / "empty") and "not a model bundle" in w.statusBar().currentMessage()
    assert not w.load_bundle(tmp_path / "nothing.zip") and not w.models_ready


def test_the_bundle_menu_and_dialogs(app, tmp_path, monkeypatch):
    b = make_bundle(tmp_path)
    w = MainWindow()
    assert [a.text() for a in w.btn_bundle.menu().actions()] == ["Bundle .zip file…", "Bundle folder…"]
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(b)))
    w.act_bundle_dir.trigger()
    assert w.models_ready
    w.clear_models()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
    w.act_bundle_zip.trigger()                                               # cancelled: nothing changes
    assert not w.models_ready
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(tmp_path / "x" / "unet.pt"), "")))
    w.rows["unet"].button.click()                                            # a row's button asks for its own kind of file
    assert not w.models_ready


# ── analyzing with loaded files, and what it writes ─────────────────────────────────────────────────────────────

def ready_window(tmp_path, monkeypatch, results=None):
    _patched_run(monkeypatch)
    monkeypatch.setattr(UNetSegmenter, "from_file", staticmethod(lambda path, config=None, **kw: FakeUNet()))
    w = MainWindow(results_dir=results or tmp_path / "results")
    assert all(load_all(w, files(tmp_path)))
    w.open_paths([str(make_recording(tmp_path / "rec one", 450))])
    return w


def test_analyze_runs_with_the_loaded_files(app, tmp_path, monkeypatch):
    w = ready_window(tmp_path, monkeypatch)
    assert w.btn_analyze.isEnabled()
    w.btn_analyze.click()
    assert wait(lambda: w.worker.isFinished())
    log = w.dash.log.toPlainText()
    assert "segmenter=U-Net (unet.pt)" in log and "classifier=loaded" in log and "result:" in log
    assert "masks come from U-Net (unet.pt)" in log


def test_no_models_are_needed_from_the_repository(app, tmp_path, monkeypatch):
    """The whole flow also works when the repository's own model folders do not exist."""
    import celldform.demo.run as runmod
    import celldform.demo.segment as segmod
    monkeypatch.setattr(runmod, "DEFAULT_CLASSIFIER", tmp_path / "does" / "not" / "exist.joblib")
    monkeypatch.setattr(segmod, "DEPLOY_DIR", tmp_path / "no" / "deploy")
    w = ready_window(tmp_path, monkeypatch)
    w.btn_analyze.click()
    assert wait(lambda: w.worker.isFinished())
    assert "result:" in w.dash.log.toPlainText()
    # and the real U-Net loader says what is missing instead of looking in the repository
    with pytest.raises(ValueError, match="needs its settings file"):
        segmod.UNetSegmenter(tmp_path / "m", "unet.pt", device="cpu")


def test_each_analysis_gets_its_own_results_folder(app, tmp_path, monkeypatch):
    w = ready_window(tmp_path, monkeypatch)
    base = tmp_path / "results"
    assert not base.exists()                                                 # nothing is created until the first analysis
    w.btn_analyze.click(); assert wait(lambda: w.worker.isFinished())
    first = w.out
    w.btn_analyze.click(); assert wait(lambda: w.worker.isFinished())
    second = w.out
    assert first != second and first.parent == base and second.parent == base
    assert first.name.endswith("_rec_one") and (first / "verdict.json").is_file() and (second / "dashboard.png").is_file()
    assert sorted(p.name for p in base.iterdir()) == sorted([first.name, second.name])


def test_results_say_no_more_about_the_user_than_the_recordings_name(app, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("USERPROFILE", str(tmp_path))
    w = ready_window(tmp_path, monkeypatch, results=tmp_path / "Documents" / "results")
    w.btn_analyze.click(); assert wait(lambda: w.worker.isFinished())
    j = json.loads((w.out / "verdict.json").read_text())
    assert j["recording"] == "rec one" and os.sep not in j["recording"] and "/" not in j["recording"]
    for f in w.out.glob("*"):
        if f.suffix in (".json", ".log", ".csv"):
            text = f.read_text()
            assert str(tmp_path) not in text, f"{f.name} contains the user's folder"
            assert privacy.find_sensitive(text) == [], f.name
    assert "in ~" in (w.out / "inference.log").read_text()                   # where it went, without who or where from


def test_everything_works_with_the_network_switched_off(app, tmp_path, monkeypatch):
    privacy.block_network()
    try:
        w = ready_window(tmp_path, monkeypatch)
        assert w.load_bundle(make_bundle(tmp_path))
        w.btn_analyze.click()
        assert wait(lambda: w.worker.isFinished())
        assert "result:" in w.dash.log.toPlainText() and "Error" not in w.dash.log.toPlainText()
    finally:
        privacy.unblock_network()


def test_the_results_folder_button_opens_the_latest_or_the_base(app, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("celldform.demo.qt.QtGui.QDesktopServices.openUrl", staticmethod(lambda url: opened.append(url.toLocalFile())))
    w = ready_window(tmp_path, monkeypatch)
    w.open_results()
    assert opened[-1] == str(tmp_path / "results")                           # created on demand before the first analysis
    w.btn_analyze.click(); assert wait(lambda: w.worker.isFinished())
    w.open_results()
    assert opened[-1] == str(w.out)


def test_gating_follows_the_models_at_every_stage(app, tmp_path, monkeypatch):
    w = ready_window(tmp_path, monkeypatch)
    assert w.btn_analyze.isEnabled()
    w.clear_models()
    assert not w.btn_analyze.isEnabled() and w.models_box.isEnabled()
    load_all(w, files(tmp_path, "again"))
    assert w.btn_analyze.isEnabled()
    w.btn_analyze.click()
    assert not w.models_box.isEnabled()                                      # locked while running
    w.btn_stop.click(); assert wait(lambda: w._run_state == "idle")
    assert w.btn_analyze.isEnabled() and w.models_box.isEnabled()
