"""Tests for celldform.demo.run (stage 7)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from celldform.demo.run import protocol_caveats, run
from celldform.demo.segment import Seg
from celldform.demo.source import Recording
from tests.test_demo_session import fake_framework

W, H = 160, 120


class DiscSegmenter:
    """Stands in for the U-Net: a disc at the trap, ignoring the pixels."""
    calls = 0

    def segment(self, gray):
        DiscSegmenter.calls += 1
        lab = np.zeros((H, W), np.uint8)
        cv2.circle(lab, (80, 60), 8, 1, -1)
        return Seg(lab, np.full((H, W), 0.99, np.float32), None)


def make_recording(folder: Path, n: int, log=None):
    folder.mkdir()
    w = cv2.VideoWriter(str(folder / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 22.0, (W, H), True)
    for _ in range(n):
        w.write(np.full((H, W, 3), 120, np.uint8))
    w.release()
    (folder / "SKBR3.txt").write_text("Cell label: SKBR3\n")
    if log is not None:
        log.to_csv(folder / "frames.csv", index=False)
    return folder


def test_short_recording_runs_and_withholds_verdict(tmp_path):
    rec = make_recording(tmp_path / "r", 450)
    DiscSegmenter.calls = 0
    v = run(rec, tmp_path / "out", DiscSegmenter(), fake_framework(), size=(W, H))
    assert v.status == "insufficient" and v.label is None
    assert DiscSegmenter.calls == 14                      # steps 0,1: 6 frames each; step 2: frames 400, 420
    out = tmp_path / "out"
    info = json.loads((out / "verdict.json").read_text())
    assert info["frames_measured"] == 14 and info["cell_label"] == "SKBR3" and info["status"] == "insufficient"
    f = pd.read_csv(out / "frames.csv")
    assert len(f) == 14 and f.passed.all()
    assert (out / "steps.csv").exists() and (out / "annotated.mp4").stat().st_size > 0


def test_long_recording_gives_verdict(tmp_path):
    rec = make_recording(tmp_path / "r", 2400)
    v = run(rec, tmp_path / "out", DiscSegmenter(), fake_framework(), size=(W, H), write_video=False)
    assert v.status == "ok" and v.label in ("high", "low")
    assert json.loads((tmp_path / "out/verdict.json").read_text())["n_steps"] == 12
    assert not (tmp_path / "out/annotated.mp4").exists()


def test_max_frames_limits_work(tmp_path):
    rec = make_recording(tmp_path / "r", 450)
    DiscSegmenter.calls = 0
    run(rec, tmp_path / "out", DiscSegmenter(), None, size=(W, H), max_frames=100, write_video=False)
    assert DiscSegmenter.calls == 3                       # frames 0, 20, 60 (stop=100 is exclusive)


def test_protocol_caveats_flag_a_different_laser_protocol(tmp_path):
    t = np.arange(0, 100, 1 / 22.0)
    cur = 350 + 50 * (t // 10).astype(int)                # 10 s holds from 350 mA
    rec = Recording(make_recording(tmp_path / "r", 5, pd.DataFrame({"time_s": t, "set_current_mA": cur})), size=(W, H))
    c = protocol_caveats(rec, "unet")
    assert any("350" in x and "mA" in x for x in c) and any("every ~10 s" in x for x in c)
    assert any("U-Net" in x for x in c)


def test_no_laser_log_is_a_caveat(tmp_path):
    rec = Recording(make_recording(tmp_path / "r", 5), size=(W, H))
    assert any("no laser log" in x for x in protocol_caveats(rec, "nnunet"))
    assert not any("U-Net" in x for x in protocol_caveats(rec, "nnunet"))


def test_one_frame_current_blips_are_not_levels(tmp_path):
    t = np.arange(0, 100, 1 / 22.0)
    cur = 500 + 50 * (t // 30).astype(int)                # the training protocol: 30 s holds
    cur[100] += 1                                         # a 1-frame blip must not look like a level change
    rec = Recording(make_recording(tmp_path / "r", 5, pd.DataFrame({"time_s": t, "set_current_mA": cur})), size=(W, H))
    assert not any("every ~" in x for x in protocol_caveats(rec, "nnunet"))


def test_stop_hook_ends_the_run_early_but_still_writes_results(tmp_path):
    rec = make_recording(tmp_path / "r", 450)
    seen = []
    v = run(rec, tmp_path / "out", DiscSegmenter(), None, size=(W, H), write_video=False,
            on_frame=lambda i, img, res, vv, s: seen.append(i), stop=lambda: len(seen) >= 3)
    assert seen == [0, 20, 60] and (tmp_path / "out/frames.csv").exists() and v.status == "insufficient"


def test_run_from_images_logs_every_measured_frame_and_saves_the_log(tmp_path):
    import cv2 as _cv
    folder = make_recording(tmp_path / "r", 450)
    d = folder / "stills_1GB"; d.mkdir()
    rows = []
    for k in range(450):
        n = f"still_{k + 1:06d}.png"
        _cv2 = _cv.imwrite(str(d / n), np.full((H, W, 3), 120, np.uint8))
        rows.append((n, k + 1))
    pd.DataFrame(rows, columns=["still_file", "video_frame"]).to_csv(folder / "stills.csv", index=False)
    seen = []
    v = run(folder, tmp_path / "out", DiscSegmenter(), fake_framework(), size=(W, H), source="images", log=seen.append)
    assert v.status == "insufficient"
    text = (tmp_path / "out" / "inference.log").read_text().splitlines()
    assert text == seen                                              # the window and the file get the same lines
    assert "source=images" in text[0] and text[0].split()[0].count(":") == 2            # timestamped
    frame_lines = [l for l in text if " frame " in l and "seg" in l]
    assert len(frame_lines) == 14 and all("PASS" in l or "REJECT" in l for l in frame_lines)
    assert any("step usable" in l for l in text) and any(l.split("  ", 1)[1].startswith("finished") for l in text)
    assert text[-1].split("  ", 1)[1].startswith("saved") and any("result: no verdict" in l for l in text)
    assert json.loads((tmp_path / "out" / "verdict.json").read_text())["frames_measured"] == 14


def test_run_accepts_a_list_of_image_files(tmp_path):
    folder = make_recording(tmp_path / "r", 450)
    d = folder / "pics"; d.mkdir()
    files = []
    for k in range(450):
        files.append(d / f"p{k:04d}.png")
        cv2.imwrite(str(files[-1]), np.full((H, W, 3), 120, np.uint8))
    seen = []
    v = run(files, tmp_path / "out", DiscSegmenter(), fake_framework(), size=(W, H), log=seen.append)
    assert v.status == "insufficient"
    assert "450 images from pics" in seen[0] and "source=images" in seen[0]
    assert json.loads((tmp_path / "out" / "verdict.json").read_text())["frames_measured"] == 14


def test_run_provisional_marks_everything_as_not_validated(tmp_path):
    rec = make_recording(tmp_path / "r", 450)
    seen = []
    v = run(rec, tmp_path / "out", DiscSegmenter(), fake_framework(), size=(W, H), provisional=True, log=seen.append)
    assert v.status == "provisional" and v.label in ("high", "low")
    j = json.loads((tmp_path / "out" / "verdict.json").read_text())
    assert j["status"] == "provisional" and j["validated"] is False and j["label"] == v.label
    assert "needs 10" in j["reason"]
    assert any("PROVISIONAL, NOT VALIDATED" in l for l in seen if "result:" in l)
    assert any("PROVISIONAL" in l for l in seen if "step usable" in l)            # also while it builds up
    assert "PROVISIONAL" in (tmp_path / "out" / "inference.log").read_text()
    assert (tmp_path / "out" / "dashboard.png").stat().st_size > 1000


def test_run_default_has_no_label_and_says_not_validated_in_json(tmp_path):
    v = run(make_recording(tmp_path / "r", 450), tmp_path / "out", DiscSegmenter(), fake_framework(), size=(W, H))
    j = json.loads((tmp_path / "out" / "verdict.json").read_text())
    assert v.status == "insufficient" and j["validated"] is False and j["label"] is None


def _cli(monkeypatch, tmp_path, *args):
    """Run main() with run() replaced by a stub that records its keyword arguments."""
    import sys
    import celldform.demo.run as runmod
    import celldform.demo.segment as segmod
    got = {}

    def fake_run(recording, out, segmenter, framework, **k):
        got.update(k, recording=recording, out=out, segmenter=segmenter, framework=framework)
        from celldform.demo.session import Verdict
        return Verdict("provisional", "r", "low", 0.3, 0.3, 0.3, 2, 0.5)
    monkeypatch.setattr(runmod, "run", fake_run)
    monkeypatch.setattr(segmod, "make_segmenter", lambda *a, **k: DiscSegmenter())
    monkeypatch.setattr(segmod.UNetSegmenter, "from_file", staticmethod(lambda p, c=None, **k: DiscSegmenter()))
    monkeypatch.setattr(runmod, "load_framework", lambda p: fake_framework())
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), "--out", str(tmp_path / "o"), *args])
    runmod.main()
    return got


def test_the_command_line_flag(tmp_path, monkeypatch):
    got = _cli(monkeypatch, tmp_path, "--segmenter", "unet", "--provisional")
    assert got["provisional"] is True
    assert _cli(monkeypatch, tmp_path, "--segmenter", "unet")["provisional"] is False


def test_the_command_line_needs_models_and_says_so(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path)
    assert "no models given: use --bundle" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, "--unet-file", "w.pt")
    assert "--unet-config and --classifier too" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, "--unet-file", "w.pt", "--unet-config", "c.yaml")
    assert "--classifier too" in capsys.readouterr().err


def test_the_command_line_takes_the_files_one_by_one(tmp_path, monkeypatch):
    got = _cli(monkeypatch, tmp_path, "--unet-file", "w.pt", "--unet-config", "c.yaml", "--classifier", "f.joblib")
    assert got["segmenter_name"] == "U-Net (w.pt)" and got["framework"] is not None


def test_the_command_line_takes_a_bundle(tmp_path, monkeypatch):
    from celldform.demo import models as M
    from tests.test_demo_models import _save_framework, tiny_unet, write_settings
    src = tmp_path / "src"; src.mkdir()
    b = M.make_bundle(tmp_path / "bundle", unet=tiny_unet(src / "w.pt"), config=write_settings(src / "c.yaml"),
                      classifier=_save_framework(src / "f.joblib"))
    got = _cli(monkeypatch, tmp_path, "--bundle", str(b))
    assert got["segmenter_name"] == "U-Net (unet.pt)"
    (b / "classifier.joblib").write_bytes(b"changed")
    with pytest.raises(M.BundleError, match="wrong size"):
        _cli(monkeypatch, tmp_path, "--bundle", str(b))                         # a damaged bundle is refused before anything runs
