"""
Headless demo: a recording in, an annotated video + tables + a verdict out.

    python -m celldform.demo.run <recording folder or video> --out <dir> [--classifier <joblib>] [--crop-center X Y]

Writes to <dir>: ``frames.csv`` (one row per measured frame: checks, passed, features), ``steps.csv`` (per laser step),
``verdict.json``, ``annotated.mp4`` (measured frames, predicted mask outlined white = cell, grey = decoy, with the
frame's verdict from the checks and the running HER2 result). Not real-time: it reports its own speed.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import pandas as pd

from celldform.demo.framework import HER2Framework, load as load_framework
from celldform.demo.session import CellSession, Verdict
from celldform.demo import privacy
from celldform.demo.source import Recording

DEFAULT_CLASSIFIER = (Path(__file__).resolve().parent.parent.parent
                      / "checkpoints/classification/nnunet1000_loocv/framework/her2_framework.joblib")
TRAINED = {"hold_s": 30.0, "start_mA": 500.0, "max_mA": 1200.0}


def protocol_caveats(rec: Recording, segmenter_name: str) -> List[str]:
    """What this recording does that the classifier was not validated for."""
    out = []
    if segmenter_name != "nnunet":
        label = "the U-Net" if segmenter_name == "unet" else segmenter_name
        out.append(f"masks come from {label}; the classifier's features were measured on nnU-Net masks")
    log = rec.frame_log
    if log is None or "set_current_mA" not in log:
        out.append("no laser log: the laser protocol is assumed to be the training one")
        return out
    lo, hi = float(log.set_current_mA.min()), float(log.set_current_mA.max())
    if lo < TRAINED["start_mA"] - 1 or hi > TRAINED["max_mA"] + 1:
        out.append(f"laser current {lo:.0f}-{hi:.0f} mA; the classifier was trained on "
                   f"{TRAINED['start_mA']:.0f}-{TRAINED['max_mA']:.0f} mA")
    cur, t = log.set_current_mA.to_numpy(), log.time_s.to_numpy()
    edges = np.r_[0, np.flatnonzero(np.diff(cur) != 0) + 1, len(cur) - 1]
    holds = np.diff(t[edges])[np.diff(edges) > 5]        # runs of constant current; 1-frame blips are not levels
    if len(holds) > 2:                                   # the hold that half of the recording's time is spent in
        order = np.sort(holds)[::-1]
        hold = float(order[np.searchsorted(np.cumsum(order), order.sum() / 2)])
        if abs(hold - TRAINED["hold_s"]) > 5:
            out.append(f"current changes every ~{hold:.0f} s; the classifier's features use 200-frame "
                       f"({200 / rec.fps:.1f} s) steps from recordings held {TRAINED['hold_s']:.0f} s per level")
    return out


def draw(frame: np.ndarray, labels: np.ndarray, region: Optional[np.ndarray], lines: List[str]) -> np.ndarray:
    """Gray frame -> BGR with the predicted mask outlined (white = trapped cell, grey = decoy) and text."""
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for cls, col in ((2, (150, 150, 150)), (1, (255, 255, 255))):
        m = (region.astype(np.uint8) if cls == 1 and region is not None else (labels == cls).astype(np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, col, 2)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, t, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return img


def run(recording, out: Path | str, segmenter, framework: Optional[HER2Framework] = None,
        segmenter_name: str = "unet", max_frames: Optional[int] = None, size=(1228, 922),
        progress: Optional[Callable[[int, Verdict], None]] = None, write_video: bool = True,
        trap_xy=None, on_frame: Optional[Callable] = None, stop: Optional[Callable[[], bool]] = None,
        source: str = "video", log: Optional[Callable[[str], None]] = None,
        paused_s: Optional[Callable[[], float]] = None, tick: Optional[Callable[[], None]] = None,
        provisional: bool = False) -> Verdict:
    """Run the models over a recording. ``source`` is ``"video"`` or ``"images"``; ``log(line)`` receives one
    timestamped line per event and per measured frame, and the same lines are saved as ``inference.log``. ``paused_s()``
    returns the seconds spent paused so far; they are left out of the timings. ``provisional=True`` also computes a verdict when the checks fail, labelled NOT VALIDATED. ``tick()`` is called for every frame
    the source steps over, measured or not, so a caller can pause (block) between any two frames, not only after a
    measured one."""
    lines: List[str] = []

    def say(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        lines.append(line)
        if log:
            log(line)

    rec = Recording.from_spec(recording, size=size, source=source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sess = CellSession(framework, fps=rec.fps, caveats=protocol_caveats(rec, segmenter_name), trap_xy=trap_xy,
                       provisional=provisional)
    say(f"start: {rec.name}, source={rec.source}, {rec.declared_frames} frames, {rec.fps:g} fps, "
        f"segmenter={segmenter_name}, classifier={'loaded' if framework is not None else 'none'}"
        + (f", trap fixed at ({trap_xy[0]:.0f}, {trap_xy[1]:.0f})" if trap_xy else ", trap learned from the frames"))
    for c in sess.caveats:
        say(f"caveat: {c}")
    writer = None
    t0, n, last_steps = time.perf_counter(), 0, 0
    wants = sess.wants if tick is None else (lambda i: (tick(), sess.wants(i))[1])
    for idx, t, gray in rec.frames(stop=max_frames, select=wants):
        t_seg = time.perf_counter()
        seg = segmenter.segment(gray)
        seg_ms = 1000 * (time.perf_counter() - t_seg)
        res = sess.add(seg, idx, gray)
        n += 1
        v = sess.verdict()
        r = res.row
        say(f"frame {idx:>6} step {sess.step_of(idx):>3}  seg {seg_ms:5.0f} ms  "
            + (f"PASS    area {r.get('area', float('nan')):.0f} px  aspect {r.get('aspect_ratio', float('nan')):.2f}"
               if res.passed else f"REJECT  {r['reasons']}"))
        if v.n_steps != last_steps:
            last_steps = v.n_steps
            say(f"step usable: {v.n_steps} step(s) so far, {v.pass_fraction:.0%} of measured frames pass  -> "
                + (v.headline() if v.label else v.reason))
        if write_video or on_frame:
            txt = [f"frame {idx}  t={t:.1f}s  step {sess.step_of(idx)}",
                   "cell: " + ("OK" if res.passed else "rejected - " + str(res.row["reasons"])),
                   v.headline()[:90] if v.label else "HER2: " + v.reason[:70]]
            img = draw(gray, seg.labels, res.region, txt)
            if on_frame:
                on_frame(idx, img, res, v, sess)
            if write_video and writer is None:
                writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                                         (img.shape[1], img.shape[0]))
            if writer is not None:
                writer.write(img)
        if progress:
            progress(idx, v)
        if stop and stop():
            break
    if writer is not None:
        writer.release()
    paused = paused_s() if paused_s else 0.0
    elapsed = time.perf_counter() - t0 - paused
    say(f"{'stopped' if stop and stop() else 'finished'}: {n} frames measured in {elapsed:.1f} s "
        f"({1000 * elapsed / max(n, 1):.0f} ms each)")
    sess.frames.to_csv(out / "frames.csv", index=False)
    sess.steps().to_csv(out / "steps.csv", index=False)
    v = sess.verdict()
    info = {**asdict(v), "recording": rec.name, "cell_label": rec.cell_label, "frames_declared": rec.declared_frames,
            "frames_measured": n, "seconds": round(elapsed, 1), "ms_per_measured_frame": round(1000 * elapsed / max(n, 1), 1),
            "segmenter": segmenter_name, "validated": v.validated, **({"paused_seconds": round(paused, 1)} if paused else {})}
    try:
        from celldform.demo import dashboard
        dashboard.save(dashboard.build_state(sess, rec, n, elapsed, segmenter_name, getattr(segmenter, "device", ""),
                                             dashboard.load_reference()), out / "dashboard.png")
    except Exception as e:                                   # the picture is a bonus: never lose the results over it
        print(f"dashboard.png not written: {type(e).__name__}: {e}")
    say("result: " + v.headline())
    say(f"saved: frames.csv, steps.csv, dashboard.png, verdict.json, inference.log in {privacy.redact(str(out))}")
    (out / "inference.log").write_text("\n".join(lines) + "\n")
    (out / "verdict.json").write_text(json.dumps(info, indent=2, default=lambda o: None if o != o else str(o)) + "\n")
    return v


def main() -> None:
    from celldform.demo import models
    from celldform.demo.segment import SEGMENTERS, make_segmenter
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="+", help="a recording folder, a video file, or several image files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["video", "images"], default="video", help="read the video or its still images")
    ap.add_argument("--bundle", default=None, help="a model bundle (folder or .zip): all the files the app needs, verified")
    ap.add_argument("--unet-file", default=None, help="the U-Net weights (.pt)")
    ap.add_argument("--unet-config", default=None, help="the U-Net's settings (config.yaml)")
    ap.add_argument("--classifier", default=None, help="the HER2 classifier (.joblib)")
    ap.add_argument("--segmenter", choices=sorted(SEGMENTERS), default=None,
                    help="developers only: use a model that lives in this repository (needs its checkpoints)")
    ap.add_argument("--crop-center", type=int, nargs=2, default=None)
    ap.add_argument("--trap", type=float, nargs=2, default=None, help="trap position x y in the 1228x922 frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--provisional", action="store_true",
                    help="also give a verdict when the checks fail; it is labelled PROVISIONAL, NOT VALIDATED")
    a = ap.parse_args()
    crop = tuple(a.crop_center) if a.crop_center else None
    if a.bundle:
        files = models.open_bundle(a.bundle)
        unet, unet_cfg, clf = files["unet"], files["unet_config"], files["classifier"]
    else:
        unet, unet_cfg, clf = a.unet_file, a.unet_config, a.classifier
    if unet:
        missing = [n for n, v in (("--unet-config", unet_cfg), ("--classifier", clf)) if not v]
        if missing:
            ap.error("the U-Net needs " + " and ".join(missing) + " too (or give --bundle with all of them)")
        from celldform.demo.segment import UNetSegmenter
        seg, name = UNetSegmenter.from_file(unet, unet_cfg, crop_center=crop), f"U-Net ({Path(unet).name})"
        fwk = load_framework(clf)
    elif a.segmenter:                                    # developer shortcut: the models in this repository
        seg, name = make_segmenter(a.segmenter, crop), a.segmenter
        fwk = load_framework(clf or DEFAULT_CLASSIFIER)
    else:
        ap.error("no models given: use --bundle, or --unet-file with --unet-config and --classifier")
    v = run(a.recording if len(a.recording) > 1 else a.recording[0], a.out, seg, fwk, segmenter_name=name, source=a.source, provisional=a.provisional, max_frames=a.max_frames, write_video=not a.no_video, trap_xy=tuple(a.trap) if a.trap else None,
            progress=lambda i, v: print(f"\rframe {i}  steps {v.n_steps}", end="", flush=True))
    print(f"\n{v.status}: {v.headline()}")
    for c in v.caveats:
        print("caveat:", c)


if __name__ == "__main__":
    main()
