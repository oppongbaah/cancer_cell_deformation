"""
Dashboard for the demo: what the app measured, why a verdict was or was not given, and where the cell sits among the
15 training cells. Every panel is a function ``panel_x(ax, state)`` that draws on a matplotlib Axes, so the window
(live, in tabs), the command line (``dashboard.png`` next to the results) and the tests all use the same code.

    state = build_state(session, recording, ...)      # a snapshot of a running or finished CellSession
    save(state, "dashboard.png")                      # every panel on one page

Colours: high HER2 orange, low HER2 blue, frame passes green, fails red, nothing measured grey. Predicted masks
elsewhere in the app are white/grey outlines (the repository's mask colour convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HIGH, LOW, OK, BAD, NONE, INK = "#d95f02", "#1b6ca8", "#2e7d32", "#c62828", "#b8b8b8", "#333333"
CAMERA_MS = 45.5                      # camera frame interval: a speed reference only, the demo is not real-time
REFERENCE_CSV = (Path(__file__).resolve().parent.parent.parent / "results/features/nnunet1000_loocv/features_cells.csv")
TRAINED_RANGE_MA = (500.0, 1200.0)
CROP = 112


@dataclass
class State:
    """Everything the panels need; plain data, so a snapshot can cross a thread boundary."""
    frames: pd.DataFrame = field(default_factory=pd.DataFrame)
    steps: pd.DataFrame = field(default_factory=pd.DataFrame)
    cell_features: Dict[str, float] = field(default_factory=dict)
    verdict: object = None
    t_s: Optional[np.ndarray] = None              # per video frame: time and set current (None without a laser log)
    mA: Optional[np.ndarray] = None
    fps: float = 22.0
    frames_per_step: int = 200
    total_frames: int = 0
    n_measured: int = 0
    elapsed_s: float = 0.0
    segmenter: str = "unet"
    device: str = ""
    crops: Dict[int, np.ndarray] = field(default_factory=dict)       # laser step -> small image of the cell
    info: Dict = field(default_factory=dict)                         # the classifier's training record
    reference: Optional[pd.DataFrame] = None                         # the 15 training cells' per-cell features
    min_steps: int = 3
    min_pass: float = 0.8
    title: str = ""
    source: str = "video"                        # where the frames were read from: video or images


def load_reference(path: Path | str = REFERENCE_CSV) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def reference_for(framework) -> Optional[pd.DataFrame]:
    """The 15 training cells' table, only when the classifier is the one trained on them (a different classifier was
    trained on different cells, so showing these would mislead)."""
    info = getattr(framework, "info", None) or {}
    return load_reference() if (framework is None or info.get("features_model") == "nnunet1000_loocv") else None


def make_crop(gray: np.ndarray, region: Optional[np.ndarray], size: int = CROP) -> Optional[np.ndarray]:
    """A small gray image of the cell with its outline, or None when there is no region."""
    import cv2
    if region is None or not region.any():
        return None
    ys, xs = np.nonzero(region)
    cx, cy = int(xs.mean()), int(ys.mean())
    h = size
    x0, y0 = min(max(cx - h, 0), gray.shape[1] - 2 * h), min(max(cy - h, 0), gray.shape[0] - 2 * h)
    img = cv2.cvtColor(np.ascontiguousarray(gray[y0:y0 + 2 * h, x0:x0 + 2 * h]), cv2.COLOR_GRAY2BGR)
    cs, _ = cv2.findContours(np.ascontiguousarray(region[y0:y0 + 2 * h, x0:x0 + 2 * h]).astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, (255, 255, 255), 2)
    cv2.drawContours(img, cs, -1, (60, 60, 60), 1)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def build_state(sess, rec=None, n_measured: int = 0, elapsed_s: float = 0.0, segmenter: str = "unet",
                device: str = "", reference: Optional[pd.DataFrame] = None) -> State:
    """Snapshot of a :class:`~celldform.demo.session.CellSession` (copies, so it is safe to hand to another thread)."""
    from celldform.demo.framework import MIN_STEPS_SHORT
    t = mA = None
    log = getattr(rec, "frame_log", None)
    if log is not None and "set_current_mA" in log:
        t, mA = log.time_s.to_numpy(float), log.set_current_mA.to_numpy(float)
    f = sess.frames.copy()
    return State(frames=f, steps=sess.steps().copy(), cell_features=dict(sess.cell_features()) if len(f) else {},
                 verdict=sess.verdict(), t_s=t, mA=mA, fps=sess.fps, frames_per_step=sess.frames_per_step,
                 total_frames=int(getattr(rec, "declared_frames", 0) or 0), n_measured=n_measured,
                 elapsed_s=elapsed_s, segmenter=segmenter, device=device, crops=dict(getattr(sess, "crops", {})),
                 info=dict(getattr(sess.framework, "info", {}) or {}), reference=reference,
                 min_steps=MIN_STEPS_SHORT, min_pass=sess.min_pass_fraction, source=str(getattr(rec, "source", "video")),
                 title=str(getattr(rec, "cell_label", "") or (getattr(rec, "name", "") if rec is not None else "")))


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────────

def _step_table(s: State) -> pd.DataFrame:
    """One row per measured laser step: frames measured, frames passing, pass fraction, mean set current."""
    f = s.frames
    if f.empty:
        return pd.DataFrame(columns=["laser_step", "n", "passed", "frac", "mA"])
    g = f.assign(passed=f.passed.astype(bool)).groupby("laser_step")
    t = g.agg(n=("passed", "size"), passed=("passed", "sum")).reset_index()
    t["frac"] = t.passed / t.n
    t["mA"] = [_step_current(s, k) for k in t.laser_step]
    return t


def _step_current(s: State, step: int) -> float:
    if s.mA is None:
        return float("nan")
    lo, hi = step * s.frames_per_step, min((step + 1) * s.frames_per_step, len(s.mA))
    return float(np.mean(s.mA[lo:hi])) if hi > lo else float("nan")


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK)


def _empty(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#777", fontsize=9, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])


def reason_group(reason: str) -> str:
    r = str(reason)
    for key, name in (("no cell near", "no cell near the trap"), ("px from the trap", "too far from the trap"),
                      ("frame edge", "cut by the frame edge"), ("solidity", "irregular outline"),
                      ("area x", "size off typical"), ("boundary confidence", "uncertain boundary")):
        if key in r:
            return name
    return "other"


# ── panels ────────────────────────────────────────────────────────────────────────────────────────────────────────

def panel_protocol(ax, s: State) -> None:
    """The laser protocol of this recording against the trained range, with each measured step coloured."""
    _style(ax, "Laser protocol and what was measured")
    if s.mA is None:
        _empty(ax, "no laser log in this recording"); return
    t = s.t_s
    ax.axhspan(*TRAINED_RANGE_MA, color=LOW, alpha=0.08, lw=0)
    ax.text(t[-1], TRAINED_RANGE_MA[1], " range the classifier was trained on (500-1200 mA)", fontsize=7, color=LOW,
            ha="right", va="bottom")
    ax.plot(t, s.mA, color=INK, lw=1)
    ax.set_xlabel("video time (s)", fontsize=8); ax.set_ylabel("set current (mA)", fontsize=8)
    ax.set_xlim(0, t[-1]); ax.set_ylim(min(s.mA.min() * 0.85, 250), max(s.mA.max(), TRAINED_RANGE_MA[1]) * 1.08)
    tab = _step_table(s)
    done = set(tab.laser_step.astype(int))
    n_steps = int(np.ceil(len(s.mA) / s.frames_per_step))
    y0, hgt = ax.get_ylim()[0], 0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for k in range(n_steps):
        x0, x1 = k * s.frames_per_step / s.fps, (k + 1) * s.frames_per_step / s.fps
        if k in done:
            fr = float(tab.loc[tab.laser_step == k, "frac"].iloc[0])
            col = OK if fr >= s.min_pass else (BAD if fr == 0 else "#e0a030")
        else:
            col = NONE
        ax.add_patch(__import__("matplotlib").patches.Rectangle((x0, y0), x1 - x0 - 0.15, hgt, color=col, lw=0))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=c, label=l)
                       for c, l in ((OK, f">= {s.min_pass:.0%} of frames pass"), ("#e0a030", "some pass"),
                                    (BAD, "none pass"), (NONE, "not measured"))],
              fontsize=7, loc="upper left", frameon=False, ncol=2)


def panel_pass_rate(ax, s: State) -> None:
    """Share of measured frames that pass the five checks, per laser step, against the 80% the classifier needs."""
    _style(ax, "Frames passing the checks, per laser step")
    tab = _step_table(s)
    if tab.empty:
        _empty(ax, "no frames measured yet"); return
    cols = [OK if f >= s.min_pass else (BAD if f == 0 else "#e0a030") for f in tab.frac]
    ax.bar(tab.laser_step, tab.frac * 100, color=cols, width=0.85)
    ax.axhline(s.min_pass * 100, color=INK, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], s.min_pass * 100 + 2, f"{s.min_pass:.0%} needed", ha="right", fontsize=7, color=INK)
    ax.set_ylim(0, 105); ax.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))
    ax.set_xlabel("laser step (200 frames each)", fontsize=8); ax.set_ylabel("% of frames", fontsize=8)


def panel_reasons(ax, s: State) -> None:
    """Why measured frames were rejected."""
    _style(ax, "Why frames were rejected")
    f = s.frames
    if f.empty:
        _empty(ax, "no frames measured yet"); return
    bad = f[~f.passed.astype(bool)]
    counts = bad.reasons.map(lambda r: reason_group(str(r).split(";")[0])).value_counts()
    n_ok = int(f.passed.astype(bool).sum())
    labels = ["passed"] + list(counts.index)
    vals = [n_ok] + [int(v) for v in counts.values]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=[OK] + [BAD] * len(counts), height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v}", va="center", fontsize=8, color=INK)
    ax.set_xlabel(f"frames (of {len(f)} measured)", fontsize=8)


def panel_shape(ax, s: State) -> None:
    """Cell size and aspect ratio of the frames that passed, per laser step, against the set current."""
    _style(ax, "Cell shape across the laser ramp (passing frames)")
    ok = s.frames[s.frames.passed.astype(bool)] if len(s.frames) else s.frames
    if ok.empty:
        _empty(ax, "no frame passed the checks, so there is no shape to show"); return
    g = ok.groupby("laser_step").agg(area=("area", "median"), ar=("aspect_ratio", "median")).reset_index()
    x = np.array([_step_current(s, int(k)) for k in g.laser_step])
    xl = "set current (mA)"
    if np.isnan(x).all():
        x, xl = g.laser_step.to_numpy(float), "laser step"
    ax.plot(x, g.area, "o-", color=LOW, ms=3, lw=1.2)
    ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("area (px)", fontsize=8, color=LOW)
    a2 = ax.twinx(); a2.plot(x, g.ar, "s-", color=HIGH, ms=3, lw=1.2)
    a2.set_ylabel("aspect ratio", fontsize=8, color=HIGH); a2.tick_params(labelsize=8)
    a2.spines["top"].set_visible(False)


def panel_verdict(ax, s: State) -> None:
    """The result: a probability gauge when there is a verdict, otherwise what is still missing."""
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Result", loc="left", fontsize=10, fontweight="bold", color=INK)
    v = s.verdict
    if v is None:
        ax.text(0.5, 0.5, "nothing run yet", ha="center", color="#777"); return
    if v.label is not None:                                  # a verdict: validated ("ok"), short-recording or provisional
        col = HIGH if v.label == "high" else LOW
        flag = v.status != "ok"
        g = 0.55 if flag else 0.30                          # bottom of the gauge; a flagged one leaves room below for why
        ax.text(0.0, 0.96 if flag else 0.86, f"{v.label.upper()} HER2", fontsize=20, fontweight="bold", color=col, va="center")
        if flag:
            banner = "SHORT RECORDING: STEP MODEL ONLY" if v.status == "short" else "PROVISIONAL: NOT VALIDATED"
            ax.text(0.0, 0.86, banner, fontsize=11, fontweight="bold", color="#b85c00", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fff3e0", ec="#b85c00"))
        ax.text(0.0, 0.77 if flag else 0.70, f"P(high) = {v.p_high:.2f}", fontsize=11, color=INK, va="center")
        xs = np.linspace(0, 1, 200)
        for lo, hi in zip(xs[:-1], xs[1:]):
            ax.axvspan(lo, hi, g, g + 0.12, color=plt_blend(LOW, HIGH, (lo + hi) / 2), lw=0)
        ax.plot([0.5, 0.5], [g - 0.04, g + 0.16], color=INK, lw=1)
        for p, name in ((v.p_step, "step model"), (v.p_cell, "cell model"), (v.p_high, "average")):
            if p == p:                                       # the cell model is absent from a short recording
                ax.plot([p], [g + 0.06], marker="v" if name == "average" else "o", ms=9 if name == "average" else 6,
                        color=INK, mec="white", zorder=5, clip_on=False)
        ax.text(0.0, g + 0.17, "low", fontsize=7, color=LOW); ax.text(1.0, g + 0.17, "high", fontsize=7, color=HIGH, ha="right")
        cell_txt = f"cell model {v.p_cell:.2f}" if v.p_cell == v.p_cell else "cell model not used"
        if flag:                                             # what is missing, so nobody takes it for a validated result
            import textwrap
            ax.text(0.0, g - 0.11, f"step model {v.p_step:.2f}    {cell_txt}    average {v.p_high:.2f}",
                    fontsize=8, color=INK, va="center")
            why = "\n".join(textwrap.wrap(("Not validated: " if v.status == "provisional" else "") + v.reason, 64))
            ax.text(0.0, g - 0.2, why, fontsize=6.5, color="#a05a00", va="top", linespacing=1.3)
        else:
            for yy, (p, name) in zip((0.18, 0.10, 0.02), ((v.p_step, "step model"), (v.p_cell, "cell model"), (v.p_high, "average"))):
                ax.text(0.0, yy, f"{name}: {p:.2f}", fontsize=8, color=INK, va="center")
        return
    ax.text(0.0, 0.88, "No verdict", fontsize=18, fontweight="bold", color=INK, va="center")
    ax.text(0.0, 0.73, "what is still missing:", fontsize=9, color="#555", va="center")
    st = len(s.steps)
    for i, (name, val, need, txt) in enumerate((
            ("usable laser steps", min(st / s.min_steps, 1), 1, f"{st} (need {s.min_steps})"),
            ("frames passing the checks", min(v.pass_fraction / s.min_pass, 1) if v.pass_fraction == v.pass_fraction else 0,
             1, f"{v.pass_fraction:.0%} of {s.min_pass:.0%}"))):
        y = 0.52 - i * 0.26
        ax.text(0.0, y + 0.10, name, fontsize=8, color=INK)
        ax.barh(y, 1, height=0.07, left=0, color="#e6e6e6")
        ax.barh(y, val, height=0.07, left=0, color=OK if val >= 1 else BAD)
        ax.text(1.0, y + 0.10, txt, fontsize=8, color=INK, ha="right")


def plt_blend(c1: str, c2: str, t: float) -> Tuple[float, float, float]:
    from matplotlib.colors import to_rgb
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * t)


def panel_reference(ax, s: State) -> None:
    """This cell among the 15 training cells, on two of the classifier's four selected features."""
    _style(ax, "Where this cell sits among the 15 training cells")
    ref, fx, fy = s.reference, "len_par_base", "fluctuation"
    if ref is None or fx not in ref or fy not in ref:
        _empty(ax, "no training-cell reference for this classifier"); return
    for lab, col in (("low", LOW), ("high", HIGH)):
        r = ref[ref.her2 == lab]
        ax.scatter(r[fx], r[fy], s=26, color=col, alpha=0.85, label=f"{lab} HER2 (training)", edgecolor="white", lw=0.5)
    cf = s.cell_features or {}
    x, y = cf.get(fx, np.nan), cf.get(fy, np.nan)
    if np.isfinite(x) and np.isfinite(y):
        ax.scatter([x], [y], s=170, marker="*", color="black", zorder=5, label="this cell")
    else:
        ax.text(0.98, 0.04, "this cell: features not available\n(too few passing frames)", transform=ax.transAxes,
                ha="right", fontsize=7, color=BAD)
    ax.set_xlabel(fx.replace("_", " "), fontsize=8); ax.set_ylabel(fy.replace("_", " "), fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="best")


def panel_filmstrip(ax, s: State) -> None:
    """The cell as the model saw it, one picture per laser step (first passing frame of the step)."""
    ax.axis("off"); ax.set_title("The cell across laser steps (outline = predicted mask)", loc="left", fontsize=10,
                                 fontweight="bold", color=INK)
    if not s.crops:
        ax.text(0.5, 0.5, "no frame has passed the checks yet", ha="center", va="center", color="#777", fontsize=9,
                transform=ax.transAxes); return
    keys = sorted(s.crops)
    if len(keys) > 10:
        keys = [keys[int(i)] for i in np.linspace(0, len(keys) - 1, 10)]
    tiles = []
    import cv2
    for k in keys:
        t = s.crops[k].copy()
        cv2.rectangle(t, (0, 0), (56, 15), (0, 0, 0), -1)
        cv2.putText(t, f"step {k}", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tiles.append(t)
    ax.imshow(np.hstack(tiles)[..., ::-1]); ax.axis("off")


def panel_run(ax, s: State) -> None:
    """Speed, volume of work, and what the classifier was trained and checked on."""
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Run and model", loc="left", fontsize=10, fontweight="bold", color=INK)
    ms = 1000 * s.elapsed_s / max(s.n_measured, 1)
    i = s.info
    lines = [("segmentation model", {"unet": "U-Net", "nnunet": "nnU-Net"}.get(s.segmenter, s.segmenter) +
              (f" ({s.device})" if s.device else "")),
             ("frames read from", {"video": "the video file", "images": "the still images"}.get(s.source, s.source)),
             ("frames in the video", f"{s.total_frames:,}" if s.total_frames else "?"),
             ("frames measured", f"{s.n_measured:,}" + (f"  ({s.n_measured / s.total_frames:.1%})" if s.total_frames else "")),
             ("time spent", f"{s.elapsed_s:.0f} s" if s.n_measured else "-"),
             ("per measured frame", f"{ms:.0f} ms  (camera interval {CAMERA_MS} ms, reference only)" if s.n_measured else "-"),
             ("classifier trained on", f"{i.get('n_high', '?')} high + {i.get('n_low', '?')} low HER2 cells" if i else "-"),
             ("leave-one-cell-out", f"{i.get('loocv_correct', '?')}/15 correct, AUC {i.get('loocv_auc', float('nan')):.2f}"
              if i else "-"),
             ("trained current range", f"{TRAINED_RANGE_MA[0]:.0f}-{TRAINED_RANGE_MA[1]:.0f} mA, 30 s per level"),
             ("this recording", (f"{s.mA.min():.0f}-{s.mA.max():.0f} mA, steps of ~9 s" if s.mA is not None else "no laser log"))]
    for k, (a, b) in enumerate(lines):
        y = 0.95 - k * 0.098
        ax.text(0.0, y, a, fontsize=8, color="#666", va="center")
        ax.text(0.36, y, b, fontsize=8, color=INK, va="center")


# ── pages ─────────────────────────────────────────────────────────────────────────────────────────────────────────

PAGES = {
    "Process": [(panel_protocol, (0, 0, 2, 1)), (panel_pass_rate, (1, 0, 1, 1)), (panel_reasons, (1, 1, 1, 1))],
    "Evidence": [(panel_shape, (0, 0, 1, 1)), (panel_reference, (0, 1, 1, 1)), (panel_filmstrip, (1, 0, 1, 2))],
    "Result and model": [(panel_verdict, (0, 0, 1, 1)), (panel_run, (0, 1, 1, 1))],
}


def draw_page(fig, s: State, page: str) -> None:
    """Redraw one page of panels onto ``fig``."""
    fig.clear()
    specs = PAGES[page]
    if page == "Process":
        gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.55, wspace=0.28, left=0.06, right=0.98, top=0.9, bottom=0.12)
        axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    elif page == "Evidence":
        gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 0.8], hspace=0.5, wspace=0.35, left=0.06, right=0.95, top=0.88, bottom=0.1)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.12, left=0.03, right=0.98, top=0.9, bottom=0.08)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for (fn, _), ax in zip(specs, axes):
        fn(ax, s)


def save(s: State, path: Path | str, dpi: int = 130) -> Path:
    """Every page on one image: the shareable summary of a run."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(13, 15.5), dpi=dpi, facecolor="white")
    top = fig.subfigures(4, 1, height_ratios=[0.35, 1.2, 1.2, 0.85], hspace=0.02)
    head = top[0].subplots(1, 1); head.axis("off")
    v = s.verdict
    res = ({"ok": "", "short": "SHORT RECORDING, STEP MODEL ONLY: "}.get(v.status if v is not None else "", "PROVISIONAL, NOT VALIDATED: ")
           + f"{v.label.upper()} HER2  (P high {v.p_high:.2f})"
           if v is not None and v.label is not None
           else "No HER2 verdict: " + (v.reason if v is not None else "nothing run"))
    head.text(0.0, 0.75, f"celldform demo: {s.title}".strip(), fontsize=16, fontweight="bold", color=INK, transform=head.transAxes)
    head.text(0.0, 0.35, res, fontsize=12, color=INK, transform=head.transAxes)
    if v is not None and v.caveats:
        head.text(0.0, -0.05, "caveats: " + "; ".join(v.caveats), fontsize=7, color="#a05a00", transform=head.transAxes, wrap=True)
    for sub, page in zip(top[1:], PAGES):
        draw_page(sub, s, page)
        sub.suptitle(page, x=0.01, y=0.995, ha="left", fontsize=12, color="#777")
    path = Path(path)
    fig.savefig(path, dpi=dpi, facecolor="white")
    return path
