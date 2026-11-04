"""
Per-frame HER2 classifier (deep learning), judged per held-out cell.

    python scripts/classification/frames_cnn.py crops      --model nnunet100_5fold   # cut the cell crops once
    python scripts/classification/frames_cnn.py frozen     --model nnunet100_5fold   # ResNet-18 features + LR
    python scripts/classification/frames_cnn.py finetune   --model nnunet100_5fold   # fine-tune the last block
    python scripts/classification/frames_cnn.py report     --model nnunet100_5fold   # table + figure
    python scripts/classification/frames_cnn.py final      --model nnunet100_5fold --seeds 0 1 2
                                     # train on ALL cells, save to checkpoints/classification/<model>/cnn/

Data: every passing feature frame of the usable cells (~3 800 frames, ~6 per laser step).

Input -- built so the network cannot read the recording session from the frame:
    masked   a CROP x CROP px window centred on the trapped cell (the region the extraction kept), every pixel
             outside the (slightly grown) mask set to zero, intensity standardised over the cell's own pixels.
             The scale is NOT normalised: the cell's size stays in the input, as in the per-cell features.
    mask     the binary mask alone, same window (shape and size only; the control).

Models (ImageNet ResNet-18, grey crop copied to 3 channels):
    frozen    the pretrained network as a fixed feature extractor (512 numbers per frame) + logistic regression.
    finetune  the last residual block and a new head trained per fold (flips, 90-degree turns, mild intensity
              change), the laser step joined to the image features before the head. Cells are weighted equally
              (each cell's frames share one unit of weight), a fixed number of epochs (no peeking at the held-out
              cell).

Test: leave-one-cell-out, the same 14 folds as benchmark_cells.py. A held-out cell's frame probabilities are
averaged into one P(high) for the cell -- what a live system watching the cell would do.

Outputs in results/classification/<model>/:  cnn_frames_<variant>.csv (every frame's held-out P(high)),
cnn_cells.csv, cnn_cells.tex, cnn_frames_seen.png (accuracy against the number of laser steps watched).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.features.quality import select_trapped_region   # noqa: E402
from celldform.analysis_config import as_dict, dump as dump_config, load as load_config   # noqa: E402
from celldform.utils.runlog import start_run_log   # noqa: E402

SEED = 0           # the reference seed: its outputs carry no _seed suffix
# Settings from configs/analysis/cnn.yaml (--config for another file):
#   CROP  px at frame resolution: the largest cell is ~45 px across, so the window holds it at any step
#   GROW  px the mask is grown by before the background is cut away (keeps the cell's edge)
#   NET   px the crop is resized to for the network
# CROP and GROW are baked into the saved crops: rerun the `crops` step after changing them.


def configure(cfg) -> None:
    global CFG, CROP, GROW, NET
    CFG, CROP, GROW, NET = cfg, cfg.crop_px, cfg.grow_px, cfg.input_px


configure(load_config("cnn"))
FACTORY = ROOT / "data" / "frames" / "03_mask_factory"
DENSE = ROOT / "data" / "frames" / "05_dense_frames"


def res_dir(model: str) -> Path:
    return ROOT / "results" / "classification" / model


def cache_path(model: str) -> Path:
    return ROOT / "data" / "features" / model / "cnn_crops.npz"


# ── crops ────────────────────────────────────────────────────────────────────

def label_and_raw(model: str, stem: str) -> tuple[Path, Path]:
    """Same lookup as extract_features.located(): the factory copy wins over the dense one."""
    for sub, raw in (("feature", FACTORY), ("dense", DENSE)):
        lab = ROOT / "data" / "features" / model / sub / "labels" / f"{stem}.png"
        if lab.exists():
            return lab, raw / f"{stem}.jpg"
    raise FileNotFoundError(stem)


def cut_crop(raw: np.ndarray, label: np.ndarray, trap_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    """The network's two inputs from one frame: the masked crop (float16) and the binary mask crop (uint8), a CROP px
    window centred on the trapped-cell region nearest the trap; None if there is no such region. `raw` is the grey
    frame, `label` the segmentation (1 = trapped cell)."""
    reg, _ = select_trapped_region(label == 1, trap_xy)        # the region the features came from
    if reg is None:
        return None
    h = CROP // 2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1, 2 * GROW + 1))
    m = reg["mask"].astype(np.uint8)
    cx, cy = (int(round(v)) for v in reg["centroid"])
    pad = lambda a: np.pad(a, h, mode="constant")
    win = (slice(cy, cy + CROP), slice(cx, cx + CROP))              # (cy, cx) shifted by the pad of h
    rc, mc = pad(raw)[win].astype(np.float32), pad(m)[win]
    grown = cv2.dilate(mc, k) > 0
    v = rc[mc > 0]
    z = np.where(grown, (rc - v.mean()) / (v.std() + 1e-6), 0.0)
    return z.astype(np.float16), mc.astype(np.uint8)


def crops(model: str) -> None:
    feat = ROOT / "results" / "features" / model
    cells = pd.read_csv(feat / "features_cells.csv")
    usable = set(cells.cell[cells.usable])
    df = pd.read_csv(feat / "features_frames.csv")
    df = df[df.passed & df.cell.isin(usable)].reset_index(drop=True)
    masked, binary, keep = [], [], []
    for i, r in df.iterrows():
        lab_p, raw_p = label_and_raw(model, r.stem)
        lab, raw = cv2.imread(str(lab_p), cv2.IMREAD_UNCHANGED), cv2.imread(str(raw_p), cv2.IMREAD_GRAYSCALE)
        cut = cut_crop(raw, lab, (r.trap_x, r.trap_y))
        if cut is None:
            continue
        masked.append(cut[0])
        binary.append(cut[1])
        keep.append(i)
        if len(keep) % 500 == 0:
            print(f"  {len(keep)}/{len(df)}")
    meta = df.loc[keep, ["stem", "cell", "her2", "laser_step", "offset", "frame_index"]].reset_index(drop=True)
    out = cache_path(model)
    np.savez_compressed(out, masked=np.stack(masked), mask=np.stack(binary), **{c: meta[c].to_numpy() for c in meta})
    print(f"{len(meta)} crops of {meta.cell.nunique()} cells -> {out}")


def load_crops(model: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    z = np.load(cache_path(model), allow_pickle=True)
    meta = pd.DataFrame({c: z[c] for c in ("stem", "cell", "her2", "laser_step", "offset", "frame_index")})
    return {"masked": z["masked"].astype(np.float32), "mask": z["mask"].astype(np.float32)}, meta


# ── network ──────────────────────────────────────────────────────────────────

def to_net(x: np.ndarray, variant: str):
    """(N, CROP, CROP) -> (N, 3, NET, NET) torch tensor in the range ResNet expects."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(x).float().unsqueeze(1)
    if variant == "mask":
        t = (t - 0.5) / 0.5
    t = F.interpolate(t, size=(NET, NET), mode="bilinear", align_corners=False)
    return t.repeat(1, 3, 1, 1)


def resnet(pretrained: bool = True):
    import torchvision
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return torchvision.models.resnet18(weights=w)


def embed(x: np.ndarray, variant: str, device: str) -> np.ndarray:
    """512-number ImageNet features per frame (frozen network)."""
    import torch
    net = resnet()
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    out = []
    with torch.no_grad():
        for b in range(0, len(x), 256):
            out.append(net(to_net(x[b:b + 256], variant).to(device)).cpu().numpy())
    return np.concatenate(out)


def folds(meta: pd.DataFrame):
    for c in sorted(meta.cell.unique(), key=lambda c: (c.startswith("low"), int(c.split("cell")[1]))):
        yield c, (meta.cell != c).to_numpy(), (meta.cell == c).to_numpy()


def cell_weights(cells: pd.Series) -> np.ndarray:
    """Each cell's frames share one unit of weight, so a cell with more frames does not count more."""
    return (1.0 / cells.map(cells.value_counts())).to_numpy()


def frozen(model: str, device: str) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, meta = load_crops(model)
    y = (meta.her2 == "high").astype(int).to_numpy()
    for variant in ("masked", "mask"):
        E = embed(X[variant], variant, device)
        p = np.zeros(len(y))
        for c, tr, te in folds(meta):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=CFG.frozen_lr_C, max_iter=5000))
            clf.fit(E[tr], y[tr], logisticregression__sample_weight=cell_weights(meta.cell[tr]))
            p[te] = clf.predict_proba(E[te])[:, 1]
            print(f"frozen/{variant:6s} {c:12s} P(high) {p[te].mean():.2f}")
        save_frames(model, f"frozen_{variant}", meta, p)


def train_predict(X: np.ndarray, y: np.ndarray, step: np.ndarray, meta: pd.DataFrame, tr: np.ndarray,
                  te: np.ndarray, variant: str, epochs: int, lr: float, seed: int, device: str) -> tuple[np.ndarray, float]:
    """Fine-tune on the frames `tr` (boolean mask), return P(high) of the frames `te` and the last batch loss."""
    net, head, loss = train(X, y, step, meta, tr, variant, epochs, lr, seed, device)
    return predict(net, head, X[te], step[te], variant, device), loss


def train(X: np.ndarray, y: np.ndarray, step: np.ndarray, meta: pd.DataFrame, tr: np.ndarray, variant: str,
          epochs: int, lr: float, seed: int, device: str):
    """Fine-tune the last block and a new head on the frames `tr`; return the network, the head and the last loss."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = resnet()
    net.fc = nn.Identity()
    trainable = tuple(CFG.trainable_blocks)
    for name, prm in net.named_parameters():
        prm.requires_grad = name.startswith(trainable)
    head = nn.Sequential(nn.Dropout(CFG.dropout), nn.Linear(512 + 1, 1))
    net.to(device), head.to(device)
    params = [q for q in net.parameters() if q.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=CFG.weight_decay)
    w = cell_weights(meta.cell[tr])
    idx_tr = np.flatnonzero(tr)
    n_draw = len(idx_tr)
    loss = torch.tensor(np.nan)
    for ep in range(epochs):
        net.train(), head.train()
        # BatchNorm stays in eval for the frozen blocks (their statistics are ImageNet's)
        for blk in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"):
            if not blk.startswith(trainable):
                getattr(net, blk).eval()
        draw = rng.choice(idx_tr, size=n_draw, p=w / w.sum())   # cell-balanced sampling
        for b in range(0, n_draw, CFG.batch_size):
            ib = draw[b:b + CFG.batch_size]
            xb = augment(X[ib], variant, rng)
            xb = to_net(xb, variant).to(device)
            sb = torch.from_numpy(step[ib]).to(device)
            logit = head(torch.cat([net(xb), sb], 1)).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[ib]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval(), head.eval()
    return net, head, float(loss.item())


def predict(net, head, X: np.ndarray, step: np.ndarray, variant: str, device: str) -> np.ndarray:
    """P(high) of every frame: crops X (N, CROP, CROP), step (N, 1) = laser step / 49."""
    import torch
    with torch.no_grad():
        out = []
        for b in range(0, len(X), 256):
            f = net(to_net(X[b:b + 256], variant).to(device))
            s = torch.from_numpy(np.ascontiguousarray(step[b:b + 256], dtype=np.float32)).to(device)
            out.append(torch.sigmoid(head(torch.cat([f, s], 1))).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def final(model: str, device: str, epochs: int, seeds: list[int], variants=("masked",)) -> None:
    """The fine-tuned CNN trained once on ALL frames of all usable cells, per seed, and saved (full weights, so
    loading needs no ImageNet download) with the input recipe and its leave-one-cell-out score. A cell is scored
    by the mean P(high) of its frames, averaged over the seeds' models (as in the report)."""
    import torch
    from celldform import final_models
    torch.backends.cudnn.benchmark = False
    X, meta = load_crops(model)
    y = (meta.her2 == "high").astype(np.float32).to_numpy()
    step = (meta.laser_step.to_numpy(np.float32) / 49.0)[:, None]
    every = np.ones(len(y), bool)
    cells = meta.groupby("cell").her2.first()
    loo = pd.read_csv(res_dir(model) / "cnn_cells.csv") if (res_dir(model) / "cnn_cells.csv").exists() else None
    folder = final_models.ckpt_dir("classification", model, "cnn")
    rows = []
    def loocv_score(variant: str, seed: int) -> dict:
        """This variant and seed's leave-one-cell-out row of cnn_cells.csv, and the mean-of-seeds row (crop only)."""
        if loo is None:
            return {}
        name = {"masked": "CNN (ResNet-18) fine-tuned, cell crop", "mask": "CNN (ResNet-18) fine-tuned, mask only"}[variant]
        rows_ = loo.model.str.strip().tolist()
        i = rows_.index(name) if name in rows_ else None
        want = {0: i} if i is not None else {}
        if variant == "masked" and i is not None:      # the seed rows follow the crop row as "same, seed k"
            want |= {k: rows_.index(f"same, seed {k}") for k in (1, 2) if f"same, seed {k}" in rows_}
            want["mean"] = rows_.index("same, mean of 3 seeds") if "same, mean of 3 seeds" in rows_ else None
        out = {}
        if want.get(seed) is not None:
            out |= {"loocv_correct": int(loo.correct.iloc[want[seed]]), "loocv_auc": float(loo.auc.iloc[want[seed]])}
        if want.get("mean") is not None:
            out |= {"loocv_seedmean_correct": int(loo.correct.iloc[want["mean"]]),
                    "loocv_seedmean_auc": float(loo.auc.iloc[want["mean"]])}
        return out

    for variant in variants:
        for seed in seeds:
            score = loocv_score(variant, seed)
            net, head, loss = train(X[variant], y, step, meta, every, variant, epochs, CFG.lr, seed, device)
            meta_ = final_models.info(model, cells.index, cells.values, variant=variant, seed=seed, epochs=epochs,
                                      lr=CFG.lr, n_frames=len(y), last_loss=loss, crop_px=CROP, grow_px=GROW,
                                      input_px=NET, step_divisor=49.0, config=as_dict(CFG), **score)
            file = f"finetune_{variant}_seed{seed}.pt"
            torch.save({"net": net.state_dict(), "head": head.state_dict(), "info": meta_}, folder / file)
            rows.append({"file": file, "variant": variant, "seed": seed, "epochs": epochs, "n_frames": len(y),
                         "last_loss": loss, **score, "created": meta_["created"]})
            print(f"final/{variant} seed {seed}: trained on {len(y)} frames of {len(cells)} cells "
                  f"(last loss {loss:.3f}) -> {folder / file}", flush=True)
    final_models.write_manifest(folder, rows)


def load_final(path: str | Path, device: str = "cpu"):
    """A saved final CNN: returns (net, head, info). P(high) of crops from cut_crop():
    predict(net, head, crops, laser_steps[:, None] / info["step_divisor"], info["variant"], device)."""
    import torch
    from torch import nn
    d = torch.load(path, map_location=device, weights_only=False)
    net = resnet(pretrained=False)
    net.fc = nn.Identity()
    net.load_state_dict(d["net"])
    head = nn.Sequential(nn.Dropout(d["info"]["config"]["dropout"]), nn.Linear(512 + 1, 1))
    head.load_state_dict(d["head"])
    return net.to(device).eval(), head.to(device).eval(), d["info"]


def finetune(model: str, device: str, epochs: int, seed: int = SEED, variants=("masked", "mask")) -> None:
    import torch
    torch.backends.cudnn.benchmark = False
    X, meta = load_crops(model)
    y = (meta.her2 == "high").astype(np.float32).to_numpy()
    step = (meta.laser_step.to_numpy(np.float32) / 49.0)[:, None]
    for variant in variants:
        p = np.zeros(len(y))
        for c, tr, te in folds(meta):
            p[te], loss = train_predict(X[variant], y, step, meta, tr, te, variant, epochs, CFG.lr, seed, device)
            print(f"finetune/{variant:6s} {c:12s} P(high) {p[te].mean():.2f}  (last loss {loss:.3f})", flush=True)
        save_frames(model, f"finetune_{variant}" + (f"_seed{seed}" if seed != SEED else ""), meta, p)


def inner_folds(cells: list[str], her2: dict, k: int) -> list[list[str]]:
    """The training cells dealt into k folds, high cells first then low, so every fold holds both where possible."""
    order = sorted(cells, key=lambda c: (her2[c] == "low", int(c.split("cell")[1])))
    folds_ = [[] for _ in range(k)]
    for i, c in enumerate(order):
        folds_[i % k].append(c)
    return folds_


def tune(model: str, device: str, seed: int) -> None:
    """Nested CV of the fine-tuned cell-crop CNN over the config's `tuning.grid` (configs/analysis/cnn.yaml): for
    each test cell, every grid point is scored on the other 13 cells in `tuning.inner_folds` folds (Brier score of
    the cells' mean P(high)); the best is retrained on the 13 and the test cell is predicted once."""
    import itertools
    import torch
    torch.backends.cudnn.benchmark = False
    X, meta = load_crops(model)
    X = X["masked"]
    y = (meta.her2 == "high").astype(np.float32).to_numpy()
    step = (meta.laser_step.to_numpy(np.float32) / 49.0)[:, None]
    grid = CFG.tuning["grid"]
    points = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    her2 = meta.groupby("cell").her2.first().to_dict()
    p, rows = np.zeros(len(y)), []
    for c, tr, te in folds(meta):
        train_cells = sorted(set(meta.cell[tr]))
        briers = []
        for pt in points:
            cell_p = {}
            for f in inner_folds(train_cells, her2, CFG.tuning["inner_folds"]):
                ite = meta.cell.isin(f).to_numpy()
                itr = tr & ~ite
                q, _ = train_predict(X, y, step, meta, itr, ite, "masked", int(pt["epochs"]), float(pt["lr"]),
                                     seed, device)
                cell_p |= pd.Series(q, index=meta.cell[ite].to_numpy()).groupby(level=0).mean().to_dict()
            yc = np.array([her2[k] == "high" for k in cell_p], float)
            briers.append(float(np.mean((np.array(list(cell_p.values())) - yc) ** 2)))
        best = points[int(np.argmin(briers))]
        p[te], _ = train_predict(X, y, step, meta, tr, te, "masked", int(best["epochs"]), float(best["lr"]), seed, device)
        rows.append({"test_cell": c, **best, **{f"brier_{i}": b for i, b in enumerate(briers)}})
        print(f"tune {c:12s} best {best}  inner Brier {min(briers):.3f}  P(high) {p[te].mean():.2f}", flush=True)
    save_frames(model, "tuned_masked", meta, p)
    pd.DataFrame(rows).to_csv(res_dir(model) / "cnn_tuning_choices.csv", index=False)
    pd.Series([str(pt) for pt in points]).to_csv(res_dir(model) / "cnn_tuning_grid.csv", header=["grid_point"])


def augment(x: np.ndarray, variant: str, rng: np.random.Generator) -> np.ndarray:
    """Flips and 90-degree turns (the cell has no preferred side in the crop apart from the beam axis, and the
    turns are kept to multiples of 90 so the pixel grid is not resampled); mild intensity change on the
    masked crop only, inside the cell."""
    aug = CFG.augment
    out = x.copy()
    for i in range(len(out)):
        a = out[i]
        if aug.flips and rng.random() < 0.5:
            a = a[:, ::-1]
        if aug.flips and rng.random() < 0.5:
            a = a[::-1, :]
        if aug.rot90:
            a = np.rot90(a, rng.integers(4))
        if variant == "masked":
            inside = a != 0
            a = np.where(inside, a * rng.uniform(*aug.intensity_scale) + rng.normal(0, aug.intensity_noise), 0)
        out[i] = a
    return np.ascontiguousarray(out)


# ── results ──────────────────────────────────────────────────────────────────

def save_frames(model: str, name: str, meta: pd.DataFrame, p: np.ndarray) -> None:
    out = res_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    meta.assign(p_high=p).to_csv(out / f"cnn_frames_{name}.csv", index=False)


def report(model: str, img: Path | None = None, tables: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    out = res_dir(model)
    rows, curves = [], {}
    labels = {"frozen_masked": "CNN (ResNet-18) frozen, cell crop", "frozen_mask": "CNN (ResNet-18) frozen, mask only",
              "finetune_masked": "CNN (ResNet-18) fine-tuned, cell crop"}
    seeds = sorted(out.glob("cnn_frames_finetune_masked_seed[0-9]*.csv"))
    for f in seeds:
        labels[f.stem.removeprefix("cnn_frames_")] = f"  same, seed {f.stem.split('seed')[1]}"
    if seeds:   # the seed runs averaged frame by frame: the seed spread taken out
        runs = [pd.read_csv(out / "cnn_frames_finetune_masked.csv")] + [pd.read_csv(f) for f in seeds]
        runs[0].assign(p_high=np.mean([r.p_high for r in runs], 0)).to_csv(
            out / "cnn_frames_finetune_masked_seedmean.csv", index=False)
        labels["finetune_masked_seedmean"] = f"  same, mean of {len(runs)} seeds"
    labels["finetune_mask"] = "CNN (ResNet-18) fine-tuned, mask only"
    for name, lab in labels.items():
        f = out / f"cnn_frames_{name}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["y"] = (d.her2 == "high").astype(int)
        cell = d.groupby("cell").agg(y=("y", "first"), p=("p_high", "mean"))
        pred = (cell.p >= 0.5).astype(int)
        frame_bacc = balanced_accuracy_score(d.y, d.p_high >= 0.5)
        rows.append({"model": lab, "correct": int((pred == cell.y).sum()), "n": len(cell),
                     "balanced_acc": balanced_accuracy_score(cell.y, pred), "auc": roc_auc_score(cell.y, cell.p),
                     "frame_balanced_acc": frame_bacc,
                     "low1_p_high": cell.p.get("low_cell1", np.nan),
                     "wrong": " ".join(c.replace("_cell", "-") for c in cell.index[pred != cell.y])})
        # accuracy after watching the first k laser steps (frames averaged over steps 0..k-1)
        acc = []
        last = int(d.laser_step.max()) + 1
        for k in range(1, last + 1):
            s = d[d.laser_step < k].groupby("cell").agg(y=("y", "first"), p=("p_high", "mean"))
            acc.append(balanced_accuracy_score(s.y, (s.p >= 0.5).astype(int)))
        curves[lab] = acc
        print(f"{lab:34s} {rows[-1]['correct']}/{len(cell)}  bal.acc {rows[-1]['balanced_acc']:.2f}  "
              f"AUC {rows[-1]['auc']:.2f}  frames {frame_bacc:.2f}  low-1 P(high) {rows[-1]['low1_p_high']:.2f}  "
              f"wrong: {rows[-1]['wrong']}")
    t = pd.DataFrame(rows)
    t.to_csv(out / "cnn_cells.csv", index=False)
    lines = [f"{r.model.replace('  same', '\\quad same')} & {r.correct}/{r.n} & {r.balanced_acc:.2f} & {r.auc:.2f} & {r.frame_balanced_acc:.2f} & "
             f"{r.low1_p_high:.2f} \\\\" for r in t.itertuples()]
    (out / "cnn_cells.tex").write_text(
        "\\begin{tabular}{lccccc}\n\\toprule\nModel & Correct & Bal.\\ acc. & AUC & Frame bal.\\ acc. & "
        "\\code{low-1} $P(\\text{high})$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    style = {"CNN (ResNet-18) frozen, cell crop": ("#c9c8c4", 1.4, "-"), "CNN (ResNet-18) frozen, mask only": ("#8a8985", 1.4, "--"),
             "CNN (ResNet-18) fine-tuned, mask only": ("#3f7a4d", 1.6, "--"),
             "CNN (ResNet-18) fine-tuned, cell crop": ("#d9b8b8", 1.0, "-")}
    for lab, acc in curves.items():
        if lab.strip().startswith("same, seed"):
            col, lw, ls = "#d9b8b8", 1.0, "-"
            lab_ = None
        elif "mean of" in lab:
            col, lw, ls, lab_ = "#b02a2a", 2.2, "-", "CNN (ResNet-18) fine-tuned, cell crop (mean of 3 seeds; thin: single seeds)"
        else:
            col, lw, ls = style[lab]
            lab_ = None if lab == "CNN (ResNet-18) fine-tuned, cell crop" else lab
        ax.plot(range(1, len(acc) + 1), acc, color=col, lw=lw, ls=ls, label=lab_)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("laser steps watched (all frames of steps 1..k averaged per cell)", fontsize=8.5)
    ax.set_ylabel(f"balanced accuracy, {int(t.n.iloc[0])} cells", fontsize=8.5)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "cnn_frames_seen.png", dpi=200)
    if img:
        fig.savefig(img / "classify_cnn.png", dpi=200)
    if tables:
        (tables / "classify_cnn.tex").write_text((out / "cnn_cells.tex").read_text())
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["crops", "frozen", "finetune", "tune", "final", "report", "all"],
                    help="tune: nested-CV hyperparameter tuning of the fine-tuned cell-crop CNN; final: train on all "
                         "cells and save to checkpoints/classification/<model>/cnn/ (neither is part of all)")
    ap.add_argument("--model", default="nnunet100_5fold")
    ap.add_argument("--config", type=Path, default=None, help="settings file (default configs/analysis/cnn.yaml)")
    ap.add_argument("--epochs", type=int, default=None, help="fine-tuning epochs per fold (default: the config's)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--img", type=Path, default=None, help="report step: also write the figure here")
    ap.add_argument("--tables", type=Path, default=None, help="report step: also write the LaTeX table here")
    ap.add_argument("--seed", type=int, default=None, help="fine-tuning seed (default: the config's)")
    ap.add_argument("--variants", nargs="+", default=None, choices=["masked", "mask"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="final step: one saved model per seed (default: the config's seed)")
    a = ap.parse_args()
    if a.step == "final" and a.variants is None:     # the final model: the cell crop (the reported CNN)
        a.variants = ["masked"]
    if a.step != "report":               # log the training runs, not the table/figure step
        start_run_log(f"classification/{a.model}", f"frames_cnn_{a.step}")
    configure(load_config("cnn", a.config))
    a.epochs = CFG.epochs if a.epochs is None else a.epochs
    a.seed = CFG.seed if a.seed is None else a.seed
    a.variants = list(CFG.variants) if a.variants is None else a.variants
    from dataclasses import replace
    configure(replace(CFG, epochs=a.epochs, seed=a.seed, variants=tuple(a.variants)))   # what this run uses
    if a.step in ("frozen", "finetune", "tune", "final", "all"):
        dump_config(CFG, res_dir(a.model) / f"config_cnn_{a.step}.yaml")
    warnings.filterwarnings("ignore", category=UserWarning)
    device = a.device
    if device is None and a.step in ("frozen", "finetune", "tune", "final", "all"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.step in ("crops", "all"):
        crops(a.model)
    if a.step in ("frozen", "all"):
        frozen(a.model, device)
    if a.step in ("finetune", "all"):
        finetune(a.model, device, a.epochs, a.seed, tuple(a.variants))
    if a.step == "tune":
        tune(a.model, device, a.seed)
    if a.step == "final":
        final(a.model, device, a.epochs, a.seeds or [a.seed], tuple(a.variants))
    if a.step in ("report", "all"):
        report(a.model, a.img, a.tables)


if __name__ == "__main__":
    main()
