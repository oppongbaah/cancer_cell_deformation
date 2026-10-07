#!/usr/bin/env python3
"""
Distil the 4-seed ensemble into a single deployable U-Net.

Why: the ensemble scores 0.859 evaluated against 0.809 for the mean single model,
and no per-cell collapse survives it. But it is an inference-time construct --
no checkpoint, 4x GPU per frame, and RealTimePipeline takes one UNet. Distillation
trains one student on the ensemble's *soft* outputs, so the student learns the
smoothed decision boundary the ensemble produces rather than only the hard masks.

How it works
------------
1. The teachers' averaged softmax is PRECOMPUTED over the training pool and cached
   to disk. Four teachers plus a student do not fit in 8 GB together, and caching
   also avoids 4 extra forward passes per batch per epoch.
2. The student trains on
       L = lambda * CE(hard masks) + (1 - lambda) * T^2 * KL(student/T || teacher/T)
   Hard labels are kept because the training pool *is* annotated -- the ensemble is
   a second opinion here, not the only source of truth. T > 1 softens both
   distributions so the student sees the teacher's relative confidences, not just
   its argmax; the T^2 factor restores the gradient scale that softening removes.
3. The split is the teachers' split (split_seed 42), so the student's validation
   slice is one no teacher trained on either.

Usage
-----
python scripts/training/distill_ensemble.py                      # cache targets, then train
python scripts/training/distill_ensemble.py --cache-only         # just build the targets
python scripts/training/distill_ensemble.py --temperature 3 --lambda-hard 0.5
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).parent))                 # for train_unet helpers

from celldform.config import load as load_config
from celldform.segmentation.unet import UNet
from celldform.utils.metrics import SegmentationMetrics
from celldform.utils.visualization import Visualizer
from train_unet import _build_groups, _group_split

REPO = Path(__file__).parent.parent.parent
TEACHERS = [f"postswap_ftl_cw142_s{s}" for s in (42, 43, 44, 45)]
CKPT_ROOT = REPO / "checkpoints/postswap/experiment_0a_seed_baseline"            # the four teachers
OUT_ROOT = REPO / "checkpoints/postswap/experiment_0b_distillation"             # the student
CACHE = REPO / "data/preprocessed/train/postswap_soft_targets"


# ---------------------------------------------------------------------------
# Step 1 — cache the ensemble's soft targets
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_cache(inputs_dir: Path, teachers: list[str], device: str, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(inputs_dir.glob("*.png"))
    print(f"[distill] caching ensemble targets for {len(frames)} frames "
          f"from {len(teachers)} teachers -> {out.relative_to(REPO)}")
    models = []
    for name in teachers:
        ck = CKPT_ROOT / name / "unet_best.pt"
        if not ck.exists():
            raise SystemExit(f"missing teacher checkpoint: {ck}")
        m = UNet.from_checkpoint(ck, in_channels=1, out_channels=3,
                                 base_features=64, device=device)
        m.eval(); models.append(m)

    t0 = time.time()
    for i, f in enumerate(frames, 1):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        x = torch.from_numpy(img)[None, None].to(device)
        # average PROBABILITIES, not logits: logits are only comparable within one
        # model, probabilities share a meaning across models.
        probs = torch.stack([torch.softmax(m(x), dim=1)[0] for m in models]).mean(0)
        np.save(out / f"{f.stem}.npy", probs.cpu().numpy().astype(np.float16))
        if i % 100 == 0:
            print(f"           {i}/{len(frames)}  ({time.time()-t0:.0f}s)")
    del models
    torch.cuda.empty_cache()
    print(f"[distill] cached {len(frames)} targets in {time.time()-t0:.0f}s")
    return len(frames)


# ---------------------------------------------------------------------------
# Step 2 — the student
# ---------------------------------------------------------------------------

class DistillDataset(Dataset):
    """Yields (image, hard mask, teacher probabilities).

    No augmentation: the cached targets are tied to the un-augmented frame, and a
    spatial transform would have to be applied identically to a 3-channel float
    probability map. Worth revisiting -- flips/rotations would transfer cleanly --
    but the first version keeps the correspondence obviously correct.
    """

    def __init__(self, pairs, cache: Path):
        self.pairs = pairs
        self.cache = cache

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        fp, mp = self.pairs[i]
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE).astype(np.int64)
        soft = np.load(self.cache / f"{fp.stem}.npy").astype(np.float32)
        return (torch.from_numpy(img)[None], torch.from_numpy(mask), torch.from_numpy(soft))


def distill_loss(logits, hard, soft, lam: float, T: float, class_weights):
    ce = F.cross_entropy(logits, hard, weight=class_weights)
    # KL(teacher || student) on temperature-softened distributions.
    #
    # NOTE reduction="batchmean" is WRONG for spatial maps: it divides by the batch
    # size only, so a (B, C, H, W) tensor sums 3 x 65536 terms per sample and the KL
    # term dwarfs CE by ~5 orders of magnitude. Sum over classes, then mean over
    # batch AND pixels, so this is a per-pixel KL directly comparable to CE.
    s_log = F.log_softmax(logits / T, dim=1)
    t_soft = F.softmax(torch.log(soft.clamp_min(1e-8)) / T, dim=1)
    kl = F.kl_div(s_log, t_soft, reduction="none").sum(dim=1).mean() * (T * T)
    return lam * ce + (1.0 - lam) * kl, ce.item(), kl.item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=REPO / "configs/postswap/experiment_0a_seed_baseline/postswap_ftl_cw142_s42.yaml")
    p.add_argument("--teachers", nargs="*", default=TEACHERS)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--lambda-hard", type=float, default=0.5,
                   help="Weight on the hard-label CE term; 1-lambda goes to the KL term")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--run-name", default="postswap_distilled")
    args = p.parse_args()

    conf = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs, labels = Path(conf.data.frames_dir), Path(conf.data.masks_dir)

    if args.rebuild_cache or not args.cache.exists() or not any(args.cache.glob("*.npy")):
        build_cache(inputs, args.teachers, device, args.cache)
    else:
        print(f"[distill] reusing cache at {args.cache.relative_to(REPO)} "
              f"({len(list(args.cache.glob('*.npy')))} targets) -- --rebuild-cache to redo")
    if args.cache_only:
        return

    pairs = [(f, labels / f.with_suffix(".png").name) for f in sorted(inputs.glob("*.png"))]
    groups = _build_groups(pairs)
    split_seed = conf.training.split_seed or conf.training.seed
    tr, va = _group_split(pairs, groups, int(len(pairs) * conf.data.val_split),
                          split_seed=split_seed)
    print(f"[distill] split (split_seed={split_seed}) — train {len(tr)}  val {len(va)}")

    seed = conf.training.seed
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = conf.training.cudnn_deterministic
    torch.backends.cudnn.benchmark = conf.training.cudnn_benchmark

    tl = DataLoader(DistillDataset(tr, args.cache), batch_size=conf.training.batch_size,
                    shuffle=True, num_workers=conf.training.num_workers, pin_memory=True)
    vl = DataLoader(DistillDataset(va, args.cache), batch_size=conf.training.batch_size,
                    shuffle=False, num_workers=conf.training.num_workers, pin_memory=True)

    student = UNet(in_channels=1, out_channels=3,
                   base_features=conf.unet.base_features, dropout=conf.unet.dropout).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=conf.training.lr,
                            weight_decay=conf.training.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=conf.training.lr * 0.01)
    cw = torch.tensor(conf.training.class_weights, dtype=torch.float32, device=device) \
        if conf.training.class_weights else None
    metrics = SegmentationMetrics()

    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = {"train_loss": [], "train_dsc": [], "val_loss": [], "val_dsc": []}
    best, best_ep, patience = -1.0, 0, conf.training.early_stopping_patience or 20

    print(f"[distill] student: base_features={conf.unet.base_features}  "
          f"lambda_hard={args.lambda_hard}  T={args.temperature}  epochs={args.epochs}")
    for ep in range(1, args.epochs + 1):
        student.train(); tot = dsc = n = 0.0
        for img, hard, soft in tl:
            img, hard, soft = img.to(device), hard.to(device), soft.to(device)
            opt.zero_grad()
            logits = student(img)
            loss, _, _ = distill_loss(logits, hard, soft, args.lambda_hard, args.temperature, cw)
            loss.backward(); opt.step()
            b = img.size(0); tot += loss.item() * b; n += b
            with torch.no_grad():
                pr = logits.softmax(1).argmax(1)
                dsc += metrics.dice((pr.cpu() == 1), (hard.cpu() == 1)) * b
        tr_loss, tr_dsc = tot / n, dsc / n

        student.eval(); tot = dsc = n = 0.0
        with torch.no_grad():
            for img, hard, soft in vl:
                img, hard, soft = img.to(device), hard.to(device), soft.to(device)
                logits = student(img)
                loss, _, _ = distill_loss(logits, hard, soft, args.lambda_hard,
                                          args.temperature, cw)
                b = img.size(0); tot += loss.item() * b; n += b
                pr = logits.softmax(1).argmax(1)
                dsc += metrics.dice((pr.cpu() == 1), (hard.cpu() == 1)) * b
        va_loss, va_dsc = tot / n, dsc / n
        sched.step()

        for k, v in zip(hist, (tr_loss, tr_dsc, va_loss, va_dsc)):
            hist[k].append(v)
        print(f"Epoch {ep:03d}/{args.epochs}  train_loss={tr_loss:.4f}  train_DSC={tr_dsc:.4f}  "
              f"val_loss={va_loss:.4f}  val_DSC={va_dsc:.4f}")

        if va_dsc > best + 1e-3:
            best, best_ep = va_dsc, ep
            torch.save({"epoch": ep, "val_dsc": va_dsc,
                        "model_state_dict": student.state_dict(),
                        "distilled_from": args.teachers,
                        "lambda_hard": args.lambda_hard,
                        "temperature": args.temperature}, out_dir / "unet_best.pt")
        elif ep - best_ep >= patience:
            print(f"[distill] early stop at epoch {ep} (best {best_ep}, val_DSC {best:.4f})")
            break

    print(f"[distill] best epoch {best_ep}  val_DSC {best:.4f}  -> {out_dir}/unet_best.pt")
    curve = REPO / "results/postswap/experiment_0b_distillation" / f"unet_training_curves_{args.run_name}.png"
    Visualizer().plot_training_curves(hist, best_epoch=best_ep, save_path=str(curve), n_classes=3)
    print(f"[distill] curves -> {curve.relative_to(REPO)}")


if __name__ == "__main__":
    main()
