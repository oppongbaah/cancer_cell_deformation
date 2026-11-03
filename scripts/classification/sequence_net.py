"""
Sequence networks on the deformation curves: can HER2 be told from HOW a cell deforms over the laser ramp?

    python scripts/classification/sequence_net.py --model nnunet1000_loocv [--config <yaml>] [--perms N]

Input, one per cell: the strain (%) of each measure (configs/analysis/sequence.yaml: length along and across the
beam, volume) at every laser step -- the median of the step's passing frames against the cell's own baseline
(celldform.biomechanics.rod.step_strain) -- a (measures x steps) curve. Missing steps are interpolated; a cell
with fewer steps is padded with its last value.

Models (deep learning, tiny on purpose: 13 training curves per fold):
    cnn1d  two 1-D convolution layers over the steps, global average pooling, a linear output
    gru    one recurrent (GRU) layer, its last state, a linear output

Training: leave-one-cell-out, the same 14 folds as every other model; each channel scaled by its SD over the
training cells; class-balanced loss; a random window of `crop` steps, noise and amplitude scaling per epoch; a
fixed number of epochs (no validation set). The held-out cell is scored on its whole curve; P(high) is averaged
over the seeds. Permutation p: labels shuffled and the whole leave-one-cell-out rerun (first seed only).

Outputs in results/classification/<model>/: seq_cells.csv (every cell's P(high), per model and seed),
seq_metrics.csv, config_sequence.yaml.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.analysis_config import dump as dump_config, load as load_config  # noqa: E402
from celldform.biomechanics.rod import step_strain  # noqa: E402
from celldform.utils.runlog import start_run_log  # noqa: E402


def curves(model: str, cfg) -> tuple[np.ndarray, pd.DataFrame]:
    """(cells, measures, steps) strain curves of the usable cells, and the cells' table."""
    res = ROOT / "results" / "features" / model
    cells = pd.read_csv(res / "features_cells.csv")
    cells = cells[cells.usable].copy()
    cells["_n"] = cells.cell.str.extract(r"(\d+)$")[0].astype(int)
    cells = cells.sort_values(["her2", "_n"]).drop(columns="_n").reset_index(drop=True)
    f = pd.read_csv(res / "features_frames.csv")
    X = np.zeros((len(cells), len(cfg.measures), cfg.steps), np.float32)
    for i, c in enumerate(cells.cell):
        g = f[f.cell == c]
        for j, m in enumerate(cfg.measures):
            st = step_strain(g, m, base_steps=cfg.base_steps).reindex(range(cfg.steps))
            X[i, j] = st.interpolate(limit_direction="both").to_numpy(np.float32)
    return X, cells


def net(arch: str, n_in: int, cfg):
    import torch
    from torch import nn

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            k = cfg.kernel
            self.f = nn.Sequential(nn.Conv1d(n_in, cfg.channels, k, padding=k // 2), nn.ReLU(), nn.Dropout(cfg.dropout),
                                   nn.Conv1d(cfg.channels, cfg.channels, k, padding=k // 2), nn.ReLU(),
                                   nn.AdaptiveAvgPool1d(1))
            self.out = nn.Linear(cfg.channels, 1)

        def forward(self, x):
            return self.out(self.f(x).squeeze(-1)).squeeze(-1)

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_in, cfg.channels, batch_first=True)
            self.drop = nn.Dropout(cfg.dropout)
            self.out = nn.Linear(cfg.channels, 1)

        def forward(self, x):
            _, h = self.rnn(x.transpose(1, 2))
            return self.out(self.drop(h[-1])).squeeze(-1)

    if arch == "cnn1d":
        return CNN1D()
    if arch == "gru":
        return GRU()
    raise KeyError(f"unknown arch {arch!r}: cnn1d or gru")


def fit_predict(arch: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, cfg, seed: int) -> np.ndarray:
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6          # per-channel scale from the training cells only
    Xs = X / sd
    m = net(arch, X.shape[1], cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ytr = torch.from_numpy(y[tr].astype(np.float32))
    pos_w = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    xtr = Xs[tr]
    for _ in range(cfg.epochs):
        m.train()
        starts = rng.integers(0, X.shape[2] - cfg.crop + 1, len(xtr))
        xb = np.stack([x[:, s:s + cfg.crop] for x, s in zip(xtr, starts)])
        xb = xb * rng.uniform(*cfg.scale, (len(xb), 1, 1)) + rng.normal(0, cfg.jitter, xb.shape) / sd
        loss = lossf(m(torch.from_numpy(xb.astype(np.float32))), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(Xs[te]))).numpy()


def loocv(arch: str, X: np.ndarray, y: np.ndarray, cfg, seed: int) -> np.ndarray:
    p = np.zeros(len(y))
    for i in range(len(y)):
        te = np.arange(len(y)) == i
        p[i] = fit_predict(arch, X, y, ~te, te, cfg, seed)[0]
    return p


def perm_p(arch: str, X: np.ndarray, y: np.ndarray, observed: float, cfg, n: int) -> float:
    rng = np.random.default_rng(cfg.seeds[0])
    shuffles = [rng.permutation(y) for _ in range(n)]

    def one(yp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return balanced_accuracy_score(yp, (loocv(arch, X, yp, cfg, cfg.seeds[0]) >= 0.5).astype(int))

    null = Parallel(n_jobs=cfg.n_jobs)(delayed(one)(yp) for yp in shuffles)
    return (1 + sum(b >= observed - 1e-12 for b in null)) / (1 + n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv", help="feature run in results/features/")
    ap.add_argument("--config", type=Path, default=None, help="settings (default configs/analysis/sequence.yaml)")
    ap.add_argument("--perms", type=int, default=None, help="label shuffles (default: the config's)")
    a = ap.parse_args()
    start_run_log(f"classification/{a.model}", "sequence_net")
    cfg = load_config("sequence", a.config)
    perms = cfg.permutations if a.perms is None else a.perms
    out = ROOT / "results" / "classification" / a.model
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out / "config_sequence.yaml")
    X, cells = curves(a.model, cfg)
    y = (cells.her2 == "high").to_numpy().astype(int)
    print(f"{len(cells)} cells, curves {X.shape[1]} measures x {X.shape[2]} steps")
    prows, mrows = [], []
    for arch in cfg.archs:
        ps = []
        for seed in cfg.seeds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = loocv(arch, X, y, cfg, seed)
            ps.append(p)
            prows.append({"arch": arch, "seed": seed, **dict(zip(cells.cell, p))})
            print(f"{arch:6s} seed {seed}: {int(((p >= 0.5) == y).sum())}/14", flush=True)
        for label, p in [(f"seed {s}", q) for s, q in zip(cfg.seeds, ps)] + [("mean of seeds", np.mean(ps, 0))]:
            pred = (p >= 0.5).astype(int)
            bacc = balanced_accuracy_score(y, pred)
            mrows.append({"arch": arch, "run": label, "correct": int((pred == y).sum()), "balanced_acc": bacc,
                          "auc": roc_auc_score(y, p), "low1_p_high": p[cells.cell.tolist().index("low_cell1")],
                          "wrong": " ".join(c.replace("_cell", "-") for c in cells.cell[pred != y])})
        pv = perm_p(arch, X, y, mrows[-len(cfg.seeds) - 1]["balanced_acc"], cfg, perms) if perms else np.nan
        for r in mrows[-len(cfg.seeds) - 1:]:
            r["p_perm_seed0"] = pv
        prows.append({"arch": arch, "seed": "mean", **dict(zip(cells.cell, np.mean(ps, 0)))})
        r = mrows[-1]
        print(f"{arch:6s} mean of seeds: {r['correct']}/14  bal.acc {r['balanced_acc']:.2f}  AUC {r['auc']:.2f}  "
              f"low-1 P {r['low1_p_high']:.2f}  p(seed 0) {pv:.3f}  wrong: {r['wrong']}", flush=True)
    pd.DataFrame(prows).to_csv(out / "seq_cells.csv", index=False)
    pd.DataFrame(mrows).to_csv(out / "seq_metrics.csv", index=False)
    print(f"wrote {out}/seq_{{cells,metrics}}.csv")


if __name__ == "__main__":
    main()
