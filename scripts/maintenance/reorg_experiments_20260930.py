#!/usr/bin/env python3
"""
Group every training run into one folder per experiment (2026-09-30, user request).

Before: configs/ checkpoints/ logs/ results/ each held a flat list per era
(preswap/, postswap/), so one experiment's files were scattered across four long
listings. After: every experiment has the SAME folder name under all four roots,

    <root>/<era>/<experiment>/...      root in {configs, checkpoints, logs, results}

    preswap/stage_1_labels_and_resolution   binary vs multiclass, 128 vs 256 (Aug)
    preswap/stage_2_domain_gap              oversampling real-cell frames
    preswap/stage_3_regularisation          width, dropout, cutout, wd, SWA, schedule (Sep 2-3)
    preswap/stage_4_augmentation            brightness, scale, combined, controlled re-run
    preswap/stage_5_loss                    class weights, background, Focal-Tversky
    postswap/experiment_0a_seed_baseline    4 seeds, ensemble, deployment test (9/24)
    postswap/experiment_0b_distillation     ensemble -> one student (failed, 0.537)
    postswap/experiment_1                   optimizer (SGD) and loss (Lovasz) vs s43

This script is the record of that move: MAP below is the complete old -> new list.
It moves each item (git mv when tracked), splits the postswap eval ledger per
experiment, and rewrites every reference to a moved path in scripts, configs, docs,
READMEs, the notebook, eval ledgers and checkpoint config snapshots. Training logs
(*.log) are NOT rewritten -- they are records of what ran at the time.

    python scripts/maintenance/reorg_experiments_20260930.py            # dry run
    python scripts/maintenance/reorg_experiments_20260930.py --apply
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PRE = {
    "stage_1_labels_and_resolution": {
        "runs": ["binary_experiment_128", "binary_experiment_256", "multiclass_experiment",
                 "multiclass_experiment_256"],
        "configs": ["binary_experiment_128", "binary_experiment_256", "multiclass_experiment",
                    "multiclass_experiment_256"],
        "logs": ["train_unet_multiclass_256_20260828_163716", "train_unet_multiclass_256_deterministic_20260828_194423",
                 "train_unet_multiclass_256_deterministic_20260828_195116"],
        "curves": list(range(1, 20)) + list(range(21, 26)),
    },
    "stage_2_domain_gap": {
        "runs": ["multiclass_experiment_256_domain_oversample"],
        "configs": ["multiclass_experiment_256_oversample"],
        "logs": [],
        "curves": [20],
    },
    "stage_3_regularisation": {
        "runs": ["multiclass_experiment_256_bf32", "multiclass_experiment_256_bf32_reg",
                 "multiclass_experiment_256_bf32_reg2", "multiclass_experiment_256_bf32_reg3",
                 "multiclass_experiment_256_bf32_reg2_det", "multiclass_experiment_256_bf32_reg2_swa",
                 "multiclass_experiment_256_bf32_reg2_split", "multiclass_experiment_256_bf32_reg2_ablate",
                 "multiclass_experiment_256_bf32_reg2_wd", "multiclass_experiment_256_bf32_reg2_wd2",
                 "multiclass_experiment_256_bf32_reg2_cos", "multiclass_experiment_256_bf64_reg"],
        "configs": [],
        "logs": ["train_unet_multiclass_256_bf32_20260902_211128", "train_unet_multiclass_256_bf32_reg_20260902_212841",
                 "train_unet_multiclass_256_bf32_reg2_20260902_214957", "train_unet_multiclass_256_bf32_reg3_20260902_221803",
                 "train_unet_multiclass_256_swa_fixed_20260902_230656", "train_unet_multiclass_256_swa_fixed_20260902_230754",
                 "train_unet_multiclass_256_bf32_reg2_det_20260902_233453",
                 "train_unet_multiclass_256_bf32_reg2_split_20260902_234808",
                 "train_unet_multiclass_256_bf32_reg2_ablate_20260903_003333",
                 "train_unet_multiclass_256_bf32_reg2_wd_20260903_004521",
                 "train_unet_multiclass_256_bf32_reg2_cos_20260903_005534",
                 "train_unet_multiclass_256_bf32_reg2_wd2_20260903_010512",
                 "train_unet_multiclass_256_bf64_reg_20260903_015403"],
        "curves": list(range(26, 38)),
    },
    "stage_4_augmentation": {
        "runs": ["multiclass_experiment_256_bf64_bc", "multiclass_experiment_256_bf64_allaug",
                 "multiclass_experiment_256_bf64_base_cos", "multiclass_experiment_256_bf64_scale_cos",
                 "multiclass_experiment_256_bf64_scale_cos_445"],
        "configs": ["multiclass_experiment_256_base_cos", "multiclass_experiment_256_scale_cos",
                    "multiclass_experiment_256_scale_cos_445"],
        "logs": [],
        "curves": list(range(38, 44)),
    },
    "stage_5_loss": {
        "runs": ["multiclass_experiment_256_bf64_scale_cos_cw133", "multiclass_experiment_256_bf64_scale_cos_nobg",
                 "multiclass_experiment_256_bf64_scale_cos_ftl", "multiclass_experiment_256_bf64_scale_cos_ftl_seed43"],
        "configs": ["multiclass_experiment_256_scale_cos_cw133", "multiclass_experiment_256_scale_cos_nobg",
                    "multiclass_experiment_256_scale_cos_ftl", "multiclass_experiment_256_scale_cos_ftl_seed43"],
        "logs": ["exp1_cw133", "exp1b_nobg", "exp1c_ftl", "exp1c_ftl_seed43"],
        "curves": list(range(44, 48)),
    },
}
SEEDS = [f"postswap_ftl_cw142_s{s}" for s in (42, 43, 44, 45)]
E0A, E0B = "experiment_0a_seed_baseline", "experiment_0b_distillation"


def build_map() -> list[tuple[str, str]]:
    """Every (old, new) repo-relative path, for all four roots."""
    moves: list[tuple[str, str]] = []
    curves = {int(m.group(1)): p.name for p in (REPO / "results/preswap").glob("unet_training_curves_v*.png")
              if (m := re.match(r"unet_training_curves_v(\d+)", p.name))}
    for stage, g in PRE.items():
        for r in g["runs"]:
            moves.append((f"checkpoints/preswap/{r}", f"checkpoints/preswap/{stage}/{r}"))
        for c in g["configs"]:
            moves.append((f"configs/preswap/{c}.yaml", f"configs/preswap/{stage}/{c}.yaml"))
        for lg in g["logs"]:
            moves.append((f"logs/preswap/{lg}.log", f"logs/preswap/{stage}/{lg}.log"))
        for v in g["curves"]:
            if v in curves:
                moves.append((f"results/preswap/{curves[v]}", f"results/preswap/{stage}/{curves[v]}"))
    for s in SEEDS:
        moves += [(f"checkpoints/postswap/{s}", f"checkpoints/postswap/{E0A}/{s}"),
                  (f"configs/postswap/{s}.yaml", f"configs/postswap/{E0A}/{s}.yaml"),
                  (f"logs/postswap/{s}.log", f"logs/postswap/{E0A}/{s}.log")]
    moves += [(f"logs/postswap/{n}", f"logs/postswap/{E0A}/{n}") for n in ("postswap_series.log", "s45_driver.log")]
    moves += [(f"results/postswap/{p.name}", f"results/postswap/{E0A}/{p.name}")
              for p in (REPO / "results/postswap").glob("unet_training_curves_v*_postswap_ftl_cw142_s*.png")]
    moves += [("results/postswap/deployment_ensemble", f"results/postswap/{E0A}/deployment_ensemble"),
              ("checkpoints/postswap/postswap_distilled", f"checkpoints/postswap/{E0B}/postswap_distilled"),
              ("logs/postswap/distill.log", f"logs/postswap/{E0B}/distill.log"),
              ("results/postswap/unet_training_curves_postswap_distilled.png",
               f"results/postswap/{E0B}/unet_training_curves_postswap_distilled.png")]
    return moves


def tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return set(out.split())


def move(old: str, new: str, git: set[str]) -> None:
    src, dst = REPO / old, REPO / new
    dst.parent.mkdir(parents=True, exist_ok=True)
    inside = [f for f in git if f == old or f.startswith(old + "/")]
    if inside:
        subprocess.run(["git", "mv", old, new], cwd=REPO, check=True)
    else:
        src.rename(dst)


def split_postswap_ledger(apply: bool) -> None:
    """results/postswap/postswap_eval_log.jsonl -> one eval_log.jsonl per experiment."""
    led = REPO / "results/postswap/postswap_eval_log.jsonl"
    if not led.exists():
        return
    buckets: dict[str, list[str]] = {E0A: [], E0B: []}
    for line in led.read_text().splitlines():
        if line.strip():
            run = Path(json.loads(line)["checkpoint"]).parent.name
            buckets[E0B if run == "postswap_distilled" else E0A].append(line)
    for exp, lines in buckets.items():
        print(f"  ledger: {len(lines):3d} entries -> results/postswap/{exp}/eval_log.jsonl")
    if apply:
        for exp, lines in buckets.items():
            dst = REPO / f"results/postswap/{exp}/eval_log.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("\n".join(lines) + "\n")
        subprocess.run(["git", "rm", "-q", "--cached", str(led.relative_to(REPO))], cwd=REPO, check=False)
        led.unlink()


def rewrite_targets(git: set[str]) -> list[Path]:
    """Text files whose path references follow the move. Training logs are left as-is."""
    exts = {".py", ".sh", ".md", ".yaml", ".yml", ".ipynb", ".jsonl", ".json", ".txt", ".toml", ".cfg"}
    files = {REPO / f for f in git if Path(f).suffix in exts and not f.startswith("data/")}
    files |= {REPO / "CLAUDE.md"} | set(REPO.glob("checkpoints/**/config.yaml"))
    files |= set(REPO.glob("results/**/eval_log.jsonl"))
    files.discard(Path(__file__).resolve())
    return sorted(f for f in files if f.exists())


def rewrite(moves: list[tuple[str, str]], files: list[Path], apply: bool) -> None:
    extra = [("results/postswap/postswap_eval_log.jsonl", f"results/postswap/{E0A}/eval_log.jsonl")]
    table = dict(moves + extra)
    # Longest first, and never inside a longer name: ".../multiclass_experiment_256" must
    # not match the front of ".../multiclass_experiment_256_bf32".
    pat = re.compile("|".join(re.escape(o) for o in sorted(table, key=len, reverse=True))
                     + r"(?![A-Za-z0-9_])")
    total = 0
    for f in files:
        text = f.read_text()
        new, n = pat.subn(lambda m: table[m.group(0)], text)
        if n:
            total += n
            print(f"  {n:3d} refs  {f.relative_to(REPO)}")
            if apply:
                f.write_text(new)
    print(f"  {total} references {'rewritten' if apply else 'to rewrite'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    git = tracked()
    moves = build_map()
    missing = [o for o, _ in moves if not (REPO / o).exists()]
    clash = [n for _, n in moves if (REPO / n).exists()]
    if missing or clash:
        raise SystemExit(f"missing sources: {missing}\nexisting targets: {clash}")
    print(f"{len(moves)} moves")
    for o, n in moves:
        if args.apply:
            move(o, n, git)
    split_postswap_ledger(args.apply)
    rewrite(moves, rewrite_targets(tracked() if args.apply else git), args.apply)
    left = [p.name for root in ("checkpoints", "configs", "logs", "results") for era in ("preswap", "postswap")
            for p in (REPO / root / era).glob("*") if not p.name.startswith(("stage_", "experiment_", "README"))
            and p.name not in {"preswap_eval_log.jsonl", "postswap_eval_log.jsonl"}]
    print(f"  left at era level (not an experiment): {left}")


if __name__ == "__main__":
    main()
