"""
The files the app needs from the user, how each is checked, and the model bundle that carries them.

The app ships no models. On a new machine nothing works until three files are loaded:

====================  =========  ==================================================================================
role                  file       what it is
====================  =========  ==================================================================================
``unet``              ``.pt``    the U-Net weights (segmentation)
``unet_config``       ``.yaml``  how frames are prepared for it: filters, crop size and crop centre of THE SETUP
``classifier``        ``.joblib``  the fitted HER2 framework (step model + cell model)
``reference``         ``.csv``   optional: the training cells' features, for the comparison chart
====================  =========  ==================================================================================

They can be loaded one by one or as one **bundle** (a folder or a ``.zip`` with ``celldform-models.json``, which lists
every file with its SHA-256, so a file that was damaged or swapped on the way is refused). A bundle holds weights only
and a minimal settings file, never a copy of the training configuration (dataset paths, splits) or the optimizer state.

Every check returns a :class:`Check` (never raises), so the window can show what is wrong and what to do.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from celldform.demo import privacy

MANIFEST = "celldform-models.json"
FORMAT = 1
MAX_BUNDLE_BYTES = 4 * 1024 ** 3
MAX_YAML_BYTES = 1024 ** 2


@dataclass(frozen=True)
class Role:
    key: str
    title: str
    suffixes: tuple
    required: bool
    hint: str


ROLES: Dict[str, Role] = {r.key: r for r in (
    Role("unet", "U-Net weights", (".pt", ".pth"), True, "the segmentation model (a .pt file)"),
    Role("unet_config", "U-Net settings", (".yaml", ".yml"), True,
         "how frames are prepared for the U-Net: filters, crop size and crop centre (a config.yaml)"),
    Role("classifier", "HER2 classifier", (".joblib",), True, "the fitted HER2 classifier (a .joblib file)"),
    Role("reference", "Training-cell reference", (".csv",), False,
         "optional: the training cells' features, for the comparison chart (a .csv)"),
)}
REQUIRED = tuple(k for k, r in ROLES.items() if r.required)
BUNDLE_NAMES = {"unet": "unet.pt", "unet_config": "unet_settings.yaml", "classifier": "classifier.joblib",
                "reference": "reference.csv"}


class BundleError(ValueError):
    """A model bundle that is damaged, tampered with, or not a bundle at all."""


@dataclass
class Check:
    role: str
    path: Optional[Path]
    ok: bool
    message: str
    sha256: str = ""
    size: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def short_hash(self) -> str:
        return self.sha256[:12]


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fail(role: str, path, message: str) -> Check:
    return Check(role, Path(path) if path else None, False, message)


def _base(role: str, path) -> Optional[Check]:
    """The checks every file passes first: it exists, is a file, is not empty, has the right kind of name."""
    r = ROLES[role]
    if not path:
        return _fail(role, None, "not loaded")
    p = Path(path)
    if not p.is_file():
        return _fail(role, p, f"file not found: {p.name}")
    if p.stat().st_size == 0:
        return _fail(role, p, f"{p.name} is empty")
    if p.suffix.lower() not in r.suffixes:
        return _fail(role, p, f"{p.name} is not a {'/'.join(r.suffixes)} file ({r.hint})")
    return None


def _done(role: str, p: Path, message: str, **detail) -> Check:
    return Check(role, p, True, message, sha256_file(p), p.stat().st_size, detail)


# ── the U-Net ─────────────────────────────────────────────────────────────────────────────────────────────────────

def check_unet(path) -> Check:
    bad = _base("unet", path)
    if bad:
        return bad
    from celldform.demo.segment import inspect_checkpoint
    p = Path(path)
    try:
        arch = inspect_checkpoint(p)
    except Exception as e:
        return _fail("unet", p, str(e))
    return _done("unet", p, f"U-Net, {arch['out_channels']} classes, base width {arch['base_features']}", **arch)


def load_unet_settings(path) -> dict:
    """The validated ``preprocessing`` settings of a settings file (a full training config.yaml also works: only the
    keys the app needs are read). Raises ValueError saying what is wrong."""
    import yaml
    p = Path(path)
    if p.stat().st_size > MAX_YAML_BYTES:
        raise ValueError(f"{p.name} is too large to be a settings file")
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"{p.name} is not valid YAML ({type(e).__name__})") from e
    pp = cfg.get("preprocessing") if isinstance(cfg, dict) else None
    if not isinstance(pp, dict):
        raise ValueError(f"{p.name} has no 'preprocessing' section")
    missing = [k for k in ("median_kernel", "clahe_clip_limit", "clahe_tile_grid", "morph_kernel_size", "target_size") if k not in pp]
    if missing:
        raise ValueError(f"{p.name}: 'preprocessing' lacks {', '.join(missing)}")

    def pair(name, kind=int, positive=True):
        v = pp[name]
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
            raise ValueError(f"{p.name}: {name} must be two numbers, got {v!r}")
        if positive and any(x <= 0 for x in v):
            raise ValueError(f"{p.name}: {name} must be positive, got {v!r}")
        return [kind(x) for x in v]

    out = {"median_kernel": pp["median_kernel"], "clahe_clip_limit": pp["clahe_clip_limit"],
           "clahe_tile_grid": pair("clahe_tile_grid"), "morph_kernel_size": pp["morph_kernel_size"],
           "target_size": pair("target_size")}
    if not (isinstance(out["median_kernel"], int) and out["median_kernel"] >= 1 and out["median_kernel"] % 2 == 1):
        raise ValueError(f"{p.name}: median_kernel must be an odd whole number, got {out['median_kernel']!r}")
    if not (isinstance(out["clahe_clip_limit"], (int, float)) and out["clahe_clip_limit"] > 0):
        raise ValueError(f"{p.name}: clahe_clip_limit must be a positive number")
    if not (isinstance(out["morph_kernel_size"], int) and out["morph_kernel_size"] >= 0):
        raise ValueError(f"{p.name}: morph_kernel_size must be a whole number")
    if any(x % 16 for x in out["target_size"]):
        raise ValueError(f"{p.name}: target_size {out['target_size']} must be multiples of 16 (the U-Net halves it four times)")
    crop = pp.get("crop_size")
    if crop is not None:
        if not (isinstance(crop, int) and 16 <= crop <= 922):
            raise ValueError(f"{p.name}: crop_size must be a whole number up to 922 (the frame height), got {crop!r}")
        out["crop_size"] = crop
        c = pp.get("crop_center")
        if c is not None:
            out["crop_center"] = pair_value(p, "crop_center", c)
    return out


def pair_value(p: Path, name: str, v) -> list:
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
        raise ValueError(f"{p.name}: {name} must be two numbers (x, y), got {v!r}")
    if not (0 <= v[0] <= 1228 and 0 <= v[1] <= 922):
        raise ValueError(f"{p.name}: {name} {list(v)!r} is outside a 1228x922 frame")
    return [int(v[0]), int(v[1])]


def check_unet_settings(path) -> Check:
    bad = _base("unet_config", path)
    if bad:
        return bad
    p = Path(path)
    try:
        pp = load_unet_settings(p)
    except Exception as e:
        return _fail("unet_config", p, str(e))
    crop = f"{pp['crop_size']} px crop" + (f" at {tuple(pp['crop_center'])}" if "crop_center" in pp else "") if "crop_size" in pp else "no crop"
    return _done("unet_config", p, f"settings OK: {crop}, target {pp['target_size'][0]}x{pp['target_size'][1]}", **pp)


def try_segmenter(unet, settings, device: Optional[str] = None):
    """``(segmenter, check)``: the segmenter built from the two files and run on a blank frame, which proves the pair works
    together on this machine (architecture, crop, device) before any recording is opened. ``segmenter`` is None on failure."""
    import numpy as np

    from celldform.demo.segment import UNetSegmenter
    t = time.perf_counter()
    try:
        seg = UNetSegmenter.from_file(unet, config=settings, device=device)
        out = seg.segment(np.full((922, 1228), 128, np.uint8))
    except Exception as e:
        return None, _fail("unet", unet, f"the U-Net and its settings do not work together: {str(e)[:200]}")
    return seg, Check("unet", Path(unet), True, f"{seg.arch['out_channels']} classes, runs on {seg.device} "
                      f"({1000 * (time.perf_counter() - t):.0f} ms incl. loading)", sha256_file(unet), Path(unet).stat().st_size,
                      {"device": seg.device, "labels_shape": tuple(out.labels.shape), **seg.arch})


def smoke_test_unet(unet, settings, device: Optional[str] = None) -> Check:
    return try_segmenter(unet, settings, device)[1]


# ── the classifier ────────────────────────────────────────────────────────────────────────────────────────────────

def check_classifier(path) -> Check:
    bad = _base("classifier", path)
    if bad:
        return bad
    from celldform.demo import framework as fw
    p = Path(path)
    try:
        model = fw.load(p)                           # safe: only what a HER2 framework contains
    except Exception as e:
        return _fail("classifier", p, str(e))
    try:                                             # does it expect the features this version of the app measures?
        n_step = model.step_model.n_features_in_
        n_cell = model.cell_model.n_features_in_
    except Exception:
        n_step = n_cell = None
    if n_step is not None and (n_step != len(fw.STEP_FEATURES) or n_cell != len(fw.SELECTED)):
        return _fail("classifier", p, f"{p.name} was trained on {n_step} step features and {n_cell} cell features; this app "
                     f"measures {len(fw.STEP_FEATURES)} and {len(fw.SELECTED)}, so it cannot be used")
    info = getattr(model, "info", {}) or {}
    bits = [f"{info['n_high']} high + {info['n_low']} low HER2 cells"] if "n_high" in info else []
    if "loocv_correct" in info:
        bits.append(f"leave-one-cell-out {info['loocv_correct']}/{info.get('n_high', 0) + info.get('n_low', 0) or '?'}")
    return _done("classifier", p, "HER2 classifier" + (": " + ", ".join(bits) if bits else ""), **{
        k: info[k] for k in ("n_high", "n_low", "loocv_correct", "loocv_auc", "features_model") if k in info})


def check_reference(path) -> Check:
    bad = _base("reference", path)
    if bad:
        return bad
    import pandas as pd
    p = Path(path)
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return _fail("reference", p, f"{p.name} could not be read as a table ({type(e).__name__})")
    need = {"her2", "len_par_base", "fluctuation"}
    if not need <= set(df.columns):
        return _fail("reference", p, f"{p.name} lacks the columns {', '.join(sorted(need - set(df.columns)))}")
    return _done("reference", p, f"{len(df)} training cells", rows=len(df))


CHECKS = {"unet": check_unet, "unet_config": check_unet_settings, "classifier": check_classifier, "reference": check_reference}


def check(role: str, path) -> Check:
    return CHECKS[role](path)


# ── the bundle ────────────────────────────────────────────────────────────────────────────────────────────────────

def make_bundle(out_dir: Path | str, *, unet, config, classifier, reference=None, description: str = "") -> Path:
    """Write a model bundle: weights only, a minimal settings file, the classifier, a manifest with SHA-256s.

    What goes in is rebuilt, not copied: the U-Net file is re-saved as a bare state dict (no optimizer or training
    history), and the settings file is rebuilt from the few keys the app reads (a training config.yaml holds dataset paths
    and split details that have no business on someone else's machine)."""
    import torch
    import yaml

    from celldform.demo import framework as fw
    from celldform.demo.segment import inspect_checkpoint, read_checkpoint
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bad = privacy.find_sensitive(description)
    if bad:
        raise BundleError("the description must not contain " + "; ".join(str(b) for b in bad))
    arch = inspect_checkpoint(unet)
    sd = read_checkpoint(unet)
    torch.save({k: v.detach().cpu() for k, v in sd.items()}, out / BUNDLE_NAMES["unet"])
    pp = load_unet_settings(config)
    (out / BUNDLE_NAMES["unet_config"]).write_text(yaml.safe_dump(
        {"preprocessing": pp, "unet": {"in_channels": arch["in_channels"], "base_features": arch["base_features"],
                                       "out_channels": arch["out_channels"]}}, sort_keys=False), encoding="utf-8")
    model = fw.load(classifier)                              # must be a real, safe HER2 framework
    leaks = privacy.find_sensitive(json.dumps(getattr(model, "info", {}) or {}, default=str))
    if leaks:
        raise BundleError("the classifier's own record contains " + "; ".join(str(b) for b in leaks))
    shutil.copyfile(classifier, out / BUNDLE_NAMES["classifier"])
    if reference:
        c = check_reference(reference)
        if not c.ok:
            raise BundleError(c.message)
        shutil.copyfile(reference, out / BUNDLE_NAMES["reference"])
    files = {}
    for role, name in BUNDLE_NAMES.items():
        f = out / name
        if f.is_file():
            files[role] = {"name": name, "sha256": sha256_file(f), "size": f.stat().st_size}
    (out / MANIFEST).write_text(json.dumps({
        "format": FORMAT, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": description, "files": files}, indent=2) + "\n", encoding="utf-8")
    return out


def pack_zip(folder: Path | str, zip_path: Path | str) -> Path:
    """A bundle folder as one ``.zip`` (the files are already compressed or tiny, so it is stored as is)."""
    folder, zip_path = Path(folder), Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.iterdir()):
            if f.is_file():
                z.write(f, f.name)
    return zip_path


def _user_cache() -> Path:
    try:
        from platformdirs import user_cache_dir
        return Path(user_cache_dir("celldform", appauthor=False)) / "bundles"
    except Exception:
        return Path.home() / ".celldform" / "bundles"


def _safe_extract(z: zipfile.ZipFile, dest: Path) -> None:
    """Unpack a bundle zip that cannot write outside ``dest``, has no links, and cannot be a zip bomb."""
    infos = z.infolist()
    if len(infos) > 20:
        raise BundleError("this zip has too many files to be a model bundle")
    if sum(i.file_size for i in infos) > MAX_BUNDLE_BYTES:
        raise BundleError("this zip is too large to be a model bundle")
    dest = dest.resolve()
    for i in infos:
        name = i.filename
        pure = PurePosixPath(name.replace("\\", "/"))
        if i.is_dir():
            continue
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1 or (i.external_attr >> 16) & 0o170000 == 0o120000:
            raise BundleError(f"the zip contains an unsafe entry ({name!r}); it was not opened")
        target = (dest / pure.name).resolve()
        if dest not in target.parents:
            raise BundleError(f"the zip contains an unsafe entry ({name!r}); it was not opened")
        with z.open(i) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)


def open_bundle(path: Path | str, cache_dir: Optional[Path] = None) -> Dict[str, Path]:
    """The verified files of a bundle (a folder, or a ``.zip`` that is unpacked into the user's cache): role -> path.
    Every file must be listed in the manifest and match its SHA-256 and size, or :class:`BundleError` says which not."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        try:
            z = zipfile.ZipFile(path)
        except zipfile.BadZipFile as e:
            raise BundleError(f"{path.name} is not a valid zip file") from e
        with z:
            key = sha256_file(path)[:16]
            folder = (cache_dir or _user_cache()) / key
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            _safe_extract(z, folder)
    elif path.is_dir():
        folder = path
    else:
        raise BundleError(f"{path.name} is neither a folder nor a .zip file")
    mf = folder / MANIFEST
    if not mf.is_file():
        raise BundleError(f"{path.name} is not a model bundle (no {MANIFEST})")
    if mf.stat().st_size > 1024 ** 2:
        raise BundleError("the bundle's manifest is too large")
    try:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        files = manifest["files"]
    except Exception as e:
        raise BundleError(f"the bundle's manifest cannot be read ({type(e).__name__})") from e
    if manifest.get("format") != FORMAT:
        raise BundleError(f"the bundle has format {manifest.get('format')!r}; this app reads format {FORMAT}")
    unknown = sorted(set(files) - set(ROLES))
    if unknown:
        raise BundleError(f"the manifest lists unknown files: {', '.join(unknown)}")
    absent = [ROLES[r].title for r in REQUIRED if r not in files]
    if absent:
        raise BundleError("the bundle lacks: " + ", ".join(absent))
    out: Dict[str, Path] = {}
    for role, meta in files.items():
        name = str(meta.get("name", ""))
        if not name or name != Path(name).name or name in (".", ".."):
            raise BundleError(f"the manifest names an unsafe file ({name!r})")
        f = folder / name
        if not f.is_file():
            raise BundleError(f"{name} is listed in the bundle but missing")
        if f.stat().st_size != meta.get("size"):
            raise BundleError(f"{name} has the wrong size: the bundle is damaged or was changed")
        if sha256_file(f) != meta.get("sha256"):
            raise BundleError(f"{name} does not match its SHA-256: the bundle is damaged or was changed; it was not used")
        out[role] = f
    return out


def describe_bundle(path: Path | str) -> str:
    """The description text of a bundle's manifest (shown when it is loaded)."""
    p = Path(path)
    try:
        if p.is_dir():
            return str(json.loads((p / MANIFEST).read_text(encoding="utf-8")).get("description", ""))
        with zipfile.ZipFile(p) as z:
            return str(json.loads(z.read(MANIFEST)).get("description", ""))
    except Exception:
        return ""


# ── what is still missing ─────────────────────────────────────────────────────────────────────────────────────────

def missing_roles(checks: Dict[str, Check]) -> List[str]:
    """Titles of the required files that are not loaded and valid yet."""
    return [ROLES[r].title for r in REQUIRED if not (checks.get(r) and checks[r].ok)]


# ── command line: make and verify a bundle ────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    """``python -m celldform.demo.models make|verify ...`` (for whoever prepares the files for others)."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="python -m celldform.demo.models", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="build a model bundle from the trained files")
    mk.add_argument("--unet", required=True, help="the U-Net checkpoint (.pt); only its weights are kept")
    mk.add_argument("--config", required=True, help="its config.yaml; only the pre-processing keys are kept")
    mk.add_argument("--classifier", required=True, help="the fitted HER2 framework (.joblib)")
    mk.add_argument("--reference", default=None, help="optional: features_cells.csv of the training cells")
    mk.add_argument("--description", default="", help="a line shown when the bundle is loaded (no names or e-mail addresses)")
    mk.add_argument("--out", required=True, help="the bundle folder to write")
    mk.add_argument("--zip", default=None, help="also pack the folder as this .zip file")
    vf = sub.add_parser("verify", help="check a bundle (folder or .zip): checksums and every file's own check")
    vf.add_argument("bundle")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "make":
            out = make_bundle(a.out, unet=a.unet, config=a.config, classifier=a.classifier, reference=a.reference,
                              description=a.description)
            if a.zip:
                pack_zip(out, a.zip)
            for role, meta in json.loads((out / MANIFEST).read_text())["files"].items():
                print(f"  {meta['name']:<22} {meta['size'] / 1e6:8.2f} MB  sha256 {meta['sha256'][:16]}...")
            print(f"bundle written to {out}" + (f" and {a.zip}" if a.zip else ""))
            return 0
        files = open_bundle(a.bundle)
        bad = 0
        for role, p in files.items():
            c = check(role, p)
            bad += not c.ok
            print(f"  {'OK ' if c.ok else 'BAD'} {ROLES[role].title:<26} {c.message}")
        print("bundle is usable" if not bad else "bundle has unusable files")
        return 1 if bad else 0
    except (BundleError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
