"""Tests for celldform.demo.segment (stage 3)."""
import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from celldform.demo.segment import DEPLOY_DIR, UNetSegmenter, estimate_crop_center, probs_to_seg


def test_probs_to_seg_pastes_into_crop():
    p = np.zeros((3, 4, 4), np.float32)
    p[0] = 0.9; p[:, 1, 2] = [0.0, 0.8, 0.2]
    s = probs_to_seg(p, (10, 8), (5, 3, 9, 7))
    assert s.labels.shape == (8, 10) and s.cell.sum() == 1 and s.labels[4, 7] == 1
    assert s.maxprob[0, 0] == 1.0 and s.maxprob[4, 7] == pytest.approx(0.8)


def test_probs_to_seg_no_crop_resizes():
    p = np.zeros((3, 4, 4), np.float32); p[1] = 1.0
    s = probs_to_seg(p, (8, 6), None)
    assert s.labels.shape == (6, 8) and (s.labels == 1).all()


def test_estimate_crop_center_clamps():
    assert estimate_crop_center([(10, 10), (12, 8)]) == (192, 192)
    assert estimate_crop_center([(600, 460)] * 3) == (600, 460)


def test_tiny_checkpoint_roundtrip(tmp_path):
    from celldform.segmentation.unet import UNet
    cfg = yaml.safe_load((DEPLOY_DIR / "config.yaml").read_text()) if DEPLOY_DIR.exists() else None
    if cfg is None:
        pytest.skip("deploy config absent")
    cfg["unet"]["base_features"] = 4
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    UNet(1, 4, 3).save(tmp_path / "m.pt")
    seg = UNetSegmenter(tmp_path, device="cpu")
    s = seg.segment(np.random.default_rng(0).integers(0, 255, (922, 1228), np.uint8))
    assert s.labels.shape == (922, 1228) and s.crop_box == (387, 318, 771, 702)
    with pytest.raises(ValueError):
        seg.segment(np.zeros((10, 10), np.uint8))


@pytest.mark.skipif(not list(DEPLOY_DIR.glob("*.pt")), reason="deployment checkpoint absent")
def test_deployment_model_finds_a_cell_on_a_real_frame():
    import cv2
    rec = DEPLOY_DIR.parent.parent.parent.parent.parent / "data/frames/03_mask_factory"
    jpgs = sorted(rec.glob("low_cell1_*.jpg"))
    if not jpgs:
        pytest.skip("factory frames absent")
    g = cv2.imread(str(jpgs[5]), cv2.IMREAD_GRAYSCALE)
    s = UNetSegmenter().segment(g)
    area = int(s.cell.sum())
    assert 300 < area < 3000, area


def test_make_segmenter_rejects_unknown_name():
    from celldform.demo.segment import make_segmenter
    with pytest.raises(ValueError, match="unknown segmenter"):
        make_segmenter("sam")


def test_nnunet_segmenter_missing_checkpoint(tmp_path):
    pytest.importorskip("nnunetv2")
    from celldform.demo.segment import NNUNetSegmenter
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        NNUNetSegmenter(folder=tmp_path)


def test_nnunet_segmenter_finds_the_cell_in_a_training_frame():
    """Real model, real frame of a training cell: guards the wrapper (input layout, class order, full-frame output)."""
    import glob
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("nnunetv2")
    from celldform.demo.segment import NNUNET_DIR, NNUNetSegmenter
    frames = sorted(glob.glob(str(DEPLOY_DIR.parents[4] / "data/frames/03_mask_factory/high_cell2_000200.jpg")))
    if not (NNUNET_DIR / "fold_all/checkpoint_final.pth").is_file() or not frames:
        pytest.skip("nnU-Net checkpoint or factory frame not on this machine")
    seg = NNUNetSegmenter().segment(cv2.imread(frames[0], 0))
    assert seg.labels.shape == (922, 1228) and 300 < seg.cell.sum() < 3000


# ── U-Net checkpoints the user supplies ─────────────────────────────────────────────────────────────────────────

def _save_tiny(path, base=4, out=3, wrapped=False):
    from celldform.segmentation.unet import UNet
    m = UNet(1, base, out)
    if wrapped:
        torch.save({"epoch": 3, "val_dsc": 0.9, "model_state_dict": m.state_dict()}, path)
    else:
        m.save(path)
    return path


def test_architecture_is_read_from_the_weights(tmp_path):
    from celldform.demo.segment import inspect_checkpoint
    assert inspect_checkpoint(_save_tiny(tmp_path / "a.pt", base=4, out=3)) == {"in_channels": 1, "base_features": 4, "out_channels": 3}
    assert inspect_checkpoint(_save_tiny(tmp_path / "b.pt", base=8, out=2, wrapped=True))["base_features"] == 8


def test_a_bare_pt_runs_with_the_deployment_preprocessing(tmp_path):
    seg = UNetSegmenter.from_file(_save_tiny(tmp_path / "m.pt"), device="cpu")
    assert seg.config_source == "deployment settings" and seg.arch["base_features"] == 4
    s = seg.segment(np.random.default_rng(0).integers(0, 255, (922, 1228), np.uint8))
    assert s.labels.shape == (922, 1228) and s.crop_box == (387, 318, 771, 702)
    assert "deployment settings" in seg.describe() and "3 classes" in seg.describe()


def test_a_config_beside_the_pt_or_given_explicitly_sets_the_preprocessing(tmp_path):
    if not DEPLOY_DIR.exists():
        pytest.skip("deploy config absent")
    cfg = yaml.safe_load((DEPLOY_DIR / "config.yaml").read_text())
    cfg["preprocessing"]["crop_size"] = 256
    d = tmp_path / "beside"; d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert UNetSegmenter.from_file(_save_tiny(d / "m.pt"), device="cpu").crop_box is not None
    seg = UNetSegmenter.from_file(_save_tiny(d / "m.pt"), device="cpu")
    x0, y0, x1, y1 = seg.crop_box
    assert (x1 - x0, y1 - y0) == (256, 256) and "config.yaml" in seg.describe()
    other = tmp_path / "elsewhere.yaml"; other.write_text(yaml.safe_dump(cfg))
    seg2 = UNetSegmenter.from_file(_save_tiny(tmp_path / "m2.pt"), config=other, device="cpu")
    assert seg2.config_source == str(other) and seg2.crop_box[2] - seg2.crop_box[0] == 256


def test_bad_checkpoints_get_a_clear_error(tmp_path):
    from celldform.demo.segment import inspect_checkpoint
    (tmp_path / "junk.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="not a PyTorch checkpoint"):
        inspect_checkpoint(tmp_path / "junk.pt")
    torch.save({"w": torch.zeros(3, 3)}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a celldform U-Net"):
        inspect_checkpoint(tmp_path / "other.pt")
    torch.save([1, 2, 3], tmp_path / "list.pt")
    with pytest.raises(ValueError, match="state dict"):
        inspect_checkpoint(tmp_path / "list.pt")
    with pytest.raises(ValueError, match="binary"):
        inspect_checkpoint(_save_tiny(tmp_path / "bin.pt", out=1))
    with pytest.raises(FileNotFoundError):
        inspect_checkpoint(tmp_path / "nope.pt")


def test_a_checkpoint_cannot_run_code_when_opened(tmp_path):
    """Checkpoints are read with weights_only: a pickled object with a __reduce__ payload is refused, not executed."""
    import os
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))
    torch.save(Evil(), tmp_path / "evil.pt")
    from celldform.demo.segment import inspect_checkpoint
    with pytest.raises(ValueError):
        inspect_checkpoint(tmp_path / "evil.pt")
    assert not marker.exists()
