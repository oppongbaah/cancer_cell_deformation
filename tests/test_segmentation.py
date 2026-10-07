"""Tests for celldform.segmentation (UNet forward pass and predict)."""

import numpy as np
import pytest
import torch

from celldform.segmentation.unet import UNet


class TestUNet:

    def test_forward_output_shape(self):
        model = UNet(in_channels=1, base_features=16)
        x = torch.zeros(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_forward_non_power_of_two_spatial(self):
        model = UNet(in_channels=1, base_features=16)
        x = torch.zeros(1, 1, 100, 100)
        out = model(x)
        assert out.shape == (1, 1, 100, 100)

    def test_predict_returns_binary_uint8(self):
        model = UNet(in_channels=1, base_features=16)
        image = np.random.rand(64, 64).astype(np.float32)
        mask = model.predict(image, device="cpu")
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})

    def test_predict_shape_matches_input(self):
        model = UNet(in_channels=1, base_features=16)
        image = np.random.rand(128, 128).astype(np.float32)
        mask = model.predict(image, device="cpu")
        assert mask.shape == (128, 128)

    def test_save_and_load_checkpoint(self, tmp_path):
        model = UNet(in_channels=1, base_features=16)
        ckpt = tmp_path / "unet.pt"
        model.save(ckpt)
        loaded = UNet.from_checkpoint(ckpt, in_channels=1, base_features=16, device="cpu")
        # Verify weights are identical.
        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2)


class TestExperiment1Additions:
    """Lovasz-softmax term and configurable SGD momentum (2026-09-30)."""

    def test_lovasz_equals_one_minus_miou_for_hard_predictions(self):
        import torch.nn.functional as F
        from celldform.segmentation.trainer import _lovasz_softmax
        torch.manual_seed(0)
        t = torch.randint(0, 3, (1, 16, 16))
        pred = torch.randint(0, 3, (1, 16, 16))
        probs = F.one_hot(pred, 3).permute(0, 3, 1, 2).float()
        ious = [((pred == c) & (t == c)).sum() / ((pred == c) | (t == c)).sum()
                for c in range(3) if (t == c).any()]
        expected = 1 - torch.stack(ious).float().mean()
        assert torch.isclose(_lovasz_softmax(probs, t), expected, atol=1e-5)

    def test_lovasz_weight_zero_is_the_old_loss(self):
        from celldform.segmentation.trainer import _FocalTverskyCELoss
        torch.manual_seed(0)
        logits, t = torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 16, 16))
        assert torch.equal(_FocalTverskyCELoss(3)(logits, t),
                           _FocalTverskyCELoss(3, lovasz_weight=0.0)(logits, t))
        assert _FocalTverskyCELoss(3, lovasz_weight=0.5)(logits, t) > _FocalTverskyCELoss(3)(logits, t)

    def test_sgd_momentum_and_nesterov_are_configurable(self):
        from celldform.config import _parse_training
        from celldform.segmentation.trainer import _build_optimizer
        cfg = _parse_training({"optimizer": "sgd", "lr": 0.01, "momentum": 0.99, "nesterov": True})
        opt = _build_optimizer(torch.nn.Linear(2, 2), cfg)
        g = opt.param_groups[0]
        assert g["momentum"] == 0.99 and g["nesterov"] is True and g["lr"] == 0.01


class TestGradAccumulation:
    """training.grad_accum_steps (2026-09-30): k micro-batches == one batch of k x size."""

    def _trainer(self, tmp_path, batch, k, x, y, init):
        from torch.utils.data import DataLoader, TensorDataset
        from celldform.config import load
        from celldform.segmentation.trainer import SegmentationTrainer
        cfg = tmp_path / f"c{batch}_{k}.yaml"
        cfg.write_text(
            "data: {frames_dir: x, masks_dir: y}\n"
            f"unet: {{out_channels: 3}}\n"
            f"training: {{optimizer: sgd, lr: 0.1, momentum: 0.0, weight_decay: 0.0, "
            f"batch_size: {batch}, grad_accum_steps: {k}, loss_type: focal_tversky, "
            f"use_amp: false, scheduler: none}}\n"
            f"device: cpu\ncheckpoint_dir: {tmp_path / f'ck{batch}'}\n")
        conf = load(cfg)
        model = torch.nn.Conv2d(1, 3, 3, padding=1)          # no BatchNorm on purpose
        model.load_state_dict(init)
        dl = DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)
        return SegmentationTrainer(model, dl, dl, conf)

    def test_four_micro_batches_equal_one_big_batch(self, tmp_path):
        torch.manual_seed(0)
        x, y = torch.randn(8, 1, 12, 12), torch.randint(0, 3, (8, 12, 12))
        init = torch.nn.Conv2d(1, 3, 3, padding=1).state_dict()
        big = self._trainer(tmp_path, 8, 1, x, y, init)
        acc = self._trainer(tmp_path, 2, 4, x, y, init)
        big._train_one_epoch(); acc._train_one_epoch()
        for a, b in zip(big.model.parameters(), acc.model.parameters()):
            assert torch.allclose(a, b, atol=1e-6)
