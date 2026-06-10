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
