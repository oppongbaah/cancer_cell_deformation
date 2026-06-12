# GPU & Hardware

---

## Which stages use the GPU?

Only **Stage 3 (U-Net segmentation)** runs on the GPU. All other stages run entirely on CPU.

| Stage | Component | GPU |
|-------|-----------|-----|
| 1 — Acquisition | `FrameExtractor` — OpenCV frame decode | No |
| 2 — Preprocessing | `PreprocessingPipeline` — OpenCV / NumPy | No |
| 3 — Segmentation | `UNet`, `SegmentationTrainer` — PyTorch | **Yes** |
| 3 — Real-time inference | `RealTimePipeline` — U-Net forward pass | **Yes** |
| 4 — Features | `MorphologyExtractor` — scikit-image / NumPy | No |
| 5 — Classification | `CellClassifier` — scikit-learn SVM/RF/DT | No |
| 6 — Biomechanics | `DeformationAnalyzer` — scipy / NumPy | No |

---

## Device configuration

The device is auto-detected at runtime everywhere using:

```python
device = conf.device or ("cuda" if torch.cuda.is_available() else "cpu")
```

### Via `configs/default.yaml`

```yaml
device: null     # auto-detect (default — uses CUDA if available)
device: "cuda"   # force GPU
device: "cpu"    # force CPU
```

### Via CLI flag

```bash
celldform-train-unet --config configs/default.yaml --device cuda
celldform-infer input/ --unet-ckpt checkpoints/unet_best.pt --device cpu
```

---

## Checking GPU availability

```python
import torch
print(torch.cuda.is_available())       # True / False
print(torch.cuda.get_device_name(0))   # e.g. "NVIDIA GeForce RTX 3080"
print(torch.cuda.memory_allocated())   # bytes currently in use
```

---

## Memory considerations

The U-Net processes `(B, 1, 256, 256)` float32 tensors. Approximate GPU memory usage at different batch sizes:

| Batch size | ~VRAM |
|-----------|-------|
| 8 | ~1 GB |
| 16 | ~2 GB |
| 32 | ~4 GB |

Reduce `training.batch_size` in `configs/default.yaml` if you run out of VRAM.

---

## Windows / WSL2

On WSL2, CUDA is supported via the NVIDIA CUDA on WSL2 driver. Verify with:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Set `training.num_workers: 0` on Windows to avoid DataLoader multiprocessing issues.
