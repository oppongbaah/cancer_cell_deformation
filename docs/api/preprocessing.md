# Preprocessing

**Module:** `celldform.preprocessing`

Stage 2 of the pipeline — converts raw greyscale frames to normalised 256×256 float32 arrays ready for U-Net input.

!!! warning "Preprocessing is manual"
    Run `scripts/preprocess_frames.py --masks` before training. `PreprocessingPipeline` is **not** applied automatically inside the training loop.

---

## PreprocessingConfig

::: celldform.preprocessing.pipeline.PreprocessingConfig

---

## PreprocessingPipeline

::: celldform.preprocessing.pipeline.PreprocessingPipeline
