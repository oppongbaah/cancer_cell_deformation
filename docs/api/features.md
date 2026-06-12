# Feature Extraction

**Module:** `celldform.features`

Stage 4 of the pipeline — extracts 15 morphological descriptors from a binary cell mask.

---

## Feature list

| Feature | Description |
|---------|-------------|
| `area` | Cell area in px² (or μm² if `pixel_size_um` is set) |
| `perimeter` | Boundary length |
| `major_axis` | Length of major ellipse axis |
| `minor_axis` | Length of minor ellipse axis |
| `aspect_ratio` | `major_axis / minor_axis` — primary deformation signal |
| `eccentricity` | Ellipse eccentricity (0 = circle, 1 = line) |
| `circularity` | `4π·area / perimeter²` (1 = perfect circle) |
| `solidity` | `area / convex-hull area` |
| `hu_0` – `hu_6` | Log-scaled Hu moment invariants |

!!! note "Feature ordering"
    Features are always sorted **alphabetically by name** before being passed to the classifier. Do not change this ordering without updating `RealTimePipeline` and `scripts/infer.py`.

---

## MorphologyExtractor

::: celldform.features.extractor.MorphologyExtractor
