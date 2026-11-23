# Per-frame inference (legacy)

**Module:** `celldform.integration`

The feature-mask contract (`feature_mask`, `FEATURE_SIZE`) that the evaluation scripts rely on, and
`RealTimePipeline`, an older per-frame loop. **`RealTimePipeline` is not real-time and does not use the final HER2
framework**; it is kept for `celldform-infer`. To classify a recording, use the [desktop app](../demo.md).

---

## feature_mask

::: celldform.integration.realtime.feature_mask

## RealTimePipeline (legacy)

::: celldform.integration.realtime.RealTimePipeline
