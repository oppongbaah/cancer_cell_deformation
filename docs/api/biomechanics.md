# Biomechanics

**Module:** `celldform.biomechanics`

Stage 6 of the pipeline — quantifies deformation rate and fits constitutive models.

---

## Key equations

| Equation | Description |
|----------|-------------|
| `RoD = dAR/dt` | Rate of deformation — linear slope of aspect ratio over time |
| `D = (a−b)/(a+b)` | Deformation index |
| `σ = E·ε + η·dε/dt` | Kelvin–Voigt constitutive model |

---

## DeformationAnalyzer

::: celldform.biomechanics.analysis.DeformationAnalyzer
