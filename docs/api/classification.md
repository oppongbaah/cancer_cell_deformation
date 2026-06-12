# Classification

**Module:** `celldform.classification`

Stage 5 of the pipeline — classifies cells as low or high HER2 expression from morphological features.

---

## Labels

| Label | Meaning |
|-------|---------|
| `0` | Low HER2 expression — stiffer cell |
| `1` | High HER2 expression — more deformable |

---

## Comparative classifiers

The thesis requires comparative analysis of SVM, RF, and DT. All classifiers implement the same interface so they can be swapped in `scripts/infer.py` and `RealTimePipeline` without any code changes:

```python
clf = CellClassifier(model_type="svm")   # or "rf", "dt"
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.evaluate(X_test, y_test)
clf.save("checkpoints/svm.pkl")

clf2 = CellClassifier.load("checkpoints/svm.pkl")
```

---

## CellClassifier

::: celldform.classification.classifiers.CellClassifier
