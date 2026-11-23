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

SVM, random-forest and decision-tree classifiers were compared. They share one interface, so they can be swapped in `scripts/inference/infer.py` without code changes (the final HER2 classifier used by the desktop app is a separate framework, `celldform.demo.framework.HER2Framework`):

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
