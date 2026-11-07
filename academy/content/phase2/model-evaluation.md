---
{
  "id": "p2-model-evaluation",
  "title": "Model Evaluation and Metrics",
  "phase": 2,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Choose appropriate metrics for classification and regression",
    "Understand confusion matrix and derived metrics",
    "Use ROC, PR curves and AUC correctly",
    "Recognize metric pitfalls: accuracy paradox, class imbalance"
  ],
  "concepts": [
    "Confusion matrix: TP, FP, TN, FN",
    "Accuracy, precision, recall, F1, specificity",
    "ROC curve and AUC; PR curve for imbalanced data",
    "Log-loss and Brier score for probabilistic predictions",
    "Regression metrics: MSE, RMSE, MAE, R², MAPE",
    "Threshold-dependent vs threshold-independent metrics",
    "Cost-sensitive evaluation",
    "Calibration assessment"
  ],
  "formulas": [
    {"name": "Precision", "latex": "P = \\frac{TP}{TP+FP}", "note": "Of predicted positives, how many correct."},
    {"name": "Recall", "latex": "R = \\frac{TP}{TP+FN}", "note": "Of actual positives, how many found."},
    {"name": "F1", "latex": "F_1 = \\frac{2PR}{P+R}", "note": "Harmonic mean of precision and recall."},
    {"name": "AUC-ROC", "latex": "\\int_0^1 TPR\\,d(FPR)", "note": "Area under ROC curve; 0.5=random, 1.0=perfect."},
    {"name": "Log-loss", "latex": "L = -\\frac{1}{n}\\sum_i [y_i \\log p_i + (1-y_i)\\log(1-p_i)]", "note": "Penalizes confident wrong predictions heavily."}
  ],
  "references": [
    "sklearn.metrics documentation",
    "'The Precision-Recall Plot Is More Informative...' (Saito & Rehmsmeier 2015)",
    "scikit-learn model evaluation guide"
  ]
}
---

## Penjelasan

Model evaluation menjawab: "seberapa baik model ini pada data yang belum pernah dilihat?" Metrik yang tepat bergantung pada jenis masalah, distribusi kelas, dan biaya kesalahan.

**Confusion matrix.** Untuk binary classification: TP (benar positif), FP (salah positif), TN (benar negatif), FN (salah negatif). Semua metrik lain diturunkan dari ini.

**Metrik klasifikasi.**
- *Accuracy* = (TP+TN)/n — menyesatkan pada imbalanced data (prediksi semua mayoritas = tinggi).
- *Precision* = TP/(TP+FP) — dari yang diprediksi positif, berapa yang benar. Penting saat false positive mahal (spam filter).
- *Recall/Sensitivity* = TP/(TP+FN) — dari positif sebenarnya, berapa yang tertangkap. Penting saat false negative mahal (deteksi penyakit).
- *F1* = harmonic mean precision & recall — kompromi saat keduanya penting.
- *Specificity* = TN/(TN+FP) — kebalikan recall untuk kelas negatif.
- *Balanced accuracy* = (TPR + TNR)/2 — rata-rata recall per kelas, lebih baik untuk imbalance.

**ROC vs PR curve.** ROC plot TPR vs FPR pada berbagai threshold; AUC = probabilitas classifier memberi skor lebih tinggi ke positif acak daripada negatif. Berguna untuk perbandingan model secara umum. *Tapi* pada data sangat imbalanced, ROC bisa misleading — PR curve (precision vs recall) lebih informatif karena fokus pada kelas minoritas.

**Probabilistic metrics.** Log-loss menghukum prediksi yang salah dengan percaya diri; Brier score = MSE pada probabilitas. Keduanya menilai kalibrasi, bukan hanya ranking.

**Regression.** MSE/RMSE (sensitif outlier, satuan sama dengan y), MAE (robust, interpretasi median-like), MAPE (persentase error — bermasalah jika y dekat 0), R² (proporsi variance dijelaskan — bisa negatif pada test), sMAPE, quantile loss untuk interval.

## Konsep Penting

1. **Accuracy paradox** — pada 99:1 imbalance, model yang prediksi semua negatif dapat 99% accuracy tapi 0 recall.
2. **Threshold adalah hyperparameter** — semua metrik threshold-dependent; ROC/PR membantu memilih operating point.
3. **Metrik harus align dengan biaya** — tentukan dulu biaya FP vs FN, baru pilih metrik.
4. **Jangan tuning di test** — semua keputusan model (threshold, model choice) harus di validation.
5. **Satu metrik tidak cukup** — laporkan confusion matrix + metrik utama + CI.

## Formula

- F1 = 2·P·R/(P+R).
- Balanced accuracy = (TPR + TNR)/2.
- Brier score = (1/n)Σ(pi − yi)².
- MAPE = (100/n)Σ|yi − ŷi|/|yi|.

## Contoh Kasus

Deteksi fraud (0.17% positif): accuracy tidak berguna. PR-AUC dan recall@precision=0.9 lebih relevan. Threshold dipilih untuk menangkap 80% fraud dengan precision 30% (investigasi manual 3x lipat lebih banyak kasus tapi fraud tertangkap). Bandingkan dengan ROC-AUC yang terlihat optimis.

## Implementasi

```python
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, log_loss,
    precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("PR-AUC:", average_precision_score(y_test, proba))
print("Log-loss:", log_loss(y_test, proba))

# PR curve for threshold selection
prec, rec, thr = precision_recall_curve(y_test, proba)
plt.plot(rec, prec); plt.xlabel('recall'); plt.ylabel('precision')
plt.show()
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A model predicts all samples as negative on a dataset with 99% negatives. Its accuracy is 99% but recall for positive class is:",
    "options": ["0%", "99%", "50%", "100%"],
    "answer": 0,
    "explanation": "Predicting all negative -> TP=0 -> recall=0/(0+FN)=0. Accuracy paradox.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "For fraud detection where false negatives are very costly, prioritize:",
    "options": ["High recall", "High precision", "High accuracy", "Low log-loss"],
    "answer": 0,
    "explanation": "High recall catches more actual fraud; accept more false positives.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `f1_score(y_true, y_pred)` for binary labels returning F1 (handle division by zero -> 0.0).",
    "starter_code": "def f1_score(y_true, y_pred):\n    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)\n    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)\n    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)\n    pass\n",
    "function": "f1_score",
    "tests": [
      {"call": "f1_score([1,1,0,0],[1,1,0,0])", "expected": 1.0},
      {"call": "f1_score([1,1,1,0],[1,0,0,0])", "expected": 0.5, "tolerance": 0.001},
      {"call": "f1_score([0,0],[1,1])", "expected": 0.0}
    ],
    "hints": ["precision = tp/(tp+fp); recall = tp/(tp+fn); f1 = 2pr/(p+r)"],
    "explanation": "F1 harmonic mean of precision and recall.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `best_threshold_pr(y_true, proba, target_recall)` returning the threshold achieving at least target_recall with maximum precision.",
    "starter_code": "import numpy as np\n\ndef best_threshold_pr(y_true, proba, target_recall=0.8):\n    y_true = np.asarray(y_true)\n    proba = np.asarray(proba)\n    pass\n",
    "function": "best_threshold_pr",
    "tests": [
      {"call": "best_threshold_pr([1,1,0,0],[0.9,0.8,0.4,0.3],0.5)", "expected": 0.8, "tolerance": 0.01}
    ],
    "hints": ["Sort unique thresholds; for each compute precision/recall; pick max precision with recall>=target"],
    "explanation": "Operating point selection from PR curve.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your classifier has ROC-AUC 0.95 but PR-AUC 0.30 on a dataset with 2% positives. A stakeholder says '95% AUC means we're almost perfect'. Explain the discrepancy and what you'd report instead.",
    "rubric": [
      "ROC-AUC optimistic on imbalance; PR-AUC more informative for minority class",
          "PR-AUC 0.30 still better than baseline (0.02) but far from great",
      "Recommend PR curve, recall@precision operating point, or F-beta",
      "Mentions calibration if probabilities used for decisions"
        ],
    "points": 2
  }
]
```

## Referensi

- sklearn.metrics documentation
- Saito & Rehmsmeier (2015) PR vs ROC paper
- 'Assessing Classifier Performance' — various
- scikit-learn model evaluation guide
