---
{
  "id": "p2-supervised-learning",
  "title": "Supervised Learning",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": [
    "Formulate supervised learning as input-output mapping",
    "Understand the role of the loss function and inductive bias",
    "Distinguish parametric vs non-parametric models",
    "Know the landscape of supervised algorithms"
  ],
  "concepts": [
    "Training data {(xi, yi)}; goal: learn f: X -> Y",
    "Loss functions: MSE, MAE, cross-entropy, hinge",
    "Empirical risk minimization (ERM)",
    "Parametric (LR, NN) vs non-parametric (kNN, trees, kernels)",
    "Inductive bias: what assumptions the model makes",
    "Decision boundaries and hypothesis space",
    "Generative vs discriminative models"
  ],
  "formulas": [
    {"name": "ERM objective", "latex": "\\hat{f} = \\arg\\min_{f \\in \\mathcal{H}} \\frac{1}{n}\\sum_i L(f(x_i), y_i)", "note": "Choose hypothesis minimizing average loss."},
    {"name": "Cross-entropy", "latex": "L = -\\sum_c y_c \\log \\hat{p}_c", "note": "Standard classification loss; punishes confident wrong predictions."}
  ],
  "references": [
    "sklearn user guide — supervised learning",
    "Hastie et al., 'The Elements of Statistical Learning' ch. 2"
  ]
}
---

## Penjelasan

Supervised learning adalah kerangka paling umum di ML: diberikan pasangan (input, output), belajar fungsi yang memetakan input ke output. Semua yang Anda lakukan di Phase 2 (regresi, klasifikasi, trees, SVM) adalah instance supervised learning.

**Formulasi.** Data training D = {(x1,y1),...,(xn,yn)}. Tujuan: temukan f ∈ H (hypothesis space) yang meminimalkan expected loss E[L(f(X),Y)]. Karena distribusi data tidak diketahui, kita minimalkan *empirical* loss pada training set — itulah ERM.

**Loss function.** Loss mengukur seberapa salah prediksi untuk satu contoh; objective = rata-rata loss. Pilihan loss mencerminkan struktur masalah:
- *Regression*: MSE (kuadrat menghukum outlier besar), MAE (robust), Huber (kompromi).
- *Classification*: cross-entropy (probabilistik, smooth), hinge (SVM), 0-1 (tidak smooth, tak bisa dioptimasi langsung).

**Inductive bias.** Setiap model membawa asumsi tentang data. Linear regression mengasumsikan hubungan linear; decision tree mengasumsikan axis-aligned splits; kNN mengasumsikan titik berdekatan sekelas. Bias yang tepat membuat generalisasi baik dengan sedikit data; bias yang salah membuat model mustahil fit.

**Parametrik vs non-parametrik.** Parametrik (linear regression, logistic regression, neural net) punya jumlah parameter tetap — kapasitas tetap berapapun n. Non-parametrik (kNN, decision tree, kernel methods) kapasitas tumbuh dengan data — bisa fit kompleksitas arbitrer tapi butuh lebih banyak data dan rawan overfit.

**Generative vs discriminative.** Generative memodelkan P(X,Y) lalu derive P(Y|X) (naive Bayes, GMM, GAN). Discriminative langsung memodelkan P(Y|X) atau decision boundary (logistic regression, SVM, trees). Discriminative biasanya lebih akurat untuk prediksi; generative berguna untuk sampling dan data regions.

## Konsep Penting

1. **ERM bisa overfit** — minimization training loss ≠ minimization test loss; regularization menyeimbangkan.
2. **Hypothesis space penting** — model tidak bisa menemukan fungsi di luar H; linear model tak akan fit data sinusoidal.
3. **Loss ≠ metric** — Anda bisa train dengan cross-entropy tapi evaluasi dengan F1; keduanya terkait tapi berbeda.
4. **IID assumption** — supervised learning klasik mengasumsikan train/test dari distribusi sama; time series melanggar ini.

## Formula

- Expected risk: R(f) = E[L(f(X), Y)].
- ERM: R̂(f) = (1/n)Σ L(f(xi), yi).
- 0-1 loss classification: L = 1[f(x) ≠ y].

## Contoh Kasus

Klasifikasi email: x = representasi email (TF-IDF vector), y ∈ {spam, ham}. Model discriminative (logistic regression) belajar boundary langsung. Bandingkan dengan generative naive Bayes yang memodelkan P(kata|kelas) — bekerja baik di data kecil tapi asumsi independensi melanggar realita.

## Implementasi

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np

# ERM in action: sklearn minimizes loss internally
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=0)
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
proba = clf.predict_proba(X_test)
print("log-loss:", log_loss(y_test, proba))
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "ERM minimizes which quantity?",
    "options": [
      "Average loss on training data",
      "Expected loss on all possible data",
      "Test set error",
      "Model complexity"
    ],
    "answer": 0,
    "explanation": "ERM = empirical risk minimization = min avg training loss.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Which model is non-parametric?",
    "options": ["Linear regression", "kNN", "Logistic regression", "Naive Bayes (Gaussian)"],
    "answer": 1,
    "explanation": "kNN's capacity grows with n; others have fixed parameter count.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Why is cross-entropy preferred over 0-1 loss for training classifiers?",
    "options": [
      "It provides smooth gradients even when predictions are wrong",
      "It is faster to compute",
      "It guarantees global optimum",
      "0-1 loss is undefined for probabilities"
    ],
    "answer": 0,
    "explanation": "0-1 loss is piecewise constant (zero gradient a.e.); CE is smooth and gives informative gradients.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "open",
    "question": "Compare generative vs discriminative approaches for the same classification task. When would you prefer each? Give one concrete example.",
    "rubric": [
      "Generative models P(X,Y); discriminative P(Y|X)",
      "Generative useful for sampling, missing data, small n; discriminative often more accurate asymptotically",
      "Concrete example: naive Bayes vs logistic regression on text"
    ],
    "points": 2
  }
]
```

## Referensi

- Hastie, Tibshirani, Friedman — ESL ch. 2
- sklearn user guide
- Ng, "Machine Learning Yearning" ch. on dev process
