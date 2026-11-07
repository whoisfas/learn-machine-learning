---
{
  "id": "p2-decision-tree",
  "title": "Decision Trees",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": [
    "Understand how decision trees partition feature space",
    "Implement CART splitting with Gini impurity and entropy",
    "Control overfitting via pruning and hyperparameters",
    "Interpret trees visually and extract rules"
  ],
  "concepts": [
    "Recursive binary partitioning",
    "Gini impurity, entropy, classification error",
    "Information gain / Gini gain for split selection",
    "Greedy top-down induction (CART algorithm)",
    "Pre-pruning (max_depth, min_samples) vs post-pruning",
    "Regression trees (MSE/MAE splitting)",
    "Feature importance via impurity decrease",
    "Decision boundaries: axis-aligned rectangles"
  ],
  "formulas": [
    {"name": "Gini impurity", "latex": "G = 1 - \\sum_k p_k^2", "note": "0 = pure node; max at uniform."},
    {"name": "Entropy", "latex": "H = -\\sum_k p_k \\log_2 p_k", "note": "Information content in bits."},
    {"name": "Information gain", "latex": "IG = H_{parent} - \\sum_{child} \\frac{n_{child}}{n_{parent}} H_{child}", "note": "Reduction in impurity from split."}
  ],
  "references": [
    "sklearn.tree docs",
    "ESL ch. 9",
    "Quinlan 'C4.5' / Breiman 'CART'"
  ]
}
---

## Penjelasan

Decision tree adalah model yang memprediksi dengan serangkaian pertanyaan if-else pada fitur. Sangat mudah diinterpretasi — setiap prediksi bisa dilacak ke jalur keputusan yang eksplisit.

**Cara kerja.** Mulai dari root node berisi semua data. Pada setiap node, cari split terbaik (fitur + threshold) yang memaksimalkan *information gain* (pengurangan impurity). Bagi data ke child nodes kiri/kanan, ulangi rekursif sampai kriteria stopping (max_depth, min_samples, pure node). Prediksi = kelas/mean mayoritas di leaf.

**Kriteria split.**
- *Classification*: Gini impurity (default sklearn, cepat) atau entropy (information gain). Keduanya mengukur ketidakmurnian node; lebih kecil = lebih murni.
- *Regression*: variance/MSE reduction — split meminimalkan within-node variance.
- *Gain ratio*: koreksi information gain yang bias ke fitur dengan banyak nilai (C4.5).

**Greedy nature.** CART adalah greedy top-down: memilih split terbaik *saat ini* tanpa melihat konsekuensi global. Ini cepat tapi tidak optimal — kombinasi split yang tampak buruk sekarang bisa menghasilkan tree global yang lebih baik.

**Overfitting.** Tree yang tumbuh sampai pure akan menghafal training data. Kontrol:
- *Pre-pruning*: max_depth, min_samples_split, min_samples_leaf, max_features.
- *Post-pruning*: cost-complexity pruning — pangkas subtree yang kontribusinya kecil.
- *Ensemble*: random forest/gradient boosting memperbaiki variance tree tunggal.

**Interpretasi.** Visualisasi tree (export_text, plot_tree) menunjukkan rules eksplisit: "jika income > 50k DAN debt_ratio < 0.3 -> approve". Feature importance dari impurity decrease menunjukkan fitur yang paling sering/sangat digunakan untuk split.

## Konsep Penting

1. **Axis-aligned boundaries** — tree hanya bisa memotong sumbu; data diagonal butuh banyak split untuk aproksimasi.
2. **Instability** — perubahan kecil data -> tree sangat berbeda; ensemble mengatasi ini.
3. **Missing values** — sklearn tree tidak handle NaN native; impute dulu atau pakai HistGradientBoosting.
4. **Extrapolation** — regression tree prediksi konstan di luar range training (tidak seperti linear).
5. **Categorical features** — perlu encoding (OneHot/Ordinal) atau native support (HistGB).

## Formula

- Gini: G = 1 − Σ pk²; entropy H = −Σ pk log2 pk.
- Split criterion: maximize impurity decrease = impurity(parent) − weighted avg impurity(children).
- Regression split: minimize Σ(y − mean_child)².

## Contoh Kasus

Underwriting kredit: features (income, credit_score, debt_ratio, employment_years). Tree depth 4 menghasilkan rules yang bisa dijelaskan ke regulator: "credit_score < 620 -> reject". Bandingkan interpretability vs accuracy dengan random forest.

## Implementasi

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

for depth in [2, 4, 6, None]:
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20,
                                random_state=42)
    scores = cross_val_score(dt, X_train, y_train, cv=5)
    print(f"depth={depth}: acc={scores.mean():.3f}")

dt.fit(X_train, y_train)
print(export_text(dt, feature_names=feature_names, max_depth=3))
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A node contains 40% class A and 60% class B. Its Gini impurity is:",
    "options": ["0.48", "0.60", "0.40", "0.24"],
    "answer": 0,
    "explanation": "G = 1 - (0.4² + 0.6²) = 1 - 0.52 = 0.48.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Decision trees handle missing values:",
    "options": [
      "Not natively in sklearn; need imputation",
      "Automatically by surrogate splits",
      "By treating NaN as a separate category",
      "By mean imputation internally"
    ],
    "answer": 0,
    "explanation": "sklearn.tree doesn't support NaN; HistGradientBoosting does.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `gini_impurity(labels)` for a list of class labels returning the Gini impurity.",
    "starter_code": "import numpy as np\nfrom collections import Counter\n\ndef gini_impurity(labels):\n    labels = np.asarray(labels)\n    pass\n",
    "function": "gini_impurity",
    "tests": [
      {"call": "gini_impurity([0,0,0,1])", "expected": 0.375, "tolerance": 0.001},
      {"call": "gini_impurity([0,0,0,0])", "expected": 0.0, "tolerance": 0.001},
      {"call": "gini_impurity([0,1])", "expected": 0.5, "tolerance": 0.001}
    ],
    "hints": ["counts = Counter(labels); probs = counts/n; G = 1 - sum(p²)"],
    "explanation": "Gini impurity measures node purity.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `best_split_gini(X, y)` for 1D feature X returning (threshold, gini_gain) of the best binary split over sorted unique midpoints.",
    "starter_code": "import numpy as np\n\ndef best_split_gini(X, y):\n    X = np.asarray(X, dtype=float)\n    y = np.asarray(y)\n    pass\n",
    "function": "best_split_gini",
    "tests": [
      {"call": "best_split_gini([1,2,3,4],[0,0,1,1])[0]", "expected": 2.5, "tolerance": 0.01}
    ],
    "hints": ["Sort by X; try midpoints between consecutive unique values", "gain = Gini(parent) - weighted avg Gini(children)"],
    "explanation": "Core of CART split selection.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "A depth-10 decision tree gets 99% train accuracy but 75% test. Explain why, and describe three strategies to improve generalization — including when you'd prefer a different model entirely.",
    "rubric": [
      "Identifies overfitting via memorization of noise",
      "Pre-pruning (depth/min_samples) or post-pruning",
      "Ensemble methods (RF, GBM)",
      "Mentions when linear/logistic model suffices or when interpretability vs accuracy tradeoff matters"
    ],
    "points": 2
  }
]
```

## Referensi

- sklearn.tree documentation
- ESL ch. 9 (Trees)
- Breiman et al., "Classification and Regression Trees" (CART)
