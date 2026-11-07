---
{
  "id": "p2-random-forest",
  "title": "Random Forest",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": [
    "Understand bagging and how Random Forest reduces variance",
    "Explain feature randomness and its role in decorrelating trees",
    "Tune key hyperparameters (n_estimators, max_depth, max_features)",
    "Interpret feature importances and their limitations"
  ],
  "concepts": [
    "Bootstrap aggregating (bagging)",
    "Random subspace method (feature subsampling)",
    "Variance reduction through ensemble averaging",
    "Out-of-bag (OOB) error estimation",
    "Feature importance: Gini importance vs permutation importance",
    "ExtraTrees vs RandomForest",
    "When RF struggles: high-dimensional sparse, smooth functions"
  ],
  "formulas": [
    {"name": "Bagging variance", "latex": "\\text{Var}(\\bar{f}) \\approx \\frac{\\sigma^2}{B} + \\text{covariance term}", "note": "Averaging B trees reduces variance if trees are decorrelated."},
    {"name": "OOB estimate", "latex": "\\hat{R}_{OOB} = \\frac{1}{n}\\sum_i L(y_i, \\hat{f}_{-i}(x_i))", "note": "Each sample predicted by trees that didn't see it."}
  ],
  "references": [
    "Breiman 2001 'Random Forests' paper",
    "sklearn.ensemble docs",
    "ESL ch. 15"
  ]
}
---

## Penjelasan

Random Forest adalah ensemble dari banyak decision tree yang dilatih pada bootstrap sample (bagging) dengan subset fitur acak di setiap split. Kombinasi dua sumber randomness ini mendekorrelasikan tree sehingga averaging sangat efektif mengurangi variance.

**Bagging.** Setiap tree dilatih pada sampel bootstrap (n sampel diambil dengan replacement dari n data). Sekitar 63.2% data unik per sampel; 36.8% sisanya = out-of-bag (OOB) yang bisa dipakai untuk validasi internal tanpa holdout terpisah. Prediksi final = majority vote (klasifikasi) atau rata-rata (regresi).

**Feature randomness.** Di setiap node split, hanya subset acak fitur (default sqrt(n_features) untuk klasifikasi, n_features/3 untuk regresi) yang dipertimbangkan. Tanpa ini, semua tree akan memilih fitur dominan yang sama -> tree sangat korelatif -> ensemble tidak banyak membantu.

**Mengapa bekerja.** Variance ensemble ≈ variance individual / B jika tree tidak berkorelasi. RF membuat tree kurang korelatif via bagging + feature randomness. Hasil: akurasi mirip boosting tapi lebih robust, tidak perlu tuning learning rate, dan parallelizable.

**Hyperparameters.**
- `n_estimators`: lebih banyak = lebih stabil, diminishing returns setelah ~500.
- `max_depth`: None = full growth; batasi untuk kontrol variance/kompleksitas.
- `max_features`: sqrt (klasifikasi) / n_features (regresi); lebih kecil = lebih decorrelated tapi tree lebih lemah.
- `min_samples_leaf`: smoothing; mencegah leaf dengan 1 sampel.
- `bootstrap`: False = pastikan semua data dipakai (mirip ExtraTrees).

**Feature importance.** Dua cara:
- *Gini/MDI importance*: penurunan impurity rata-rata saat fitur digunakan; bias ke fitur dengan banyak nilai unik dan high cardinality.
- *Permutation importance*: acak ulang fitur, ukur penurunan performa; lebih reliable tapi butuh data held-out dan lebih lambat.

## Konsep Penting

1. **RF = variance reducer** — boosting = bias reducer; keduanya sering komplementer.
2. **OOB score** — gratis validasi internal; bandingkan dengan CV.
3. **Feature importance bias** — MDI importance menyesatkan untuk fitur kontinu/korelatif; pakai permutation.
4. **Tidak extrapolate** — seperti tree, prediksi konstan di luar range training.
5. **Scalability** — RF lambat untuk data sangat besar; HistGradientBoosting lebih cepat.

## Formula

- Bootstrap sample: P(sample not in bootstrap) = (1−1/n)^n → e⁻¹ ≈ 36.8%.
- RF prediction: ŷ = (1/B)Σ f_b(x) untuk regresi; majority vote untuk klasifikasi.

## Contoh Kasus

Fraud detection: 28 fitur numerik PCA + Amount. RF dengan 500 trees, class_weight balanced, OOB score untuk monitoring. Feature importance menunjukkan V14, V4, V11 paling prediktif — konsisten dengan domain knowledge fraud detection.

## Implementasi

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=2,
    max_features='sqrt', oob_score=True,
    class_weight='balanced_subsample', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

print("OOB score:", rf.oob_score_)
print("Test AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# permutation importance (more reliable)
from sklearn.inspection import permutation_importance
imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0)
for i in imp.importances_mean.argsort()[::-1][:10]:
    print(f"{feature_names[i]}: {imp.importances_mean[i]:.4f}")
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Random Forest primarily reduces which source of error compared to a single decision tree?",
    "options": ["Variance", "Bias", "Noise in labels", "Feature noise"],
    "answer": 0,
    "explanation": "Bagging + feature randomness decorrelate trees; averaging reduces variance, not bias.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Approximately what fraction of training samples are out-of-bag (not in a given bootstrap sample)?",
    "options": ["~37%", "~50%", "~63%", "~10%"],
    "answer": 0,
    "explanation": "(1-1/n)^n -> e^-1 ≈ 0.368.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Why does feature subsampling at each split help Random Forest?",
    "options": [
      "It decorrelates trees so averaging is more effective",
      "It speeds up training",
      "It prevents overfitting entirely",
      "It handles missing values"
    ],
    "answer": 0,
    "explanation": "Without it, all trees pick the same dominant features -> highly correlated -> little ensemble benefit.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `bootstrap_sample(X, y, rng)` returning indices of a bootstrap sample (n draws with replacement) and the OOB indices.",
    "starter_code": "import numpy as np\n\ndef bootstrap_sample(X, y, rng):\n    n = len(y)\n    idx = rng.integers(0, n, n)\n    pass\n",
    "function": "bootstrap_sample",
    "tests": [
      {"call": "a,b=bootstrap_sample([0]*10,[0]*10,np.random.default_rng(0)); len(a)==10", "expected": true}
    ],
    "hints": ["in-bag = idx; oob = sorted(set(range(n)) - set(idx))", "return (idx, oob_idx)"],
    "explanation": "Bootstrap resampling core of bagging.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Gini importance ranks feature A highest, but permutation importance ranks it 5th. Explain the discrepancy and which you'd trust for a report to stakeholders.",
    "rubric": [
      "Gini importance biased toward high-cardinality/correlated features",
      "Permutation importance measures actual performance drop — more reliable",
          "Mentions need for held-out data and variance across repeats",
      "Recommends permutation for reporting, possibly SHAP for deeper analysis"
        ],
    "points": 2
  }
]
```

## Referensi

- Breiman, 'Random Forests' (2001)
- sklearn.ensemble.RandomForestClassifier docs
- ESL ch. 15 (Bagging, RF, Boosting)
