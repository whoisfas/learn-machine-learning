---
{
  "id": "p2-cross-validation",
  "title": "Cross-Validation",
  "phase": 2,
  "difficulty": "intermediate",
  "estimated_hours": 8,
  "objectives": [
    "Implement k-fold, stratified k-fold, and leave-one-out CV",
    "Understand when each CV strategy is appropriate",
    "Avoid data leakage in CV pipelines",
    "Interpret CV scores and their variance"
  ],
  "concepts": [
    "k-fold cross-validation mechanics",
    "StratifiedKFold for classification",
    "GroupKFold for grouped data",
    "TimeSeriesSplit for temporal data",
    "Leave-one-out (LOO) and leave-p-out",
    "Repeated CV for stability",
    "Nested CV for unbiased model selection",
    "Pipeline + CV: preprocessing inside fold"
  ],
  "formulas": [
    {"name": "CV estimate", "latex": "\\hat{R}_{CV} = \\frac{1}{k}\\sum_{i=1}^{k} R_i", "note": "Average test error across folds."},
    {"name": "Standard error of CV", "latex": "SE = \\frac{sd(R_i)}{\\sqrt{k}}", "note": "Uncertainty of CV estimate."}
  ],
  "references": [
    "sklearn.model_selection docs",
    "ESL ch. 7.10",
    "Kohavi 1995 'A study of cross-validation'"
  ]
}
---

## Penjelasan

Cross-validation (CV) adalah teknik evaluasi yang membagi data menjadi k fold, melatih pada k-1 fold dan menguji pada fold tersisa, lalu merata-rata hasil. Ini memberikan estimasi performa yang lebih stabil daripada single train/test split.

**Varian CV.**
- *k-fold*: bagi data jadi k bagian sama; setiap fold jadi test sekali. Default k=5 atau 10.
- *StratifiedKFold*: untuk klasifikasi — menjaga proporsi kelas di setiap fold (penting untuk imbalance).
- *GroupKFold*: jika data punya grup (mis. pasien, user), semua sampel grup yang sama harus di fold yang sama — mencegah leakage.
  
- *TimeSeriesSplit*: untuk data temporal — fold training selalu sebelum fold test (tidak boleh shuffle).
- *Leave-one-out (LOO)*: k=n; melatih n model; mahal tapi unbiased untuk dataset kecil.
- *Repeated CV*: ulangi k-fold dengan shuffle berbeda; mengurangi varians estimasi.

**Nested CV.** Untuk model selection + evaluation yang tidak bias: outer CV mengukur performa, inner CV memilih hyperparameter. Tanpa nested, hyperparameter tuning pada outer fold test set menyebabkan optimistic bias.

**CV + Pipeline.** Preprocessing (scaling, encoding) HARUS di dalam pipeline yang di-CV — jika di-fit pada seluruh data sebelum CV, statistics (mean, std) bocor dari test fold ke training fold. sklearn `Pipeline` + `cross_val_score` menangani ini dengan benar.

**Interpretasi.** CV scores punya varians; perbedaan kecil antar model bisa noise. Gunakan paired tests (McNemar, Wilcoxon) atau repeated CV untuk membandingkan model secara signifikan.

## Konsep Penting

1. **Stratify untuk klasifikasi** — tanpa stratifikasi, fold bisa tidak representatif untuk kelas minoritas.
2. **Group leakage** — data dari grup yang sama di train dan test = leakage; GroupKFold mencegah.
3. **Temporal leakage** — time series tidak boleh di-shuffle; TimeSeriesSplit menjaga urutan.
4. **Pipeline dalam CV** — preprocessing di dalam fold, bukan sebelum.
5. **CV bukan pengganti test set** — CV untuk model selection/tuning; final test set untuk estimasi performa unbiased.

## Formula

- CV estimate: mean of k fold scores.
- Nested CV: outer loop evaluates, inner loop tunes.

## Contoh Kasus

Dataset pasien dengan multiple measurements per patient: GroupKFold by patient_id mencegah model "mengenali" pasien dari pengukuran lain di training. Atau time series forecasting: TimeSeriesSplit memastikan model tidak "melihat masa depan".

## Implementasi

```python
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
    GroupKFold, TimeSeriesSplit, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# proper pipeline inside CV
pipe = Pipeline([('scaler', StandardScaler()),
                 ('clf', LogisticRegression(max_iter=1000))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
print(f"AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

# grouped data
cv_group = GroupKFold(n_splits=5)
scores_g = cross_val_score(pipe, X, y, groups=groups, cv=cv_group)

# detailed per-fold results
res = cross_validate(pipe, X, y, cv=cv,
                     scoring=['accuracy','roc_auc','neg_log_loss'],
                     return_train_score=True)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Why use StratifiedKFold instead of regular KFold for classification?",
    "options": [
      "To maintain class proportions in each fold, especially for imbalanced data",
      "It's faster",
      "It prevents overfitting",
      "It handles missing values"
    ],
    "answer": 0,
    "explanation": "Stratification ensures each fold has representative class distribution.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "When must you use GroupKFold?",
    "options": [
      "When samples from the same group appear in both train and test would leak information",
      "When data is too large for k-fold",
      "When classes are imbalanced",
      "Always for regression"
    ],
    "answer": 0,
    "explanation": "GroupKFold ensures all samples from a group stay in the same fold — prevents group-level leakage.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Why must preprocessing be inside the CV pipeline?",
    "options": [
      "Fitting scaler on all data before CV leaks test fold statistics into training",
      "It's faster",
      "sklearn requires it",
      "It improves accuracy"
    ],
    "answer": 0,
    "explanation": "Scaler fit on full data uses test fold info; inside pipeline, each fold's scaler sees only train fold.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `manual_kfold_cv(model_fn, X, y, k, scoring_fn)` performing k-fold CV manually and returning list of scores.",
    "starter_code": "import numpy as np\n\ndef manual_kfold_cv(model_fn, X, y, k, scoring_fn):\n    X = np.asarray(X); y = np.asarray(y)\n    n = len(y)\n    indices = np.arange(n)\n    scores = []\n    pass\n",
    "function": "manual_kfold_cv",
    "tests": [
      {"call": "manual_kfold_cv(lambda: __import__('sklearn.linear_model',fromlist=['LinearRegression']).LinearRegression(), [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],[1,2,3,4,5,6,7,8,9,10], 5, lambda m,X,y: m.score(X,y))", "expected": "check"}
    ],
    "hints": ["Split indices into k folds; for each fold, train on rest, predict on fold, score", "return list of scores"],
    "explanation": "Understanding CV mechanics manually.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You compare Model A (CV acc 0.85±0.02) vs Model B (0.86±0.03) on 5-fold CV. Is B better? What statistical test or procedure would give a more confident answer?",
    "rubric": [
      "Notes difference within noise (overlapping CIs)",
          "Proposes paired comparison: same folds, per-fold difference test (Wilcoxon/t-test on fold scores)",
      "Or repeated CV / bootstrap for more stable estimate",
      "Or nested CV if hyperparameters involved"
        ],
    "points": 2
  }
]
```

## Referensi

- sklearn.model_selection documentation
- ESL ch. 7.10 (model assessment)
- Kohavi (1995) 'A study of cross-validation and bootstrap'
- 'Cross-validation strategies' sklearn guide
