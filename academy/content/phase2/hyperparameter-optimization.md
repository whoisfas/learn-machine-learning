---
{
  "id": "p2-hyperparameter-optimization",
  "title": "Hyperparameter Optimization",
  "phase": 2,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Distinguish hyperparameters from parameters",
    "Apply grid search, random search, and Bayesian optimization",
    "Use successive halving and early stopping for efficiency",
    "Avoid overfitting the validation set during search"
  ],
  "concepts": [
    "Parameters vs hyperparameters",
    "Grid search: exhaustive but expensive",
    "Random search: often as good, more efficient",
    "Bayesian optimization: model-based, sample-efficient",
    "Successive halving / Hyperband",
    "Early stopping as implicit regularization",
    "Search space design: log vs linear scales",
    "Nested CV to avoid optimistic bias"
  ],
  "formulas": [
    {"name": "Random search efficiency", "latex": "P(\\text{hit good region}) \\propto \\frac{\\text{budget}}{\\text{dim}}", "note": "Random search scales better than grid in high dimensions."}
  ],
  "references": [
    "sklearn.model_selection docs (GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV)",
    "Bergstra & Bengio 2012 'Random Search for Hyper-Parameter Optimization'",
    "optuna documentation"
  ]
}
---

## Penjelasan

Hyperparameters adalah konfigurasi model yang di-set *sebelum* training (learning rate, max_depth, regularization strength) — berbeda dengan parameters (weights) yang dipelajari dari data. HPO adalah proses mencari kombinasi hyperparameter yang memaksimalkan performa validasi.

**Grid search.** Coba semua kombinasi dalam grid yang dispesifikasikan. Kelebihan: exhaustive, reproducible. Kekurangan: biaya kombinatorial meledak; banyak evaluasi terbuang pada region yang jelas buruk.

**Random search.** Sampel kombinasi acak dari distribusi. Bergstra & Bengio (2012) menunjukkan random search sering sama baiknya dengan grid dengan budget jauh lebih kecil — terutama ketika hanya beberapa hyperparameter yang penting. Untuk d hyperparameters, grid butuh k^d evaluasi; random butuh O(d·log k) untuk coverage yang sama.

**Bayesian optimization.** Bangun surrogate model (Gaussian process) dari hasil trial sebelumnya; pilih trial berikutnya yang memaksimalkan acquisition function (expected improvement, UCB). Jauh lebih sample-efficient untuk fungsi mahal — standar untuk deep learning HPO. Tools: Optuna, Hyperopt, scikit-optimize.

**Successive halving / Hyperband.** Evaluasi banyak konfigurasi dengan resource kecil dulu (sedikit epoch/data), eliminasi yang buruk, alokasikan lebih banyak resource ke yang promising. Hyperband mengulang dengan budget berbeda — adaptif terhadap ketidakpastian.

**Search space design.** Untuk parameter yang span orde magnitudo (lr: 1e-5 s/d 1e-1), gunakan log-uniform bukan linear — linear grid akan menghabiskan budget di region besar. Untuk integer (max_depth), uniform integer.

**Early stopping.** Hentikan training saat validation metric berhenti membaik — implisit regularization + efisiensi. Bisa dipandang sebagai HPO gratis untuk training duration.

## Konsep Penting

1. **Nested CV untuk unbiased estimate** — tuning pada outer test fold = optimistic bias.
2. **Search space > algorithm** — mendefinisikan range yang masuk akal lebih penting dari metode search.
3. **Log scales untuk lr/regularization** — parameter orde-magnitudo perlu log sampling.
4. **Budget allocation** — successive halving lebih efisien dari full training semua konfigurasi.
5. **Reproducibility** — catat semua trials (params, metrics, seed); gunakan experiment tracking.

## Formula

- Grid search cost: O(Π |values_i|) configurations.
- Successive halving: keep top 1/η configurations each round.
- Expected improvement: EI(x) = E[max(f(x) − f*, 0)].

## Contoh Kasus

Tune XGBoost pada tabular data: 6 hyperparameters (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight). Grid = 4^6 = 4096 runs (mahal). Random search 100 runs mencapai ~95% performa grid. Optuna Bayesian 50 trials melampaui keduanya dengan budget lebih kecil.

## Implementasi

```python
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import loguniform, randint, uniform
from sklearn.pipeline import Pipeline
import numpy as np

# log-uniform for learning rate
param_dist = {
    'clf__learning_rate': loguniform(1e-4, 0.3),
    'clf__max_depth': randint(3, 12),
    'clf__n_estimators': randint(50, 500),
    'clf__subsample': uniform(0.5, 1.0),
}
search = RandomizedSearchCV(
    pipe, param_dist, n_iter=100, cv=5,
    scoring='roc_auc', n_jobs=-1, random_state=42,
    refit=True)
search.fit(X_train, y_train)
print("Best:", search.best_params_)
print("CV score:", search.best_score_)

# log all trials for analysis
results = pd.DataFrame(search.cv_results_)
results[['param_clf__learning_rate','param_clf__max_depth',
         'mean_test_score']].sort_values('mean_test_score', ascending=False).head()
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Hyperparameters vs parameters:",
    "options": [
      "Hyperparameters are set before training; parameters are learned from data",
      "They are the same thing",
      "Hyperparameters are learned, parameters are set",
      "Parameters are only for neural networks"
    ],
    "answer": 0,
    "explanation": "Weights/biases = parameters (learned); lr, max_depth = hyperparameters (tuned).",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Why is random search often better than grid search for HPO?",
    "options": [
      "Only a few hyperparameters matter; random covers important dimensions more efficiently",
      "It's always faster",
      "It guarantees better results",
      "Grid search is biased"
    ],
    "answer": 0,
    "explanation": "Bergstra & Bengio 2012: random search explores important dimensions better with same budget.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "For learning_rate spanning 1e-5 to 1e-1, the best sampling strategy is:",
    "options": ["Log-uniform", "Linear uniform", "Always 0.01", "Exponential decay"],
    "answer": 0,
    "explanation": "Log-uniform samples uniformly in log space — covers orders of magnitude evenly.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `halving_schedule(n_configs, eta=3, min_resource=1)` returning list of (n_configs_to_keep, resource_multiplier) for successive halving rounds.",
    "starter_code": "import math\n\ndef halving_schedule(n_configs, eta=3, min_resource=1):\n    pass\n",
    "function": "halving_schedule",
    "tests": [
      {"call": "halving_schedule(27, eta=3)", "expected": [[27,1],[9,3],[3,9],[1,27]]}
    ],
    "hints": ["Each round: keep top n/eta, multiply resource by eta", "Stop when 1 config remains"],
    "explanation": "Successive halving allocates more resources to promising configs.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You run 500-trial random search and find best config with CV score 0.91 vs baseline 0.85. Before reporting 'we improved by 6%', what validation steps do you take to ensure this isn't overfitting the search process?",
    "rubric": [
      "Nested CV or held-out test set untouched during search",
          "Checks if improvement is within noise (repeated CV, confidence intervals)",
      "Verifies best config isn't a lucky outlier among 500 trials",
      "Documents search space and budget for reproducibility"
        ],
    "points": 2
  }
]
```

## Referensi

- sklearn.model_selection documentation (search classes)
- Bergstra & Bengio (2012) 'Random Search for Hyper-Parameter Optimization'
- Optuna documentation
- 'Algorithms for Hyper-Parameter Optimization' — Bergstra et al. 2011
