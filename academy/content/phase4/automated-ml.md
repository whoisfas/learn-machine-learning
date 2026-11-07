---
{
  "id": "p4-automated-ml",
  "title": "Automated Machine Learning (AutoML)",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 12,
  "objectives": [
    "Understand what AutoML automates: preprocessing, model selection, HPO",
    "Build a simple AutoML pipeline",
    "Compare AutoML tools: auto-sklearn, AutoGluon, H2O",
    "Know when AutoML helps vs when manual tuning is better"
  ],
  "concepts": [
    "AutoML pipeline: preprocessing + model zoo + HPO + ensemble",
    "meta-learning for warm-starting",
    "Bayesian optimization for HPO",
    "Successive halving for efficiency",
    "Stacking and ensemble selection",
    "AutoGluon, auto-sklearn, H2O AutoML comparison",
    "When AutoML fails: domain-specific features, tiny data"
  ],
  "formulas": [],
  "references": [
    "auto-sklearn documentation",
    "AutoGluon documentation",
    "H2O AutoML documentation",
    "Feurer et al. 2015 'Efficient and Robust Automated Machine Learning'"
  ]
}
---

## Penjelasan

AutoML mengotomatisasi pipeline ML: memilih preprocessing, model, hyperparameter, dan ensemble — mengubah proses yang biasanya butuh expert knowledge menjadi satu perintah. Penting dipahami: AutoML bukan sulap; ia mengotomatiskan *pencarian* dalam ruang konfigurasi yang sudah ditentukan.

**Komponen AutoML.**
- *Preprocessing search*: memilih imputation strategy, encoding, scaling, feature selection.
- *Model zoo*: himpunan model kandidat (linear, trees, SVM, NN, GBM).
- *HPO*: mencari hyperparameter optimal per model (Bayesian opt, random search, successive halving).
- *Ensemble*: menggabungkan top models (stacking, weighted average, Caruana ensemble selection).
- *Meta-learning*: menggunakan metadata dari dataset serupa untuk warm-start (mis. auto-sklearn).

**Tools utama.**
- *auto-sklearn*: sklearn-compatible; SMAC untuk HPO; meta-learning; mature untuk tabular.
- *AutoGluon*: multi-layer stacking ensemble; sangat kuat untuk tabular + time series; `TabularPredictor.fit()` satu baris.
- *H2O AutoML*: Java-based, scalable; bagus untuk production pipelines.
- *TPOT*: genetic programming untuk pipeline discovery — paling fleksibel tapi paling lambat.

**Kapan AutoML membantu.** Tabular data dengan banyak model candidates; tim tanpa ML expertise; baseline yang kuat dengan effort minimal; benchmark untuk membandingkan manual approach.

**Kapan AutoML gagal.** Domain sangat spesifik dengan fitur engineering kunci; data sangat kecil (AutoML butuh cukup data untuk validasi); interpretability requirements; real-time constraints; problems di luar tabular (vision, NLP custom).

## Konsep Penting

1. **AutoML = strong baseline, bukan final answer** — hasilnya sering sangat baik tapi manual tuning bisa melampaui dengan domain knowledge.
2. **Ensemble > single model** — AutoML tools hampir selalu return ensemble; ini bagian dari kekuatan mereka.
3. **Time budget matters** — AutoML butuh waktu; untuk quick baseline gunakan auto-sklearn 60s atau AutoGluon best_quality.
4. **Leakage still possible** — AutoML tidak menyembunyikan leakage; preprocessing tetap harus benar.
5. **Interpretability trade-off** — ensemble AutoML complex; untuk regulated domains, pilih single interpretable model.

## Formula

- Ensemble selection: greedily add models that maximize validation score.
- Successive halving: keep top 1/η configs each round.

## Contoh Kasus

Kaggle-style tabular competition: AutoGluon `predictor.fit(train_data, label='target', time_limit=3600)` menghasilkan ensemble dengan CV score yang sering mengalahkan manual pipeline dalam waktu singkat. Bandingkan dengan manual: feature engineering + XGBoost tuning — AutoML competitive tapi manual bisa lebih baik dengan domain insight.

## Implementasi

```python
# AutoGluon example (simplest AutoML)
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='target').fit(
    train_data, time_limit=600,  # 10 minutes
    presets='best_quality')

predictions = predictor.predict(test_data)
print(predictor.leaderboard())

# auto-sklearn example
import autosklearn.classification
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=600, per_run_time_limit=60)
automl.fit(X_train, y_train)
print("CV score:", automl.score(X_test, y_test))

# manual AutoML-style pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())]), numeric_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])
pipe = Pipeline([('pre', preprocessor),
                 ('clf', HistGradientBoostingClassifier())])
scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "AutoML automates which parts of the ML pipeline?",
    "options": [
      "Preprocessing, model selection, hyperparameter tuning, and ensembling",
      "Only model training",
      "Only data collection",
      "Feature engineering from scratch"
    ],
    "answer": 0,
    "explanation": "AutoML covers the full pipeline: preprocessing choices, model zoo, HPO, ensemble.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "When does AutoML typically struggle?",
    "options": [
      "Domain-specific feature engineering is critical, or data is very small",
      "Tabular data with many rows",
      "Standard classification tasks",
      "When you have lots of time"
    ],
    "answer": 0,
    "explanation": "AutoML can't invent domain knowledge; tiny data limits validation reliability.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The main advantage of AutoML ensembles is:",
    "options": [
      "Combining diverse models reduces variance and improves robustness",
      "They're faster to train",
      "They use less memory",
      "They're more interpretable"
    ],
    "answer": 0,
    "explanation": "Stacking/weighted ensembles leverage model diversity for better generalization.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `greedy_ensemble_select(model_preds, y_true, metric_fn, max_models=5)` performing Caruana-style ensemble selection: repeatedly add the model (with replacement) that maximizes ensemble score.",
    "starter_code": "import numpy as np\n\ndef greedy_ensemble_select(model_preds, y_true, metric_fn, max_models=5):\n    # model_preds: list of prediction arrays\n    # metric_fn: callable(ensemble_preds, y_true) -> score (higher better)\n    pass\n",
    "function": "greedy_ensemble_select",
    "tests": [
      {"call": "greedy_ensemble_select([np.array([1,0,1]),np.array([0,0,1])], np.array([1,0,1]), lambda p,y: float(np.mean(p==y)), 3)", "expected": "check"}
    ],
    "hints": ["Start empty; each step try adding each model (with replacement), keep best; average predictions of selected"],
    "explanation": "Caruana ensemble selection — greedy forward selection with replacement.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your AutoML pipeline achieves 0.89 AUC on validation but manual XGBoost tuning reaches 0.92. Explain possible reasons and when you'd invest in manual tuning vs accepting AutoML results.",
    "rubric": [
      "AutoML search space may not include optimal features/architecture",
          "Domain-specific feature engineering can beat generic preprocessing",
      "Manual tuning finds task-specific insights AutoML misses",
      "Accept AutoML for baseline/quick deployment; manual for maximum performance"
        ],
    "points": 2
  }
]
```

## Referensi

- auto-sklearn documentation
- AutoGluon documentation (https://auto.gluon.ai)
- H2O AutoML documentation
- Feurer et al. (2015) 'Efficient and Robust Automated Machine Learning'
- Thornton et al. (2013) 'Auto-WEKA'
