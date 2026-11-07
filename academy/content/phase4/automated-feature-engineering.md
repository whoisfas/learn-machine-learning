---
{
  "id": "p4-automated-feature-engineering",
  "title": "Automated Feature Engineering",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 12,
  "objectives": [
    "Understand automated feature engineering approaches",
    "Use Featuretools for relational data feature synthesis",
    "Apply TSFresh for time series feature extraction",
    "Evaluate when automated FE helps vs manual engineering"
  ],
  "concepts": [
    "Feature synthesis: aggregations, transforms, interactions",
    "Featuretools primitives and DFS algorithm",
    "TSFresh: comprehensive time series features",
    "Feature selection after synthesis",
    "Computational cost of feature explosion",
    "Domain-specific vs generic primitives",
    "AutoFeat: polynomial feature expansion"
  ],
  "formulas": [],
  "references": [
    "Featuretools documentation",
    "TSFresh documentation (Christ et al. 2018)",
    "AutoFeat paper (Horn et al. 2019)",
    "Kaggle automated feature engineering discussions"
  ]
}
---

## Penjelasan

Automated Feature Engineering (AFE) menggunakan algoritma untuk menghasilkan kandidat fitur dari data mentah — mengurangi beban manual engineering. Pendekatan utama: template-based synthesis (Featuretools), statistical feature extraction (TSFresh), dan symbolic expansion (AutoFeat).

**Featuretools.** Dirancang untuk *relational data* (multiple tabel dengan relationships). Deep Feature Synthesis (DFS) menggenerate fitur dengan menerapkan *primitives* (aggregation, transform) secara rekursif di sepanjang relasi. Contoh: tabel `orders` + `customers` -> fitur "total spent by customer", "average order value", "days since last order".

**TSFresh.** Untuk time series: mengekstrak ~800 statistical features per series (mean, std, skewness, FFT coefficients, autocorrelation, entropy, dll). Berguna untuk classification/regression pada time series tanpa domain expertise. Feature selection penting karena 800 fitur per series bisa overfit.

**AutoFeat.** Symbolic approach: generate polynomial features (x_i*x_j, x_i², dll) dan apply symbolic regression untuk nonlinear transforms; select via L1 regularization.

**Kapan AFE membantu.** Relational databases dengan banyak tabel (Featuretools sangat kuat); time series classification (TSFresh baseline kuat); rapid prototyping; tim tanpa domain expertise.

**Kapan manual lebih baik.** Domain sangat spesifik (fitur fisika/bisnis yang tidak ada di primitive library); interpretability requirements; computational constraints (feature explosion); problems dengan struktur unconventional.

## Konsep Penting

1. **Feature explosion** — AFE bisa menghasilkan ribuan fitur; feature selection wajib.
2. **Leakage risk** — beberapa primitive bisa bocorkan informasi target; validasi cermat.
3. **Computational cost** — DFS pada relational data dalam bisa sangat mahal; batasi depth dan primitive set.
4. **Generic vs domain primitives** — menambah domain-specific primitives meningkatkan kualitas.
5. **AFE + manual hybrid** — AFE untuk baseline; manual untuk domain insight.

## Formula

- DFS depth: recursion depth for feature synthesis.
- TSFresh: ~800 features per time series.

## Contoh Kasus

Dataset e-commerce relational: customers, orders, order_items, products. Featuretools DFS depth=2 menghasilkan 500+ fitur: "customer total_spent", "product avg_rating", "order_items count per order", "days_between_first_last_order". Setelah feature selection (SelectKBest), 50 fitur -> XGBoost -> +12% AUC vs manual features.

## Implementasi

```python
# Featuretools example
import featuretools as ft
import pandas as pd

# define entityset
dataframes = {
    'customers': customers_df,
    'orders': orders_df,
    'order_items': order_items_df,
}
relationships = [
    ('customers', 'customer_id', 'orders', 'customer_id'),
    ('orders', 'order_id', 'order_items', 'order_id'),
]
es = ft.EntitySet(id='ecommerce')
for name, df in dataframes.items():
    es.add_dataframe(dataframe_name=name, dataframe=df,
                     index=name + '_id' if name != 'customers' else 'customer_id')
for parent, parent_col, child, child_col in relationships:
    es.add_relationship(parent, parent_col, child, child_col)

# deep feature synthesis
fm, features = ft.dfs(entityset=es, target_dataframe_name='customers',
                      max_depth=2, features_only=False)
print(f"Generated {len(features)} features")

# feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(fm.fillna(0), y)

# TSFresh example
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

settings = ComprehensiveFCParameters()
X_ts = extract_features(time_series_df, column_id='id',
                        column_sort='time', column_value='value',
                        default_fc_parameters=settings, n_jobs=-1)
X_ts = X_ts.dropna(axis=1)  # remove features with NaN
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Featuretools' Deep Feature Synthesis is designed for:",
    "options": [
      "Relational databases with multiple linked tables",
      "Single flat tables",
      "Image data",
      "Text data"
    ],
    "answer": 0,
    "explanation": "DFS traverses entity relationships to generate features across tables.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "TSFresh extracts approximately how many features per time series?",
    "options": ["~800", "~10", "~50", "~5000"],
    "answer": 0,
    "explanation": "ComprehensiveFCParameters generates ~800 statistical features.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The main risk of automated feature engineering is:",
    "options": [
      "Feature explosion leading to overfitting and computational cost",
      "Too few features",
      "Missing domain knowledge always",
      "Slow inference"
    ],
    "answer": 0,
    "explanation": "AFE generates thousands of candidates; without selection, overfitting is likely.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `feature_cross(df, col1, col2)` creating interaction feature as product of two numeric columns, handling non-numeric gracefully.",
    "starter_code": "import pandas as pd\nimport numpy as np\n\ndef feature_cross(df, col1, col2):\n    df = df.copy()\n    pass\n",
    "function": "feature_cross",
    "tests": [
      {"call": "feature_cross(pd.DataFrame({'a':[1,2],'b':[3,4]}),'a','b')['a_x_b'].tolist()", "expected": [3,8]}
    ],
    "hints": ["df[f'{col1}_x_{col2}'] = pd.to_numeric(df[col1]) * pd.to_numeric(df[col2])"],
    "explanation": "Manual feature cross as baseline for AFE.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your Featuretools pipeline generates 2000 features and XGBoost overfits. Describe your strategy to select the most useful features while preserving predictive power.",
    "rubric": [
      "Feature importance from model (XGBoost built-in)",
          "Univariate statistical tests (mutual info, ANOVA)",
      "L1 regularization for embedded selection",
      "Cross-validated selection to avoid leakage",
      "Domain knowledge to prune irrelevant features"
        ],
    "points": 2
  }
]
```

## Referensi

- Featuretools documentation (https://featuretools.alteryx.com)
- Christ et al. (2018) 'Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests' (TSFresh)
- Horn et al. (2019) 'AutoFeat: An Automatic Feature Engineering Method'
- 'Feature Engineering for Machine Learning' — Zheng & Casari
