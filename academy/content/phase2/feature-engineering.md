---
{
  "id": "p2-feature-engineering",
  "title": "Feature Engineering",
  "phase": 2,
  "difficulty": "intermediate",
  "estimated_hours": 12,
  "objectives": [
    "Create informative features from raw data",
    "Handle categorical, text, datetime, and missing data",
    "Understand encoding schemes and their tradeoffs",
    "Use domain knowledge and automated feature tools"
  ],
  "concepts": [
    "Numeric transforms: log, sqrt, polynomial, binning",
    "Categorical encoding: one-hot, ordinal, target encoding, embeddings",
    "Text features: TF-IDF, n-grams, hashing",
    "Datetime features: cyclical encoding, time since event",
    "Missing value indicators and imputation strategies",
    "Interaction terms and feature crosses",
    "Feature selection: filter, wrapper, embedded methods",
    "Automated feature engineering (Featuretools, TSFresh)"
  ],
  "formulas": [
    {"name": "TF-IDF", "latex": "tfidf(t,d) = tf(t,d) \\cdot \\log\\frac{N}{df(t)}", "note": "Term frequency × inverse document frequency."},
    {"name": "Target encoding", "latex": "E[y|x=c] \\text{ smoothed}", "note": "Replace category with mean target; needs CV to avoid leakage."}
  ],
  "references": [
    "sklearn.preprocessing docs",
    "'Feature Engineering for Machine Learning' by Zheng & Casari",
    "Featuretools documentation"
  ]
}
---

## Penjelasan

Feature engineering adalah seni mengubah data mentah menjadi representasi yang membuat pola lebih mudah dipelajari model. Di kompetisi ML dan praktik nyata, feature engineering sering menentukan 80% keberhasilan — model canggih dengan fitur buruk kalah dari model sederhana dengan fitur bagus.

**Transformasi numerik.** Log transform untuk data skewed (pendapatan, populasi); sqrt untuk count data; polynomial features (x², x³) untuk non-linearitas; binning untuk ordinal categories; clipping/winsorizing untuk outlier.

**Encoding kategorikal.**
- *One-hot*: binary indicator per kategori; aman tapi high-dimensional untuk cardinality tinggi.
- *Ordinal*: integer per kategori; hanya jika ada ordering natural.
- *Target encoding*: ganti kategori dengan mean target; powerful tapi leakage risk — harus pakai CV/out-of-fold encoding.
- *Frequency encoding*: ganti kategori dengan frekuensi kemunculan.
- *Embeddings*: learned dense vectors (untuk high-cardinality seperti user_id).

**Datetime.** Ekstrak komponen: hour, dayofweek, month, year; cyclical encoding sin/cos untuk periodic patterns (jam, hari); time since last event; rolling aggregates.

**Missing values.** Strategi: mean/median/mode imputation, model-based (KNN imputer, iterative), indicator features untuk missingness yang informatif. Jangan drop rows kecuali missingness MCAR (missing completely at random).

**Text.** TF-IDF vectorization adalah baseline kuat; n-grams menangkap frasa; hashing vectorizer untuk vocabulary besar; embeddings (Word2Vec, pretrained) untuk semantik.

**Interaksi.** Fitur baru dari kombinasi: income/debt_ratio, price_per_sqft, age×income. Trees bisa menemukan interaksi sendiri tapi linear models butuh eksplisit.

## Konsep Penting

1. **Data leakage** — fitur yang mengandung informasi target masa depan atau target itu sendiri merusak validasi.
2. **Target encoding butuh CV** — hitung mean target hanya dari fold training, bukan seluruh data.
3. **High cardinality** — one-hot untuk 10k kategori = 10k kolom; gunakan target/frequency encoding atau hashing.
4. **Feature selection bukan gratis** — mengurangi dimensi bisa menghilangkan sinyal; validasi pada held-out set.
5. **Domain knowledge > automation** — memahami proses bisnis menghasilkan fitur yang lebih bermakna dari tools otomatis.

## Formula

- TF-IDF: tf·idf = (count/total) · log(N/df).
- Cyclical encoding: sin(2π·x/period), cos(2π·x/period).
- Target encoding (out-of-fold): mean(y_train_fold | category) + smoothing.

## Contoh Kasus

Dataset e-commerce: features mentah (user_id, timestamp, product_id, price, category). Engineered: hour_of_day (cyclical), days_since_signup, price_per_unit_weight, user_purchase_frequency, category_target_encoded (CV), product_price_zscore. Hasil: +15% AUC vs baseline.

## Implementasi

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# datetime features
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['days_since'] = (df['now'] - df['signup_date']).dt.days

# interaction
df['price_per_kg'] = df['price'] / df['weight_kg']

# out-of-fold target encoding
from sklearn.model_selection import KFold
enc = pd.Series(index=df.index, dtype=float)
for tr, va in KFold(5).split(df):
    means = df.iloc[tr].groupby('category')['target'].mean()
    global_mean = df.iloc[tr]['target'].mean()
    enc.iloc[va] = df.iloc[va]['category'].map(
        lambda c: (means.get(c, 0)*len(means) + global_mean*10) / (len(means)+10))
df['cat_te'] = enc
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Why is target encoding dangerous if computed on the full dataset before splitting?",
    "options": [
      "It leaks target information into features, inflating validation scores",
      "It's too slow",
      "It creates too many columns",
      "It only works for regression"
    ],
    "answer": 0,
    "explanation": "Target encoding uses y; computing on full data leaks y into train/val -> overoptimistic CV.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "For a 'hour of day' feature (0-23) that is cyclical, the best encoding is:",
    "options": [
      "sin(2π·hour/24) and cos(2π·hour/24)",
      "One-hot for each hour",
      "Raw integer 0-23",
      "Log transform"
    ],
    "answer": 0,
    "explanation": "Cyclical encoding preserves circular distance (23 is close to 0, not far).",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `cyclical_encode(values, period)` returning two arrays (sin, cos) for cyclical features.",
    "starter_code": "import numpy as np\n\ndef cyclical_encode(values, period):\n    values = np.asarray(values, dtype=float)\n    pass\n",
    "function": "cyclical_encode",
    "tests": [
      {"call": "s,c=cyclical_encode([0,6,12,18],24); s.tolist()", "expected": [0.0,1.0,0.0,-1.0]},
      {"call": "s,c=cyclical_encode([0],7); c.tolist()", "expected": [1.0]}
    ],
    "hints": ["sin(2*pi*v/period), cos(2*pi*v/period)", "return (sin_array, cos_array)"],
    "explanation": "Cyclical encoding for periodic features.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `add_missing_indicators(df, cols)` returning a copy of df with new binary columns `{col}_missing` = 1 where original is NaN.",
    "starter_code": "import pandas as pd\nimport numpy as np\n\ndef add_missing_indicators(df, cols):\n    df = df.copy()\n    pass\n",
    "function": "add_missing_indicators",
    "tests": [
      {"call": "add_missing_indicators(pd.DataFrame({'a':[1.0,np.nan]}),['a'])['a_missing'].tolist()", "expected": [0,1]}
    ],
    "hints": ["df[col].isna().astype(int)", "name new column f'{col}_missing'"],
    "explanation": "Missingness indicators capture informative missingness.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You have a 'user_id' feature with 50,000 unique values. One-hot is infeasible. Describe three encoding strategies with their tradeoffs, and which you'd choose for a gradient boosting model vs a linear model.",
    "rubric": [
      "Target encoding (with CV), frequency encoding, embeddings/hash",
          "Notes leakage risk of target encoding",
      "GBM can handle ordinal-encoded high-cardinality natively in some libs",
      "Linear model needs dense numeric representation"
        ],
    "points": 2
  }
]
```

## Referensi

- 'Feature Engineering for Machine Learning' — Zheng & Casari
- sklearn.preprocessing documentation
- Featuretools (automated feature engineering)
- Kaggle feature engineering discussions
