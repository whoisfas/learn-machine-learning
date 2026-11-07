---
{
  "id": "p6-data-collection-preprocessing",
  "title": "Data Collection and Preprocessing",
  "phase": 6,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Source datasets systematically using the academy dataset manager",
    "Design preprocessing pipelines that avoid leakage",
    "Handle missing data, outliers, and class imbalance",
    "Document preprocessing for reproducibility"
  ],
  "concepts": [
    "Dataset selection criteria: relevance, quality, license, size",
    "Train/validation/test split strategies",
    "Preprocessing: normalization, encoding, imputation",
    "Data leakage prevention: preprocessing inside CV folds",
    "Class imbalance handling: SMOTE, class weights, threshold tuning",
    "Outlier detection and treatment",
    "Data documentation: provenance, preprocessing steps, transformations"
  ],
  "formulas": [],
  "references": [
    "Phase 1-2: Pandas, Feature Engineering, Model Evaluation",
    "scikit-learn preprocessing documentation",
    "'The Elements of Statistical Learning' ch. 2"
  ]
}
---

## Penjelasan

Data collection dan preprocessing adalah fondasi research — data yang buruk atau preprocessing yang bocor (leakage) menghasilkan kesimpulan yang salah meski model terlihat bagus. Modul ini memandu pipeline data yang rigorous.

**Dataset selection.** Kriteria: (1) *Relevance*: align dengan research question. (2) *Quality*: dokumentasi baik, tidak ada known issues. (3) *License*: compatible dengan research use. (4) *Size*: cukup untuk task; tidak terlalu besar untuk compute constraints. (5) *Representativeness*: reflect real-world distribution.

**Menggunakan academy dataset manager.** `academy datasets list` untuk melihat available datasets; `academy datasets info <name>` untuk metadata; `academy datasets download <name>` untuk fetch large datasets; `academy datasets generate <name>` untuk synthetic.

**Preprocessing pipeline.** Urutan penting: (1) split data FIRST (train/val/test) — sebelum preprocessing apapun. (2) Fit preprocessing pada train saja (scaler, encoder, imputer). (3) Apply ke val/test. Ini mencegah leakage.

**Missing data.** Strategies: deletion (jika <5% missing), mean/median imputation, model-based imputation (KNN, MICE), atau indicator variables. Untuk time series: forward-fill atau interpolation.

**Class imbalance.** Jika class ratio > 10:1: SMOTE oversampling, class weights dalam loss, threshold tuning, atau focal loss. Jangan hapus majority class — information loss.

**Outliers.** Detection: IQR method, Z-score, isolation forest. Treatment: investigate (data error? genuine extreme?), winsorize, atau remove jika confirmed error.

## Konsep Penting

1. **Split before preprocess** — leakage adalah silent killer; selalu split dulu.
2. **Fit on train only** — scaler/encoder/imputer hanya learn dari train data.
3. **Document everything** — preprocessing steps harus reproducible.
4. **Imbalance != noise** — class imbalance adalah challenge, bukan data quality issue.
5. **Missing data mechanism** — MCAR/MAR/MNAR menentukan strategi imputation.

## Implementasi

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocessing_pipeline(X_train, X_val, X_test, numerical_cols, categorical_cols):
    """Fit preprocessing on train, apply to all splits."""
    # numerical: impute + scale
    num_imputer = SimpleImputer(strategy='median')
    num_scaler = StandardScaler()
    X_train_num = num_scaler.fit_transform(num_imputer.fit_transform(X_train[numerical_cols]))
    X_val_num = num_scaler.transform(num_imputer.transform(X_val[numerical_cols]))
    X_test_num = num_scaler.transform(num_imputer.transform(X_test[numerical_cols]))

    # categorical: impute + one-hot
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_cat = cat_encoder.fit_transform(cat_imputer.fit_transform(X_train[categorical_cols]))
    X_val_cat = cat_encoder.transform(cat_imputer.transform(X_val[categorical_cols]))
    X_test_cat = cat_encoder.transform(cat_imputer.transform(X_test[categorical_cols]))

    # combine
    from scipy.sparse import hstack
    X_train_proc = hstack([X_train_num, X_train_cat]).tocsr()
    X_val_proc = hstack([X_val_num, X_val_cat]).tocsr()
    X_test_proc = hstack([X_test_num, X_test_cat]).tocsr()
    return X_train_proc, X_val_proc, X_test_proc

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Train/val/test split."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size),
        random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "The correct order for data preprocessing is:",
    "options": [
      "Split data first, then fit preprocessing on train, then apply to all",
      "Preprocess all data, then split",
      "Split, preprocess train and test together",
      "Preprocess test first"
    ],
    "answer": 0,
    "explanation": "Preprocessing on full data leaks test information into training.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Data leakage occurs when:",
    "options": [
      "Information from test/validation set influences training",
      "Model is too complex",
      "Dataset is too small",
      "Features are correlated"
    ],
    "answer": 0,
    "explanation": "Leakage = using information that wouldn't be available at prediction time.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "For class imbalance, the best first approach is:",
    "options": [
      "Class weights in loss function or threshold tuning",
      "Delete majority class",
      "Duplicate minority class randomly",
      "Ignore the problem"
    ],
    "answer": 0,
    "explanation": "Class weights/threshold preserve all data while addressing imbalance.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `safe_impute_fit_transform(train, val, test, strategy='median')` fitting imputer on train only.",
    "starter_code": "import numpy as np\nfrom sklearn.impute import SimpleImputer\n\ndef safe_impute_fit_transform(train, val, test, strategy='median'):\n    imputer = SimpleImputer(strategy=strategy)\n    train_imp = imputer.fit_transform(train)\n    val_imp = imputer.transform(val)\n    test_imp = imputer.transform(test)\n    return train_imp, val_imp, test_imp\n",
    "function": "safe_impute_fit_transform",
    "tests": [
      {"call": "safe_impute_fit_transform([[1,np.nan],[3,4]],[[np.nan,2]],[[5,6]])[0]", "expected": "check"}
    ],
    "hints": ["fit_transform on train; transform on val/test"],
    "explanation": "Leakage-free imputation.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You discover your preprocessing pipeline computed feature means on the entire dataset before splitting. Explain the impact and how you'd fix it.",
    "rubric": [
      "This is data leakage — test distribution influenced training",
          "Fix: split first, then fit preprocessing on train only",
      "Re-run all experiments with corrected pipeline",
      "Report the error transparently if results change significantly"
        ],
    "points": 2
  }
]
```

## Referensi

- scikit-learn preprocessing documentation
- 'The Elements of Statistical Learning' ch. 2
- Phase 2: Feature Engineering, Model Evaluation modules
