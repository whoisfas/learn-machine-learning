---
{
  "id": "p2-logistic-regression",
  "title": "Logistic Regression",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": [
    "Derive logistic regression as a probabilistic classifier",
    "Interpret coefficients as log-odds and odds ratios",
    "Understand the sigmoid function and decision threshold",
    "Handle multiclass via softmax and class imbalance"
  ],
  "concepts": [
    "Binary classification: P(y=1|x) = σ(w·x + b)",
    "Sigmoid/logistic function properties",
    "Log-odds interpretation: logit(p) = w·x + b",
    "Odds ratio: exp(βj) = multiplicative change in odds per unit xj",
    "Decision threshold: default 0.5, tunable for precision/recall tradeoff",
    "Multinomial logistic regression (softmax)",
    "Class weights and imbalance",
    "L1/L2 regularization in logistic regression",
    "Calibration: predicted probabilities vs observed frequencies"
  ],
  "formulas": [
    {"name": "Sigmoid", "latex": "\\sigma(z) = \\frac{1}{1+e^{-z}}", "note": "Maps R -> (0,1); σ(0)=0.5."},
    {"name": "Log-odds", "latex": "\\log\\frac{p}{1-p} = w^\\top x + b", "note": "Linear in log-odds space."},
    {"name": "Odds ratio", "latex": "OR_j = e^{\\beta_j}", "note": "Per-unit change in odds."},
    {"name": "Cross-entropy loss", "latex": "L = -\\frac{1}{n}\\sum_i [y_i \\log p_i + (1-y_i)\\log(1-p_i)]", "note": "Negative log-likelihood of Bernoulli."}
  ],
  "references": [
    "sklearn linear_model.LogisticRegression docs",
    "ESL ch. 4.4",
    "'Applied Logistic Regression' by Hosmer"
  ]
}
---

## Penjelasan

Logistic regression adalah linear regression untuk klasifikasi. Alih-alih memprediksi nilai kontinu, ia memprediksi *probabilitas* kelas lewat fungsi sigmoid yang memetakan skor linear ke (0,1).

**Model.** Untuk binary: z = w·x + b (linear score); p = σ(z) = 1/(1+e^(−z)). Prediksi kelas: ŷ = 1 jika p > threshold (default 0.5), else 0. z disebut *logit* atau *log-odds*: log(p/(1−p)) = z — inilah mengapa namanya logistic regression: hubungan linear ada di ruang log-odds, bukan probabilitas.

**Interpretasi koefisien.** βj adalah perubahan log-odds per unit kenaikan xj (fitur lain tetap). exp(βj) = odds ratio — faktor pengali odds. Contoh: β_age = 0.05 -> OR = e^0.05 ≈ 1.05, odds naik 5% per tahun usia.

**Training.** Maksimalkan log-likelihood = minimalkan cross-entropy. Convex problem -> convex optimization -> global optimum terjamin (tidak seperti neural network). sklearn menggunakan solver L-BFGS/ saga untuk L2, coordinate descent untuk L1.

**Multiclass.** Softmax regression: K kelas -> K vektor bobot; P(y=k|x) = exp(wk·x)/Σj exp(wj·x). sklearn otomatis pakai multinomial untuk >2 kelas.

**Threshold tuning.** Default 0.5 bukan selalu optimal. Untuk imbalanced data atau asymmetric cost (false negative mahal), turunkan threshold; untuk high precision, naikkan. ROC/PR curve membantu memilih operating point.

**Regularization.** L1 -> sparse coefficients (feature selection); L2 -> shrinkage. `class_weight='balanced'` untuk imbalance. `C` = 1/λ (kebalikan regulasi: C kecil = lebih banyak regulasi).

**Kalibrasi.** Probabilitas yang di-output belum tentu terkalibrasi; `CalibratedClassifierCV` atau isotonic/sigmoid calibration memperbaiki ini — penting jika probabilitas dipakai untuk keputusan (mis. risk score).

## Konsep Penting

1. **Linear decision boundary** — logistic regression hanya bisa memisahkan data linearly; untuk non-linear perlu feature engineering atau kernel.
2. **Perfect separation** — jika kelas terpisah sempurna, MLE tidak ada (koefisien -> ∞); regularisasi menyelesaikan ini.
3. **Log-odds bukan probabilitas** — koefisien diinterpretasi di log-odds, bukan langsung di probability.
4. **Threshold adalah hyperparameter** — treat seperti model selection, tune di validation.
5. **Multicollinearity** — seperti OLS, koefisien tidak stabil tapi prediksi tetap valid.

## Formula

- σ(z) = 1/(1+e^(−z)); logit(p) = log(p/(1−p)).
- Softmax: P(y=k|x) = e^(wk·x) / Σj e^(wj·x).
- Log-likelihood: Σ [yi log pi + (1−yi) log(1−pi)].

## Contoh Kasus

Prediksi default kredit: fitur (income, debt_ratio, credit_history). Logistic regression memberi probability default + interpretasi koefisien (OR per fitur). Threshold 0.3 dipilih untuk recall tinggi (tangkap lebih banyak defaulter dengan biaya false positive yang dapat diterima).

## Implementasi

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(C=1.0, max_iter=1000,
                                        class_weight='balanced'))
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, proba))

# coefficient interpretation (on standardized features)
coefs = pipe.named_steps['logisticregression'].coef_[0]
for name, c in sorted(zip(feature_names, coefs), key=lambda t: -abs(t[1])):
    print(f"{name}: log-odds={c:.3f}, OR={np.exp(c):.3f}")
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "For logistic regression, the decision boundary in feature space is:",
    "options": [
      "Linear (a hyperplane)",
      "Always nonlinear",
      "Circular",
      "Depends on activation function"
    ],
    "answer": 0,
    "explanation": "Logistic regression is linear in log-odds -> linear boundary in x-space.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "math",
    "question": "If β1 = 0.7 for feature x1, what is the odds ratio for a 1-unit increase in x1?",
    "answer": 2.0137527074704766,
    "tolerance": 0.01,
    "hints": ["OR = exp(0.7)"],
    "explanation": "e^0.7 ≈ 2.014 — odds roughly double per unit increase.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "σ(0) equals:",
    "options": ["0.5", "0", "1", "Depends on weights"],
    "answer": 0,
    "explanation": "Sigmoid at 0 = 1/(1+e^0) = 1/2.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `sigmoid(z)` returning 1/(1+exp(-z)) element-wise for arrays, numerically stable for large |z|.",
    "starter_code": "import numpy as np\n\ndef sigmoid(z):\n    z = np.asarray(z, dtype=float)\n    pass\n",
    "function": "sigmoid",
    "tests": [
      {"call": "sigmoid(0.0).tolist()", "expected": [0.5]},
      {"call": "sigmoid([0.0, 1.0])[0]", "expected": 0.5},
      {"call": "sigmoid([-1000.0, 1000.0]).tolist()", "expected": [0.0, 1.0]}
    ],
    "hints": ["Use np.where(z>=0, 1/(1+exp(-z)), exp(z)/(1+exp(z))) for stability", "or clip z first"],
    "explanation": "Numerically stable sigmoid — naive exp(-z) overflows for very negative z.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your logistic regression has AUC 0.92 but at threshold 0.5 recall is only 0.4. The business cost of a false negative is 10x a false positive. Explain how to choose the operating threshold and what metric/curve guides this.",
    "rubric": [
      "Lowers threshold to increase recall (accept more false positives)",
      "Uses PR curve / ROC to find operating point",
      "Considers cost-sensitive threshold or class_weight",
      "Validates threshold on held-out set, not test"
    ],
    "points": 2
  }
]
```

## Referensi

- sklearn LogisticRegression documentation
- ESL ch. 4.4
- "Applied Logistic Regression" Hosmer & Lemeshow
- scikit-learn calibration docs
