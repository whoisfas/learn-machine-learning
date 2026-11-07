---
{
  "id": "p2-linear-regression",
  "title": "Linear Regression",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 10,
  "objectives": [
    "Derive and interpret the linear regression model",
    "Fit OLS, Ridge, Lasso, and ElasticNet",
    "Diagnose assumptions: linearity, homoscedasticity, independence",
    "Understand regularization and its effect on coefficients"
  ],
  "concepts": [
    "Model: y = Xβ + ε; β via least squares",
    "Normal equations and closed-form solution",
    "Residual analysis: fitted vs residuals, QQ plot",
    "R², adjusted R², RMSE, MAE",
    "Ridge (L2), Lasso (L1), ElasticNet",
    "Multicollinearity and its effect on coefficients",
    "Polynomial features for nonlinearity",
    "Gauss-Markov assumptions"
  ],
  "formulas": [
    {"name": "OLS estimate", "latex": "\\hat{\\beta} = (X^\\top X)^{-1} X^\\top y", "note": "Closed-form least squares."},
    {"name": "Ridge", "latex": "\\hat{\\beta} = (X^\\top X + \\lambda I)^{-1} X^\\top y", "note": "L2 penalty shrinks coefficients."},
    {"name": "R²", "latex": "R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}", "note": "Fraction of variance explained; can be negative on test."},
    {"name": "Lasso", "latex": "\\min_\\beta \\|y-X\\beta\\|^2 + \\lambda \\|\\beta\\|_1", "note": "L1 induces sparsity (feature selection)."}
  ],
  "references": [
    "sklearn linear models docs",
    "ESL ch. 3",
    "Statsmodels for detailed diagnostics"
  ]
}
---

## Penjelasan

Linear regression memodelkan hubungan linear y = β0 + β1x1 + ... + βdxd + ε. Meski sederhana, ia adalah baseline penting, mudah diinterpretasi, dan fondasi untuk memahami model lebih kompleks.

**Ordinary Least Squares (OLS).** Meminimalkan jumlah kuadrat residual: min_β ||y − Xβ||². Solusi closed-form β̂ = (XᵀX)⁻¹Xᵀy. Setiap koefisien βj = perubahan y per unit xj *dengan fitur lain tetap* (ceteris paribus).

**Asumsi Gauss-Markov.** (1) Linearity: E[y|X] linear. (2) Independence residuals. (3) Homoscedasticity: Var(ε) konstan. (4) Eksogenitas: tidak ada korelasi X dan ε. (5) Tidak ada multikolinearitas sempurna. Pelanggaran tidak selalu membuat β bias, tapi mempengaruhi efisiensi dan validitas inference.

**Diagnostik.** Plot residuals vs fitted: pola = non-linearitas atau heteroscedasticity. QQ plot residuals vs normal: ekor berat = outlier/non-normal. VIF (Variance Inflation Factor) > 10 = multikolinearitas bermasalah. Cook's distance untuk influential points.

**Regularization.**
- *Ridge (L2)*: tambah λ||β||² — shrink semua koefisien ke 0; stabil untuk multikolinearitas; tidak menghasilkan sparse.
- *Lasso (L1)*: tambah λ||β||₁ — mendorong koefisien tepat 0; feature selection otomatis.
- *ElasticNet*: kombinasi L1+L2 — baik untuk p >> n atau fitur korelatif.
- *λ tuning*: cross-validation; λ besar = lebih sederhana.

**Evaluasi.** R² = 1 − SSres/SStot (proporsi variance dijelaskan); hati-hati: R² test bisa negatif. RMSE dalam satuan y — intuitif tapi sensitif outlier. MAE robust. Adjusted R² penalti jumlah fitur. Bandingkan dengan baseline (mean predictor).

## Konsep Penting

1. **Scale fitur** — regularization butuh fitur terskalakan (StandardScaler) agar penalti adil.
2. **Interaksi & polinomial** — hubungan non-linear bisa dimodelkan dengan x², x1·x2 terms.
3. **Extrapolasi berbahaya** — linear model di luar range training bisa prediksi absurd.
4. **Residual bukan noise murni** — bisa mengandung sinyal yang belum dimodelkan.
5. **Lasso path** — koefisien sebagai fungsi λ menunjukkan feature importance.

## Formula

- Normal equations: XᵀXβ = Xᵀy.
- Ridge: (XᵀX + λI)β = Xᵀy.
- VIFj = 1/(1 − R²j) dari regresi xj pada fitur lain.

## Contoh Kasus

Prediksi harga rumah (California Housing): 8 fitur numerik + kategorikal (ocean_proximity). Pipeline: OneHotEncoder + StandardScaler -> Ridge. Bandingkan R² vs Lasso — Lasso mungkin drop fitur lemah. Residual plot menunjukkan heteroscedasticity (error lebih besar di harga tinggi) -> pertimbangkan log-transform target.

## Implementasi

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# manual OLS (for understanding)
def ols(X, y):
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta

# sklearn pipeline
pipe = make_pipeline(StandardScaler(),
                     Ridge(alpha=1.0))
scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("CV RMSE:", -scores.mean())
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "In y = 2 + 3x1 - 0.5x2, what does coefficient 3 mean?",
    "options": [
      "y increases by 3 per unit x1 holding x2 fixed",
      "x1 causes y to increase by 3",
      "Correlation between x1 and y is 3",
      "The model explains 3% of variance"
    ],
    "answer": 0,
    "explanation": "Coefficient = partial effect ceteris paribus; correlation/causation conflated in other options.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Why scale features before Ridge/Lasso?",
    "options": [
      "Penalty λ||β|| treats all coefficients equally; unscaled features get unfair penalty",
      "Scaling improves R²",
      "Required by sklearn",
      "Makes coefficients interpretable"
    ],
    "answer": 0,
    "explanation": "L2/L1 penalize coefficient magnitude; features on different scales get penalized differently.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `ridge_regression(X, y, lam)` returning coefficients β for (XᵀX + λI)β = Xᵀy with intercept handled by centering.",
    "starter_code": "import numpy as np\n\ndef ridge_regression(X, y, lam):\n    X = np.asarray(X, dtype=float)\n    y = np.asarray(y, dtype=float)\n    pass\n",
    "function": "ridge_regression",
    "tests": [
      {"call": "ridge_regression([[1],[2],[3]],[2,4,6], 0.001).tolist()", "expected": [0.0, 2.0]},
      {"call": "ridge_regression([[1,1],[1,2],[1,3]],[1,2,3], 1.0)[0]", "expected": "check"}
    ],
    "hints": ["Center X and y; solve for slope; intercept = y_mean - X_mean @ slopes", "Return np.array([intercept, *slopes])"],
    "explanation": "Closed-form ridge via normal equations.",
    "points": 3
  },
  {
    "id": "ex4",
    "type": "math",
    "question": "For a simple linear regression, R² = 0.85 on training data. What does this mean?",
    "options": [
      "The model explains 85% of variance in training y",
      "85% of predictions are correct",
      "The model will get 85% on test",
      "Correlation is 0.85"
    ],
    "answer": 0,
    "explanation": "R² = proportion of variance explained; on training data it can be optimistic.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your residual plot shows a clear U-shape (residuals positive at low/high fitted values, negative in middle). Diagnose the problem and propose two concrete fixes.",
    "rubric": [
      "Identifies non-linearity / omitted nonlinear terms",
      "Proposes polynomial features or spline basis",
      "Or proposes different model class (trees, kernel)",
      "Mentions checking for omitted variable bias"
    ],
    "points": 2
  }
]
```

## Referensi

- sklearn.linear_model documentation
- ESL ch. 3 (Linear Methods)
- statsmodels for diagnostics (summary, influence plots)
