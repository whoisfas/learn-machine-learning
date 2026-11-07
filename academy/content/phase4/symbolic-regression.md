---
{
  "id": "p4-symbolic-regression",
  "title": "Symbolic Regression",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Formulate symbolic regression as program search",
    "Compare GP-based and other SR approaches",
    "Evaluate SR results: accuracy vs complexity",
    "Apply SR to scientific law discovery"
  ],
  "concepts": [
    "SR task: find f such that y ≈ f(x)",
    "Expression representation and search space",
    "GP for SR vs sparse regression (LASSO/SINDy)",
    "Complexity measures: tree size, depth, operations",
    "Parsimony vs accuracy tradeoff",
    "Feynman equations benchmark",
    "SR for physics-informed ML",
    "Limitations: extrapolation, identifiability"
  ],
  "formulas": [
    {"name": "SR objective", "latex": "\\min_f \\|y - f(x)\\|^2 + \\lambda \\cdot \\text{complexity}(f)", "note": "Fit + parsimony penalty."},
    {"name": "SINDy", "latex": "\\dot{x} = \\Theta(x)\\xi", "note": "Sparse regression on library of candidate functions."}
  ],
  "references": [
    "Udrescu & Tegmark 2020 'AI Feynman'",
    "Schmidt & Lipson 2009 'Distilling Free-Form Natural Laws'",
    "gplearn documentation",
    "PySR documentation"
  ]
}
---

## Penjelasan

Symbolic Regression (SR) adalah tugas menemukan persamaan simbolik f(x) yang menjelaskan data — bukan hanya memprediksi, melainkan *menemukan hukum*. Berbeda dengan regression biasa yang fit parameter pada model form yang sudah ditentukan, SR mencari *bentuk fungsi itu sendiri*.

**Formulasi.** Diberikan data {(x1,y1),...,(xn,yn)}, temukan f ∈ F (space semua fungsi) yang meminimalkan error + complexity penalty. F bisa berupa expression trees (GP), sparse linear combinations (SINDy), atau grammar-constrained search.

**Pendekatan utama.**
- *Genetic Programming*: evolusi expression trees; fleksibel tapi lambat dan rawan bloat.
- *Sparse regression (SINDy)*: library Θ(x) berisi kandidat fungsi (polynomial, sin, cos, dll); cari sparse coefficient vector ξ via LASSO — hanya sedikit fungsi aktif. Cepat dan robust tapi terbatas pada library.
- *AI Feynman*: kombinasi neural network + symbolic regression + dimensional analysis; state-of-art untuk physics law discovery.
- *PySR*: GP + multi-population + complexity frontier; menghasilkan frontier akurasi vs kompleksitas.

**Evaluasi.** RMSE pada held-out test untuk akurasi; tree size / operation count untuk kompleksitas. Pareto frontier menunjukkan trade-off: persamaan sederhana dengan error sedikit lebih tinggi vs kompleks dengan error minimal.

**Keunggulan vs neural network.** Hasil SR adalah persamaan eksplisit — bisa dibaca manusia, diekstrapolasi dengan benar jika struktur tepat, dan memberikan insight ilmiah. NN bisa akurat tapi black-box.

**Keterbatasan.** SR sulit untuk fungsi sangat kompleks atau high-dimensional; search space besar; identifiability (banyak fungsi bisa fit data dengan baik); extrapolation bisa gagal jika bentuk fungsi salah meski fit training baik.

## Konsep Penting

1. **SR = program synthesis untuk matematika** — GP adalah approach paling general tapi SINDy lebih efisien untuk library terbatas.
2. **Complexity penalty essential** — tanpa itu, SR menemukan fungsi yang overfit dengan ekspresi raksasa.
3. **Held-out test crucial** — SR bisa menemukan fungsi yang fit training sempurna tapi salah secara fundamental.
4. **Domain knowledge helps** — primitive set yang baik (fungsi fisika yang relevan) mempercepat dan memperbaiki SR.
5. **SR untuk science** — telah mereproduksi hukum fisika (gravitasi, relativitas) dari data — bukti konsep automated discovery.

## Formula

- SR objective: min ||y−f(x)||² + λ·complexity(f).
- SINDy: ẋ = Θ(x)ξ, ξ sparse via LASSO.
- Complexity: number of nodes/operations in expression tree.

## Contoh Kasus

Data generated dari y = 3x1² + 2sin(x2) + noise. SINDy dengan library {x1, x1², x1³, sin(x2), cos(x2), x1*x2, 1} menemukan ξ dengan hanya 3 non-zero coefficients — persamaan yang benar. GP menemukan ekspresi ekuivalen tapi lebih kompleks.

## Implementasi

```python
import numpy as np
from sklearn.linear_model import Lasso, LassoLarsIC

# SINDy-style sparse regression
def sindy_fit(X, y, library_funcs, alpha=0.01):
    """Fit y = sum(theta_i * f_i(X)) with sparse coefficients."""
    Theta = np.column_stack([f(X) for f in library_funcs])
    # normalize columns
    norms = np.linalg.norm(Theta, axis=0)
    Theta_norm = Theta / (norms + 1e-10)
    lasso = Lasso(alpha=alpha, fit_intercept=True)
    lasso.fit(Theta_norm, y)
    # recover coefficients
    coef = lasso.coef_ / (norms + 1e-10)
    return coef, library_funcs

# example: discover y = 2*x1^2 + sin(x2)
def make_library():
    return [
        lambda X: X[:,0],      # x1
        lambda X: X[:,0]**2,   # x1^2
        lambda X: X[:,1],      # x2
        lambda X: np.sin(X[:,1]),  # sin(x2)
        lambda X: np.cos(X[:,1]),  # cos(x2)
        lambda X: X[:,0]*X[:,1],   # x1*x2
        lambda X: np.ones(len(X)), # constant
    ]

# GP-based SR (simplified using gplearn if available)
try:
    from gplearn.genetic import SymbolicRegressor
    def gp_sr(X, y):
        gp = SymbolicRegressor(
            population_size=1000, generations=50,
            function_set=['add','sub','mul','div','sin','cos','sqrt','log'],
            const_range=(-1,1), parsimony_coefficient=0.001,
            max_samples=0.9, n_jobs=-1, random_state=42)
        gp.fit(X, y)
        return gp
except ImportError:
    print("gplearn not installed; use pip install gplearn")
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Symbolic regression differs from standard regression because it:",
    "options": [
      "Searches for the functional form itself, not just parameters",
      "Uses neural networks",
      "Only works for linear models",
      "Requires labeled data only"
    ],
    "answer": 0,
    "explanation": "SR discovers the equation structure; standard regression fits parameters to a given form.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "SINDy approach to SR uses:",
    "options": [
      "Sparse regression over a library of candidate functions",
      "Genetic programming only",
      "Neural networks",
      "Brute force search"
    ],
    "answer": 0,
    "explanation": "SINDy = Sparse Identification of Nonlinear Dynamics; LASSO on function library.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Why is complexity penalty important in SR?",
    "options": [
      "Without it, SR finds overly complex expressions that overfit",
      "It speeds up computation",
      "It reduces memory",
      "It's required by the algorithm"
    ],
    "answer": 0,
    "explanation": "Parsimony pressure prevents bloat and favors interpretable solutions.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `expression_complexity(tree_str)` estimating complexity as count of operators and functions in a prefix expression string (rough heuristic).",
    "starter_code": "def expression_complexity(expr):\n    # count operators/functions: + - * / sin cos exp log sqrt pow\n    pass\n",
    "function": "expression_complexity",
    "tests": [
      {"call": "expression_complexity('(+ x1 (* x2 2))')", "expected": 2},
      {"call": "expression_complexity('(sin (+ x1 x2))')", "expected": 2}
    ],
    "hints": ["Count occurrences of operator tokens in prefix notation"],
    "explanation": "Complexity measure for parsimony.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You run SR on physics data and find an expression with RMSE=0.001 on training but RMSE=0.5 on test. Diagnose what happened and propose three ways to improve robustness.",
    "rubric": [
      "Overfitting — expression fits noise, not underlying law",
          "Increase parsimony penalty / complexity constraint",
      "Use more diverse training data / cross-validation",
      "Domain knowledge: constrain primitive set to physically meaningful functions"
        ],
    "points": 2
  }
]
```

## Referensi

- Schmidt & Lipson (2009) 'Distilling Free-Form Natural Laws from Experimental Data'
- Udrescu & Tegmark (2020) 'AI Feynman: A physics-inspired method for symbolic regression'
- gplearn documentation
- PySR documentation (https://pysr.readthedocs.io)
- 'Sparse Identification of Nonlinear Dynamics' — Brunton et al. 2016
