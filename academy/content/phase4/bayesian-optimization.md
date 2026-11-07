---
{
  "id": "p4-bayesian-optimization",
  "title": "Bayesian Optimization",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Understand surrogate models and acquisition functions",
    "Implement BO with Gaussian processes",
    "Apply BO to hyperparameter optimization",
    "Compare BO vs random/grid search for expensive objectives"
  ],
  "concepts": [
    "Black-box optimization problem",
    "Surrogate model: Gaussian process regression",
    "Acquisition functions: UCB, EI, PI",
    "Exploration vs exploitation tradeoff",
    "BO for HPO, neural architecture search, materials design",
    "Multi-fidelity BO",
    "Constrained BO",
    "Tools: Optuna, scikit-optimize, BoTorch"
  ],
  "formulas": [
    {"name": "GP posterior", "latex": "p(f|x, D) = \\mathcal{N}(\\mu(x), \\sigma^2(x))", "note": "Mean and uncertainty from kernel regression."},
    {"name": "UCB acquisition", "latex": "\\alpha_{UCB}(x) = \\mu(x) + \\kappa \\sigma(x)", "note": "Balance mean prediction and uncertainty."},
    {"name": "EI acquisition", "latex": "\\alpha_{EI}(x) = (\\mu(x) - f^* - \\xi)\\Phi(Z) + \\sigma(x)\\phi(Z)", "note": "Expected improvement over best-so-far."}
  ],
  "references": [
    "Snoek et al. 2012 'Practical Bayesian Optimization of Machine Learning Algorithms'",
    "Optuna documentation",
    "scikit-optimize documentation",
    "BoTorch documentation"
  ]
}
---

## Penjelasan

Bayesian Optimization (BO) adalah metode untuk mengoptimasi fungsi mahal (expensive black-box) dengan meminimalkan jumlah evaluasi. Alih-alih mencoba semua titik (grid) atau acak (random), BO membangun *surrogate model* dari hasil sebelumnya dan memilih titik berikutnya yang paling informatif.

**Masalah.** f(x) mahal untuk dievaluasi (mis. training neural network = 1 jam). Tujuan: temukan x* = argmax f(x) dengan budget minimal evaluasi.

**Surrogate model.** Gaussian Process (GP) regression memodelkan f(x) sebagai distribusi Gaussian: p(f|x,D) = N(μ(x), σ²(x)). μ(x) = prediksi mean, σ²(x) = ketidakpastian. GP memberikan *uncertainty estimate* — BO tahu di mana dia "tidak tahu".

**Acquisition function.** Fungsi yang menentukan titik berikutnya untuk dievaluasi:
- *UCB (Upper Confidence Bound)*: α(x) = μ(x) + κσ(x). κ besar = eksplorasi; kecil = eksploitasi.
- *EI (Expected Improvement)*: expected improvement over best-so-far f*. Menyeimbangkan eksploitasi (μ tinggi) dan eksplorasi (σ tinggi).
- *PI (Probability of Improvement)*: probabilitas f(x) > f*.

**Iterasi BO.** Mulai dengan n0 random points -> fit GP -> maximize acquisition -> evaluate f(x_new) -> update GP -> repeat sampai budget habis.

**Kapan BO unggul.** Fungsi mahal (training NN, materials simulation, chemistry), dimensi rendah-sedang (<20), noisy observations. Untuk fungsi murah atau dimensi tinggi, random search sering cukup.

## Konsep Penting

1. **Sample efficiency** — BO jauh lebih efisien dari grid/random untuk expensive objectives.
2. **GP assumptions** — smoothness assumption; jika fungsi sangat discontinuous, GP mungkin tidak akurat.
3. **κ tuning** — exploration-exploitation tradeoff; κ besar = lebih banyak eksplorasi.
4. **Multi-fidelity** — evaluate cheap proxy first, expensive full evaluation only for promising candidates.
5. **BO untuk HPO** — standar untuk tuning neural networks, XGBoost, SVM.

## Formula

- GP: μ(x) = k(x,X)K⁻¹y; σ²(x) = k(x,x) - k(x,X)K⁻¹k(X,x).
- UCB: α = μ + κσ.
- EI: α = (μ-f*-ξ)Φ(Z) + σφ(Z), Z = (μ-f*-ξ)/σ.

## Contoh Kasus

Hyperparameter optimization untuk XGBoost: 8 hyperparameters (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha). Grid = 4^8 = 65k runs. BO dengan Optuna: 100 trials mencapai ~95% performa grid dalam 1/600 waktu.

## Implementasi

```python
# Optuna example (easiest BO)
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return cross_val_score(model, X_val, y_val, cv=3).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print("Best params:", study.best_params)
print("Best score:", study.best_value)

# manual GP-BO with scikit-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np

def gp_bo(fitness_fn, bounds, n_init=10, n_iter=50):
    rng = np.random.default_rng(42)
    X_sample = rng.uniform(bounds[:,0], bounds[:,1], (n_init, bounds.shape[0]))
    y_sample = np.array([fitness_fn(x) for x in X_sample])
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(bounds.shape[0]))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    for i in range(n_iter):
        gp.fit(X_sample, y_sample)
        # sample candidates and pick max UCB
        X_cand = rng.uniform(bounds[:,0], bounds[:,1], (1000, bounds.shape[0]))
        mu, sigma = gp.predict(X_cand, return_std=True)
        ucb = mu + 2.0 * sigma
        x_new = X_cand[np.argmax(ucb)]
        y_new = fitness_fn(x_new)
        X_sample = np.vstack([X_sample, x_new])
        y_sample = np.append(y_sample, y_new)
    return X_sample[np.argmax(y_sample)], y_sample.max()
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Bayesian Optimization is most useful when:",
    "options": [
      "Function evaluations are expensive (e.g., neural network training)",
      "Function is cheap to evaluate",
      "You have unlimited compute",
      "The function is linear"
    ],
    "answer": 0,
    "explanation": "BO minimizes number of expensive evaluations via surrogate model.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "The acquisition function in BO balances:",
    "options": [
      "Exploration (high uncertainty) vs exploitation (high predicted value)",
      "Speed vs accuracy",
      "Memory vs compute",
      "Training vs inference"
    ],
    "answer": 0,
    "explanation": "UCB/EI trade off between trying uncertain regions and exploiting known good regions.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "math",
    "question": "For UCB with mu=0.8, sigma=0.1, kappa=2, compute the acquisition value.",
    "answer": 1.0,
    "tolerance": 0.001,
    "hints": ["UCB = mu + kappa*sigma = 0.8 + 2*0.1"],
    "explanation": "0.8 + 0.2 = 1.0.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `ucb_acquisition(mu, sigma, kappa)` returning UCB values for arrays.",
    "starter_code": "import numpy as np\n\ndef ucb_acquisition(mu, sigma, kappa=2.0):\n    mu = np.asarray(mu, dtype=float)\n    sigma = np.asarray(sigma, dtype=float)\n    pass\n",
    "function": "ucb_acquisition",
    "tests": [
      {"call": "ucb_acquisition([0.5,0.8],[0.1,0.2],2.0).tolist()", "expected": [0.7,1.2]}
    ],
    "hints": ["return mu + kappa * sigma"],
    "explanation": "UCB acquisition function.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You have 50 trials budget for HPO. Compare random search, grid search, and Bayesian optimization for tuning 8 hyperparameters. Which do you choose and what are the tradeoffs?",
    "rubric": [
      "BO most sample-efficient for expensive evals; uses surrogate model",
          "Random search surprisingly competitive baseline",
      "Grid infeasible for 8 params (needs thousands of trials)",
      "Mentions Optuna/GP-based tools and warm-starting"
        ],
    "points": 2
  }
]
```

## Referensi

- Snoek, Larochelle, Adams (2012) 'Practical Bayesian Optimization of Machine Learning Algorithms'
- Optuna documentation (https://optuna.org)
- scikit-optimize documentation
- BoTorch documentation (Facebook/Meta)
- Shahriari et al. (2016) 'Taking the Human Out of the Loop: A Review of Bayesian Optimization'
