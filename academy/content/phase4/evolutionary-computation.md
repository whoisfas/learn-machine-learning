---
{
  "id": "p4-evolutionary-computation",
  "title": "Evolutionary Computation",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Survey the evolutionary computation family: GA, GP, ES, DE, NSGA",
    "Implement evolution strategies and differential evolution",
    "Understand multi-objective optimization with Pareto fronts",
    "Compare EC methods for different problem types"
  ],
  "concepts": [
    "Evolution strategies (ES): self-adaptive step sizes",
    "Differential evolution (DE): vector differences for mutation",
    "Multi-objective EC: Pareto dominance, NSGA-II",
    "CMA-ES: covariance matrix adaptation",
    "Memetic algorithms: EC + local search",
    "Island models and parallel EC",
    "Comparison: GA vs ES vs DE vs PSO"
  ],
  "formulas": [
    {"name": "DE mutation", "latex": "v_i = x_{r1} + F(x_{r2} - x_{r3})", "note": "Difference vector perturbation."},
    {"name": "ES update", "latex": "x_i' = x_i + \\sigma N(0, I)", "note": "Gaussian perturbation; σ self-adapted."},
    {"name": "Pareto dominance", "latex": "x \\succ y \\iff f_i(x) \\leq f_i(y) \\forall i \\text{ and } \\exists j: f_j(x) < f_j(y)", "note": "x dominates y if better in all objectives."}
  ],
  "references": [
    "Eiben & Smith 'Introduction to Evolutionary Computing'",
    "Deb 'Multi-Objective Optimization using Evolutionary Algorithms'",
    "pycma documentation (CMA-ES)",
    "DEAP documentation"
  ]
}
---

## Penjelasan

Evolutionary Computation (EC) adalah keluarga metode optimasi yang meniru evolusi. Selain GA dan GP, ada varian penting untuk continuous optimization dan multi-objective problems.

**Evolution Strategies (ES).** Fokus pada continuous optimization. Setiap individual adalah vector real; mutasi = Gaussian perturbation x' = x + σN(0,I). σ (step size) bisa self-adaptive: setiap individual membawa σ sendiri yang ikut berevolusi. ES sangat baik untuk continuous black-box optimization.

**Differential Evolution (DE).** Varian ES yang menggunakan *difference vectors* untuk mutasi: v = x_r1 + F·(x_r2 − x_r3). F adalah scaling factor (0.5-0.9). DE sangat efisien untuk continuous optimization, robust, dan sederhana — sering mengalahkan GA dan PSO pada benchmark.

**CMA-ES.** Covariance Matrix Adaptation ES — state-of-art untuk continuous black-box optimization. Belajar covariance matrix dari successful mutations -> adapt step size dan direction. Sangat baik untuk non-separable, ill-conditioned problems. pycma library.

**Multi-objective EC.** Banyak masalah punya multiple conflicting objectives (mis. accuracy vs model size, cost vs performance). Pareto dominance: x dominates y jika lebih baik di semua objectives. NSGA-II adalah algoritma standar: non-dominated sorting + crowding distance untuk menjaga diversity di Pareto front.

**Memetic algorithms.** Kombinasi EC global search + local search (gradient descent, hill climbing) — best of both worlds.

## Konsep Penting

1. **EC vs gradient methods** — EC tidak butuh gradient; cocok untuk non-differentiable, noisy, multi-modal objectives.
2. **Population diversity** — kunci untuk menghindari premature convergence; island models dan niching membantu.
3. **Self-adaptation** — ES/CMA-ES belajar step size dari experience; lebih efisien dari fixed mutation rate.
4. **Pareto front** — untuk multi-objective, tidak ada single optimum; front adalah set trade-off solutions.
5. **EC untuk NAS/HPO** — neural architecture search dan hyperparameter optimization adalah use case EC yang sedang aktif diteliti.

## Formula

- DE mutation: v = x_r1 + F(x_r2 - x_r3).
- ES: x' = x + σ·N(0,I); σ self-adapted.
- CMA-ES: adapts full covariance matrix C.
- Pareto: x ≻ y iff f_i(x) ≤ f_i(y) ∀i and ∃j f_j(x) < f_j(y).

## Contoh Kasus

Optimasi multi-objective: minimize (model_size, error) untuk neural network compression. NSGA-II menemukan Pareto front: model kecil dengan error sedikit lebih tinggi vs model besar dengan error minimal. Decision maker memilih trade-off point.

## Implementasi

```python
import numpy as np

def differential_evolution(fitness_fn, dim, pop_size=50, generations=100,
                           F=0.8, CR=0.9, bounds=(-5, 5), seed=42):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(bounds[0], bounds[1], (pop_size, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])
    for gen in range(generations):
        for i in range(pop_size):
            idxs = rng.choice(pop_size, 3, replace=False)
            while i in idxs: idxs = rng.choice(pop_size, 3, replace=False)
            mutant = pop[idxs[0]] + F * (pop[idxs[1]] - pop[idxs[2]])
            cross_mask = rng.random(dim) < CR
            trial = np.where(cross_mask, mutant, pop[i])
            trial_fit = fitness_fn(trial)
            if trial_fit > fitness[i]:  # maximization
                pop[i] = trial
                fitness[i] = trial_fit
    best = np.argmax(fitness)
    return pop[best], fitness[best]

# simple ES
def evolution_strategy(fitness_fn, dim, pop_size=50, generations=100,
                       sigma=0.1, seed=42):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(-5, 5, (pop_size, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])
    for gen in range(generations):
        # select top half
        elite_idx = np.argsort(fitness)[-pop_size//2:]
        elites = pop[elite_idx]
        # mutate
        offspring = elites + sigma * rng.standard_normal((len(elites), dim))
        pop = np.vstack([elites, offspring])[:pop_size]
        fitness = np.array([fitness_fn(ind) for ind in pop])
    return pop[np.argmax(fitness)], fitness.max()
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Differential Evolution mutates individuals using:",
    "options": [
      "Difference vectors between other population members",
      "Gaussian noise only",
      "Crossover with random strings",
      "Gradient information"
    ],
    "answer": 0,
    "explanation": "DE mutation: v = x_r1 + F(x_r2 - x_r3) — self-adaptive step sizes from population diversity.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "In multi-objective optimization, a solution x dominates y when:",
    "options": [
      "x is better or equal in ALL objectives and strictly better in at least one",
      "x has higher fitness",
      "x is simpler",
      "x was found first"
    ],
    "answer": 0,
    "explanation": "Pareto dominance: no objective worse, at least one better.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `pareto_front(points)` for 2-objective minimization returning indices of non-dominated points.",
    "starter_code": "import numpy as np\n\ndef pareto_front(points):\n    points = np.asarray(points)\n    n = len(points)\n    dominated = np.zeros(n, dtype=bool)\n    pass\n    return np.where(~dominated)[0]\n",
    "function": "pareto_front",
    "tests": [
      {"call": "pareto_front([[1,2],[2,1],[3,3],[0.5,2.5]]).tolist()", "expected": [0,1,3]}
    ],
    "hints": ["For each i,j: if all(points[j] <= points[i]) and any(points[j] < points[i]): i dominated"],
    "explanation": "Non-dominated sorting for multi-objective optimization.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `cma_es_step(mean, sigma, C, population, fitness)` performing one CMA-ES generation: sample population, select elite, update mean and covariance (simplified).",
    "starter_code": "import numpy as np\n\ndef cma_es_step(mean, sigma, C, pop_size=10, elite_frac=0.5):\n    pass\n",
    "function": "cma_es_step",
    "tests": [
      {"call": "cma_es_step(np.array([0.0]), 1.0, np.array([[1.0]]))", "expected": "check"}
    ],
    "hints": ["Sample from N(mean, sigma^2 * C); select top elite_frac; update mean = elite mean; update C from elite covariance"],
    "explanation": "CMA-ES core update — state-of-art continuous optimization.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You need to optimize 100 hyperparameters of a deep neural network (mixed discrete/continuous). Compare GA, Bayesian optimization, and random search for this task — which would you choose and why?",
    "rubric": [
      "Bayesian optimization most sample-efficient for expensive evaluations",
          "GA good for discrete/mixed spaces but needs many evaluations",
      "Random search baseline — surprisingly competitive for low budgets",
      "Mentions multi-fidelity, successive halving, or surrogate models"
        ],
    "points": 2
  }
]
```

## Referensi

- Eiben & Smith, 'Introduction to Evolutionary Computing'
- Deb, 'Multi-Objective Optimization using Evolutionary Algorithms'
- pycma documentation (CMA-ES)
- DEAP documentation
- 'Differential Evolution: A Simple and Efficient Heuristic' — Storn & Price 1997
