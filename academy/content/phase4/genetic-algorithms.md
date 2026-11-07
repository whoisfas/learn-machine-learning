---
{
  "id": "p4-genetic-algorithms",
  "title": "Genetic Algorithms",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Implement GA from scratch: selection, crossover, mutation",
    "Understand how GA explores solution spaces",
    "Apply GA to optimization and hyperparameter search",
    "Analyze convergence and diversity preservation"
  ],
  "concepts": [
    "Population-based stochastic search",
    "Chromosome encoding (binary, real-valued, permutation)",
    "Fitness function and selection pressure",
    "Selection: tournament, roulette, rank-based, truncation",
    "Crossover: single-point, uniform, arithmetic",
    "Mutation: bit flip, Gaussian perturbation",
    "Elitism and diversity maintenance",
    "Convergence analysis and premature convergence"
  ],
  "formulas": [
    {"name": "Schema theorem (building block)", "latex": "m(H,t+1) \\geq m(H,t)\\frac{f(H)}{\\bar{f}}[1-p_c-p_m]", "note": "Short schemata with above-average fitness grow exponentially."},
    {"name": "Tournament selection", "latex": "P(select) \\propto \\text{rank or fitness among } k \\text{ random}", "note": "k controls selection pressure."}
  ],
  "references": [
    "Holland 1975 'Adaptation in Natural and Artificial Systems'",
    "Goldberg 'Genetic Algorithms in Search, Optimization and Machine Learning'",
    "DEAP documentation (deap.readthedocs.io)"
  ]
}
---

## Penjelasan

Genetic Algorithm (GA) adalah metode optimasi yang meniru evolusi alami: populasi solusi berevolusi melalui seleksi, crossover, dan mutasi untuk memaksimalkan fitness function. GA sangat baik untuk masalah non-convex, discontinuous, atau dengan constraint yang sulit di-handle gradient-based methods.

**Komponen GA.**
- *Chromosome*: representasi solusi (bit string, real vector, permutation).
- *Population*: himpunan chromosome, ukuran tetap (mis. 100).
- *Fitness function*: mengukur kualitas solusi; lebih tinggi lebih baik.
- *Selection*: memilih parent untuk reproduksi; tekanan seleksi menentukan eksploitasi vs eksplorasi.
- *Crossover*: menggabungkan dua parent -> offspring (rekominasi gen).
- *Mutation*: mengubah gen acak -> menjaga diversity, mencegah premature convergence.
- *Elitism*: simpan top-k solutions apa adanya ke generasi berikutnya.

**Cara kerja satu generasi.** Evaluasi fitness semua individuals -> select parents -> crossover -> mutate -> form new population -> repeat sampai stopping criterion (max generations, convergence, target fitness).

**Selection methods.**
- *Roulette wheel*: probabilitas proporsional fitness; bisa didominasi individual sangat fit.
- *Tournament*: pilih k random, ambil terbaik; k mengontrol tekanan seleksi.
- *Rank-based*: probabilitas berdasarkan rank, bukan fitness — lebih robust terhadap outlier fitness.
- *Truncation*: top p% langsung jadi parent.

**Encoding matters.** Binary encoding untuk combinatorial problems; real-valued untuk continuous optimization; permutation untuk TSP/scheduling. Encoding yang baik membuat operator crossover/mutation bermakna.

**Kapan GA unggul.** Non-convex/multimodal optimization, constraint satisfaction, symbolic regression, neural architecture search, hyperparameter tuning di ruang diskrit/campuran. GA lebih lambat dari gradient methods untuk smooth convex problems tapi lebih robust untuk masalah "black-box".

## Konsep Penting

1. **Premature convergence** — populasi kehilangan diversity terlalu cepat -> stuck di local optimum; mitigasi: mutation rate cukup, island models, niching.
2. **Selection pressure balance** — terlalu tinggi -> premature convergence; terlalu rendah -> random search.
3. **Fitness landscape** — GA bekerja baik jika ada building blocks (schema) yang bisa dikombinasi.
4. **Encoding = search space** — representasi yang buruk membuat operator tidak efektif.
5. **GA vs gradient methods** — GA tidak butuh gradient; cocok untuk non-differentiable, discrete, atau noisy objectives.

## Formula

- Schema theorem: m(H,t+1) ≥ m(H,t)·(f(H)/f̄)·[1−p_c−p_m].
- Tournament selection: P = C(k,i)·i^k style probabilities.
- Mutation rate: p_m ≈ 1/L (L = chromosome length) rule of thumb.

## Contoh Kasus

Optimasi portfolio: chromosome = bobot alokasi aset (real vector, sum=1); fitness = Sharpe ratio - transaction cost penalty. GA menemukan alokasi yang robust terhadap multiple market scenarios. Atau TSP: chromosome = permutation kota; fitness = -total distance; crossover = order crossover (OX).

## Implementasi

```python
import numpy as np

def tournament_select(population, fitness, k=3, rng=None):
    rng = rng or np.random.default_rng()
    idx = rng.integers(0, len(population), (len(population), k))
    winners = [population[i[np.argmax(fitness[i])]] for i in idx]
    return winners

def uniform_crossover(p1, p2, rng):
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return child

def gaussian_mutation(individual, sigma=0.1, rng=None):
    rng = rng or np.random.default_rng()
    mask = rng.random(len(individual)) < 0.1  # mutation rate
    individual[mask] += rng.normal(0, sigma, mask.sum())
    return individual

def genetic_algorithm(fitness_fn, dim, pop_size=100, generations=200,
                      mutation_rate=0.1, elite_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    population = rng.uniform(-5, 5, (pop_size, dim))
    for gen in range(generations):
        fitness = np.array([fitness_fn(ind) for ind in population])
        # elitism
        elite_n = int(pop_size * elite_frac)
        elite_idx = np.argsort(fitness)[-elite_n:]
        elites = population[elite_idx].copy()
        # selection + reproduction
        parents = tournament_select(population, fitness, k=3, rng=rng)
        offspring = []
        while len(offspring) < pop_size - elite_n:
            p1, p2 = rng.choice(len(parents), 2, replace=False)
            child = uniform_crossover(parents[p1], parents[p2], rng)
            child = gaussian_mutation(child, rng=rng)
            offspring.append(child)
        population = np.vstack([elites] + offspring[:pop_size-elite_n])
        if gen % 50 == 0:
            print(f"gen {gen}: best fitness={fitness.max():.4f}")
    return population[np.argmax(fitness)]
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Genetic algorithms are particularly well-suited for problems that are:",
    "options": [
      "Non-convex, discontinuous, or lack gradient information",
      "Small and convex",
      "Linear only",
      "Already solved by gradient descent"
    ],
    "answer": 0,
    "explanation": "GA doesn't need gradients; handles black-box, discrete, multimodal objectives.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Premature convergence in GA refers to:",
    "options": [
      "Population loses diversity and gets stuck in local optimum before finding global",
      "Finding the optimal solution too quickly",
      "Mutation rate too high",
      "Selection too weak"
    ],
    "answer": 0,
    "explanation": "Loss of diversity -> all individuals similar -> no exploration of other regions.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `roulette_wheel_select(fitness, n)` returning n indices selected proportionally to fitness (fitness must be non-negative).",
    "starter_code": "import numpy as np\n\ndef roulette_wheel_select(fitness, n, rng=None):\n    fitness = np.asarray(fitness, dtype=float)\n    rng = rng or np.random.default_rng(0)\n    pass\n",
    "function": "roulette_wheel_select",
    "tests": [
      {"call": "roulette_wheel_select([1,1,1], 5, np.random.default_rng(0))", "expected": "check"}
    ],
    "hints": ["probs = fitness/fitness.sum(); rng.choice(len(fitness), size=n, p=probs)"],
    "explanation": "Fitness-proportional selection.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `order_crossover(p1, p2, rng)` for permutation chromosomes: preserve relative order from p1 within a random slice, fill remaining from p2 in order.",
    "starter_code": "import numpy as np\n\ndef order_crossover(p1, p2, rng=None):\n    rng = rng or np.random.default_rng(0)\n    p1 = list(p1); p2 = list(p2)\n    n = len(p1)\n    pass\n",
    "function": "order_crossover",
    "tests": [
      {"call": "sorted(order_crossover([1,2,3,4,5],[5,4,3,2,1],np.random.default_rng(0)))", "expected": [1,2,3,4,5]}
    ],
    "hints": ["Pick random slice [a:b]; copy p1[a:b]; fill rest with p2 elements not already used in order"],
    "explanation": "OX crossover preserves permutation validity for TSP-like problems.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You apply GA to optimize 50 hyperparameters of a neural network and it takes 3 days per run. Propose three strategies to make it tractable while preserving search quality.",
    "rubric": [
      "Surrogate-assisted GA (evaluate promising candidates only)",
          "Multi-fidelity evaluation (evaluate on subset/early epochs first)",
      "Parallel/island GA across machines",
      "Reduce search space via prior knowledge or two-stage search"
        ],
    "points": 2
  }
]
```

## Referensi

- Holland (1975) 'Adaptation in Natural and Artificial Systems'
- Goldberg, 'Genetic Algorithms in Search, Optimization and Machine Learning'
- DEAP documentation
- 'Introduction to Evolutionary Computing' by Eiben & Smith
