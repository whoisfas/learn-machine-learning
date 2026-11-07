---
{
  "id": "p4-genetic-programming",
  "title": "Genetic Programming",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 16,
  "objectives": [
    "Represent programs as expression trees",
    "Implement subtree crossover and mutation",
    "Apply GP to symbolic regression and program synthesis",
    "Control bloat and parsimony pressure"
  ],
  "concepts": [
    "Expression trees: functions (internal) and terminals (leaves)",
    "Primitive set design",
    "Subtree crossover and mutation",
    "Fitness for symbolic regression (error + complexity)",
    "Bloat problem and parsimony pressure",
    "Strongly typed GP (type constraints)",
    "Multi-objective GP (accuracy vs simplicity)",
    "Comparison with neural networks for function approximation"
  ],
  "formulas": [
    {"name": "GP fitness", "latex": "f = -\\text{RMSE}(\\hat{y}, y) - \\lambda \\cdot \\text{tree\\_size}", "note": "Error + parsimony penalty."},
    {"name": "Subtree crossover", "latex": "\\text{swap random subtree of } T_1 \\text{ with random subtree of } T_2", "note": "Preserves syntactic validity."}
  ],
  "references": [
    "Koza 1992 'Genetic Programming'",
    "Poli et al. 'A Field Guide to Genetic Programming'",
    "gplearn documentation",
    "DEAP gp documentation"
  ]
}
---

## Penjelasan

Genetic Programming (GP) adalah evolusi program komputer — bukan parameter, melainkan *struktur program itu sendiri*. Program direpresentasikan sebagai expression tree, dan evolusi menemukan kombinasi fungsi yang memecahkan masalah.

**Expression trees.** Node internal = fungsi/operator (+, *, sin, exp); leaf = terminal (variabel x1, x2, konstanta). Contoh: `(x1 + 2) * sin(x2)` direpresentasikan sebagai tree dengan root `*`, children `+` dan `sin`, dst.

**Primitive set.** Himpunan fungsi dan terminal yang diizinkan menentukan search space. Terlalu kecil -> tidak bisa merepresentasikan solusi; terlalu besar -> search space meledak. Desain primitive set adalah seni: sertakan operasi yang relevan dengan domain.

**Operasi evolusi.**
- *Subtree crossover*: pilih random subtree di parent1 dan parent2, tukar -> offspring valid secara sintaks.
- *Subtree mutation*: ganti random subtree dengan tree baru acak.
- *Hoist mutation*: ganti subtree dengan salah satu child-nya (mengurangi ukuran).
- *Point mutation*: ganti satu node dengan node lain dari primitive set.

**Fitness.** Untuk symbolic regression: fitness = -RMSE(prediksi, target) - λ·tree_size. Tree size penalizes bloat — program yang lebih sederhana lebih disukai jika akurasi sama.

**Bloat problem.** Tanpa penalti, GP cenderung menghasilkan tree yang sangat besar (bloat) karena subtree besar cenderung tidak merusak fitness. Solusi: parsimony pressure (fitness - λ·size), depth limits, bloat control operators.

**Kapan GP unggul.** Symbolic regression (menemukan persamaan dari data), program synthesis (menemukan algoritma), feature construction otomatis, scientific discovery (menemukan hukum fisika dari data — seperti yang dilakukan Schmidt & Lipson 2009 untuk mereproduksi hukum Newton).

## Konsep Penting

1. **GP = search over programs** — ruang pencarian adalah semua tree yang mungkin, bukan parameter kontinu.
2. **Building blocks hypothesis** — subtree kecil yang berguna bisa dikombinasi menjadi solusi lebih besar.
3. **Bloat adalah musuh** — tanpa parsimony pressure, tree tumbuh tidak terkendali.
4. **Strong typing** — type constraints pada node memastikan type correctness tanpa runtime errors.
5. **Interpretability advantage** — hasil GP adalah persamaan eksplisit, bisa dibaca manusia — keunggulan vs neural network.

## Formula

- Fitness: f = -RMSE - λ·size.
- Subtree crossover: swap(T1_subtree, T2_subtree).
- Depth limit: max_depth untuk mencegah tree terlalu dalam.

## Contoh Kasus

Symbolic regression pada data fisika: y = G·M1·M2/r². GP dengan primitive set {+,-,*,/,sin,cos,exp,log,x1,x2,x3} menemukan ekspresi yang mendekati hukum gravitasi universal dari data orbit planet — tanpa diberi tahu bentuk fungsinya.

## Implementasi

```python
import numpy as np
import operator

# Simple GP for symbolic regression
class Node:
    def __init__(self, op=None, children=None, value=None):
        self.op = op          # function name or None for leaf
        self.children = children or []
        self.value = value    # constant or variable name

    def evaluate(self, X):
        if self.op is None:
            if self.value in X: return X[self.value]
            return np.full(X['x1'].shape, float(self.value))
        args = [c.evaluate(X) for c in self.children]
        return OPS[self.op](*args)

    def size(self):
        return 1 + sum(c.size() for c in self.children)

OPS = {
    '+': lambda a,b: a+b, '-': lambda a,b: a-b,
    '*': lambda a,b: a*b, '/': lambda a,b: a/(b+1e-10),
    'sin': lambda a: np.sin(a), 'cos': lambda a: np.cos(a),
    'exp': lambda a: np.exp(np.clip(a,-10,10)),
    'log': lambda a: np.log(np.abs(a)+1e-10),
    'neg': lambda a: -a,
}

def random_tree(depth, rng, vars=['x1','x2'], consts=[0.5,1.0,2.0]):
    if depth <= 0 or rng.random() < 0.3:
        if rng.random() < 0.7:
            return Node(value=rng.choice(vars))
        return Node(value=str(rng.choice(consts)))
    op = rng.choice(list(OPS.keys()))
    n_args = 2 if op in ['+','-','*','/'] else 1
    children = [random_tree(depth-1, rng, vars, consts) for _ in range(n_args)]
    return Node(op=op, children=children)

def subtree_nodes(tree):
    nodes = []
    def collect(n):
        nodes.append(n)
        for c in n.children: collect(c)
    collect(tree)
    return nodes

def crossover(t1, t2, rng):
    # pick random subtrees and swap
    n1 = rng.choice(subtree_nodes(t1))
    n2 = rng.choice(subtree_nodes(t2))
    # simplified: return new tree (proper impl needs parent pointers)
    return t1, t2

def fitness(tree, X, y, lam=0.001):
    try:
        pred = tree.evaluate(X)
        rmse = np.sqrt(np.mean((pred - y)**2))
        return -rmse - lam * tree.size()
    except Exception:
        return -1e9

def run_gp(X, y, pop_size=500, generations=50, seed=42):
    rng = np.random.default_rng(seed)
    pop = [random_tree(4, rng) for _ in range(pop_size)]
    for gen in range(generations):
        fits = [fitness(t, X, y) for t in pop]
        # select top 20% as parents
        elite_idx = np.argsort(fits)[-pop_size//5:]
        parents = [pop[i] for i in elite_idx]
        # breed new population
        new_pop = [pop[i] for i in elite_idx]  # elitism
        while len(new_pop) < pop_size:
            p1, p2 = rng.choice(len(parents), 2, replace=False)
            # mutation: replace with new random subtree
            child = random_tree(4, rng)
            new_pop.append(child)
        pop = new_pop
        if gen % 10 == 0:
            print(f"gen {gen}: best fitness={max(fits):.4f}")
    return pop[np.argmax([fitness(t,X,y) for t in pop])]
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "In genetic programming, solutions are represented as:",
    "options": [
      "Expression trees of functions and terminals",
      "Fixed-length parameter vectors",
      "Neural network weights",
      "Binary strings only"
    ],
    "answer": 0,
    "explanation": "GP evolves program structure (trees), not just parameters.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Subtree crossover in GP works by:",
    "options": [
      "Swapping random subtrees between two parent trees",
      "Swapping single nodes",
      "Mutating leaf values",
      "Crossing over at the root only"
    ],
    "answer": 0,
    "explanation": "Subtree crossover exchanges random subtrees — preserves syntactic validity.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The 'bloat problem' in GP refers to:",
    "options": [
      "Trees growing excessively large without fitness improvement",
      "Population growing too fast",
      "Mutation rate too high",
      "Too many terminals"
    ],
    "answer": 0,
    "explanation": "Without parsimony pressure, trees bloat; size penalty needed.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `tree_size(node)` returning total number of nodes in expression tree.",
    "starter_code": "class Node:\n    def __init__(self, op=None, children=None, value=None):\n        self.op = op\n        self.children = children or []\n        self.value = value\n\ndef tree_size(node):\n    pass\n",
    "function": "tree_size",
    "tests": [
      {"call": "tree_size(Node(value='x'))", "expected": 1},
      {"call": "tree_size(Node(op='+',children=[Node(value='x'),Node(value='y')]))", "expected": 3}
    ],
    "hints": ["1 + sum(tree_size(c) for c in node.children)"],
    "explanation": "Tree size for parsimony pressure.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Compare GP symbolic regression vs neural network for discovering physical laws from data. Discuss interpretability, data efficiency, extrapolation, and when you'd choose each.",
    "rubric": [
      "GP produces explicit equations (interpretable, extrapolates correctly if structure right)",
          "NN more flexible for complex/high-dimensional data but black-box",
      "GP needs good primitive set; NN needs lots of data",
      "GP better for scientific discovery; NN for prediction tasks"
        ],
    "points": 2
  }
]
```

## Referensi

- Koza (1992) 'Genetic Programming: On the Programming of Computers by Means of Natural Selection'
- Poli, Langdon, McPhee — 'A Field Guide to Genetic Programming'
- gplearn documentation (Python GP library)
- Schmidt & Lipson (2009) 'Distilling Free-Form Natural Laws'
