---
{
  "id": "p4-neural-architecture-search",
  "title": "Neural Architecture Search (NAS)",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Understand NAS as automated architecture design",
    "Compare search strategies: random, evolutionary, RL, gradient-based",
    "Implement a simple NAS cell search",
    "Evaluate NAS results and computational cost"
  ],
  "concepts": [
    "Search space: macro (whole architecture) vs micro (cells)",
    "Weight sharing for efficient evaluation",
    "Random search NAS baseline",
    "Evolutionary NAS ( tournament selection on architectures)",
    "RL-based NAS (controller RNN)",
    "Gradient-based NAS (DARTS)",
    "Hardware-aware NAS",
    "NASBench benchmark datasets"
  ],
  "formulas": [
    {"name": "NAS objective", "latex": "\\min_\\alpha \\text{ValLoss}(\\alpha, w^*(\\alpha))", "note": "Find architecture α minimizing validation loss after training weights w*."}
  ],
  "references": [
    "Elsken et al. 2019 'Neural Architecture Search: A Survey'",
    "Liu et al. 2019 'DARTS: Differentiable Architecture Search'",
    "NASBench dataset (Ying et al. 2019)",
    "nni documentation (Microsoft)"
  ]
}
---

## Penjelasan

NAS mengotomatisasi desain arsitektur neural network — mencari struktur layer, koneksi, dan operasi yang optimal untuk task tertentu. Ini memindahkan "seninya" arsitektur ke algoritma pencarian.

**Search space.**
- *Macro search*: cari seluruh arsitektur (jumlah layer, tipe layer, koneksi) — ruang besar, mahal.
- *Micro search (cell-based)*: cari *cell* yang diulang — mis. normal cell + reduction cell untuk CNN; ruang lebih kecil, lebih transferable.
- *Operations*: pilih dari primitive set (conv3x3, conv5x5, maxpool, skip connection, dll).

**Evaluasi mahal.** Setiap arsitektur kandidat harus di-train untuk dinilai — mahal. Solusi: *weight sharing* (semua arsitektur berbagi subset weights dalam supernetwork), *low-fidelity evaluation* (train sebentar saja), *proxy metrics* (validation accuracy setelah few epochs).

**Search strategies.**
- *Random search*: baseline yang surprisingly kuat; sample arsitektur acak, train, pilih terbaik.
- *Evolutionary*: populasi arsitektur berevolusi via mutation/crossover pada encoding; tournament selection.
- *RL-based*: controller (RNN) menghasilkan arsitektur; reward = validation accuracy; policy gradient update.
- *Gradient-based (DARTS)*: relax arsitektur ke continuous space; train weights + architecture parameters bersamaan via gradient descent; pilih operasi dengan bobot tertinggi.

**Hasil penting.** NAS menemukan arsitektur yang mengalahkan manual design: AmoebaNet, EfficientNet (compound scaling), Vision Transformer variants. Tapi biaya komputasi sangat tinggi (ribuan GPU hours) — praktis hanya untuk lab besar atau dengan weight sharing.

## Konsep Penting

1. **NAS = bilevel optimization** — outer loop cari arsitektur, inner loop train weights; sangat mahal.
2. **Weight sharing essential** — tanpa itu, NAS tidak scalable; DARTS supernetwork adalah contoh.
3. **Proxy gap** — arsitektur yang baik setelah 10 epochs belum tentu terbaik setelah 100 epochs.
4. **Transferability** — cell-based NAS transferable ke dataset lain; macro search kurang.
5. **Hardware-aware NAS** — incorporate latency/memory ke objective untuk deployment constraints.

## Formula

- Bilevel: min_α L_val(w*(α), α) s.t. w*(α) = argmin_w L_train(w, α).
- DARTS: architecture parameters α relaxed via softmax over operations.

## Contoh Kasus

DARTS pada CIFAR-10: supernetwork dengan semua operasi kandidat; train weights + architecture parameters; setelah convergence, pilih operasi dengan α tertinggi per edge -> derive final architecture -> train dari scratch -> ~97% accuracy. Bandingkan dengan manual ResNet — DARTS menemukan arsitektur lebih efisien.

## Implementasi

```python
import numpy as np
import random

# Simple random search NAS for MLP architectures
class Arch:
    def __init__(self, layers, activations):
        self.layers = layers        # list of hidden dims
        self.activations = activations  # list of 'relu','tanh','gelu'

    def to_string(self):
        return f"layers={self.layers}, acts={self.activations}"

def random_arch(rng, max_layers=5):
    n_layers = rng.integers(1, max_layers+1)
    layers = [rng.integers(32, 512) for _ in range(n_layers)]
    acts = [rng.choice(['relu','tanh','gelu','leaky_relu']) for _ in range(n_layers)]
    return Arch(layers, acts)

def mutate_arch(arch, rng):
    new_layers = arch.layers.copy()
    new_acts = arch.activations.copy()
    op = rng.random()
    if op < 0.3 and len(new_layers) > 1:
        # remove a layer
        idx = rng.integers(0, len(new_layers))
        new_layers.pop(idx); new_acts.pop(idx)
    elif op < 0.6:
        # change a layer size
        idx = rng.integers(0, len(new_layers))
        new_layers[idx] = int(new_layers[idx] * rng.uniform(0.5, 2.0))
    elif op < 0.8:
        # change activation
        idx = rng.integers(0, len(new_acts))
        new_acts[idx] = rng.choice(['relu','tanh','gelu','leaky_relu'])
    else:
        # add a layer
        idx = rng.integers(0, len(new_layers)+1)
        new_layers.insert(idx, rng.integers(32, 512))
        new_acts.insert(idx, rng.choice(['relu','tanh','gelu','leaky_relu']))
    return Arch(new_layers, new_acts)

def nas_random_search(evaluate_fn, n_trials=50, seed=42):
    rng = np.random.default_rng(seed)
    best_arch, best_score = None, -float('inf')
    for i in range(n_trials):
        arch = random_arch(rng)
        score = evaluate_fn(arch)  # train and return val accuracy
        if score > best_score:
            best_score = score
            best_arch = arch
        print(f"trial {i}: {score:.4f} - {arch.to_string()}")
    return best_arch, best_score
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "NAS differs from manual architecture design by:",
    "options": [
      "Automating the search for optimal architecture using algorithms",
      "Using bigger models",
      "Training faster",
      "Removing the need for data"
    ],
    "answer": 0,
    "explanation": "NAS = automated architecture search via random/evolutionary/RL/gradient methods.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Weight sharing in NAS addresses:",
    "options": [
      "The prohibitive cost of training every candidate architecture from scratch",
      "Memory usage",
      "Gradient vanishing",
      "Overfitting"
    ],
    "answer": 0,
    "explanation": "Supernetwork shares weights across architectures -> efficient evaluation.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "DARTS makes NAS differentiable by:",
    "options": [
      "Relaxing architecture choices to continuous parameters optimized via gradient descent",
      "Using smaller models",
      "Removing validation",
      "Random sampling only"
    ],
    "answer": 0,
    "explanation": "DARTS mixes operations with softmax weights -> gradients flow to architecture params.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `architecture_encoding(arch)` converting an architecture dict to a fixed-length vector for evolutionary search.",
    "starter_code": "import numpy as np\n\ndef architecture_encoding(arch, max_layers=8):\n    # arch: dict with 'layers' (list of ints), 'activations' (list of str)\n    act_map = {'relu':0,'tanh':1,'gelu':2,'leaky_relu':3}\n    pass\n",
    "function": "architecture_encoding",
    "tests": [
      {"call": "len(architecture_encoding({'layers':[64,128],'activations':['relu','gelu']}))", "expected": "check"}
    ],
    "hints": ["Fixed length = max_layers*2 (size + activation per slot); pad with zeros"],
    "explanation": "Encoding for evolutionary NAS.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "NAS discovered an architecture with 98% accuracy but requires 500 GPU-hours to find. Your team has 8 GPUs and 2 weeks. Is NAS feasible? Propose a pragmatic alternative.",
    "rubric": [
      "Full NAS likely infeasible; propose weight-sharing methods (DARTS, Once-for-All)",
          "Or use existing NAS results/benchmarks (NASBench, AutoML tools)",
      "Or manual architecture based on similar problems + transfer learning",
      "Or progressive NAS with early stopping and low-fidelity evaluation"
        ],
    "points": 2
  }
]
```

## Referensi

- Elsken, Metzen, Hutter (2019) 'Neural Architecture Search: A Survey'
- Liu et al. (2019) 'DARTS: Differentiable Architecture Search'
- Ying et al. (2019) 'NAS-Bench-101' dataset
- Real et al. (2019) 'Regularized Evolution for Image Classifier Architecture Search'
- Microsoft NNI documentation
