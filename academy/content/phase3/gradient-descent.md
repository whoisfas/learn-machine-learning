---
{
  "id": "p3-gradient-descent",
  "title": "Gradient Descent and Variants",
  "phase": 3,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Implement batch, stochastic, and mini-batch gradient descent",
    "Understand learning rate effects and scheduling",
    "Analyze convergence behavior on convex vs non-convex problems",
    "Apply momentum and adaptive methods"
  ],
  "concepts": [
    "Batch vs stochastic vs mini-batch gradient descent",
    "Learning rate: too high diverges, too low slow",
    "Learning rate schedules: step decay, cosine, warmup",
    "Momentum: velocity accumulation",
    "Adaptive learning rates: AdaGrad, RMSProp, Adam",
    "Convergence analysis for convex functions",
    "Local minima and saddle points in non-convex landscapes"
  ],
  "formulas": [
    {"name": "GD update", "latex": "\\theta \\leftarrow \\theta - \\eta \\nabla L(\\theta)", "note": "Move opposite gradient."},
    {"name": "SGD update", "latex": "\\theta \\leftarrow \\theta - \\eta \\nabla L_i(\\theta)", "note": "Single sample gradient — noisy but fast."},
    {"name": "Momentum", "latex": "v \\leftarrow \\beta v + \\nabla L; \\theta \\leftarrow \\theta - \\eta v", "note": "Accumulates velocity for faster convergence."},
    {"name": "Adam", "latex": "\\theta \\leftarrow \\theta - \\eta \\frac{\\hat{m}}{\\sqrt{\\hat{v}} + \\epsilon}", "note": "Adaptive per-parameter learning rate."}
  ],
  "references": [
    "Ruder 'An overview of gradient descent optimization algorithms'",
    "Goodfellow et al. ch. 8.1-8.2",
    "Kingma & Ba 2015 'Adam: A Method for Stochastic Optimization'"
  ]
}
---

## Penjelasan

Gradient descent adalah algoritma optimasi fundamental ML: iteratif melangkah berlawanan arah gradient untuk meminimalkan loss. Semua training neural network pada dasarnya adalah gradient descent (atau variannya) yang diterapkan pada jutaan parameter.

**Varian.**
- *Batch GD*: gradient dihitung pada seluruh dataset per iterasi; stabil tapi mahal untuk data besar.
- *Stochastic GD (SGD)*: gradient pada satu sampel per iterasi; sangat noisy tapi cepat per step; butuh lebih banyak epoch.
- *Mini-batch GD*: kompromi — batch 32-256 sampel; noise cukup untuk escape local minima tapi stabil untuk konvergen. Standar praktis.

**Learning rate.** η mengontrol ukuran langkah. Terlalu besar: overshoot, divergen (loss naik). Terlalu kecil: konvergensi lambat, stuck di plateau. Schedule: *step decay* (turunkan per N epoch), *cosine annealing* (smooth decay ke 0), *warmup* (naikkan lr di awal untuk stabilitas).

**Momentum.** Menambahkan "kecepatan" yang diakumulasi: v = βv + ∇L; θ -= ηv. Momentum mengatasi zigzag di valley sempit dan mempercepat konvergensi di arah konsisten. β=0.9 standar.

**Adaptive methods.**
- *AdaGrad*: akumulasi squared gradients; lr menurun untuk fitur frekuen; masalah: lr -> 0 terlalu cepat.
- *RMSProp*: exponential moving average squared gradients; mengatasi AdaGrad's diminishing lr.
- *Adam*: kombinasi momentum (m) + RMSProp (v) + bias correction; default untuk kebanyakan deep learning; lr=0.001 standar.
- *AdamW*: Adam + decoupled weight decay; standar modern untuk transformers.

**Non-convex landscapes.** Neural network loss surface non-convex: banyak local minima dan saddle points. SGD noise membantu escape saddle points; momentum mempercepat traversing plateau. Konvergensi global tidak dijamin tapi dalam praktik SGD menemukan solusi baik.

## Konsep Penting

1. **lr adalah hyperparameter paling penting** — tuning lr lebih impactful dari arsitektur dalam banyak kasus.
2. **Mini-batch = sweet spot** — noise bermanfaat (regularisasi implisit + escape saddle), efisiensi GPU.
3. **Schedule > fixed lr** — cosine/warmup standar untuk transformer training.
4. **Adam default, SGD+momentum untuk CNN tradisional** — SGD kadang generalisasi lebih baik dengan tuning tepat.
5. **Gradient clipping** — norm clipping untuk mencegah exploding gradients di RNN/transformer.

## Formula

- Vanilla SGD: θ -= η∇L.
- Momentum: v = βv + ∇L; θ -= ηv.
- Adam: m = β1m + (1-β1)g; v = β2v + (1-β2)g²; θ -= η·m̂/(√v̂+ε).
- Cosine schedule: η_t = η_min + 0.5(η_max-η_min)(1+cos(πt/T)).

## Contoh Kasus

Training ResNet pada ImageNet: SGD + momentum 0.9, lr=0.1 dengan step decay per 30 epoch, batch 256. Bandingkan dengan Adam lr=0.001 — Adam konvergen lebih cepat tapi SGD+momentum sering generalisasi lebih baik untuk vision tasks.

## Implementasi

```python
import torch
import torch.nn as nn

def train_sgd(model, X, y, lr=0.01, epochs=100, batch_size=32, momentum=0.9):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    n = len(X)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    return model

# Adam alternative
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# AdamW for transformers
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Mini-batch gradient descent is preferred over full-batch because:",
    "options": [
      "It balances gradient noise (helps escape saddle points) with computational efficiency",
      "It's always more accurate",
      "It uses less memory",
      "Full-batch doesn't work"
    ],
    "answer": 0,
    "explanation": "Mini-batch noise acts as implicit regularization and helps escape saddle points; full-batch is slow and can get stuck.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "If loss oscillates wildly and diverges, the most likely cause is:",
    "options": ["Learning rate too high", "Model too simple", "Not enough data", "Wrong loss function"],
    "answer": 0,
    "explanation": "Large lr overshoots minima -> oscillation/divergence.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `sgd_update(params, grads, lr)` performing vanilla SGD update on dict of params.",
    "starter_code": "import numpy as np\n\ndef sgd_update(params, grads, lr):\n    new_params = {}\n    for k in params:\n        pass\n    return new_params\n",
    "function": "sgd_update",
    "tests": [
      {"call": "sgd_update({'w':np.array([1.0,2.0])},{'w':np.array([0.1,0.2])},0.1)['w'].tolist()", "expected": [0.99,1.98]}
    ],
    "hints": ["new_params[k] = params[k] - lr*grads[k]"],
    "explanation": "Core SGD update rule.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `cosine_lr(epoch, total_epochs, lr_max, lr_min=0)` returning learning rate at given epoch using cosine annealing.",
    "starter_code": "import numpy as np\n\ndef cosine_lr(epoch, total_epochs, lr_max, lr_min=0.0):\n    pass\n",
    "function": "cosine_lr",
    "tests": [
      {"call": "cosine_lr(0, 100, 0.1)", "expected": 0.1, "tolerance": 0.001},
      {"call": "cosine_lr(50, 100, 0.1)", "expected": 0.0, "tolerance": 0.001},
      {"call": "cosine_lr(100, 100, 0.1)", "expected": 0.1, "tolerance": 0.001}
    ],
    "hints": ["lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*epoch/total))"],
    "explanation": "Cosine annealing schedule.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You're training a transformer and notice loss spikes at the start then stabilizes. Explain the warmup phenomenon and why it's needed for Adam/transformers specifically.",
    "rubric": [
      "Adam's adaptive lr needs warmup to avoid large early updates with unreliable gradient estimates",
          "Warmup linearly increases lr over first N steps",
      "Prevents early divergence and improves final convergence",
      "Standard practice: ~2000-4000 warmup steps for large transformers"
        ],
    "points": 2
  }
]
```

## Referensi

- Ruder (2016) 'An overview of gradient descent optimization algorithms'
- Kingma & Ba (2015) 'Adam: A Method for Stochastic Optimization'
- Goodfellow et al. ch. 8
- Loshchilov & Hutter (2017) 'Decoupled Weight Decay Regularization' (AdamW)
