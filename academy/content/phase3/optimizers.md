---
{
  "id": "p3-optimizers",
  "title": "Optimization Algorithms",
  "phase": 3,
  "difficulty": "advanced",
  "estimated_hours": 12,
  "objectives": [
    "Compare SGD, momentum, AdaGrad, RMSProp, Adam, AdamW",
    "Understand adaptive learning rate mechanisms",
    "Choose appropriate optimizer for different architectures",
    "Implement custom optimizers and learning rate schedules"
  ],
  "concepts": [
    "SGD with momentum (heavy ball, Nesterov)",
    "AdaGrad: per-parameter adaptive lr",
    "RMSProp: exponential moving average of squared gradients",
    "Adam: momentum + RMSProp + bias correction",
    "AdamW: decoupled weight decay",
    "LAMB, LARS for large batch training",
    "Learning rate warmup and decay schedules",
    "Optimizer state and memory requirements"
  ],
  "formulas": [
    {"name": "SGD+momentum", "latex": "v = \\beta v + \\nabla L; \\theta -= \\eta v", "note": "Heavy ball; β=0.9 typical."},
    {"name": "Nesterov", "latex": "v = \\beta v + \\nabla L(\\theta - \\eta \\beta v); \\theta -= \\eta v", "note": "Look-ahead gradient."},
    {"name": "Adam update", "latex": "\\theta -= \\eta \\frac{\\hat{m}}{\\sqrt{\\hat{v}}+\\epsilon}", "note": "m=momentum, v=RMSProp estimates."},
    {"name": "AdamW", "latex": "\\theta -= \\eta(\\frac{\\hat{m}}{\\sqrt{\\hat{v}}+\\epsilon} + \\lambda\\theta)", "note": "Decoupled weight decay."}
  ],
  "references": [
    "Kingma & Ba (2015) 'Adam: A Method for Stochastic Optimization'",
    "Loshchilov & Hutter (2019) 'Decoupled Weight Decay Regularization'",
    "PyTorch optim documentation"
  ]
}
---

## Penjelasan

Optimizers adalah algoritma yang menentukan bagaimana parameter diperbarui berdasarkan gradient. Pilihan optimizer sangat memengaruhi kecepatan konvergensi dan generalisasi akhir.

**SGD + Momentum.** Vanilla SGD lambat di valley sempit. Momentum menambahkan velocity: v = βv + ∇L; θ -= ηv. β=0.9 berarti velocity mengingat 90% iterasi sebelumnya -> mempercepat di arah konsisten, meredam zigzag. Nesterov momentum "melihat ke depan": menghitung gradient di posisi yang diprediksi, bukan posisi saat ini.

**AdaGrad.** Menyesuaikan lr per parameter berdasarkan frekuensi historis: G += g²; θ -= η·g/(√G+ε). Parameter yang sering update -> lr menurun; jarang update -> lr tetap. Masalah: G tumbuh monoton -> lr -> 0 terlalu cepat untuk training panjang.

**RMSProp.** Perbaikan AdaGrad: gunakan exponential moving average G = βG + (1-β)g² alih-alih akumulasi total — mencegah lr -> 0. β=0.9 atau 0.999.

**Adam.** Menggabungkan momentum (first moment m) + RMSProp (second moment v) + bias correction:
- m = β1·m + (1-β1)·g; v = β2·v + (1-β2)·g²
- m̂ = m/(1-β1^t); v̂ = v/(1-β2^t) — koreksi bias inisialisasi
- θ -= η·m̂/(√v̂ + ε)
Default: β1=0.9, β2=0.999, ε=1e-8, lr=0.001. Adam adalah default untuk kebanyakan deep learning tasks.

**AdamW.** Memisahkan weight decay dari adaptive lr: weight decay di-apply langsung ke parameter (decoupled), bukan sebagai gradient tambahan. Lebih baik untuk generalisasi di transformers dan vision models.

**Kapan pakai apa.**
- Adam/AdamW: default aman untuk MLP, transformer, NLP, tabular.
- SGD+momentum: sering lebih baik untuk CNN vision dengan lr schedule yang tepat; generalisasi lebih baik tapi butuh tuning.
- AdaGrad/RMSProp: historis penting; jarang dipakai langsung sekarang.

## Konsep Penting

1. **Adam bukan selalu terbaik** — untuk vision tasks, SGD+momentum dengan cosine schedule sering mengalahkan Adam pada generalisasi.
2. **lr schedule interaksi** — Adam lr=0.001 dengan cosine decay; SGD lr=0.1 dengan step decay.
3. **Weight decay ≠ L2 di Adam** — di Adam, L2 regularization dan weight decay berbeda; AdamW memisahkan dengan benar.
4. **Memory cost** — Adam menyimpan 2 state per parameter (m, v); untuk model besar ini signifikan.
5. **Batch size interaction** — lr harus diskalakan dengan batch size (linear scaling rule untuk SGD).

## Formula

- Adam bias correction: m̂ = m/(1-β1^t), v̂ = v/(1-β2^t).
- Effective lr per param: η/√v̂ — adaptif per parameter.
- LAMB/LARS: layer-wise adaptive lr untuk large batch.

## Contoh Kasus

Training BERT-base: AdamW lr=1e-4, weight_decay=0.01, warmup 10k steps, cosine decay, batch 256. Bandingkan dengan SGD lr=0.1 momentum — SGD konvergen lebih lambat dan hasil akhir sering lebih buruk untuk NLP.

## Implementasi

```python
import torch
import torch.nn as nn

# optimizer comparison on same model
def get_optimizer(name, model):
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    elif name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=1e-3)
    elif name == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=0.01)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
# or warmup + cosine
from torch.optim.lr_scheduler import LambdaLR
def lr_lambda(step):
    if step < warmup: return step/warmup
    return 0.5*(1+np.cos(np.pi*(step-warmup)/(total-warmup)))
scheduler = LambdaLR(opt, lr_lambda)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Adam combines which two ideas?",
    "options": [
      "Momentum (first moment) + RMSProp (second moment adaptive lr)",
      "SGD + batch normalization",
      "Dropout + weight decay",
      "AdaGrad + Nesterov"
    ],
    "answer": 0,
    "explanation": "Adam = momentum estimate m + RMSProp variance estimate v + bias correction.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Why does AdamW often outperform Adam for transformers?",
    "options": [
      "Decoupled weight decay works better with adaptive learning rates",
      "It uses a larger learning rate",
      "It has fewer parameters",
      "It converges faster"
    ],
    "answer": 0,
    "explanation": "In Adam, L2 penalty interacts with adaptive lr; AdamW applies decay directly to weights.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Momentum helps optimization by:",
    "options": [
      "Accumulating velocity in consistent directions, reducing zigzag",
      "Increasing the learning rate",
      "Adding noise to gradients",
      "Reducing model size"
    ],
    "answer": 0,
    "explanation": "Momentum smooths gradient noise and accelerates in consistent directions.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `adam_step(params, grads, m, v, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8)` performing one Adam update. Return updated params, m, v.",
    "starter_code": "import numpy as np\n\ndef adam_step(params, grads, m, v, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):\n    new_params, new_m, new_v = {}, {}, {}\n    for k in params:\n        pass\n    return new_params, new_m, new_v\n",
    "function": "adam_step",
    "tests": [
      {"call": "p,m,v=adam_step({'w':np.array([1.0])},{'w':np.array([0.5])},{'w':np.array([0.0])},{'w':np.array([0.0])},1,0.1); p['w'].tolist()", "expected": [0.9], "tolerance": 0.01}
    ],
    "hints": ["m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g^2; m_hat=m/(1-b1^t); v_hat=v/(1-b2^t); p -= lr*m_hat/(sqrt(v_hat)+eps)"],
    "explanation": "Core Adam update with bias correction.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You train a CNN on CIFAR-10: Adam converges fast but test accuracy plateaus at 85%; SGD+momentum with cosine schedule reaches 90% but takes 3x longer. Explain the generalization gap and when you'd choose each optimizer.",
    "rubric": [
      "Adam's adaptive lr may find sharper minima; SGD finds flatter minima that generalize better",
          "SGD+momentum with proper schedule often better for vision",
      "Adam better for quick experiments, transformers, NLP",
      "Mentions weight decay interaction and lr schedule importance"
        ],
    "points": 2
  }
]
```

## Referensi

- Kingma & Ba (2015) 'Adam: A Method for Stochastic Optimization'
- Loshchilov & Hutter (2019) 'Decoupled Weight Decay Regularization' (AdamW)
- Ruder (2016) 'An overview of gradient descent optimization algorithms'
- PyTorch optim documentation
