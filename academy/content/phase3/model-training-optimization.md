---
{
  "id": "p3-model-training-optimization",
  "title": "Model Training and Optimization",
  "phase": 3,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Design effective training pipelines",
    "Apply regularization techniques: dropout, weight decay, early stopping",
    "Use learning rate schedules and warmup",
    "Diagnose and fix common training problems"
  ],
  "concepts": [
    "Dropout: random neuron deactivation",
    "Weight decay (L2 regularization)",
    "Early stopping and patience",
    "Batch normalization and layer normalization",
    "Gradient clipping",
    "Mixed precision training",
    "Data augmentation strategies",
    "Learning rate warmup and schedules",
    "Training diagnostics: loss curves, gradient norms"
  ],
  "formulas": [
    {"name": "Dropout", "latex": "h_i = \\begin{cases} 0 & \\text{prob } p \\\\ \\frac{h_i}{1-p} & \\text{else} \\end{cases}", "note": "Inverted dropout keeps expected value."},
    {"name": "Weight decay", "latex": "L_{total} = L_{task} + \\lambda \\|\\theta\\|^2", "note": "L2 penalty on weights."},
    {"name": "Early stopping", "latex": "\\text{stop if } \\min_{t-k..t} L_{val} \\text{ not improved for } p \\text{ epochs}", "note": "Patience-based stopping."}
  ],
  "references": [
    "Srivastava et al. 2014 'Dropout: A Simple Way to Prevent Overfitting'",
    "Ioffe & Szegedy 2015 'Batch Normalization'",
    "Goodfellow et al. ch. 7 (regularization)",
    "PyTorch docs — optim, lr_scheduler"
  ]
}
---

## Penjelasan

Training neural network yang efektif membutuhkan lebih dari sekadar "compile dan jalankan". Regularization, optimization schedule, dan diagnostics menentukan apakah model generalisasi dengan baik.

**Regularization techniques.**
- *Dropout*: saat training, randomly set activations ke 0 dengan probability p; scale sisanya dengan 1/(1-p) (inverted dropout). Mencegah co-adaptation neuron; ensemble effect. p=0.5 untuk hidden layers klasik; p=0.1-0.2 untuk modern deep nets.
- *Weight decay (L2)*: tambah λ||θ||² ke loss; mendorong bobot kecil -> smoother function. Di AdamW, decoupled weight decay lebih baik.
- *Early stopping*: monitor validation loss; stop jika tidak membaik selama `patience` epochs; restore best weights.
- *Data augmentation*: transformasi input (flip, crop, noise) -> model melihat variasi -> generalisasi lebih baik.
- *BatchNorm/LayerNorm*: normalisasi activations -> stabilkan training, memungkinkan lr lebih besar, regularisasi implisit.

**Learning rate schedules.**
- *Warmup*: naikkan lr linearly dari 0 ke max selama N steps — mencegah early divergence dengan gradient estimates yang belum stabil.
- *Cosine annealing*: lr menurun smooth ke 0 mengikuti cosinus — standar untuk vision/transformer.
- *Step decay*: turunkan lr by factor per N epochs — klasik CNN.
- *OneCycle*: naik-turun sekali — cepat konvergen.

**Gradient clipping.** Batasi norm gradient: if ||g|| > clip, scale g. Mencegah exploding gradients di RNN/transformer; wajib untuk training stabil.

**Mixed precision.** Pakai float16 untuk forward/backward, float32 untuk update — 2x faster di GPU modern, setengah memory. `torch.cuda.amp.autocast()` + `GradScaler`.

**Diagnostics.** Plot train/val loss per epoch: diverging = lr terlalu tinggi; plateau = lr terlalu rendah atau model stuck; val loss naik sementara train turun = overfitting. Gradient norms: exploding = clipping; vanishing = arsitektur/aktivasi bermasalah.

## Konsep Penting

1. **Regularization stack** — dropout + weight decay + augmentation + early stopping bekerja bersama; satu saja sering tidak cukup.
2. **lr schedule > fixed lr** — cosine/warmup standar untuk modern training.
3. **BatchNorm interplay** — BN punya regularisasi implisit; menambah dropout + BN bisa over-regularize.
4. **Early stopping butuh patience** — terlalu kecil = premature stop; terlalu besar = wasted compute.
5. **Mixed precision butuh GradScaler** — loss scaling mencegah gradient underflow di fp16.

## Formula

- Dropout inverted: h/(1-p).
- Weight decay: L += λ||θ||².
- Grad clipping: g *= clip_norm/||g|| if ||g||>clip_norm.
- Cosine: η_t = η_min + 0.5(η_max-η_min)(1+cos(πt/T)).

## Contoh Kasus

Training ResNet-50 pada ImageNet: SGD momentum 0.9, lr=0.1 batch 256, step decay /10 per 30 epochs, weight decay 1e-4, label smoothing 0.1, augmentation (crop+flip+color jitter). 90 epochs -> 76% top-1. Bandingkan dengan Adam lr=1e-3 — konvergen lebih cepat tapi akurasi akhir lebih rendah.

## Implementasi

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def train_with_regularization(model, train_loader, val_loader, epochs=50,
                              lr=1e-3, weight_decay=0.01, patience=10):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # warmup + cosine
    warmup = LinearLR(opt, start_factor=0.01, total_iters=5)
    cosine = CosineAnnealingLR(opt, T_max=epochs-5)
    sched = SequentialLR(opt, [warmup, cosine], milestones=[5])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            opt.step()
        sched.step()

        # validation + early stopping
        model.eval()
        val_loss = sum(criterion(model(xb), yb).item()
                       for xb, yb in val_loader) / len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Dropout prevents overfitting by:",
    "options": [
      "Randomly deactivating neurons during training, preventing co-adaptation",
      "Reducing learning rate",
      "Adding noise to labels",
      "Removing features"
    ],
    "answer": 0,
    "explanation": "Dropout forces neurons to be robust individually; ensemble effect at test time.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Learning rate warmup is important because:",
    "options": [
      "Early gradient estimates are unreliable; large lr causes divergence",
      "It speeds up training",
      "It reduces model size",
      "It's required by Adam"
    ],
    "answer": 0,
    "explanation": "Warmup linearly increases lr from 0 to max, stabilizing early training.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Early stopping monitors:",
    "options": [
      "Validation loss/metric and stops when it stops improving",
      "Training loss",
      "Gradient norm",
      "Learning rate"
    ],
    "answer": 0,
    "explanation": "Early stopping uses validation performance to prevent overfitting.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `early_stopping_check(val_loss, best_val, wait, patience)` returning (should_stop, new_best_val, new_wait).",
    "starter_code": "def early_stopping_check(val_loss, best_val, wait, patience):\n    pass\n",
    "function": "early_stopping_check",
    "tests": [
      {"call": "early_stopping_check(0.5, 0.6, 0, 5)", "expected": [false, 0.5, 0]},
      {"call": "early_stopping_check(0.7, 0.5, 4, 5)", "expected": [false, 0.5, 5]},
      {"call": "early_stopping_check(0.8, 0.5, 5, 5)", "expected": [true, 0.5, 6]}
    ],
    "hints": ["If val_loss < best_val: update best, reset wait; else increment wait; stop if wait >= patience", "return (stop, best_val, wait)"],
    "explanation": "Early stopping logic.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your training loss decreases steadily but validation loss increases after epoch 20. Diagnose the problem and propose a prioritized list of fixes to try.",
    "rubric": [
      "Identifies overfitting (train improves, val worsens)",
          "Proposes: early stopping, more dropout, weight decay, data augmentation",
      "Or simpler model / fewer parameters",
      "Or more training data / regularization"
        ],
    "points": 2
  }
]
```

## Referensi

- Srivastava et al. (2014) 'Dropout: A Simple Way to Prevent Overfitting'
- Ioffe & Szegedy (2015) 'Batch Normalization'
- Loshchilov & Hutter (2017) 'SGDR: Stochastic Gradient Descent with Warm Restarts'
- PyTorch docs — nn.Dropout, optim, lr_scheduler
