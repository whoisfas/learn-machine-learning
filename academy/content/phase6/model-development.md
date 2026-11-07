---
{
  "id": "p6-model-development",
  "title": "Model Development for Research",
  "phase": 6,
  "difficulty": "advanced",
  "estimated_hours": 12,
  "objectives": [
    "Select appropriate model families for research problems",
    "Implement model variants and baselines",
    "Manage training: hyperparameters, early stopping, checkpoints",
    "Balance model complexity with generalization"
  ],
  "concepts": [
    "Model selection: matching architecture to problem",
    "Baseline establishment: simple models first",
    "Training loop: forward, loss, backward, update",
    "Early stopping and learning rate scheduling",
    "Checkpoint management for long training",
    "Regularization: dropout, weight decay, data augmentation",
    "Model complexity vs generalization tradeoff",
    "Ensemble methods for research"
  ],
  "formulas": [],
  "references": [
    "Phase 2-4: all model-related modules",
    "'Deep Learning' by Goodfellow et al.",
    "PyTorch documentation"
  ]
}
---

## Penjelasan

Model development adalah fase di mana Anda mengimplementasikan dan melatih model untuk research question Anda. Prinsip: mulai dari baseline sederhana, iterasi berdasarkan hasil, dan dokumentasikan setiap keputusan.

**Model selection.** Pilih model family berdasarkan problem characteristics: tabular data -> gradient boosting (XGBoost, LightGBM); image -> CNN/ViT; text -> transformer; time series -> LSTM/transformer/ARIMA; small data -> simple models + regularization.

**Baseline first.** Sebelum model canggih, train baseline: majority class, linear regression, simple neural net. Baseline memberi referensi — improvement harus signifikan terhadap baseline, bukan hanya absolut.

**Training loop.** Iterasi: forward pass -> loss computation -> backward pass -> parameter update. Monitor: training loss, validation loss/metrics, learning rate, gradient norms. Early stopping ketika validation metric tidak improve untuk N epochs.

**Hyperparameter management.** Jangan tune manual satu-satu — gunakan grid search, random search, atau Bayesian optimization (Phase 4). Dokumentasikan semua trials — bukan hanya yang terbaik.

**Regularization.** Untuk mencegah overfitting: dropout, weight decay, data augmentation, early stopping, batch normalization. Untuk underfitting: model lebih besar, train lebih lama, less regularization.

**Checkpoint management.** Save model setiap improvement validation metric; restore best checkpoint untuk final evaluation. Untuk long training: save periodically, resume dari checkpoint jika crash.

## Konsep Penting

1. **Baseline is mandatory** — tanpa baseline, improvement tidak meaningful.
2. **Start simple, iterate** — complex model hanya jika simple tidak cukup.
3. **Document every decision** — hyperparameters, seeds, preprocessing, results.
4. **Monitor training dynamics** — loss curves, gradient norms, learning rate.
5. **Generalization > training performance** — validation/test metrics adalah truth.

## Implementasi

```python
import torch
import torch.nn as nn
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer,
                n_epochs=50, patience=10, device='cpu'):
    """Training loop with early stopping and checkpointing."""
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(n_epochs):
        # training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
                correct += (output.argmax(1) == y).sum().item()
                total += y.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Why establish a baseline before training complex models?",
    "options": [
      "To measure whether improvements are meaningful relative to simple approaches",
      "To save time",
      "To reduce memory",
      "Baselines are optional"
    ],
    "answer": 0,
    "explanation": "Baseline provides reference point; improvement must be significant vs baseline.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Early stopping prevents:",
    "options": [
      "Overfitting by halting training when validation metric plateaus",
      "Underfitting",
      "Slow training",
      "Memory issues"
    ],
    "answer": 0,
    "explanation": "Early stopping = regularization via training duration.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The training loop consists of:",
    "options": [
      "Forward pass, loss computation, backward pass, parameter update",
      "Data loading, visualization, reporting",
      "Split, preprocess, evaluate",
      "Download, train, publish"
    ],
    "answer": 0,
    "explanation": "Core ML training iteration.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `compute_grad_norm(model)` returning total gradient norm across all parameters.",
    "starter_code": "import torch\n\ndef compute_grad_norm(model):\n    total_norm = 0.0\n    for p in model.parameters():\n        if p.grad is not None:\n            total_norm += p.grad.data.norm(2).item() ** 2\n    return total_norm ** 0.5\n",
    "function": "compute_grad_norm",
    "tests": [
      {"call": "compute_grad_norm(torch.nn.Linear(2,1))", "expected": 0.0}
    ],
    "hints": ["Sum squared norms of all parameter gradients; return sqrt"],
    "explanation": "Gradient norm monitoring for training stability.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your model achieves 95% training accuracy but 70% validation accuracy. Diagnose the problem and describe three specific interventions.",
    "rubric": [
      "Overfitting: model memorizes training data",
          "Add regularization: dropout, weight decay, data augmentation",
      "Reduce model complexity or increase data",
      "Early stopping or reduce training epochs"
        ],
    "points": 2
  }
]
```

## Referensi

- Phase 2-4 modules: all model-related content
- Goodfellow et al., 'Deep Learning'
- PyTorch documentation
