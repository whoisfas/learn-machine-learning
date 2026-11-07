---
{
  "id": "p3-pytorch-fundamentals",
  "title": "PyTorch Fundamentals",
  "phase": 3,
  "difficulty": "intermediate",
  "estimated_hours": 12,
  "objectives": [
    "Work with tensors, autograd, and nn.Module",
    "Build and train models using PyTorch idioms",
    "Use DataLoader for batching and shuffling",
    "Understand device management (CPU/GPU)"
  ],
  "concepts": [
    "Tensors: creation, operations, broadcasting",
    "Autograd: requires_grad, backward(), computational graph",
    "nn.Module: layers, forward(), parameters()",
    "nn.Sequential vs custom Module",
    "Dataset and DataLoader",
    "Loss functions and optimizers",
    "Training loop structure",
    "Device management: .to(device), .cuda(), .cpu()"
  ],
  "formulas": [],
  "references": [
    "PyTorch official tutorials (pytorch.org/tutorials)",
    "PyTorch nn documentation",
    "Goodfellow et al. (conceptual)"
  ]
}
---

## Penjelasan

PyTorch adalah framework deep learning paling populer untuk research. Desainnya imperative (eager execution): kode berjalan langsung, debug dengan Python biasa, dan autograd membangun computational graph secara otomatis.

**Tensors.** Tensor = array N-dimensi (seperti NumPy) dengan kemampuan GPU acceleration dan autograd tracking. `torch.tensor([[1,2],[3,4]])`, `torch.zeros`, `torch.randn`, `torch.arange`. Operasi mirip NumPy: `+`, `*`, `@`, `.sum()`, `.mean()`. Broadcasting rules sama dengan NumPy.

**Autograd.** Setiap tensor dengan `requires_grad=True` melacak operasi yang menghasilkannya. `loss.backward()` menghitung gradient semua parameter yang terlibat — inilah backpropagation otomatis. `tensor.grad` berisi gradient; `optimizer.zero_grad()` membersihkan sebelum iterasi baru.

**nn.Module.** Base class untuk semua model. Definisikan layers di `__init__`, computation di `forward()`. `model.parameters()` mengembalikan semua bobot untuk optimizer. `model.train()` / `model.eval()` mengontrol dropout/batchnorm behavior.

**Dataset & DataLoader.** `Dataset` class: implement `__len__` dan `__getitem__`. `DataLoader` membungkus dataset: batching, shuffling, num_workers untuk parallel loading, pin_memory untuk GPU transfer.

**Training loop.** Pola standar:
```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    # validation
    model.eval()
    with torch.no_grad():
        ...
```

**Device management.** `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Semua tensor dan model harus `.to(device)`. `torch.no_grad()` untuk inference (hemat memory, lebih cepat).

## Konsep Penting

1. **Eager vs graph** — PyTorch eager execution memudahkan debug; TorchScript/JIT untuk production.
2. **In-place operations** — `x += 1` vs `x = x + 1` berbeda untuk autograd; hati-hati dengan in-place pada tensor yang butuh grad.
3. **zero_grad wajib** — gradient terakumulasi; lupa zero_grad = gradient dari iterasi sebelumnya ikut terbawa.
4. **model.eval() vs train()** — dropout dan batchnorm berperilaku berbeda; lupa eval() -> dropout aktif saat inference.
5. **DataLoader workers** — num_workers>0 parallelize data loading; terlalu banyak bisa bottleneck.

## Formula

- Tensor shape conventions: (batch, channels, height, width) untuk gambar; (batch, seq_len, features) untuk sequence.
- CrossEntropyLoss expects logits, not probabilities.

## Contoh Kasus

Training MLP pada digits dataset: Dataset class wrap numpy arrays -> DataLoader batch 64 -> model MLP -> CrossEntropyLoss -> Adam -> 10 epochs. ~98% accuracy dalam hitungan menit di CPU.

## Implementasi

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim))
    def forward(self, x): return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(NumpyDataset(X_train, y_train),
                          batch_size=64, shuffle=True, num_workers=2)
model = MLP(64, 10).to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"epoch {epoch}: loss={total_loss/len(train_loader):.4f}")
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "What does loss.backward() do in PyTorch?",
    "options": [
      "Computes gradients of loss w.r.t. all tensors with requires_grad=True via autograd",
      "Updates model parameters",
      "Forward pass",
      "Clears gradients"
    ],
    "answer": 0,
    "explanation": "backward() traverses computational graph computing gradients; optimizer.step() applies them.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Why must you call optimizer.zero_grad() before loss.backward()?",
    "options": [
      "PyTorch accumulates gradients by default; zero_grad clears them",
      "To speed up training",
      "To reset the model",
      "It's optional"
    ],
    "answer": 0,
    "explanation": "Gradients accumulate in .grad; without zeroing, old gradients contaminate new updates.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "model.eval() vs model.train() differ because:",
    "options": [
      "Dropout and BatchNorm behave differently (no dropout, use running stats)",
      "eval() freezes parameters",
      "train() is faster",
      "eval() uses less memory"
    ],
    "answer": 0,
    "explanation": "eval() disables dropout and uses running statistics in BN — critical for correct inference.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `manual_epoch(model, loader, opt, criterion, device)` performing one training epoch and returning mean loss.",
    "starter_code": "import torch\n\ndef manual_epoch(model, loader, opt, criterion, device):\n    model.train()\n    total_loss = 0.0\n    count = 0\n    pass\n    return total_loss / max(count, 1)\n",
    "function": "manual_epoch",
    "tests": [
      {"call": "manual_epoch(torch.nn.Linear(2,2), [(torch.randn(4,2),torch.randint(0,2,(4,)))], torch.optim.SGD(torch.nn.Linear(2,2).parameters(),0.01), torch.nn.CrossEntropyLoss(), 'cpu')", "expected": "check"}
    ],
    "hints": ["Loop batches; xb,yb = batch; move to device; zero_grad; forward; loss; backward; step; accumulate"],
    "explanation": "Standard training epoch loop.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Explain the difference between `requires_grad=True` tensors and regular tensors in PyTorch. When would you NOT want gradient tracking? Give two examples.",
    "rubric": [
      "requires_grad tracks operations for autograd; regular tensors don't",
          "Inference/prediction: no_grad() saves memory and speeds up",
      "Data preprocessing: no need for gradients",
      "Freezing layers: requires_grad=False for fine-tuning"
        ],
    "points": 2
  }
]
```

## Referensi

- PyTorch official tutorials (pytorch.org/tutorials)
- PyTorch nn documentation
- 'Deep Learning with PyTorch' by Stevens, Antiga, Viehmann
- PyTorch forums for common issues
