---
{
  "id": "p3-forward-propagation",
  "title": "Forward Propagation",
  "phase": 3,
  "difficulty": "intermediate",
  "estimated_hours": 8,
  "objectives": [
    "Trace data flow through a neural network layer by layer",
    "Compute activations manually for small networks",
    "Understand the role of weights, biases, and activations",
    "Implement forward pass in NumPy and PyTorch"
  ],
  "concepts": [
    "Affine transformation: z = Wx + b",
    "Activation application: a = σ(z)",
    "Layer-by-layer computation",
    "Batch processing and vectorization",
    "Computational graph (forward direction)",
    "Output layer: logits vs probabilities (softmax)",
    "Shape tracking through layers"
  ],
  "formulas": [
    {"name": "Affine transform", "latex": "z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}", "note": "Linear part of layer l."},
    {"name": "Activation", "latex": "a^{(l)} = \\sigma(z^{(l)})", "note": "Nonlinear part."},
    {"name": "Full forward pass", "latex": "a^{(L)} = \\sigma_L(W_L \\cdots \\sigma_1(W_1 x + b_1) \\cdots + b_L)", "note": "Composition through all layers."}
  ],
  "references": [
    "Goodfellow et al. ch. 6.4",
    "PyTorch nn.Module documentation",
    "3Blue1Brown neural network series"
  ]
}
---

## Penjelasan

Forward propagation adalah proses data mengalir dari input ke output melalui neural network. Setiap layer melakukan dua operasi: transformasi affine (perkalian matriks + bias) lalu aktivasi non-linear.

**Layer computation.** Untuk layer l dengan input a^(l−1):
1. z^(l) = W^(l) · a^(l−1) + b^(l) — affine transform (linear).
2. a^(l) = σ(z^(l)) — aktivasi (nonlinear).

Output layer biasanya tidak pakai aktivasi tersembunyi: untuk klasifikasi, z^(L) = logits -> softmax -> probabilitas; untuk regresi, output langsung.

**Shape tracking.** Input x shape (batch_size, input_dim). Layer Linear(d_in, d_out): W shape (d_out, d_in) di PyTorch (atau (d_in, d_out) di NumPy manual), b shape (d_out). Output z shape (batch_size, d_out). Ini penting untuk debugging shape mismatch.

**Batch processing.** Neural network memproses seluruh batch sekaligus via matrix multiplication — GPU parallelism. x shape (n, d) @ W.T shape (d, m) -> (n, m).

**Computational graph.** Forward pass membangun graph komputasi: setiap operasi mencatat input dan outputnya. Backpropagation menggunakan graph ini untuk menghitung gradient mundur. PyTorch autograd membangun graph secara otomatis saat tensor operations dieksekusi dengan `requires_grad=True`.

**Numerical example.** Untuk x=[1,2], W=[[0.5,0.3],[0.2,0.7]], b=[0.1,-0.2], ReLU:
- z1 = x@W.T + b = [1·0.5+2·0.2+0.1, 1·0.3+2·0.7-0.2] = [0.6, 1.5]
- a1 = ReLU(z1) = [0.6, 1.5]
- (continue to next layer...)

## Konsep Penting

1. **Shape discipline** — setiap layer transformasi shape secara predictable; mismatch = bug.
2. **Vectorization** — proses seluruh batch sekaligus; loop per sample sangat lambat.
3. **Logits vs probabilities** — output layer menghasilkan logits; softmax untuk probabilitas; cross-entropy bisa digabung dengan softmax (log_softmax + NLL).
4. **Deterministic vs stochastic** — dropout menambahkan randomness saat training tapi identity saat eval (`model.eval()`).
5. **Forward = inference** — setelah training, hanya forward pass yang diperlukan untuk prediksi.

## Formula

- z = Wx + b; a = σ(z).
- Softmax: p_i = e^(zi) / Σ e^(zj).
- Batch: X (n,d) @ W.T (d,m) -> Z (n,m).

## Contoh Kasus

MLP 784->256->128->10 untuk MNIST: input flatten 28x28 -> 784; layer1: 784->256 ReLU; layer2: 256->128 ReLU; layer3: 128->10 logits; softmax -> probabilitas 10 kelas. Forward pass untuk batch 64 gambar: tensor operations di GPU.

## Implementasi

```python
import numpy as np

def relu(z): return np.maximum(0, z)
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def mlp_forward_numpy(X, params):
    """2-hidden-layer MLP forward pass."""
    a = X
    for i in range(1, 3):  # hidden layers
        z = a @ params[f'W{i}'].T + params[f'b{i}']
        a = relu(z)
    # output layer
    logits = a @ params['W3'].T + params['b3']
    return softmax(logits)

# PyTorch equivalent
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # logits
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "In forward propagation, each layer performs:",
    "options": [
      "Affine transform followed by activation function",
      "Only activation function",
      "Only matrix multiplication",
      "Gradient computation"
    ],
    "answer": 0,
    "explanation": "z = Wx + b (affine), a = σ(z) (activation).",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "For a batch of 32 samples through Linear(784, 256), output shape is:",
    "options": ["(32, 256)", "(256, 32)", "(32, 784)", "(784, 256)"],
    "answer": 0,
    "explanation": "Batch dim preserved; feature dim transforms to output dim.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `softmax(z)` for 2D array (batch, classes) with numerical stability.",
    "starter_code": "import numpy as np\n\ndef softmax(z):\n    z = np.asarray(z, dtype=float)\n    pass\n",
    "function": "softmax",
    "tests": [
      {"call": "softmax([[1,2,3]]).tolist()", "expected": [[0.09003057317038046,0.24472847105479764,0.6652409557748219]], "tolerance": 1e-6},
      {"call": "softmax([[0,0]]).tolist()", "expected": [[0.5,0.5]], "tolerance": 1e-9}
    ],
    "hints": ["Subtract max for stability; exp then normalize per row"],
    "explanation": "Softmax converts logits to probabilities.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `forward_2layer(X, W1, b1, W2, b2)` for MLP: ReLU(X@W1+b1) then sigmoid(X2@W2+b2). Return output.",
    "starter_code": "import numpy as np\n\ndef forward_2layer(X, W1, b1, W2, b2):\n    X = np.asarray(X, dtype=float)\n    pass\n",
    "function": "forward_2layer",
    "tests": [
      {"call": "forward_2layer([[1,0]], [[1,0],[0,1]],[0,0], [[1,1]],[0]).tolist()", "expected": [[0.7310585786300049]], "tolerance": 0.001}
    ],
    "hints": ["h = relu(X@W1.T+b1) or X@W1+b1 depending on convention; z2 = h@W2+b2; return 1/(1+exp(-z2))"],
    "explanation": "Manual 2-layer forward pass.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Trace a single image through a CNN: input 28x28x1 -> conv -> pool -> conv -> pool -> flatten -> FC -> softmax. Describe what happens to spatial dimensions and channel counts at each stage.",
    "rubric": [
      "Conv preserves spatial dims (with padding) or reduces (valid); channels increase",
          "Pool reduces spatial dims by factor 2",
      "Flatten converts spatial to vector; FC outputs class logits",
      "Mentions parameter sharing in conv layers"
        ],
    "points": 2
  }
]
```

## Referensi

- Goodfellow et al. ch. 6.4 (feedforward networks)
- PyTorch nn.Linear, nn.ReLU documentation
- 3Blue1Brown 'But what is a neural network?' series
- NumPy broadcasting for manual implementation
