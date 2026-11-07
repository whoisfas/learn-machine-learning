---
{
  "id": "p3-neural-networks",
  "title": "Neural Networks: Foundations",
  "phase": 3,
  "difficulty": "intermediate",
  "estimated_hours": 14,
  "objectives": [
    "Understand the perceptron-to-MLP progression",
    "Explain universal approximation and depth vs width",
    "Trace forward propagation through layers",
    "Recognize the role of nonlinearity"
  ],
  "concepts": [
    "Perceptron and its limitations (XOR problem)",
    "Multi-layer perceptron (MLP) architecture",
    "Linear vs nonlinear transformations",
    "Universal approximation theorem",
    "Depth vs width tradeoffs",
    "Parameter counting and model capacity",
    "Representation learning",
    "Comparison with kernel methods and trees"
  ],
  "formulas": [
    {"name": "MLP forward pass", "latex": "h^{(l)} = \\sigma(W^{(l)} h^{(l-1)} + b^{(l)})", "note": "Layer l: affine transform + activation."},
    {"name": "Universal approximation", "latex": "\\forall f \\exists \\text{MLP}: |f - \\text{MLP}| < \\epsilon", "note": "One hidden layer suffices in theory; depth helps efficiency."}
  ],
  "references": [
    "Goodfellow et al., 'Deep Learning' ch. 6 (free)",
    "Rumelhart et al. 1986 backpropagation paper",
    "Cybenko 1989 universal approximation"
  ]
}
---

## Penjelasan

Neural network adalah model yang terdiri dari lapisan-lapisan transformasi linear diikuti fungsi non-linear. Mulai dari perceptron sederhana (1943) hingga deep network modern (ResNet, Transformer), inti arsitekturnya sama: belajar representasi hierarkis dari data.

**Dari perceptron ke MLP.** Perceptron = satu neuron: output = step(w·x + b). Hanya bisa memisahkan data linearly (tidak bisa XOR). MLP menumpuk neuron dalam hidden layers dengan aktivasi non-linear, memungkinkan aproksimasi fungsi arbitrer. Universal approximation theorem: satu hidden layer dengan aktivasi non-linear cukup untuk aproksimasi fungsi kontinu apa pun — dalam praktik, depth (lebih banyak layer) lebih efisien untuk data terstruktur.

**Forward propagation.** Input x -> hidden layers -> output. Setiap layer: z = Wx + b (affine), a = σ(z) (aktivasi). Komposisi: f(x) = σ_L(W_L · σ_{L-1}(...σ_1(W_1 x + b_1)...)). Kedalaman memungkinkan hierarki fitur: layer awal deteksi edges, layer tengah shapes, layer akhir semantic concepts.

**Parameter counting.** Layer dense d_in -> d_out punya d_in·d_out + d_out parameter (weights + biases). Contoh: 784->256->128->10 = 784·256 + 256·128 + 128·10 + biases ≈ 235k parameters.

**Kapasitas vs generalisasi.** Model besar bisa menghafal data (overfit) tapi dengan regularization yang tepat (dropout, weight decay, data augmentation, early stopping) justru generalisasi lebih baik. Deep learning "double descent": performa test bisa membaik lagi setelah melewati regime overfitting klasik.

**Representasi belajar.** Keunggulan utama deep learning vs model klasik: fitur dipelajari dari data, bukan direkayasa manual. CNN belajar edge detectors; embeddings belajar semantic similarity; autoencoder belajar compressed representation.

## Konsep Penting

1. **Nonlinearitas esensial** — tanpa aktivasi non-linear, komposisi layer linear = linear; tidak lebih kuat dari linear regression.
2. **Depth vs width** — depth lebih efisien untuk compositional functions; width lebih mudah dioptimasi.
3. **Symmetry breaking** — inisialisasi acak penting; bobot identik = gradient identik = tidak belajar.
4. **Scale matters** — neural network sensitif terhadap skala input; normalisasi (BatchNorm, LayerNorm) stabilkan training.
5. **NN vs kernel methods** — SVM/kernel sangat baik di data kecil; NN unggul di data besar dan high-dimensional (gambar, teks).

## Formula

- Dense layer: z = Wx + b; params = d_in·d_out + d_out.
- ReLU: σ(z) = max(0, z).
- Sigmoid: σ(z) = 1/(1+e^(−z)).
- Parameter count example: 784→256→10 ≈ 203k.

## Contoh Kasus

Klasifikasi digit MNIST dengan MLP 784->256->128->10: ~235k parameters, ~98% accuracy. Bandingkan dengan logistic regression (~79%) — MLP belajar fitur non-linear otomatis. Atau autoencoder untuk anomaly detection: kompresi ke latent space, reconstruction error tinggi = anomali.

## Implementasi

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(784, [256, 128], 10)
print(sum(p.numel() for p in model.parameters()), "parameters")
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Why can't a single-layer perceptron solve XOR?",
    "options": [
      "It can only learn linear decision boundaries",
      "It has too few parameters",
      "XOR is not a valid function",
      "It requires more training data"
    ],
    "answer": 0,
    "explanation": "Perceptron = linear classifier; XOR is not linearly separable. MLP with hidden layer solves it.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "A neural network with only linear activations is equivalent to:",
    "options": [
      "A single linear transformation (no more powerful than linear regression)",
      "A very deep model",
      "A kernel method",
      "A decision tree"
    ],
    "answer": 0,
    "explanation": "Composition of linear maps is linear; nonlinearity is essential for expressiveness.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "math",
    "question": "How many parameters does a dense layer 100 -> 50 have (weights + biases)?",
    "answer": 5050,
    "tolerance": 0.1,
    "hints": ["100*50 weights + 50 biases"],
    "explanation": "5050 parameters.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `mlp_forward(X, params)` performing forward pass through a 2-layer MLP with ReLU: X @ W1 + b1 -> ReLU -> @ W2 + b2 -> sigmoid. params is dict with W1,b1,W2,b2.",
    "starter_code": "import numpy as np\n\ndef mlp_forward(X, params):\n    X = np.asarray(X, dtype=float)\n    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']\n    pass\n",
    "function": "mlp_forward",
    "tests": [
      {"call": "mlp_forward([[1.0,0.0]], {'W1':[[1,0],[0,1]],'b1':[0,0],'W2':[[1,1]],'b2':[0]})", "expected": "check"}
    ],
    "hints": ["h = relu(X@W1+b1); out = sigmoid(h@W2+b2)", "return array shape (n,1) or (n,)"],
    "explanation": "Manual forward propagation.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Universal approximation theorem says one hidden layer suffices. Why do deep networks (many layers) often outperform shallow wide networks in practice? Discuss efficiency, hierarchical features, and optimization.",
    "rubric": [
      "Depth enables compositional/hierarchical representations",
          "Deeper nets can represent same function with fewer total parameters",
      "Gradient flow and residual connections help optimization",
      "Empirical evidence across vision/NLP domains"
        ],
    "points": 2
  }
]
```

## Referensi

- Goodfellow, Bengio, Courville — 'Deep Learning' (free online)
- Rumelhart, Hinton, Williams (1986) — backpropagation
- Cybenko (1989) — universal approximation
- LeCun, Bengio, Hinton (2015) — deep learning Nature review
