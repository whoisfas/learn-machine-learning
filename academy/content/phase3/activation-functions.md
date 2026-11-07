---
{
  "id": "p3-activation-functions",
  "title": "Activation Functions",
  "phase": 3,
  "difficulty": "intermediate",
  "estimated_hours": 8,
  "objectives": [
    "Understand why nonlinearity is essential",
    "Compare sigmoid, tanh, ReLU, Leaky ReLU, GELU, SiLU",
    "Analyze vanishing/exploding gradient causes",
    "Choose appropriate activations for different architectures"
  ],
  "concepts": [
    "Step, linear, sigmoid, tanh activations",
    "ReLU and its variants (Leaky, PReLU, ELU, GELU, SiLU/Swish)",
    "Vanishing gradient problem",
    "Dead ReLU problem",
    "Smooth vs nonsmooth activations",
    "Activation in output layer (softmax, sigmoid)"
  ],
  "formulas": [
    {"name": "Sigmoid", "latex": "\\sigma(z) = \\frac{1}{1+e^{-z}}", "note": "Output (0,1); derivative max 0.25."},
    {"name": "Tanh", "latex": "\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}", "note": "Output (-1,1); zero-centered."},
    {"name": "ReLU", "latex": "\\text{ReLU}(z) = \\max(0, z)", "note": "Derivative 1 for z>0, 0 else; sparse activation."},
    {"name": "GELU", "latex": "\\text{GELU}(z) = z \\cdot \\Phi(z)", "note": "Gaussian Error Linear Unit; smooth ReLU approximation."},
    {"name": "SiLU/Swish", "latex": "\\text{SiLU}(z) = z \\cdot \\sigma(z)", "note": "z times sigmoid; used in modern architectures."}
  ],
  "references": [
    "Goodfellow et al. ch. 6.3",
    "Hendrycks & Gimpel 2016 GELU paper",
    "Ramachandran et al. 2017 Swish paper"
  ]
}
---

## Penjelasan

Activation functions memperkenalkan non-linearitas ke neural network. Tanpa mereka, komposisi layer linear = linear — model tidak lebih kuat dari linear regression berapapun dalamnya.

**Sigmoid.** σ(z) = 1/(1+e^(−z)) -> output (0,1). Masalah: saturasi di ekor -> derivative -> 0 -> vanishing gradient di deep networks; output tidak zero-centered -> zigzag optimization. Sekarang jarang dipakai di hidden layers; masih dipakai di output untuk binary classification.

**Tanh.** Output (−1,1), zero-centered -> sedikit lebih baik dari sigmoid untuk optimization. Tapi saturasi yang sama -> vanishing gradient. Jarang dipakai di modern architectures.

**ReLU.** max(0,z) -> output [0,∞). Keunggulan: derivative = 1 untuk z>0 (tidak saturasi untuk positif), sparse activation (banyak neuron mati -> efisiensi komputasi), mencegah vanishing gradient untuk sinyal positif. Masalah: *dying ReLU* — jika z selalu negatif, gradient selalu 0 -> neuron mati permanen. Juga non-smooth di 0.

**Varian ReLU.**
- *Leaky ReLU*: max(αz, z) dengan α kecil (0.01) -> gradient kecil untuk z<0, mencegah dead neuron.
- *PReLU*: α dipelajari per neuron.
- *ELU*: smooth untuk z<0, output mendekati −α.
- *GELU*: z·Φ(z) — smooth approximation, dipakai di BERT/ViT/transformer modern.
- *SiLU/Swish*: z·σ(z) — smooth, non-monotonic, dipakai di EfficientNet/dll.

**Pemilihan.** ReLU default aman untuk CNN/MLP. GELU/SiLU standar di transformers. Output layer: softmax (multiclass), sigmoid (binary/multi-label), linear (regresi). Hindari sigmoid/tanh di hidden layers untuk network dalam.

## Konsep Penting

1. **Vanishing gradient** — sigmoid/tanh derivative max 0.25; produk banyak layer -> gradient -> 0; inilah alasan ReLU menang.
2. **Dead ReLU** — inisialisasi buruk atau lr terlalu tinggi bisa membunuh neuron; Leaky/ELU lebih aman.
3. **Smoothness** — GELU/SiLU smooth -> gradient well-behaved; ReLU piecewise linear -> subgradient di 0.
4. **Sparse vs dense activation** — ReLU menghasilkan sparse representation (efisien); sigmoid/tanh dense.
5. **Output activation ≠ hidden activation** — output layer ditentukan oleh task (softmax/sigmoid/linear), bukan pilihan bebas.

## Formula

- σ'(z) = σ(z)(1−σ(z)); max 0.25 at z=0.
- ReLU'(z) = 1 if z>0 else 0.
- GELU(z) = z·Φ(z) ≈ z·0.5·(1+tanh(√(2/π)(z+0.044715z³))).

## Contoh Kasus

MLP untuk MNIST: ReLU di hidden layers, softmax di output. Bandingkan dengan sigmoid-only network — ReLU konvergen jauh lebih cepat dan akurasi lebih tinggi. Untuk transformer (BERT), GELU standar karena smooth dan empiris lebih baik dari ReLU.

## Implementasi

```python
import torch
import torch.nn as nn
import numpy as np

activations = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.01),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
}

# compare activation functions on same architecture
for name, act in activations.items():
    model = nn.Sequential(
        nn.Linear(784, 256), act,
        nn.Linear(256, 128), act,
        nn.Linear(128, 10))
    # train and compare...
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Why does ReLU help with vanishing gradients compared to sigmoid?",
    "options": [
      "Its derivative is 1 for positive inputs, not bounded by 0.25",
      "It outputs larger values",
      "It is smoother",
      "It uses fewer parameters"
    ],
    "answer": 0,
    "explanation": "ReLU gradient doesn't shrink for positive activations; sigmoid max derivative 0.25 compounds over layers.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "The 'dying ReLU' problem occurs when:",
    "options": [
      "A neuron's pre-activation is always negative, so gradient is always 0",
      "ReLU outputs are too large",
      "The learning rate is too small",
      "There are too many layers"
    ],
    "answer": 0,
    "explanation": "Once z<0 always, derivative=0 forever -> neuron permanently inactive.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "math",
    "question": "Compute sigmoid'(0) — the maximum derivative of sigmoid.",
    "answer": 0.25,
    "tolerance": 0.001,
    "hints": ["σ'(z)=σ(z)(1-σ(z)); at z=0, σ=0.5"],
    "explanation": "0.5*0.5 = 0.25 — this is why sigmoid vanishes in deep nets.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `gelu_approx(z)` returning the tanh approximation of GELU: 0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z^3))).",
    "starter_code": "import numpy as np\n\ndef gelu_approx(z):\n    z = np.asarray(z, dtype=float)\n    pass\n",
    "function": "gelu_approx",
    "tests": [
      {"call": "gelu_approx(0.0).tolist()", "expected": [0.0]},
      {"call": "gelu_approx(1.0).tolist()", "expected": [0.8413447460685429], "tolerance": 0.01},
      {"call": "gelu_approx([-1.0,0.0,1.0]).tolist()", "expected": [-0.15865526393145707,0.0,0.8413447460685429], "tolerance": 0.01}
    ],
    "hints": ["Use np.tanh; formula: 0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z**3)))"],
    "explanation": "GELU smooth activation used in transformers.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You're building a deep MLP (20 layers) and training diverges with sigmoid activations but works with ReLU. Explain the mechanism and why BatchNorm + residual connections further help.",
    "rubric": [
      "Sigmoid derivative <=0.25 compounds multiplicatively over 20 layers -> vanishing gradient",
          "ReLU keeps gradient ~1 for active neurons",
      "BatchNorm stabilizes activation distributions; residuals provide gradient highway",
      "Mentions initialization schemes (He init for ReLU)"
        ],
    "points": 2
  }
]
```

## Referensi

- Goodfellow et al., 'Deep Learning' ch. 6.3
- Hendrycks & Gimpel (2016) 'Gaussian Error Linear Units'
- Ramachandran et al. (2017) 'Searching for Activation Functions'
- PyTorch nn activations documentation
