---
{
  "id": "p3-perceptron",
  "title": "The Perceptron",
  "phase": 3,
  "difficulty": "beginner",
  "estimated_hours": 6,
  "objectives": [
    "Implement the perceptron learning rule",
    "Understand its convergence guarantee for linearly separable data",
    "Recognize the XOR limitation",
    "Connect perceptron to logistic regression and modern neurons"
  ],
  "concepts": [
    "McCulloch-Pitts neuron model",
    "Weights, bias, activation step function",
    "Perceptron learning rule: w += η(y - ŷ)x",
    "Convergence theorem (Novikoff)",
    "XOR problem and linear separability",
    "Perceptron vs ADALINE vs logistic regression",
    "Historical significance (1957 Rosenblatt)"
  ],
  "formulas": [
    {"name": "Perceptron output", "latex": "\\hat{y} = \\text{sign}(w^\\top x + b)", "note": "Step function at 0."},
    {"name": "Learning rule", "latex": "w \\leftarrow w + \\eta (y - \\hat{y}) x", "note": "Update only on mistakes."},
    {"name": "Convergence bound", "latex": "N_{mistakes} \\leq (R/\\gamma)^2", "note": "Novikoff: bounded by margin and radius."}
  ],
  "references": [
    "Rosenblatt 1957 'The Perceptron'",
    "Novikoff 1962 convergence proof",
    "Minsky & Papert 1969 'Perceptrons'"
  ]
}
---

## Penjelasan

Perceptron adalah model neuron paling sederhana — nenek moyang semua neural network modern. Ditemukan Rosenblatt (1957), sempat memicu hype besar, lalu dihangatkan oleh kritik Minsky & Papert tentang keterbatasannya.

**Model.** Neuron menerima input x ∈ R^d, menghitung weighted sum z = w·x + b, lalu output step function: ŷ = 1 jika z > 0, else 0 (atau ±1). Parameter: bobot w dan bias b.

**Learning rule.** Perceptron belajar dengan iterasi: untuk setiap contoh (x, y), prediksi ŷ; jika salah, update w += η(y − ŷ)x dan b += η(y − ŷ). Intuisinya: geser decision boundary mendekati titik yang salah klasifikasi. Ini online learning — satu pass per epoch.

**Konvergensi.** Novikoff (1962) membuktikan: jika data linearly separable dengan margin γ dan radius R, perceptron berhenti membuat kesalahan setelah maksimal (R/γ)² update. Jika tidak konvergen setelah iterasi wajar -> data tidak separable.

**Keterbatasan.** Perceptron hanya linear -> tidak bisa XOR. Minsky & Papert (1969) menunjukkan secara formal bahwa satu layer perceptron tidak dapat merepresentasikan fungsi non-linear sederhana. Ini memicu "AI winter" pertama — sampai backpropagation (1986) menunjukkan multi-layer bisa.

**Evolusi.** ADALINE (Widrow-Hoff) mengganti step dengan linear activation dan memakai LMS (least mean squares) — gradient descent pada MSE. Logistic regression menambahkan sigmoid untuk probabilitas. Perceptron modern = neuron dengan activation function pilihan (ReLU, tanh) dalam MLP.

## Konsep Penting

1. **Online vs batch** — perceptron update per contoh; batch gradient descent update setelah semua data.
2. **Margin vs accuracy** — perceptron berhenti begitu benar, tidak memaksimalkan margin; SVM lebih baik di sini.
3. **Linearly separable** — syarat konvergensi; perceptron akan loop selamanya pada data non-separable.
4. **Step function tidak differentiable** — tidak bisa backprop; digantikan sigmoid/tanh/ReLU.
5. **Historical lesson** — satu counterexample (XOR) bisa mematikan field riset; kompleksitas model harus match kompleksitas masalah.

## Formula

- z = w·x + b; ŷ = sign(z).
- Update: Δw = η(y − ŷ)x; Δb = η(y − ŷ).
- Convergence: mistakes ≤ (R/γ)².

## Contoh Kasus

Klasifikasi AND gate: w=(1,1), b=-1.5; input (1,1)->z=0.5->1; (1,0)->z=-0.5->0. Perceptron belajar ini dalam beberapa iterasi. XOR: tidak ada (w,b) yang memisahkan — inilah motivasi hidden layer.

## Implementasi

```python
import numpy as np

def train_perceptron(X, y, lr=0.1, epochs=1000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y) * 2 - 1  # {0,1} -> {-1,1}
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for epoch in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            pred = 1 if w @ xi + b > 0 else -1
            if pred != yi:
                w += lr * (yi - pred) * xi
                b += lr * (yi - pred)
                errors += 1
        if errors == 0:
            print(f"Converged at epoch {epoch}")
            break
    return w, b

# XOR fails
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])
w, b = train_perceptron(X_xor, y_xor, epochs=100)
print("XOR train acc:", np.mean((X_xor @ w + b > 0).astype(int) == y_xor))
# ~50% — linearly inseparable
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "The perceptron convergence theorem guarantees termination when:",
    "options": [
      "Data is linearly separable",
      "Data is very large",
      "Learning rate is small",
      "Always"
    ],
    "answer": 0,
    "explanation": "Novikoff's bound applies only to linearly separable data.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Why did Minsky & Papert's XOR result matter historically?",
    "options": [
      "It showed single-layer perceptrons can't represent non-linear functions, dampening neural net research",
      "It proved neural nets always work",
      "It introduced backpropagation",
      "It showed perceptrons are faster than SVM"
    ],
    "answer": 0,
    "explanation": "The limitation led to 'AI winter'; multi-layer + backprop later revived the field.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `perceptron_predict(X, w, b)` returning binary predictions (0/1) using step function at 0.",
    "starter_code": "import numpy as np\n\ndef perceptron_predict(X, w, b):\n    X = np.asarray(X, dtype=float)\n    w = np.asarray(w, dtype=float)\n    pass\n",
    "function": "perceptron_predict",
    "tests": [
      {"call": "perceptron_predict([[1,1],[0,0]], [1,1], -1.5).tolist()", "expected": [1,0]},
      {"call": "perceptron_predict([[0.5,0.5]], [1,1], -1.0).tolist()", "expected": [0]}
    ],
    "hints": ["z = X@w + b; return (z > 0).astype(int)"],
    "explanation": "Forward pass of perceptron.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `perceptron_train(X, y, lr=0.1, epochs=100)` returning (w, b) trained via perceptron rule.",
    "starter_code": "import numpy as np\n\ndef perceptron_train(X, y, lr=0.1, epochs=100):\n    X = np.asarray(X, dtype=float)\n    y = np.asarray(y) * 2 - 1\n    pass\n",
    "function": "perceptron_train",
    "tests": [
      {"call": "w,b=perceptron_train([[0,0],[0,1],[1,0],[1,1]],[0,0,0,1],0.5,200); float(np.mean((np.array([[0,0],[0,1],[1,0],[1,1]])@w+b>0).astype(int)==np.array([0,0,0,1])))", "expected": 1.0}
    ],
    "hints": ["Loop epochs; for each sample if pred != y: w += lr*(y-pred)*x; b += lr*(y-pred)", "Return (w, b)"],
    "explanation": "Perceptron learning algorithm.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Compare perceptron, logistic regression, and a single ReLU neuron in terms of (a) decision boundary shape, (b) output type, (c) training algorithm. When would you choose each?",
    "rubric": [
      "All produce linear boundaries; differ in activation and training",
          "Perceptron: step output, online rule; logistic: probability output, gradient descent; ReLU neuron: part of larger network",
      "Perceptron for simple online learning; logistic for probabilistic classification; ReLU neuron as building block"
        ],
    "points": 2
  }
]
```

## Referensi

- Rosenblatt (1957) 'The Perceptron'
- Novikoff (1962) 'On convergence proofs for perceptrons'
- Minsky & Papert (1969) 'Perceptrons'
- Goodfellow et al. ch. 6 (historical context)
