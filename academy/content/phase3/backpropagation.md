---
{
  "id": "p3-backpropagation",
  "title": "Backpropagation",
  "phase": 3,
  "difficulty": "advanced",
  "estimated_hours": 16,
  "objectives": [
    "Derive gradients using the chain rule through network layers",
    "Implement backprop manually for a small MLP",
    "Understand computational graph and gradient flow",
    "Debug gradient issues: vanishing, exploding, dead neurons"
  ],
  "concepts": [
    "Chain rule applied to composite functions",
    "Computational graph (forward storage, backward traversal)",
    "Gradient of loss w.r.t. weights and biases",
    "Backprop through linear, activation, and loss layers",
    "Vector-Jacobian products (VJP)",
    "Gradient checking with finite differences",
    "Vanishing/exploding gradients",
    "Automatic differentiation (autograd)"
  ],
  "formulas": [
    {"name": "Chain rule", "latex": "\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w}", "note": "Gradients flow backward through layers."},
    {"name": "MLP weight gradient", "latex": "\\frac{\\partial L}{\\partial W^{(l)}} = \\delta^{(l)} (a^{(l-1)})^\\top", "note": "δ = upstream gradient."},
    {"name": "Delta recursion", "latex": "\\delta^{(l)} = (W^{(l+1)})^\\top \\delta^{(l+1)} \\odot \\sigma'(z^{(l)})", "note": "Backpropagate error signal."}
  ],
  "references": [
    "Rumelhart, Hinton, Williams (1986) 'Learning representations by back-propagating errors'",
    "Goodfellow et al. ch. 6.5",
    "3Blue1Brown 'Backpropagation calculus'"
  ]
}
---

## Penjelasan

Backpropagation adalah algoritma untuk menghitung gradient loss terhadap semua parameter network secara efisien menggunakan chain rule. Tanpa backprop, network dalam mustahil dilatih — komputasi gradient akan tumbuh eksponensial.

**Intuisi.** Forward pass menyimpan semua intermediate activations. Backward pass mulai dari loss, menghitung seberapa besar loss berubah jika setiap parameter sedikit berubah — dengan memanfaatkan chain rule untuk meneruskan gradient mundur layer per layer.

**Matematika.** Untuk layer l: z = Wx + b, a = σ(z). Jika δ^(l) = ∂L/∂z^(l) (gradien terhadap pre-activation), maka:
- ∂L/∂W^(l) = δ^(l) · (a^(l−1))ᵀ
- ∂L/∂b^(l) = δ^(l)
- δ^(l−1) = (W^(l))ᵀ · δ^(l) ⊙ σ'(z^(l−1)) — teruskan ke layer sebelumnya

Output layer: untuk softmax+cross-entropy, δ^(L) = p − y (sangat elegan); untuk MSE+linear, δ = 2(y−ŷ)/n.

**Kompleksitas.** Forward + backward = O(parameters) per sample — sama order dengan forward saja. Inilah mengapa deep learning feasible.

**Autograd.** PyTorch/TensorFlow membangun computational graph saat operasi dieksekusi (dengan requires_grad=True), lalu `.backward()` menghitung semua gradient otomatis. Manual backprop berguna untuk debugging dan pemahaman.

**Masalah umum.**
- *Vanishing gradient*: produk banyak σ'<1 -> gradient -> 0 di layer awal; mitigasi: ReLU, residual connections, BatchNorm, proper init.
- *Exploding gradient*: gradient membesar; mitigasi: gradient clipping.
- *Dead neurons*: ReLU dengan z<0 selalu -> gradient 0.

## Konsep Penting

1. **Chain rule adalah segalanya** — backprop = chain rule yang diorganisir efisien.
2. **Cache forward activations** — backward butuh nilai forward; simpan di memory (trade-off memory vs recompute).
3. **δ notation** — δ^(l) = ∂L/∂z^(l); semua formula diturunkan darinya.
4. **Softmax+CE simplification** — δ^(L) = p − y sangat penting untuk implementasi benar.
5. **Gradient checking** — bandingkan analytic gradient vs finite difference untuk verifikasi.

## Formula

- Output layer (softmax+CE): δ = p − y.
- Hidden layer: δ^(l) = (W^(l+1))ᵀ δ^(l+1) ⊙ σ'(z^(l)).
- Weight grad: ∂L/∂W = δ · a_prevᵀ.
- Bias grad: ∂L/∂b = δ.

## Contoh Kasus

MLP 2-2-1 untuk XOR: forward -> loss -> backward menghitung gradient W1,b1,W2,b2 -> update dengan SGD -> iterasi sampai konvergen. Manual backprop untuk network sekecil ini bisa diverifikasi dengan finite difference.

## Implementasi

```python
import numpy as np

def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)

def sigmoid(z): return 1/(1+np.exp(-z))
def sigmoid_grad(z):
    s = sigmoid(z); return s*(1-s)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z); return e/e.sum(axis=-1, keepdims=True)

def mse_loss(y, yhat): return np.mean((y-yhat)**2)

def forward(X, params):
    z1 = X @ params['W1'].T + params['b1']
    a1 = relu(z1)
    z2 = a1 @ params['W2'].T + params['b2']
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def backward(X, y, params, cache):
    z1, a1, z2, a2 = cache
    n = len(y)
    # output layer (MSE + sigmoid)
    delta2 = (2*(a2 - y)/n) * sigmoid_grad(z2)
    grads = {}
    grads['W2'] = delta2.T @ a1
    grads['b2'] = delta2.sum(axis=0)
    # hidden layer
    delta1 = (delta2 @ params['W2']) * relu_grad(z1)
    grads['W1'] = delta1.T @ X
    grads['b1'] = delta1.sum(axis=0)
    return grads
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Backpropagation computes gradients by:",
    "options": [
      "Applying the chain rule backward through the computational graph",
      "Trying all possible weight values",
      "Numerical differentiation only",
      "Random search"
    ],
    "answer": 0,
    "explanation": "Backprop = efficient chain rule application from loss to parameters.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "For softmax + cross-entropy output layer, the delta (gradient w.r.t. logits) simplifies to:",
    "options": ["p - y", "y - p", "p*(1-p)", "log(p/y)"],
    "answer": 0,
    "explanation": "Beautiful simplification: δ = predicted probabilities - true labels.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "math",
    "question": "For ReLU activation, what is σ'(z) when z = -2?",
    "answer": 0.0,
    "tolerance": 0.001,
    "hints": ["ReLU derivative is 0 for z<0"],
    "explanation": "Dead gradient for negative pre-activation.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `relu_backward(dout, z)` returning gradient w.r.t. pre-activation z given upstream gradient dout (element-wise).",
    "starter_code": "import numpy as np\n\ndef relu_backward(dout, z):\n    dout = np.asarray(dout, dtype=float)\n    z = np.asarray(z, dtype=float)\n    pass\n",
    "function": "relu_backward",
    "tests": [
      {"call": "relu_backward([1,1,1], [1,-1,0]).tolist()", "expected": [1,0,0]},
      {"call": "relu_backward([2,3], [5,-5]).tolist()", "expected": [2,0]}
    ],
    "hints": ["dout * (z > 0)"],
    "explanation": "Chain rule through ReLU.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "coding",
    "question": "Implement `linear_backward(dout, input_activation, weight_shape)` returning (dW, db) for layer z = x@W.T + b.",
    "starter_code": "import numpy as np\n\ndef linear_backward(dout, x, W_shape):\n    dout = np.asarray(dout, dtype=float)\n    x = np.asarray(x, dtype=float)\n    pass\n",
    "function": "linear_backward",
    "tests": [
      {"call": "dW,db=linear_backward([[1,1]],[[1,2,3]],[2,3]); dW.tolist()", "expected": [[1,1,1],[1,1,1]]},
      {"call": "dW,db=linear_backward([[1,1]],[[1,2,3]],[2,3]); db.tolist()", "expected": [1,1]}
    ],
    "hints": ["dW = dout.T @ x; db = dout.sum(axis=0)", "return (dW, db)"],
    "explanation": "Gradient of affine layer.",
    "points": 2
  },
  {
    "id": "ex6",
    "type": "open",
    "question": "Explain why backpropagation is O(parameters) per sample — same order as forward pass — and why this makes deep learning computationally feasible. What would happen without the chain rule reuse?",
    "rubric": [
      "Each parameter's gradient computed once via shared subexpressions",
          "Without reuse, gradient computation would recompute paths exponentially",
      "Dynamic programming / memoization analogy",
      "Connects to computational graph efficiency"
        ],
    "points": 2
  }
]
```

## Referensi

- Rumelhart, Hinton, Williams (1986) — original backprop paper
- Goodfellow et al. ch. 6.5 (backprop algorithm)
- 3Blue1Brown 'Backpropagation calculus' & 'Backpropagation building blocks'
- PyTorch autograd documentation
