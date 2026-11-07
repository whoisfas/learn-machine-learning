---
{
  "id": "p1-calculus",
  "title": "Calculus for Machine Learning",
  "phase": 1,
  "difficulty": "intermediate",
  "estimated_hours": 14,
  "objectives": [
    "Understand derivatives as slopes and gradients as directions of steepest ascent",
    "Compute partial derivatives and chain rule for composite functions",
    "Connect calculus to optimization: gradient descent, backpropagation",
    "Use numerical differentiation for verification"
  ],
  "concepts": [
    "Limit and derivative definition",
    "Rules: power, product, quotient, chain",
    "Partial derivatives and gradient vector",
    "Directional derivative, Jacobian",
    "Taylor series intuition",
    "Convexity and local vs global minima",
    "Numerical differentiation (finite differences)",
    "sympy for symbolic differentiation"
  ],
  "formulas": [
    {"name": "Derivative", "latex": "f'(x) = \\lim_{h\\to 0} \\frac{f(x+h)-f(x)}{h}", "note": "Slope of tangent line."},
    {"name": "Chain rule", "latex": "\\frac{d}{dx} f(g(x)) = f'(g(x))\\, g'(x)", "note": "Backbone of backpropagation."},
    {"name": "Gradient", "latex": "\\nabla f = \\left(\\frac{\\partial f}{\\partial x_1}, \\dots, \\frac{\\partial f}{\\partial x_d}\\right)", "note": "Points in direction of steepest increase."},
    {"name": "Taylor expansion", "latex": "f(x+h) \\approx f(x) + \\nabla f(x)^\\top h + \\frac{1}{2} h^\\top H h", "note": "Local quadratic approximation; H = Hessian."}
  ],
  "references": [
    "3Blue1Brown 'Essence of Calculus'",
    "sympy docs",
    "Khan Academy calculus"
  ]
}
---

## Penjelasan

Calculus menjawab pertanyaan: bagaimana fungsi berubah? Turunan mengukur laju perubahan sesaat; integral mengukur akumulasi. Di ML, kalkulus muncul di mana-mana: loss function diminimalkan dengan *gradient descent*, neural network dilatih dengan *backpropagation* (rantai aturan turunan), dan model linear di-fit dengan menurunkan error ke nol.

**Turunan.** f'(x) = lim_{h→0} [f(x+h)−f(x)]/h. Interpretasi: kemiringan garis singgung. Aturan dasar: (xⁿ)' = nxⁿ⁻¹, (uv)' = u'v + uv', (u/v)' = (u'v − uv')/v², dan **chain rule**: d/dx f(g(x)) = f'(g(x))·g'(x). Chain rule adalah alasan backpropagation bekerja: gradien loss terhadap bobot dihitung dengan meneruskan gradien mundur lewat komposisi fungsi.

**Multivariabel.** Untuk f(x,y), partial derivative ∂f/∂x mengubah satu variabel sambil menahan lainnya tetap. Gradient ∇f adalah vektor semua partial derivative — menunjuk arah kenaikan tersepat. Gradient descent melangkah *berlawanan* gradient: x ← x − η∇f(x).

**Konveksitas.** Fungsi konveks punya satu minimum global (mangkuk). Non-konveks punya banyak lokal minimum/saddle — relevan untuk neural network. Hessian (matriks turunan kedua) memberitahu kelengkungan; eigenvalue Hessian positif = konveks lokal.

**Numerik.** Turunan analitik bisa diverifikasi dengan finite difference: f'(x) ≈ [f(x+h)−f(x−h)]/(2h) (central difference, error O(h²)). sympy bisa menurunkan simbolik dan memverifikasi gradien manual.

## Konsep Penting

1. **Learning rate** — langkah gradient descent; terlalu besar divergen, terlalu kecil lambat.
2. **Chain rule bersarang** — network dalam = chain rule panjang; inilah "backprop".
3. **Gradient = arah, bukan besar** — besar gradien menentukan kecepatan konvergensi alami.
4. **Kondisi optimum** — ∇f=0 kandidat; cek Hessian positif definite untuk minimum.
5. **Finite difference untuk debugging** — verifikasi gradien manual vs numerik sebelum debugging network.

## Formula

- Gradient descent update: θ ← θ − η∇L(θ).
- Central difference: f'(x) ≈ [f(x+h) − f(x−h)] / 2h.
- Backprop chain rule: ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w.

## Contoh Kasus

Loss MSE untuk linear regression: L = (1/n)Σ(yi − w·xi − b)². Turunan parsial: ∂L/∂w = (2/n)Σ(xi·ŷi − xi·yi)... hitung analitik, bandingkan dengan finite difference di kode. Gradient descent iteratif konvergen ke least-squares solution.

## Implementasi

```python
import sympy as sp
import numpy as np

x, y = sp.symbols('x y')
f = x**2 * sp.sin(y) + sp.exp(x)
print(sp.diff(f, x))   # symbolic derivative
print(sp.diff(f, y))

# numerical gradient check
def f_np(v):
    return np.sum(v**2 * np.sin(v[1]) + np.exp(v[0]))

def numeric_grad(f, x, h=1e-5):
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2*h)
    return g
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "For f(x) = x^3 - 3x, at which x values does f'(x) = 0?",
    "options": ["x = ±1", "x = 0 only", "x = ±3", "x = 1 only"],
    "answer": 0,
    "explanation": "f'(x)=3x²-3=0 -> x²=1 -> x=±1.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "math",
    "question": "Compute the derivative of f(x) = 3x^2 + 2x at x = 2.",
    "answer": 14.0,
    "tolerance": 0.01,
    "hints": ["f'(x) = 6x + 2"],
    "explanation": "f'(2) = 12 + 2 = 14.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `gradient_descent(f, grad, x0, lr, iters)` performing batch gradient descent and returning the final x. f is the loss, grad returns the gradient vector.",
    "starter_code": "import numpy as np\n\ndef gradient_descent(f, grad, x0, lr, iters):\n    x = np.asarray(x0, dtype=float).copy()\n    pass\n",
    "function": "gradient_descent",
    "tests": [
      {"call": "gradient_descent(lambda x: float(np.sum(x**2)), lambda x: 2*x, [3.0,4.0], 0.1, 100).tolist()", "expected": [0.0,0.0], "tolerance": 0.01},
      {"call": "gradient_descent(lambda x: float((x[0]-2)**2), lambda x: np.array([2*(x[0]-2)]), [0.0], 0.1, 50)[0]", "expected": 2.0, "tolerance": 0.01}
    ],
    "hints": ["x = x - lr * grad(x) each iteration", "Return final x as numpy array"],
    "explanation": "Core optimization loop underlying all of ML training.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `central_difference(f, x, h=1e-5)` returning the numerical derivative of scalar f at scalar x.",
    "starter_code": "def central_difference(f, x, h=1e-5):\n    pass\n",
    "function": "central_difference",
    "tests": [
      {"call": "central_difference(lambda t: t**2, 3.0)", "expected": 6.0, "tolerance": 1e-4},
      {"call": "central_difference(lambda t: t**3, 2.0)", "expected": 12.0, "tolerance": 1e-3}
    ],
    "hints": ["(f(x+h)-f(x-h))/(2h)"],
    "explanation": "O(h^2) accurate numerical derivative.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Backpropagation computes ∂L/∂w for every weight w in a network. Using the chain rule, explain (a) why the computation is organized as a forward pass then backward pass, and (b) why deeper networks suffer vanishing/exploding gradients.",
    "rubric": [
      "Forward pass caches activations needed for backward",
      "Backward applies chain rule layer-by-layer (reverse order)",
      "Product of many Jacobian terms -> exponential growth/shrink"
    ],
    "points": 2
  }
]
```

## Referensi

- 3Blue1Brown 'Essence of Calculus' & 'Neural Networks' series
- sympy documentation
- Goodfellow et al., Deep Learning book, ch. 2 & 6 (free online)
