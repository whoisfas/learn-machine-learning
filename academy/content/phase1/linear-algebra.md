---
{
  "id": "p1-linear-algebra",
  "title": "Linear Algebra for Machine Learning",
  "phase": 1,
  "difficulty": "intermediate",
  "estimated_hours": 16,
  "objectives": [
    "Work with vectors, matrices, and their operations fluently",
    "Understand eigenvalues/eigenvectors, SVD, and rank",
    "Connect linear algebra to ML concepts: projections, PCA, least squares",
    "Use np.linalg for solving systems and decompositions"
  ],
  "concepts": [
    "Vector spaces, span, linear independence, basis",
    "Matrix multiplication, transpose, trace",
    "Linear systems Ax=b, singularity, condition number",
    "Determinant, inverse, orthogonality",
    "Eigen decomposition, symmetric matrices, positive definite",
    "Singular Value Decomposition (SVD)",
    "Rank, null space, column space",
    "Least squares and projections"
  ],
  "formulas": [
    {"name": "Matrix multiplication", "latex": "C_{ij} = \\sum_k A_{ik} B_{kj}", "note": "Inner dimensions must match: (m,k)@(k,n)->(m,n)."},
    {"name": "Eigenvalue", "latex": "Av = \\lambda v", "note": "v is eigenvector; lambda eigenvalue. PCA uses eig of covariance."},
    {"name": "Least squares", "latex": "\\hat{\\beta} = (X^\\top X)^{-1} X^\\top y", "note": "Minimizes ||y - X\\beta||^2."},
    {"name": "SVD", "latex": "A = U \\Sigma V^\\top", "note": "U,V orthogonal; Sigma diagonal singular values."}
  ],
  "references": [
    "3Blue1Brown 'Essence of Linear Algebra' (YouTube)",
    "Strang, 'Introduction to Linear Algebra'",
    "NumPy linalg docs"
  ]
}
---

## Penjelasan

Linear algebra adalah bahasa geometri ML. Dataset = matriks X (n×d); model linear = perkalian matriks; PCA = proyeksi ke basis eigen; neural network = komposisi fungsi affine.

**Vektor & ruang.** Vektor = arah+magnitudo di R^d. *Span* = semua kombinasi linear; *basis* = span minimal; *independen* = tidak ada yang redundant. *Norm* ||v|| = sqrt(Σvi²) (Euclidean/L2). *Dot product* a·b = Σai·bi = |a||b|cos θ — mengukur kesamaan arah; *cosine similarity* = (a·b)/(|a||b|) dipakai di NLP dan retrieval.

**Matriks.** A (m×n) @ B (n×p) = C (m×p). `A.T` transpose. Trace = jumlah diagonal. Inverse A⁻¹ memenuhi AA⁻¹=I — hanya ada untuk square nonsingular. Determinant det(A) mengukur volume scaling; det=0 berarti singular (tak invertible).

**Sistem linear.** Ax=b: jika A invertible, solusi unik x=A⁻¹b. Jika underdetermined (lebih variabel dari persamaan), tak hingga solusi -> *minimum-norm* solution via pseudoinverse. *Condition number* κ(A)=||A||·||A⁻¹|| mengukur sensitivitas solusi terhadap noise — penting untuk stabilitas numerik.

**Eigen & SVD.** Av=λv: eigenvector v adalah arah yang hanya di-scale λ oleh A. Untuk matriks simetris (seperti covariance), eigenvector ortogonal dan eigenvalue real. PCA: eigenvector covariance matrix = principal components; eigenvalue = variance explained. *SVD* A=UΣVᵀ berlaku untuk semua matriks; singular values σi = akar eigenvalue AᵀA; SVD adalah cara stabil numerik untuk pseudoinverse dan rank.

**Least squares.** Untuk y≈Xβ, solusi β̂=(XᵀX)⁻¹Xᵀy adalah proyeksi ortogonal y ke column space X. Residual e=y−Xβ̂ ortogonal terhadap semua kolom X (normal equations Xᵀe=0).

## Konsep Penting

1. **PCA ≠ whitening otomatis** — PCA memutar sumbu ke arah variance maksimal; whitening juga menskalakan ke unit variance.
2. **Rank deficiency** — XᵀX singular jika fitur kolinear; ridge regression menambah λI untuk stabilitas.
3. **Condition number** — κ besar berarti solusi numerik tidak stabil; normalisasi fitur membantu.
4. **SVD untuk pseudoinverse** — pinv(A) = VΣ⁺Uᵀ; bekerja untuk non-square dan rank-deficient.
5. **Orthogonality** — vektor ortogonal -> dot=0; QR decomposition (Q ortogonal, R upper triangular) dipakai di least squares stabil.

## Formula

- Proyeksi ortogonal: proj_U(y) = UUᵀy untuk U basis ortonormal.
- Variance explained PCA: λi / Σλj.
- Cosine similarity: cos θ = (a·b)/(|a||b|).

## Contoh Kasus

Least squares regression: X (n×d) dengan n>d, β̂=(XᵀX)⁻¹Xᵀy. PCA pada data ternormalisasi: hitung covariance C=XᵀX/(n−1), eigen decomposition, proyeksikan X ke 2 komponen utama. Kernel intuition: data tidak linearly separable bisa dipetakan ke ruang di mana linear algebra berlaku.

## Implementasi

```python
import numpy as np

# least squares via SVD (stable)
def lstsq_svd(X, y):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    cutoff = s.max() * max(X.shape) * np.finfo(s.dtype).eps
    s_inv = np.where(s > cutoff, 1/s, 0)
    return (Vt.T * s_inv) @ U.T @ y

# PCA via SVD on centered data
def pca(X, k):
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, s**2 / (len(X)-1)

# cosine similarity matrix
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
sim = Xn @ Xn.T
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "If A is a 3x5 matrix and B is 5x2, what is the shape of A@B and what must hold about inner dimensions?",
    "options": ["3x2, inner dims 5==5", "3x5, no constraint", "Error always", "5x2"],
    "answer": 0,
    "explanation": "(m,k)@(k,n): k must match. Here k=5, result (3,2).",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "math",
    "question": "Given v = [3, 4], compute its L2 norm ||v||.",
    "answer": 5.0,
    "tolerance": 0.001,
    "hints": ["sqrt(3^2 + 4^2)"],
    "explanation": "3-4-5 triangle: sqrt(9+16)=5.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `cosine_similarity(a, b)` for 1D vectors: (a·b)/(|a||b|), return 0.0 if either norm is 0.",
    "starter_code": "import numpy as np\n\ndef cosine_similarity(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    pass\n",
    "function": "cosine_similarity",
    "tests": [
      {"call": "cosine_similarity([1,0],[0,1])", "expected": 0.0},
      {"call": "cosine_similarity([1,2],[1,2])", "expected": 1.0},
      {"call": "cosine_similarity([1,0],[2,0])", "expected": 1.0},
      {"call": "cosine_similarity([0,0],[1,1])", "expected": 0.0}
    ],
    "hints": ["np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))", "guard zero norms"],
    "explanation": "Cosine similarity ignores magnitude — used in embeddings/IR.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `solve_linear(A, b)` returning x solving Ax=b using np.linalg.solve; raise ValueError if A is singular.",
    "starter_code": "import numpy as np\n\ndef solve_linear(A, b):\n    pass\n",
    "function": "solve_linear",
    "tests": [
      {"call": "solve_linear([[2,0],[0,2]],[4,6]).tolist()", "expected": [2.0, 3.0]},
      {"call": "solve_linear([[1,2],[2,4]],[1,2])", "expected": "error"}
    ],
    "hints": ["np.linalg.solve raises LinAlgError on singular; catch and re-raise ValueError", "for the singular case your function should raise — the test harness expects an exception"],
    "explanation": "Direct solver for square systems; singular matrices have no unique solution.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Explain why PCA uses eigenvectors of the covariance matrix (or equivalently right singular vectors of centered X). What does the eigenvalue tell you, and how do you choose the number of components?",
    "rubric": [
      "Eigenvectors = directions of max variance (orthogonal)",
      "Eigenvalue = variance captured along that direction",
      "Cumulative explained variance threshold (e.g., 95%) or elbow in scree plot"
    ],
    "points": 2
  }
]
```

## Referensi

- 3Blue1Brown Linear Algebra series
- Gilbert Strang, MIT 18.06 lectures (free)
- NumPy linalg documentation
- "Mathematics for Machine Learning" (free book, mml-book.github.io)
