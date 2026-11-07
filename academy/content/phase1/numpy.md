---
{
  "id": "p1-numpy",
  "title": "NumPy: Arrays, Broadcasting, Vectorization",
  "phase": 1,
  "difficulty": "beginner",
  "estimated_hours": 12,
  "objectives": [
    "Create and manipulate ndarrays efficiently",
    "Use broadcasting rules correctly",
    "Apply vectorized operations, reductions, and linear algebra",
    "Understand dtype, memory layout, and view vs copy semantics"
  ],
  "concepts": [
    "ndarray structure: shape, dtype, strides",
    "Creation: zeros, ones, arange, linspace, random",
    "Indexing: basic, boolean masks, fancy indexing",
    "Broadcasting rules (align trailing dims, size-1 expand)",
    "Reductions: sum, mean, argmax, axis parameter",
    "Vectorization vs Python loops (speed)",
    "Views vs copies (`.copy()`, `np.may_share_memory`)",
    "np.linalg: dot, inv, eig, svd, norm"
  ],
  "formulas": [
    {"name": "Dot product", "latex": "c_i = \\sum_j a_{ij} b_j", "note": "np.dot / @ operator; O(m·n) flops."},
    {"name": "Broadcasting", "latex": "(m,k) op (k,) -> (m,k)", "note": "Trailing dimensions must match or be 1."}
  ],
  "references": [
    "NumPy docs — broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html",
    "NumPy 100: https://numpy.org/doc/stable/user/absolute_beginners.html"
  ]
}
---

## Penjelasan

NumPy adalah fondasi numerik Python. Semua library ML (sklearn, PyTorch) pada akhirnya beroperasi pada array NumPy atau tensor yang kompatibel. Kunci NumPy: **vectorization** (operasi pada seluruh array sekaligus, diimplementasikan dalam C) dan **broadcasting** (operasi antar array dengan shape berbeda tanpa loop manual).

**ndarray.** Array N-dimensi homogen (satu dtype). Atribut penting: `shape`, `dtype`, `ndim`, `size`, `nbytes`. `np.array([[1,2],[3,4]])` membuat array 2x2 int64. Perhatikan dtype: `np.array([1,2,3])` int64, tapi ML umumnya float64/float32.

**Pembuatan.** `np.zeros((2,3))`, `np.ones`, `np.arange(0,10,0.5)`, `np.linspace(0,1,100)` (100 titik inklusif), `np.random.default_rng(seed)` untuk RNG modern (PCG64), `np.eye`, `np.diag`.

**Indexing.** `a[0,1]`, `a[:, 1]` (semua baris, kolom 1), `a[mask]` boolean indexing mengembalikan copy, `a[[0,2]]` fancy indexing. Boolean mask sangat penting untuk filtering data: `X[y == 1]`.

**Broadcasting.** Aturan: mulai dari dimensi terakhir; dimensi harus sama atau salah satunya 1. Contoh: `(3,4) + (4,)` -> `(3,4)`; `(3,1) * (1,4)` -> `(3,4)`. Ini memungkinkan operasi seperti normalisasi per-baris: `(X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)`.

**Reducing & axis.** `np.sum(a, axis=0)` menjumlahkan sepanjang baris (menghasilkan array per kolom). `axis=None` meratakan semua. `argmax/argmin` mengembalikan index, `cumsum`, `percentile`.

**View vs copy.** Slicing `a[1:3]` menghasilkan *view* (berbagi memori); `a[[0,1]]` fancy indexing menghasilkan *copy*. Ubah view mengubah aslinya. `a.copy()` eksplisit.

**Linear algebra.** `A @ B` matmul, `np.linalg.inv`, `np.linalg.det`, `np.linalg.eig` (nilai eigen), `np.linalg.svd`, `np.linalg.norm`, `np.linalg.solve(A,b)` untuk sistem linear.

## Konsep Penting

1. **Broadcasting gagal = error shape** — `ValueError: operands could not be broadcast together` berarti trailing dims tidak kompatibel.
2. **`axis` membingungkan** — axis=0 berarti "collapse dimensi 0" (per kolom untuk 2D); gambar mental: axis menunjuk dimensi yang *dihapus*.
3. **dtype float32 vs float64** — float32 setengah memori, lebih cepat di GPU, tapi presisi lebih rendah. Default NumPy float64.
4. **Integer division** — `7 // 2 = 3`; `/` selalu float.
5. **`np.random` legacy vs `default_rng`** — gunakan `default_rng(seed)` untuk reproduksibilitas modern.

## Formula

- Matmul: C = A·B, C[i,j] = Σ_k A[i,k]·B[k,j], kompleksitas O(m·n·p).
- Broadcasting: shape (a1,...,an) op (b1,...,bn) kompatibel jika ai==bi atau ai==1 atau bi==1.

## Contoh Kasus

Normalisasi dataset: `X_norm = (X - X.mean(0)) / X.std(0)` — satu baris broadcasting menggantikan loop per fitur. Atau komputasi jarak antar titik: `d = np.linalg.norm(X[:,None,:] - X[None,:,:], axis=2)` menghasilkan matriks jarak pairwise.

## Implementasi

```python
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (1000, 5))

# standardization per feature
mu, sigma = X.mean(axis=0), X.std(axis=0)
Xz = (X - mu) / sigma

# boolean mask
outliers = Xz[np.abs(Xz) > 3].shape[0]

# pairwise distances (chunked for memory)
def pdist_chunked(A, chunk=256):
    out = np.empty((len(A), len(A)))
    for i in range(0, len(A), chunk):
        diff = A[i:i+chunk, None, :] - A[None, :, :]
        out[i:i+chunk] = np.linalg.norm(diff, axis=2)
    return out
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "What is the shape of `np.zeros((2,3,4)) + np.ones((3,4))`?",
    "options": ["(2,3,4)", "(3,4)", "Error", "(2,3)"],
    "answer": 0,
    "explanation": "(3,4) broadcasts against trailing dims of (2,3,4) -> (2,3,4).",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "coding",
    "question": "Implement `standardize(X)` that z-score normalizes each column (axis=0) of a 2D array and returns the result.",
    "starter_code": "import numpy as np\n\ndef standardize(X):\n    X = np.asarray(X, dtype=float)\n    pass\n",
    "function": "standardize",
    "tests": [
      {"call": "standardize([[1,2],[3,4],[5,6]]).tolist()", "expected": [[-1.224744871391589,-1.224744871391589],[0.0,0.0],[1.224744871391589,1.224744871391589]], "tolerance": 1e-6},
      {"call": "standardize([[5,5],[5,5]]).tolist()", "expected": [[0.0,0.0],[0.0,0.0]], "tolerance": 1e-9}
    ],
    "hints": ["mean/std along axis=0, keepdims=True", "Handle zero std to avoid NaN"],
    "explanation": "Column-wise z-score with zero-variance guard.",
    "points": 2
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `top_k_indices(scores, k)` returning indices of the k largest values in a 1D array, descending.",
    "starter_code": "import numpy as np\n\ndef top_k_indices(scores, k):\n    scores = np.asarray(scores)\n    pass\n",
    "function": "top_k_indices",
    "tests": [
      {"call": "top_k_indices([0.1, 0.9, 0.5, 0.7], 2).tolist()", "expected": [1, 3]},
      {"call": "top_k_indices([1,2,3], 3).tolist()", "expected": [2, 1, 0]}
    ],
    "hints": ["np.argsort then reverse", "or np.argpartition for large arrays"],
    "explanation": "Ranking indices by score — core of evaluation metrics like top-k accuracy.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "math",
    "question": "A (1000, 784) float32 matrix is stored in memory. How many megabytes (MB, decimal) does it occupy?",
    "answer": 3.136,
    "tolerance": 0.01,
    "hints": ["1000*784 elements * 4 bytes = ? bytes; divide by 1e6"],
    "explanation": "784,000 elements × 4 bytes = 3,136,000 bytes = 3.136 MB.",
    "points": 1
  }
]
```

## Referensi

- NumPy user guide — broadcasting, indexing, linear algebra
- "Python for Data Analysis" by Wes McKinney (NumPy chapters)
- NumPy 100 exercises: https://numpy.org/doc/stable/user/basics.rec.html
