---
{
  "id": "p2-pca",
  "title": "Principal Component Analysis (PCA)",
  "phase": 2,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Derive PCA as variance-maximizing orthogonal projection",
    "Connect PCA to eigendecomposition and SVD",
    "Choose number of components via explained variance",
    "Use PCA for visualization, compression, and preprocessing"
  ],
  "concepts": [
    "PCA as orthogonal projection onto principal components",
    "Covariance matrix and eigendecomposition",
    "Explained variance ratio and cumulative variance",
    "Scree plot for component selection",
    "PCA whitening and its effect",
    "Kernel PCA for nonlinear data",
    "PCA limitations: assumes linearity, variance != importance",
    "PCA vs t-SNE vs UMAP for visualization"
  ],
  "formulas": [
    {"name": "PCA projection", "latex": "Z = XW_k", "note": "W_k = top-k eigenvectors of covariance."},
    {"name": "Explained variance", "latex": "\\frac{\\lambda_i}{\\sum_j \\lambda_j}", "note": "Fraction of total variance captured."},
    {"name": "Covariance", "latex": "C = \\frac{1}{n-1}X_c^\\top X_c", "note": "X_c = centered data."},
    {"name": "Reconstruction", "latex": "\\hat{X} = ZW_k^\\top + \\mu", "note": "Inverse transform projects back."}
  ],
  "references": [
    "sklearn.decomposition.PCA docs",
    "Jolliffe 'Principal Component Analysis'",
    "ESL ch. 14.5"
  ]
}
---

## Penjelasan

PCA menemukan arah (principal components) di mana data memiliki variance maksimum, kemudian memproyeksikan data ke subspace berdimensi lebih rendah yang dibentang arah-arah tersebut. Hasil: representasi kompak yang mempertahankan sebagian besar informasi.

**Intuisi geometris.** Data cloud di R^d. PC1 = arah variance terbesar; PC2 = arah variance terbesar yang ortogonal terhadap PC1; dst. Setiap PC adalah kombinasi linear fitur asli — interpretasi sebagai "axis" baru.

**Komputasi.** Center data (mean=0). Hitung covariance matrix C = XᵀX/(n−1). Eigen decomposition C = VΛVᵀ; eigenvector = principal components, eigenvalue = variance pada arah itu. Atau SVD langsung pada X_c = UΣVᵀ; V = PCs. sklearn pakai SVD (lebih stabil numerik).

**Explained variance.** λi/Σλj = fraksi variance total yang dijelaskan PCi. Cumulative explained variance menunjukkan berapa komponen dibutuhkan untuk menangkap mis. 95% variance. Scree plot (explained variance vs component index) menunjukkan "elbow" — setelah itu tiap PC menambah sedikit informasi.

**Transform & inverse.** `fit` belajar mean + komponen; `transform` memproyeksikan X ke Z (n×k); `inverse_transform` merekonstruksi aproksimasi X̂. Reconstruction error = variance yang dibuang.

**Whitening.** Bagi tiap komponen dengan sqrt(eigenvalue) -> semua komponen unit variance. Berguna sebelum clustering atau ketika skala komponen tidak seimbang. Hati-hati: amplifikasi noise pada komponen variance kecil.

**Kapan PCA membantu.** Reduksi dimensi untuk visualisasi (2D/3D), kompresi, menghilangkan multikolinearitas, preprocessing sebelum model yang tidak scalable ke dimensi tinggi. TIDAK membantu jika: hubungan non-linear (butuh kernel PCA), interpretasi fitur asli penting (komponen adalah campuran), atau variance tinggi bukan berarti informatif (bisa jadi noise).

## Konsep Penting

1. **PCA ≠ feature selection** — PC adalah kombinasi semua fitur; tidak ada "fitur yang dibuang", hanya dimensi yang dikurangi.
2. **Variance ≠ signal** — PC dengan variance tinggi bisa noise; domain knowledge tetap diperlukan.
3. **PCA linear** — hanya menangkap struktur linear; data pada manifold non-linear butuh kernel PCA atau autoencoder.
4. **PCA tidak supervised** — tidak tahu label; supervised dimensionality reduction (LDA) lebih baik untuk klasifikasi.
5. **Scree plot + cumulative variance** — dua cara komplementer memilih k.

## Formula

- C = X_cᵀX_c/(n−1); eigenvectors V, eigenvalues λ.
- Explained variance ratio: λi / Σλj.
- Projection: Z = X_c @ V_k.
- Reconstruction: X̂ = Z @ V_kᵀ + μ.

## Contoh Kasus

Dataset gambar 784 dimensi (MNIST): PCA ke 50 komponen mempertahankan ~95% variance, mempercepat training SVM 10x dengan akurasi turun <1%. Atau EDA: PCA ke 2D untuk visualisasi cluster pada data high-dimensional.

## Implementasi

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# scale first! PCA is variance-based
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95)  # keep 95% variance
X_reduced = pca.fit_transform(X_scaled)
print("components:", pca.n_components_, "of", X.shape[1])
print("explained:", pca.explained_variance_ratio_.sum())

# scree plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('n_components'); plt.ylabel('cumulative explained variance')
plt.show()

# 2D visualization
pca2 = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(pca2[:,0], pca2[:,1], c=y, cmap='viridis', alpha=0.5)
plt.show()
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "PCA finds directions that:",
    "options": [
      "Maximize variance in the data",
      "Minimize reconstruction error to mean",
      "Separate classes",
      "Remove outliers"
    ],
    "answer": 0,
    "explanation": "PCs are eigenvectors of covariance = directions of maximum variance.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "PCA components are:",
    "options": [
      "Linear combinations of original features",
      "Original features selected",
      "Nonlinear transformations",
      "Independent random variables"
    ],
    "answer": 0,
    "explanation": "Each PC = weighted sum of all original features; not a subset.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "If first 2 PCs explain 90% variance, keeping only them means:",
    "options": [
      "We lose 10% of information but reduce dimension from d to 2",
      "We keep 90% of features",
      "Model will be 90% accurate",
      "Data becomes 90% smaller"
    ],
    "answer": 0,
    "explanation": "Variance captured = information proxy; dimensionality reduced to 2.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `pca_fit_transform(X, k)` returning (components, transformed) where components are top-k principal directions (eigenvectors of covariance) and transformed = centered X @ components.",
    "starter_code": "import numpy as np\n\ndef pca_fit_transform(X, k):\n    X = np.asarray(X, dtype=float)\n    pass\n",
    "function": "pca_fit_transform",
    "tests": [
      {"call": "t,c=pca_fit_transform([[1,2],[3,4],[5,6],[7,8]], 1); t.shape", "expected": [4,1]},
      {"call": "t,c=pca_fit_transform([[1,0],[0,1],[1,1]], 2); t.shape", "expected": [3,2]}
    ],
    "hints": ["Center X; cov = Xc.T@Xc/(n-1); eigval,eigvec=np.linalg.eigh(cov); take top-k eigenvectors", "return (Xc @ V_k, V_k) or (transformed, components) — define clearly"],
    "explanation": "Manual PCA via eigendecomposition.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your dataset has 500 features and you apply PCA keeping 95% variance -> 120 components. A stakeholder asks 'which original features are most important?' Explain why PCA can't answer this directly and propose two better approaches.",
    "rubric": [
      "PCs are mixtures of all features — no direct mapping to original features",
          "Proposes feature importance methods: permutation importance, SHAP, L1 regularization",
      "Or supervised dimensionality reduction (LDA) if labels exist",
      "Mentions examining PC loadings as partial interpretability"
        ],
    "points": 2
  }
]
```

## Referensi

- sklearn.decomposition.PCA documentation
- Jolliffe, 'Principal Component Analysis' (2nd ed)
- ESL ch. 14.5
- 'Kernel PCA' — Bishop PRML ch. 12
