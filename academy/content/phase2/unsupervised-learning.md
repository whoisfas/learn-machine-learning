---
{
  "id": "p2-unsupervised-learning",
  "title": "Unsupervised Learning",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": [
    "Understand clustering, dimensionality reduction, and density estimation",
    "Apply K-Means, hierarchical clustering, DBSCAN, Gaussian Mixture",
    "Use PCA and other dimensionality reduction techniques",
    "Evaluate unsupervised results without labels"
  ],
  "concepts": [
    "Clustering objective: within-cluster vs between-cluster variance",
    "K-Means algorithm and its assumptions",
    "Elbow method, silhouette score, gap statistic",
    "Hierarchical clustering and dendrograms",
    "DBSCAN: density-based, finds arbitrary shapes, noise points",
    "Gaussian Mixture Models (soft clustering)",
    "Dimensionality reduction: PCA, t-SNE, UMAP",
    "Anomaly detection as unsupervised problem"
  ],
  "formulas": [
    {"name": "K-Means objective", "latex": "J = \\sum_{k} \\sum_{x \\in C_k} \\|x - \\mu_k\\|^2", "note": "Minimize within-cluster sum of squares."},
    {"name": "Silhouette", "latex": "s(i) = \\frac{b(i)-a(i)}{\\max(a(i),b(i))}", "note": "a=mean intra-cluster dist, b=mean nearest-cluster dist; range [-1,1]."},
    {"name": "PCA variance", "latex": "\\text{explained} = \\frac{\\lambda_i}{\\sum_j \\lambda_j}", "note": "Fraction of variance captured by component i."}
  ],
  "references": [
    "sklearn clustering docs",
    "scikit-learn user guide — clustering",
    "'Pattern Recognition and ML' Bishop ch. 9"
  ]
}
---

## Penjelasan

Unsupervised learning bekerja tanpa label: menemukan struktur tersembunyi dalam data. Tiga keluarga utama: **clustering** (grup titik serupa), **dimensionality reduction** (proyeksikan ke ruang lebih kecil), **density estimation** (model probabilitas data).

**Clustering.** Tujuan: partisi data ke k cluster sehingga titik dalam cluster sama dan antar cluster berbeda. K-Means adalah algoritma paling terkenal: iterasi assign-titik-ke-centroid-terdekat lalu update centroid = mean cluster. K-Means mengasumsikan cluster berbentuk bola (isotropic variance) — gagal pada bentuk elips/irregular.

**Memilih k.** Tidak ada jawaban pasti; gunakan:
- *Elbow method*: plot inertia (within-cluster SS) vs k; pilih "siku".
- *Silhouette score*: seberapa mirip titik dengan clusternya vs cluster tetangga; range [-1,1], >0.5 bagus.
- *Gap statistic*: bandingkan inertia dengan data acak.
- *Domain knowledge*: sering lebih penting dari metrik.

**Alternatif clustering.**
- *Hierarchical*: bangun dendrogram (agglomerative = bottom-up merge); bisa potong di level k.
- *DBSCAN*: berbasis kepadatan; menemukan cluster bentuk arbitrer + label noise; parameter eps & min_samples.
- *Gaussian Mixture*: soft clustering — setiap titik punya probabilitas ke tiap cluster; lebih fleksibel dari K-Means (cluster elips).
- *Spectral clustering*: untuk non-convex shapes.

**Dimensionality reduction.** PCA (linear, variance-maximizing) dibahas detail di modul sendiri. t-SNE dan UMAP untuk visualisasi non-linear — hati-hati: keduanya tidak preserve global structure dengan baik dan tidak bisa di-transform data baru secara straightforward.

**Evaluasi tanpa label.** Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz), stability analysis (cluster konsisten di subsample?), dan visualisasi. Jika ada label parsial, bisa pakai *clustering accuracy* setelah matching cluster->label optimal (Hungarian algorithm).

## Konsep Penting

1. **Clustering ≠ classification** — cluster adalah struktur data, bukan label yang benar; interpretasi domain tetap diperlukan.
2. **Scaling wajib** — clustering berbasis jarak sangat sensitif terhadap skala fitur; standardisasi dulu.
3. **K-Means random init** — hasil bergantung pada inisialisasi; pakai `n_init` > 1 atau k-means++.
4. **"Benar" tidak unik** — data yang sama bisa punya banyak partisi valid; tidak ada ground truth objektif.
5. **High-dimensional curse** — jarak menjadi tidak informatif di dimensi tinggi; kurangi dimensi dulu.

## Formula

- Silhouette: s(i) = (b(i) − a(i)) / max(a(i), b(i)).
- Davies-Bouldin: rata-rata similarity antar cluster; lebih kecil lebih baik.
- DBSCAN: titik core jika ≥ min_samples dalam eps.

## Contoh Kasus

Segmentasi pelanggan e-commerce: features (recency, frequency, monetary, categories). Standardize -> K-Means k=4 -> silhouette pilih k -> profilkan cluster ("whales", "browsers", "new", "dormant") -> beri nama bisnis. Bandingkan dengan GMM untuk probabilitas keanggotaan.

## Implementasi

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

X_scaled = StandardScaler().fit_transform(X)

for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_)
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.3f}")
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "K-Means assumes clusters are approximately:",
    "options": [
      "Spherical with similar variance",
      "Arbitrary shapes",
      "Linearly separable only",
      "Gaussian mixtures with different covariances"
    ],
    "answer": 0,
    "explanation": "K-Means minimizes Euclidean distance to centroid — works best on compact, isotropic clusters.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Silhouette score of -0.3 means:",
    "options": [
      "Points are on average closer to a neighboring cluster than their own",
      "Perfect clustering",
      "All points are noise",
      "k is too small"
    ],
    "answer": 0,
    "explanation": "Negative silhouette = likely misassigned; range [-1,1], higher better.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `choose_k_elbow(inertias, ks)` returning the k at the 'elbow' — approximate by max distance from line connecting first and last points.",
    "starter_code": "import numpy as np\n\ndef choose_k_elbow(inertias, ks):\n    inertias = np.asarray(inertias, dtype=float)\n    ks = np.asarray(ks)\n    pass\n",
    "function": "choose_k_elbow",
    "tests": [
      {"call": "choose_k_elbow([100, 60, 30, 25, 24], [2,3,4,5,6])", "expected": 4}
    ],
    "hints": ["Normalize coords to [0,1]", "Elbow = max perpendicular distance to line from first to last point"],
    "explanation": "Elbow method picks k where inertia curve bends.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `dbscan_simple(points, eps, min_samples)` returning cluster labels using a simple density-reachability implementation.",
    "starter_code": "import numpy as np\n\ndef dbscan_simple(points, eps, min_samples):\n    pass\n",
    "function": "dbscan_simple",
    "tests": [
      {"call": "dbscan_simple(np.array([[0,0],[0.1,0],[0,0.1],[10,10],[10.1,10],[10,10.1]]), 0.5, 2)", "expected": "check"}
    ],
    "hints": ["Core point: >=min_samples neighbors within eps", "Expand clusters via density-reachable; label noise as -1"],
    "explanation": "DBSCAN finds arbitrary shapes without specifying k.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You cluster customer data with K-Means k=5 and get silhouette 0.3. Your manager wants 'actionable segments'. Describe your process to validate, interpret, and communicate these clusters — and what could make the clusters misleading.",
    "rubric": [
      "Checks stability across seeds/subsamples",
      "Profiles clusters with domain features (not just centroids)",
      "Names segments meaningfully for business",
      "Acknowledges clusters reflect feature scaling/choice, not ground truth"
    ],
    "points": 2
  }
]
```

## Referensi

- sklearn clustering user guide
- "Pattern Recognition and ML" Bishop ch. 9
- scikit-learn docs — clustering comparison
