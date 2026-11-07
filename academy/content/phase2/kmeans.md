---
{
  "id": "p2-kmeans",
  "title": "K-Means Clustering",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": [
    "Derive the K-Means objective and Lloyd's algorithm",
    "Understand initialization methods (k-means++, random)",
    "Evaluate clustering quality without labels",
    "Handle scaling, outliers, and empty clusters"
  ],
  "concepts": [
    "Within-cluster sum of squares (WCSS/inertia)",
    "Lloyd's algorithm: assign-update iterations",
    "k-means++ initialization",
    "Convergence guarantees (monotone WCSS decrease)",
    "Elbow method, silhouette score, gap statistic",
    "Empty clusters and dead centroids",
    "Mini-batch K-Means for large data",
    "K-Means assumptions and failure modes"
  ],
  "formulas": [
    {"name": "WCSS objective", "latex": "J = \\sum_{k=1}^{K} \\sum_{x \\in C_k} \\|x - \\mu_k\\|^2", "note": "Minimize total within-cluster variance."},
    {"name": "Centroid update", "latex": "\\mu_k = \\frac{1}{|C_k|}\\sum_{x \\in C_k} x", "note": "Mean of assigned points minimizes WCSS."},
    {"name": "Silhouette", "latex": "s(i) = \\frac{b(i)-a(i)}{\\max(a(i),b(i))}", "note": "Cluster quality metric."}
  ],
  "references": [
    "sklearn.cluster.KMeans docs",
    "Arthur & Vassilvitskii 2007 k-means++ paper",
    "ESL ch. 14"
  ]
}
---

## Penjelasan

K-Means adalah algoritma clustering paling banyak digunakan. Tujuannya: partisi n titik ke k cluster sehingga jumlah kuadrat jarak ke centroid diminimalkan.

**Lloyd's algorithm.** Iterasi dua langkah sampai konvergen:
1. *Assignment*: tiap titik -> centroid terdekat (Euclidean distance).
2. *Update*: centroid baru = mean titik yang di-assign.
WCSS monoton menurun setiap iterasi -> dijamin konvergen ke local optimum (tidak global). Kompleksitas O(n·k·d·iter).

**Inisialisasi.** Random init bisa terjebak di local optimum buruk. *k-means++* memilih centroid pertama acak, lalu centroid berikutnya dengan probabilitas proporsional jarak kuadrat ke centroid terdekat yang sudah ada — mendorong centroid tersebar. sklearn default `n_init=10` (10 run, ambil terbaik) + k-means++.

**Memilih k.** Tidak ada metode sempurna:
- *Elbow*: plot inertia vs k; pilih titik di mana penurunan melandai.
- *Silhouette*: seberapa mirip titik dengan cluster sendiri vs tetangga; >0.5 bagus, <0.25 lemah.
- *Gap statistic*: bandingkan inertia dengan referensi data acak.
- *Domain*: sering paling penting — "berapa segmen bisnis yang actionable?"

**Masalah umum.**
- *Empty clusters*: centroid tanpa anggota; sklearn re-init centroid ke titik acak.
- *Outliers*: satu outlier jauh bisa menarik centroid; pertimbangkan trimming atau robust scaling.
- *Non-spherical clusters*: K-Means gagal pada cluster elips/irregular; GMM atau spectral clustering lebih baik.
- *High dimensions*: jarak tidak informatif; kurangi dimensi dulu.

**Variasi.** Mini-batch K-Means (sample subset per iterasi) untuk data besar; K-Medoids (PAM) menggunakan titik data sebagai centroid — lebih robust terhadap outlier tapi lebih lambat.

## Konsep Penting

1. **Scaling wajib** — K-Means pakai Euclidean distance; fitur skala besar mendominasi.
2. **k-means++ bukan sihir** — mengurangi local optimum buruk tapi tidak menghilangkannya; n_init>1 tetap penting.
3. **Inertia selalu turun dengan k** — lebih banyak cluster = WCSS lebih kecil; elbow/silhouette membantu pilih trade-off.
4. **Cluster != label** — interpretasi domain tetap diperlukan; cluster adalah struktur data, bukan kebenaran.
5. **Stability check** — jalankan dengan seed berbeda; cluster yang stabil lebih.

## Formula

- Assignment: c_i = argmin_k ||xi − μk||².
- Update: μk = mean{x : c_i = k}.
- WCSS: J = Σ||xi − μ_{c_i}||².

## Contoh Kasus

Segmentasi pelanggan retail: features (RFM: recency, frequency, monetary + categories purchased). StandardScaler -> K-Means k=5 -> silhouette 0.42 -> profilkan cluster -> beri nama (VIP, loyal, at-risk, new, dormant) -> gunakan untuk targeted marketing.

## Implementasi

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

X_scaled = StandardScaler().fit_transform(X)

# elbow + silhouette
inertias, sils = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))

plt.plot(range(2,11), inertias, 'o-'); plt.xlabel('k'); plt.ylabel('inertia')
plt.show()

best_k = np.argmax(sils) + 2
km = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit(X_scaled)
print("chosen k:", best_k, "silhouette:", sils[best_k-2])
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Lloyd's algorithm for K-Means is guaranteed to:",
    "options": [
      "Converge to a local optimum (WCSS monotonically decreases)",
      "Find the global optimum",
      "Always find the true clusters",
      "Run in O(n) time"
    ],
    "answer": 0,
    "explanation": "WCSS decreases each iteration -> converges, but to local optimum depending on init.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "k-means++ initialization improves over random init by:",
    "options": [
      "Choosing initial centroids spread apart (probability proportional to distance²)",
      "Using more clusters",
      "Removing outliers",
      "Guaranteeing global optimum"
    ],
    "answer": 0,
    "explanation": "k-means++ samples centroids with probability proportional to squared distance to nearest existing centroid.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `assign_clusters(X, centroids)` returning cluster index (0..k-1) for each point in X (nearest centroid by Euclidean distance).",
    "starter_code": "import numpy as np\n\ndef assign_clusters(X, centroids):\n    X = np.asarray(X, dtype=float)\n    centroids = np.asarray(centroids, dtype=float)\n    pass\n",
    "function": "assign_clusters",
    "tests": [
      {"call": "assign_clusters([[0,0],[10,10]], [[0,0],[10,10]]).tolist()", "expected": [0,1]},
      {"call": "assign_clusters([[5,5]], [[0,0],[10,10]]).tolist()", "expected": [0]}
    ],
    "hints": ["Compute pairwise distances; argmin along axis=1", "Use broadcasting: (X[:,None,:]-C[None,:,:])**2 sum axis=2 sqrt"],
    "explanation": "Assignment step of Lloyd's algorithm.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `update_centroids(X, labels, k)` returning new centroids as mean of assigned points; empty clusters get re-initialized to a random data point.",
    "starter_code": "import numpy as np\n\ndef update_centroids(X, labels, k, rng=None):\n    X = np.asarray(X, dtype=float)\n    labels = np.asarray(labels)\n    rng = rng or np.random.default_rng(0)\n    pass\n",
    "function": "update_centroids",
    "tests": [
      {"call": "update_centroids([[0,0],[2,2],[10,10]], [0,0,1], 2).tolist()", "expected": [[1.0,1.0],[10.0,10.0]]}
    ],
    "hints": ["For each cluster k: mask = labels==k; if any, mean of X[mask]; else random point"],
    "explanation": "Update step minimizes WCSS.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You run K-Means with k=10 on customer data and get silhouette 0.15. Your stakeholder wants '5 customer segments'. Explain how you'd reconcile the statistical result with the business request.",
    "rubric": [
      "Acknowledges silhouette 0.15 indicates weak cluster structure",
          "Tests k=5 specifically and reports its metrics",
      "Discusses stability across seeds and alternative algorithms (GMM, hierarchical)",
      "Frames business need vs data structure honestly; may suggest more features or different problem formulation"
        ],
    "points": 2
  }
]
```

## Referensi

- sklearn.cluster.KMeans documentation
- Arthur & Vassilvitskii (2007) k-means++ paper
- ESL ch. 14 (unsupervised learning)
- 'k-Means++: The Advantages of Careful Seeding' — detailed analysis
