---
{
  "id": "p2-knn",
  "title": "K-Nearest Neighbors",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 6,
  "objectives": [
    "Understand instance-based learning and the k-NN algorithm",
    "Choose distance metrics and scaling appropriately",
    "Tune k and weight functions",
    "Recognize kNN's strengths and limitations"
  ],
  "concepts": [
    "Lazy learning / instance-based learning",
    "Distance metrics: Euclidean, Manhattan, Minkowski, cosine",
    "Curse of dimensionality",
    "k selection: bias-variance tradeoff",
    "Distance weighting vs uniform voting",
    "kd-tree and ball-tree for efficient search",
    "kNN for regression vs classification",
    "kNN as non-parametric density estimator"
  ],
  "formulas": [
    {"name": "Euclidean distance", "latex": "d(x,z) = \\sqrt{\\sum_i (x_i - z_i)^2}", "note": "L2 norm; most common."},
    {"name": "Manhattan distance", "latex": "d(x,z) = \\sum_i |x_i - z_i|", "note": "L1 norm; robust to outliers."},
    {"name": "Minkowski", "latex": "d_p(x,z) = (\\sum_i |x_i - z_i|^p)^{1/p}", "note": "p=1 Manhattan, p=2 Euclidean."},
    {"name": "kNN regression", "latex": "\\hat{y} = \\frac{1}{k}\\sum_{i \\in N_k(x)} y_i", "note": "Average of k nearest neighbors."}
  ],
  "references": [
    "sklearn.neighbors docs",
    "ESL ch. 2 (kNN as canonical non-parametric)",
    "Cover & Hart 1967 original kNN paper"
  ]
}
---

## Penjelasan

k-Nearest Neighbors adalah algoritma *lazy learning* paling sederhana: tidak ada training; untuk prediksi baru, cari k titik training terdekat dan voting (klasifikasi) atau rata-rata (regresi) label mereka.

**Cara kerja.** Simpan seluruh training set. Untuk query x: hitung jarak ke semua training points -> ambil k terdekat -> prediksi = majority class atau mean. "Lazy" karena semua komputasi terjadi saat inference, bukan training.

**Jarak.** Pilihan metrik sangat memengaruhi hasil:
- *Euclidean (L2)*: default; baik untuk data dense isotropic.
- *Manhattan (L1)*: lebih robust terhadap outlier dimensi.
- *Minkowski p*: generalisasi; p<1 tidak memenuhi triangle inequality (hati-hati).
- *Cosine*: untuk text/high-dim sparse; mengukur arah bukan magnitudo.
- *Jaccard/Hamming*: untuk binary/categorical.

**Scaling wajib.** kNN sepenuhnya berbasis jarak; fitur dengan skala besar mendominasi. Selalu StandardScaler/MinMaxScaler dulu.

**Memilih k.** k=1 -> overfit (boundary noisy); k=n -> underfit (prediksi konstan). Bias-variance: k kecil variance tinggi, k besar bias tinggi. Rule of thumb k≈√n tapi validasi lebih baik. Nilai k optimal biasanya ganjil untuk klasifikasi biner.

**Bobot jarak.** `weights='distance'` memberi bobot 1/d pada tetangga — tetangga dekat lebih berpengaruh. Berguna ketika kepadatan tidak uniform.

**Curse of dimensionality.** Di dimensi tinggi, jarak antar titik menjadi relatif seragam — semua titik "jauh" dan "dekat". kNN performa menurun tajam; solusinya: kurangi dimensi (PCA, feature selection), atau pakai metrik/algorithm berbeda.

**Kompleksitas.** Brute force O(n·d) per query; kd-tree/ball-tree O(log n) rata-rata untuk dimensi rendah; untuk dimensi tinggi kd-tree degenerasi ke brute force.

## Konsep Penting

1. **Lazy != cheap** — inference mahal untuk dataset besar; training gratis tapi prediksi O(n).
2. **kNN bukan probabilistik** — output adalah vote/mean, bukan probabilitas terkalibrasi (kecuali pakai distance-weighted voting).
3. **k harus di-tune per dataset** — tidak ada universal optimal.
4. **Noise sensitivity** — outlier di training set bisa "mencuri" tetangga; smoothing via k>1 membantu.
5. **Decision boundary non-linear** — kNN bisa fit boundary kompleks tanpa asumsi linearitas.

## Formula

- Euclidean: d = sqrt(Σ(xi−zi)²).
- kNN classifier: ŷ = argmax_c Σ_{i∈N_k} w_i · 1[y_i=c].
- kNN regressor: ŷ = Σ w_i y_i / Σ w_i.

## Contoh Kasus

Rekomendasi buku: representasikan user sebagai vector rating (dimensi = buku), query user baru -> k tetangga terdekat -> rekomendasikan buku yang dibeli tetangga tapi belum dibeli user. Atau deteksi anomali: titik dengan rata-rata jarak ke k tetangga besar = outlier.

## Implementasi

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

# tune k
for k in [1, 3, 5, 7, 11, 15]:
    knn = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=k, weights='distance'))
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k}: acc={scores.mean():.3f}")
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "kNN is called 'lazy learning' because:",
    "options": [
      "It doesn't learn a model during training; computation happens at prediction",
      "It is slow to predict",
      "It requires no data",
      "It uses lazy evaluation of features"
    ],
    "answer": 0,
    "explanation": "kNN stores training data and does all work at inference time.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Why must you scale features before kNN?",
    "options": [
      "Distance is dominated by features with larger magnitude",
      "kNN requires normalized probabilities",
      "To speed up kd-tree",
      "To prevent overfitting"
    ],
    "answer": 0,
    "explanation": "Euclidean distance sums squared differences; unscaled features with large ranges dominate.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "As k increases in kNN classification, bias typically:",
    "options": ["Increases", "Decreases", "Stays same", "Becomes zero"],
    "answer": 0,
    "explanation": "Larger k = smoother boundary = more bias, less variance.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `knn_predict(X_train, y_train, x_test, k, distance='euclidean')` returning predicted class label (majority vote; ties -> smallest label).",
    "starter_code": "import numpy as np\n\ndef knn_predict(X_train, y_train, x_test, k, distance='euclidean'):\n    X_train = np.asarray(X_train, dtype=float)\n    y_train = np.asarray(y_train)\n    x_test = np.asarray(x_test, dtype=float)\n    pass\n",
    "function": "knn_predict",
    "tests": [
      {"call": "knn_predict([[0,0],[0,1],[1,0]], [0,0,1], [0.1,0.1], 2)", "expected": 0},
      {"call": "knn_predict([[0],[10]], [0,1], [9], 1)", "expected": 1}
    ],
    "hints": ["Compute distances to all train points", "Take k smallest indices, majority vote"],
    "explanation": "Core kNN algorithm.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You apply kNN to a dataset with 10,000 features (text TF-IDF) and get poor accuracy. Diagnose why and propose three concrete remedies.",
    "rubric": [
      "Identifies curse of dimensionality / distance concentration",
      "Proposes feature selection or dimensionality reduction (PCA, TruncatedSVD)",
      "Suggests alternative metric (cosine) or algorithm",
          "Possibly mentions approximate NN for scalability"
        ],
    "points": 2
  }
]
```

## Referensi

- sklearn.neighbors documentation
- ESL ch. 2, 13
- Cover & Hart (1967) 'Nearest neighbor pattern classification'
