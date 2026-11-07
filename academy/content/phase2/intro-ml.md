---
{
  "id": "p2-intro-ml",
  "title": "Introduction to Machine Learning",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 6,
  "objectives": [
    "Define ML and distinguish it from rule-based programming and statistics",
    "Identify tasks: classification, regression, clustering, ranking, generation",
    "Understand the ML pipeline end-to-end",
    "Recognize when ML is (and isn't) the right tool"
  ],
  "concepts": [
    "Supervised vs unsupervised vs self-supervised vs reinforcement learning",
    "Features, labels, training/validation/test splits",
    "Generalization, overfitting, underfitting",
    "The bias-variance tradeoff",
    "No free lunch theorem",
    "Inductive vs transductive vs unsupervised learning",
    "Curse of dimensionality"
  ],
  "formulas": [
    {"name": "Generalization gap", "latex": "R(h) = R_{emp}(h) + \\text{generalization error}", "note": "Test error >> train error signals overfitting."}
  ],
  "references": [
    "Mitchell, 'Machine Learning' (free)",
    "sklearn docs — Choosing the right estimator",
    "'The Hundred-Page Machine Learning Book' (free)"
  ]
}
---

## Penjelasan

Machine Learning = program yang memperbaiki performa pada tugas T berdasarkan pengalaman E (data), diukur oleh metrik P. Bukan "program yang menulis program" — melainkan sistem yang *belajar* pola dari data, bukan aturan eksplisit.

**Tiga paradigma utama.**
- *Supervised*: data berlabel (X, y); belajar mapping X->y. Klasifikasi (label diskrit) dan regresi (label kontinu).
- *Unsupervised*: data tanpa label; menemukan struktur — clustering, dimensionality reduction, density estimation.
- *Reinforcement learning*: agen belajar kebijakan melalui reward dari lingkungan (dibahas di Phase 4).

**Pipeline ML standar.** (1) Definisikan masalah & metrik sukses. (2) Kumpulkan data. (3) EDA & pembersihan. (4) Feature engineering. (5) Split train/val/test. (6) Pilih baseline sederhana. (7) Latih model kandidat. (8) Tune hyperparameter pada validation. (9) Evaluasi final pada test (sekali). (10) Deploy + monitor.

**Generalization.** Tujuan bukan memprediksi data training, melainkan data baru. *Overfitting*: model menghafal noise training; test error tinggi. *Underfitting*: model terlalu sederhana untuk menangkap pola; train & test error sama-sama tinggi. *Bias-variance tradeoff*: kompleksitas model menukar bias vs variance — sweet spot di tengah.

**No Free Lunch.** Tidak ada algoritma yang terbaik untuk semua masalah. Rata-rata atas semua kemungkinan masalah, semua algoritma sama. Ini berarti: pahami data Anda, coba beberapa model, dan biarkan validasi memutuskan.

**Kapan ML bukan jawaban?** Jika aturan bisnis eksplisit sudah cukup ("flag transaksi > $10k"), gunakan aturan. Jika data tidak mengandung sinyal (target acak), ML hanya akan menghafal noise. Jika interpretabilitas mutlak dibutuhkan dan akurasi bisa dikorbankan, model sederhana menang.

## Konsep Penting

1. **Baseline dulu** — selalu bandingkan dengan model trivial (majority class, mean) sebelum model kompleks.
2. **Test set hanya disentuh sekali** — tuning pada test = leakage.
3. **Label noise** — label salah meracuni training; deteksi via cross-validation disagreement.
4. **Distribution shift** — data test beda distribusi dari train; validasi harus mencerminkan target deployment.
5. **Metric alignment** — optimalkan metrik yang selaras dengan biaya bisnis, bukan sekadar accuracy.

## Formula

- Empirical risk: R_emp(h) = (1/n)Σ L(h(xi), yi).
- Generalization error ≈ complexity penalty + estimation error.

## Contoh Kasus

Deteksi email spam: supervised classification dengan features (kata, header, metadata). Alternatif rule-based gagal karena spammer beradaptasi — ML belajar pola baru dari data terbaru. Bandingkan dengan kasus "deteksi kemacetan" yang lebih baik sebagai time-series forecasting daripada klasifikasi statis.

## Implementasi

```python
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
print("baseline acc:", accuracy_score(y_test, baseline.predict(X_test)))
# Any real model must beat this number.
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A model achieves 99% training accuracy but 60% test accuracy. Diagnosis?",
    "options": ["Overfitting", "Underfitting", "Good generalization", "Label noise only"],
    "answer": 0,
    "explanation": "Large train-test gap = model memorized training data.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Which is a valid ML problem definition?",
    "options": [
      "T=classify spam, E=email corpus, P=precision@k",
      "T=be smart, E=read docs, P=happiness",
      "T=predict everything, E=all data, P=accuracy",
      "T=write code, E=github, P=stars"
    ],
    "answer": 0,
    "explanation": "Well-posed ML needs concrete task, experience data, measurable performance.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "No Free Lunch theorem implies:",
    "options": [
      "Algorithm choice doesn't matter",
      "No algorithm dominates all others across all problems",
      "More data never helps",
      "Deep learning is always best"
    ],
    "answer": 1,
    "explanation": "NFL: averaged over all possible problems, all algorithms perform equally. Choose based on your data.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "open",
    "question": "Describe the full ML pipeline for predicting customer churn. For each step, state what could go wrong and how you'd detect it.",
    "rubric": [
      "Covers problem definition, data collection, EDA, features, split, baseline, model, tuning, final eval",
      "Identifies leakage risks (e.g., features from future, target in features)",
      "Mentions distribution shift / temporal validation for churn",
      "Notes metric choice (churn is imbalanced -> PR-AUC/F1, not accuracy)"
    ],
    "points": 2
  }
]
```

## Referensi

- Tom Mitchell, "Machine Learning" (free PDF)
- sklearn — Choosing the right estimator flowchart
- "The Hundred-Page Machine Learning Book" by Andriy Burkov
