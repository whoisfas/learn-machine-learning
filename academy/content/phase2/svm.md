---
{
  "id": "p2-svm",
  "title": "Support Vector Machines",
  "phase": 2,
  "difficulty": "intermediate",
  "estimated_hours": 12,
  "objectives": [
    "Understand the maximum-margin classifier and support vectors",
    "Derive the soft-margin formulation and hinge loss",
    "Apply the kernel trick for non-linear boundaries",
    "Choose kernels and tune C/gamma parameters"
  ],
  "concepts": [
    "Maximum margin hyperplane",
    "Support vectors: points on or violating the margin",
    "Hard margin vs soft margin (slack variables)",
    "Hinge loss: max(0, 1 - y·f(x))",
    "Kernel trick: implicit feature mapping via kernel function",
    "Common kernels: linear, RBF/Gaussian, polynomial, sigmoid",
    "Dual formulation and support vector expansion",
    "One-vs-rest for multiclass"
  ],
  "formulas": [
    {"name": "Decision function", "latex": "f(x) = w^\\top x + b", "note": "Sign gives class; |f| = distance to boundary."},
    {"name": "Hinge loss", "latex": "L = \\max(0, 1 - y f(x))", "note": "Zero if correct with margin; linear penalty otherwise."},
    {"name": "RBF kernel", "latex": "K(x,z) = \\exp(-\\gamma \\|x-z\\|^2)", "note": "γ controls locality; small γ = smooth boundary."},
    {"name": "Primal SVM", "latex": "\\min \\frac{1}{2}\\|w\\|^2 + C\\sum_i \\xi_i", "note": "C trades margin width vs violations."}
  ],
  "references": [
    "sklearn.svm docs",
    "Cortes & Vapnik 1995 original paper",
    "ESL ch. 12"
  ]
}
---

## Penjelasan

Support Vector Machine mencari hyperplane yang memisahkan kelas dengan *margin* maksimum — jarak terbesar ke titik data terdekat. Intuisinya: boundary yang jauh dari data lebih robust terhadap noise.

**Maximum margin.** Untuk data linearly separable, ada banyak hyperplane pembatas; SVM memilih yang margin-nya terbesar. Margin = 2/||w||; maksimalkan margin = minimalkan ||w||². Hanya titik *tepat di* margin atau melanggarnya (support vectors) yang menentukan boundary — titik lain tidak berpengaruh.

**Soft margin.** Data real jarang separable sempurna. Soft-margin SVM memperkenalkan slack variable ξi ≥ 0 yang mengizinkan pelanggaran margin dengan penalti. Parameter C mengontrol trade-off: C besar = sedikit pelanggaran (margin sempit, bisa overfit); C kecil = banyak pelanggaran (margin lebar, lebih smooth).

**Hinge loss.** Loss per titik: L = max(0, 1 − y·f(x)). Jika y·f(x) ≥ 1 (benar dengan margin), loss=0; jika salah atau margin tipis, loss linear. Ini convex -> global optimum.

**Kernel trick.** Data tidak linearly separable bisa dipetakan ke ruang fitur lebih tinggi di mana linear separation mungkin — tanpa pernah menghitung mapping eksplisit. Kernel function K(x,z) = φ(x)·φ(z) menghitung inner product di ruang fitur. SVM dual formulation hanya butuh K, bukan φ.

**Kernel umum.**
- *Linear*: K = x·z — setara linear SVM.
- *RBF/Gaussian*: K = exp(−γ||x−z||²) — paling populer; γ kecil = decision boundary smooth/global, γ besar = lokal/complex.
- *Polynomial*: K = (γ x·z + r)^d.
- *Sigmoid*: K = tanh(γ x·z + r) — mirip neural net activation.

**Multiclass.** sklearn SVC menggunakan one-vs-rest: K binary classifier, tiap kelas vs sisanya.

## Konsep Penting

1. **Support vectors** — hanya subset data yang menentukan model; ini membuat SVM efisien di memory dan robust terhadap outlier jauh.
2. **C dan gamma tuning** — grid search CV; C terlalu besar -> overfit, gamma terlalu besar -> overfit lokal.
3. **Scaling wajib** — SVM sangat sensitif terhadap skala fitur; StandardScaler essential.
4. **Kernel bukan gratis** — kernel trick menghindari komputasi eksplisit tapi tetap O(n²) memory untuk kernel matrix; tidak scalable ke dataset sangat besar.
5. **Probabilitas** — SVC tidak output probabilitas native; `probability=True` pakai Platt scaling (internal CV, mahal).

## Formula

- Margin maximization: min ||w||² s.t. yi(w·xi+b) ≥ 1.
- Soft margin: min (1/2)||w||² + CΣξi.
- Dual: max Σαi − (1/2)ΣΣαiαj yi yj K(xi,xj).

## Contoh Kasus

Klasifikasi digit MNIST: RBF kernel SVM mencapai ~98% accuracy — historis SVM adalah state-of-art sebelum deep learning. Bandingkan dengan MLP pada data yang sama; SVM unggul di data kecil-menengah, MLP butuh data lebih banyak tapi lebih scalable.

## Implementasi

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
param_grid = {'svc__C': [0.1, 1, 10, 100],
              'svc__gamma': ['scale', 0.01, 0.1, 1]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best:", grid.best_params_, "acc:", grid.best_score_)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "In SVM, which training points determine the decision boundary?",
    "options": [
      "Support vectors only",
      "All training points equally",
      "The outliers only",
      "Points with highest loss"
    ],
    "answer": 0,
    "explanation": "Only points on/violating margin (SVs) have nonzero dual coefficients αi.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Increasing C in soft-margin SVM tends to:",
    "options": [
      "Decrease margin width, allow fewer violations (less regularization)",
      "Increase margin width",
      "Make the model linear",
      "Reduce support vectors to zero"
    ],
    "answer": 0,
    "explanation": "C is inverse regularization: large C = less slack penalty = narrower margin.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The kernel trick allows SVM to:",
    "options": [
      "Compute inner products in high-dimensional feature space without explicit mapping",
      "Train without data",
      "Avoid the curse of dimensionality",
      "Output calibrated probabilities"
    ],
    "answer": 0,
    "explanation": "K(x,z)=φ(x)·φ(z) computes implicitly; dual formulation needs only K.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "math",
    "question": "For RBF kernel K(x,z)=exp(-γ||x-z||²), what happens to K when x=z?",
    "answer": 1.0,
    "tolerance": 0.001,
    "hints": ["distance=0 -> exp(0)"],
    "explanation": "K(x,x)=1 always for RBF; self-similarity maximal.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your RBF SVM gets 95% train accuracy but 70% test. Walk through your diagnosis and tuning strategy — which hyperparameters do you adjust first and why?",
    "rubric": [
      "Identifies overfitting (high train, low test)",
      "Considers reducing C (more regularization) and/or gamma (smoother boundary)",
      "Mentions scaling check",
      "Proposes CV-based grid search over C/gamma with proper validation"
    ],
    "points": 2
  }
]
```

## Referensi

- Cortes & Vapnik (1995), 'Support-Vector Networks'
- sklearn.svm documentation
- ESL ch. 12 (SVM and kernels)
- 'Kernel Methods for Pattern Analysis' by Shawe-Taylor & Cristianini
