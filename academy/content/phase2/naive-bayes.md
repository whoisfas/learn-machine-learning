---
{
  "id": "p2-naive-bayes",
  "title": "Naive Bayes",
  "phase": 2,
  "difficulty": "beginner",
  "estimated_hours": 6,
  "objectives": [
    "Derive the naive Bayes classifier from Bayes' theorem",
    "Understand the conditional independence assumption and why it works",
    "Implement Gaussian, Multinomial, and Bernoulli variants",
    "Apply Laplace smoothing and handle zero probabilities"
  ],
  "concepts": [
    "Generative model: P(y|x) ∝ P(y)·ΠP(xi|y)",
    "Conditional independence assumption ('naive')",
    "GaussianNB for continuous features",
    "MultinomialNB for count data (text)",
    "BernoulliNB for binary features",
    "Laplace/add-alpha smoothing",
    "Log-space computation for numerical stability",
    "When naive Bayes beats discriminative models"
  ],
  "formulas": [
    {"name": "NB posterior", "latex": "P(y|x) \\propto P(y) \\prod_i P(x_i|y)", "note": "Product of class prior and feature likelihoods."},
    {"name": "GaussianNB", "latex": "P(x_i|y) = \\mathcal{N}(\\mu_{y,i}, \\sigma^2_{y,i})", "note": "Per-class per-feature Gaussian."},
    {"name": "MultinomialNB", "latex": "P(x_i|y) = \\frac{N_{yi} + \\alpha}{N_y + \\alpha V}", "note": "Smoothed word counts; alpha=1 Laplace."}
  ],
  "references": [
    "sklearn.naive_bayes docs",
    "ESL ch. 6.6",
    "Maron 1961 original NB paper"
  ]
}
---

## Penjelasan

Naive Bayes adalah classifier generatif yang memodelkan P(X,Y) lalu derive P(Y|X) via Bayes. Namanya "naive" karena asumsi keras: semua fitur independen *diberikan* kelas. Asumsi ini hampir selalu salah di dunia nyata, tapi classifier tetap bekerja sangat baik — fenomena yang masih dipelajari.

**Derivasi.** P(y|x) = P(y)P(x|y)/P(x). Model P(x|y) = Π P(xi|y) (conditional independence). P(x) konstan untuk semua kelas -> bisa diabaikan. Jadi: pilih kelas dengan P(y)·ΠP(xi|y) terbesar. Untuk stabilitas numerik, kerja di log-space: log P(y|x) = log P(y) + Σ log P(xi|y).

**Varian.**
- *GaussianNB*: fitur kontinu diasumsikan Gaussian per kelas: P(xi|y) = N(μyi, σ²yi). Parameter diestimasi dari mean/var per kelas. Cepat tapi sensitif terhadap outlier.
- *MultinomialNB*: fitur = hitungan (word counts di teks); P(xi|y) = (count + α)/(total + αV). Standar untuk text classification.
- *BernoulliNB*: fitur binary (ada/tidak); P(xi=1|y) = (count + α)/(n + 2α).
- *CategoricalNB*: fitur kategorikal.

**Smoothing.** Kata/fitur yang tidak pernah muncul di kelas y punya P=0 -> produk jadi 0 -> log = -inf. Laplace smoothing (α>0) menambahkan pseudo-count: P = (count + α)/(total + α·num_features). α=1 standar; α besar -> lebih smooth, mendekati uniform.

**Kapan NB unggul.** Dataset kecil (sedikit data untuk estimasi parameter diskrit), teks (MultinomialNB sangat kuat), data high-dimensional sparse. NB sering jadi baseline yang sulit dikalahkan di NLP. Kelemahannya: asumsi independensi melanggar korelasi fitur nyata; GaussianNB sensitif terhadap non-Gaussian distribusi.

## Konsep Penting

1. **Generative, bukan discriminative** — memodelkan P(X,Y), bukan P(Y|X) langsung; butuh lebih banyak asumsi tapi bisa generate data.
2. **Independence assumption dilanggar tapi tetap akurat** — argmax sering benar meski probabilitas tidak terkalibrasi; untuk ranking, NB sangat baik.
3. **Log-space wajib** — produk banyak probabilitas kecil -> underflow; selalu pakai log.
4. **Zero frequency problem** — smoothing bukan opsional untuk MultinomialNB.
5. **GaussianNB vs LDA** — terkait erat; LDA shared covariance, QDA per-class covariance.

## Formula

- Posterior ∝ P(y)·Π P(xi|y).
- GaussianNB: log P(xi|y) = −0.5·log(2πσ²) − (xi−μ)²/(2σ²).
- Multinomial smoothed: P(wi|y) = (Nyi + α)/(Ny + αV).

## Contoh Kasus

Filter spam: MultinomialNB pada TF-IDF counts kata. "viagra" muncul di 80% email spam vs 0.1% ham -> log-likelihood ratio besar. Meskipun kata-kata berkorelasi ("free" dan "offer" sering bersama), NB tetap akurat karena argmax robust.

## Implementasi

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# text classification
gnb_text = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
gnb_text.fit(X_train_text, y_train)

# continuous features
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("classes:", gnb.classes_)
print("priors:", gnb.class_prior_)
# inspect learned Gaussian params per class
print("means:", gnb.theta_)   # (n_classes, n_features)
print("vars:", gnb.var_)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Naive Bayes assumes features are conditionally independent given the class. This assumption is:",
    "options": [
      "Almost always violated in practice, but the classifier often still works well",
      "Always true for text data",
      "Required for the math to work",
      "Only needed for GaussianNB"
    ],
    "answer": 0,
    "explanation": "The 'naive' assumption is rarely true; NB works because argmax is robust to violated independence.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Why is log-space used in naive Bayes computation?",
    "options": [
      "Product of many small probabilities underflows to 0; sum of logs is stable",
      "Logs are faster to compute",
      "It simplifies the math",
      "To handle negative probabilities"
    ],
    "answer": 0,
    "explanation": "log(Πpi) = Σlog(pi); avoids floating-point underflow.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `gaussian_log_likelihood(x, mu, var)` returning log N(x; mu, var) for scalar x.",
    "starter_code": "import numpy as np\n\ndef gaussian_log_likelihood(x, mu, var):\n    pass\n",
    "function": "gaussian_log_likelihood",
    "tests": [
      {"call": "gaussian_log_likelihood(0.0, 0.0, 1.0)", "expected": -0.9189385332046727, "tolerance": 1e-6},
      {"call": "gaussian_log_likelihood(1.0, 0.0, 1.0)", "expected": -1.4189385332046727, "tolerance": 1e-6}
    ],
    "hints": ["-0.5*log(2*pi*var) - (x-mu)^2/(2*var)"],
    "explanation": "Gaussian log-density used in GaussianNB.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `multinomial_nb_log_proba(word_counts, class_word_counts, class_totals, alpha)` returning dict of log P(wi|y) per class with Laplace smoothing.",
    "starter_code": "import numpy as np\n\ndef multinomial_nb_log_proba(word_counts, class_word_counts, class_totals, alpha=1.0):\n    pass\n",
    "function": "multinomial_nb_log_proba",
    "tests": [
      {"call": "multinomial_nb_log_proba([2,1], {0:[10,5],1:[1,9]}, {0:15,1:10}, 1.0)", "expected": "check"}
    ],
    "hints": ["For each class: P(wi|y) = (count_yi + alpha) / (total_y + alpha * vocab_size)", "return dict class->list of log probs"],
    "explanation": "Smoothed multinomial likelihood.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "GaussianNB assumes each feature is Gaussian within each class. Your data has a bimodal feature within one class. What happens to predictions, and what alternatives would you try?",
    "rubric": [
      "GaussianNB will poorly model the bimodal distribution, biased likelihoods",
          "Alternatives: discretize feature, use kernel density estimation, different classifier (QDA, trees, NN)",
      "Mentions checking feature distributions per class during EDA"
        ],
    "points": 2
  }
]
```

## Referensi

- sklearn.naive_bayes documentation
- ESL ch. 6.6
- 'The Elements of Statistical Learning' — generative vs discriminative
- scikit-learn text classification tutorial
