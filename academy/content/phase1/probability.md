---
{
  "id": "p1-probability",
  "title": "Probability and Random Variables",
  "phase": 1,
  "difficulty": "intermediate",
  "estimated_hours": 14,
  "objectives": [
    "Compute probabilities using axioms, conditional probability, Bayes' rule",
    "Work with discrete and continuous random variables",
    "Understand expectation, variance, covariance, and common distributions",
    "Apply probability to ML concepts: likelihood, MLE, naive Bayes"
  ],
  "concepts": [
    "Sample space, events, axioms of probability",
    "Conditional probability P(A|B), independence",
    "Bayes' theorem and posterior odds",
    "Discrete RVs: PMF, Bernoulli, Binomial, Poisson, Categorical",
    "Continuous RVs: PDF, CDF, Uniform, Normal, Exponential",
    "Expectation E[X], variance Var(X), covariance",
    "Joint, marginal, conditional distributions",
    "Law of large numbers, central limit theorem"
  ],
  "formulas": [
    {"name": "Bayes", "latex": "P(A|B) = \\frac{P(B|A)P(A)}{P(B)}", "note": "Posterior ∝ likelihood × prior."},
    {"name": "Expectation", "latex": "E[X] = \\sum_x x\\,p(x) \\;\\text{or}\\; \\int x f(x)\\,dx", "note": "Mean of distribution."},
    {"name": "Variance", "latex": "\\text{Var}(X) = E[(X-\\mu)^2] = E[X^2]-\\mu^2", "note": "Spread; std = sqrt(var)."},
    {"name": "Normal PDF", "latex": "f(x)=\\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-(x-\\mu)^2/2\\sigma^2}", "note": "Bell curve; ~68/95/99.7 rule."},
    {"name": "Binomial PMF", "latex": "P(X=k)=\\binom{n}{k}p^k(1-p)^{n-k}", "note": "n trials, k successes."}
  ],
  "references": [
    "Khan Academy probability",
    "'Introduction to Probability' by Bertsekas & Tsitsiklis (free)",
    "scipy.stats docs"
  ]
}
---

## Penjelasan

Probability memformalkan ketidakpastian — fondasi ML karena data adalah sampel acak dari distribusi yang tidak diketahui. Hampir semua algoritma ML bisa dibaca sebagai inferensi probabilistik.

**Aksioma & aturan.** P(Ω)=1, P(A)∈[0,1], P mutually exclusive events dijumlahkan. P(A∪B)=P(A)+P(B)−P(A∩B). *Conditional probability* P(A|B)=P(A∩B)/P(B). *Independence*: P(A∩B)=P(A)P(B). **Bayes**: posterior = likelihood × prior / evidence — inti naive Bayes, spam filter, dan kalibrasi model.

**Random variable.** Pemetaan outcome -> angka. **Discrete**: PMF p(x)=P(X=x). Distribusi penting:
- *Bernoulli(p)*: satu trial biner.
- *Binomial(n,p)*: jumlah sukses dalam n trial.
- *Poisson(λ)*: jumlah kejadian langka per interval; mean=var=λ.
- *Categorical*: distribusi atas k kelas (softmax output).

**Continuous**: PDF f(x) dengan ∫f=1; CDF F(x)=P(X≤x). Distribusi:
- *Uniform[a,b]*: konstan di interval.
- *Normal(μ,σ²)*: simetris bell curve; CLT bilang rata-rata sampel -> normal.
- *Exponential(λ)*: waktu antar kejadian Poisson; memoryless.

**Momen.** E[X]=Σx·p(x) (discrete) atau ∫x·f(x)dx (continuous). Var(X)=E[X²]−E[X]². Cov(X,Y)=E[(X−μx)(Y−μy)] — mengukur linear co-movement; correlation = cov/(σxσy) dinormalisasi ke [−1,1].

**LLN & CLT.** Law of large numbers: rata-rata sampel -> E[X] saat n→∞ (dasar konsistensi estimator). Central limit theorem: (X̄−μ)/(σ/√n) -> N(0,1) — alasan uji-z, confidence interval, dan noise Gaussian di ANN.

## Konsep Penting

1. **MLE** — pilih parameter yang memaksimalkan likelihood data; untuk Gaussian -> mean/variance MLE.
2. **Naive Bayes assumption** — fitur independen given class; bekerja baik meski asumsi salah (terutama teks).
3. **Prior vs posterior** — prior dari keyakinan awal; update dengan data via Bayes.
4. **Independence ≠ uncorrelated** — uncorrelated hanya linier; dependence non-linier bisa ada.
5. **Log-likelihood** — jumlah log lebih stabil numerik; produk probabilitas kecil -> underflow.

## Formula

- Naive Bayes: P(y|x) ∝ P(y)·Π P(xi|y).
- MLE Gaussian: μ̂ = mean, σ̂² = (1/n)Σ(xi−μ)².
- Conditional: P(A|B) = P(A∩B)/P(B).
- Covariance: Cov(X,Y) = E[XY] − E[X]E[Y].

## Contoh Kasus

Spam classifier naive Bayes: P(spam|words) ∝ P(spam)·Π P(wi|spam). Estimasi P(wi|spam) dari frekuensi kata di email spam dengan smoothing (Laplace) untuk kata yang belum pernah muncul. Bandingkan dengan posterior threshold.

## Implementasi

```python
import numpy as np
from scipy import stats

# MLE for Gaussian
def gaussian_mle(x):
    return x.mean(), x.var(ddof=0)

# log-likelihood of data under N(mu, sigma)
def gaussian_loglik(x, mu, sigma):
    return np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))

# Bayes update: prior odds * likelihood ratio
prior_odds = 0.1 / 0.9
lik_ratio = 0.95 / 0.10   # P(test+|disease)/P(test+|healthy)
post_odds = prior_odds * lik_ratio
posterior = post_odds / (1 + post_odds)
print(f"P(disease|+) = {posterior:.3f}")
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A disease has prevalence 1%. A test has sensitivity 99% and specificity 95%. Given a positive test, what is approximately P(disease | +)?",
    "options": ["~17%", "~99%", "~50%", "~1%"],
    "answer": 0,
    "explanation": "Bayes: (0.99*0.01)/(0.99*0.01 + 0.05*0.99) = 0.0099/0.0594 ≈ 16.7%.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "math",
    "question": "Compute E[X] for a fair die (values 1-6).",
    "answer": 3.5,
    "tolerance": 0.001,
    "hints": ["(1+2+3+4+5+6)/6"],
    "explanation": "Mean of discrete uniform 1..6 = 3.5.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `bayes_posterior(prior, likelihood_pos, likelihood_neg)` returning P(hypothesis | positive evidence) via Bayes' rule.",
    "starter_code": "def bayes_posterior(prior, likelihood_pos, likelihood_neg):\n    pass\n",
    "function": "bayes_posterior",
    "tests": [
      {"call": "bayes_posterior(0.5, 0.9, 0.1)", "expected": 0.9, "tolerance": 1e-9},
      {"call": "bayes_posterior(0.01, 0.99, 0.05)", "expected": 0.16666666666666666, "tolerance": 1e-6}
    ],
    "hints": ["posterior = prior*lik_pos / (prior*lik_pos + (1-prior)*lik_neg)"],
    "explanation": "Bayes' theorem in odds form.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `normal_pdf(x, mu, sigma)` returning the Gaussian PDF value.",
    "starter_code": "import numpy as np\n\ndef normal_pdf(x, mu, sigma):\n    pass\n",
    "function": "normal_pdf",
    "tests": [
      {"call": "normal_pdf(0.0, 0.0, 1.0)", "expected": 0.3989422804014327, "tolerance": 1e-6},
      {"call": "normal_pdf(1.0, 0.0, 1.0)", "expected": 0.24197072451914337, "tolerance": 1e-6}
    ],
    "hints": ["1/(sigma*sqrt(2pi)) * exp(-(x-mu)^2/(2 sigma^2))"],
    "explanation": "Gaussian density — appears in likelihoods everywhere.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Explain why multiplying many small probabilities underflows to 0 in floating point, and why naive Bayes implementations use log-probabilities. What is the mathematical justification for summing log-likelihoods?",
    "rubric": [
      "Product of small numbers < machine epsilon",
      "log(ab) = log a + log b converts product to sum",
      "Numerical stability without changing argmax"
    ],
    "points": 2
  }
]
```

## Referensi

- Bertsekas & Tsitsiklis, "Introduction to Probability" (free PDF)
- scipy.stats documentation
- "Pattern Recognition and Machine Learning" (Bishop), ch. 1-2
