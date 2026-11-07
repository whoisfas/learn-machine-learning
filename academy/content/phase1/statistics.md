---
{
  "id": "p1-statistics",
  "title": "Statistics: Inference, Estimation, Hypothesis Testing",
  "phase": 1,
  "difficulty": "intermediate",
  "estimated_hours": 14,
  "objectives": [
    "Distinguish population parameters from sample statistics",
    "Compute confidence intervals and interpret them correctly",
    "Perform hypothesis tests (t-test, chi-square, ANOVA) and interpret p-values",
    "Understand bias, variance, and the bias-variance tradeoff"
  ],
  "concepts": [
    "Descriptive vs inferential statistics",
    "Sampling distributions and standard error",
    "Confidence intervals (z, t)",
    "Hypothesis testing: null/alternative, p-value, Type I/II errors, power",
    "t-test, chi-square test, ANOVA, Mann-Whitney",
    "Correlation vs causation",
    "Bias-variance decomposition",
    "Multiple comparisons problem"
  ],
  "formulas": [
    {"name": "Standard error", "latex": "SE = \\frac{s}{\\sqrt{n}}", "note": "Std of sampling distribution of mean."},
    {"name": "Confidence interval", "latex": "\\bar{x} \\pm t_{\\alpha/2} \\cdot SE", "note": "95% CI captures true mean 95% of time under repeated sampling."},
    {"name": "t-statistic", "latex": "t = \\frac{\\bar{x}-\\mu_0}{s/\\sqrt{n}}", "note": "Compare to t-distribution with n-1 df."},
    {"name": "Bias-variance", "latex": "E[(y-\\hat{f})^2] = \\text{Bias}^2 + \\text{Var} + \\sigma^2", "note": "Decomposition of prediction error."}
  ],
  "references": [
    "OpenStax Introductory Statistics (free)",
    "scipy.stats documentation",
    "'All of Statistics' by Wasserman"
  ]
}
---

## Penjelasan

Statistics menjembatani data sampel dan kesimpulan tentang populasi. Di ML, statistik dipakai di mana-mana: confidence interval untuk metrik, uji hipotesis untuk membandingkan model, dan bias-variance untuk memahami generalisasi.

**Deskriptif vs inferensial.** Mean, median, mode, variance, quantiles = deskriptif (menggambarkan data yang ada). Inferensial membuat klaim tentang populasi dari sampel: estimasi parameter, confidence interval, hypothesis test.

**Sampling distribution.** Jika kita ambil banyak sampel dan hitung mean tiap sampel, distribusi mean-mean itu = sampling distribution. Standar error SE = s/√n mengukur seberapa jauh sample mean bisa menyimpang dari population mean. CLT: untuk n besar, sampling distribution ~ Normal.

**Confidence interval.** CI 95% = x̄ ± t*·SE. Interpretasi yang benar: jika eksperimen diulang banyak kali, 95% CI mengandung parameter sebenarnya. BUKAN "95% peluang mean sebenarnya ada di interval ini" (parameter fixed, interval random). Untuk mean: pakai t-distribution (df=n−1) jika σ tidak diketahui.

**Hypothesis testing.** H0 (null): tidak ada efek. H1: ada efek. p-value = P(data sekurang-kurangnya ekstrem | H0 benar). Tolak H0 jika p < α (biasanya 0.05). Type I error = false positive (tolak H0 padahal benar); Type II = false negative. Power = 1 − P(Type II). *Effect size* (Cohen's d) mengukur besarnya efek — p-value saja tidak cukup.

**Uji umum.** t-test (2 mean), paired t-test (before/after), chi-square (kategorikal independence), ANOVA (k mean), Mann-Whitney (non-parametric). Korelasi ≠ kausalitas — confounding variables.

**Bias-variance.** Error prediksi = Bias² + Variance + σ² (noise tak tereduksi). Model kompleks: variance tinggi (overfit); model sederhana: bias tinggi (underfit). Regularization menurunkan variance dengan sedikit menambah bias.

## Konsep Penting

1. **p-value bukan probabilitas hipotesis benar** — ini probabilitas data|H0, sering disalahartikan.
2. **Signifikansi ≠ signifikansi praktis** — dengan n besar, efek kecil pun signifikan; laporkan effect size + CI.
3. **Multiple testing** — uji 20 hipotesis pada α=0.05 -> ~64% peluang minimal satu false positive; pakai Bonferroni/FDR.
4. **Sampling bias** — sampel tidak representatif merusak semua inferensi downstream.
5. **CI vs SE** — CI = estimasi ± margin; SE adalah komponen ketidakpastian.

## Formula

- CI mean: x̄ ± t_{n-1,α/2} · s/√n.
- t-stat: t = (x̄ − μ0)/(s/√n).
- Cohen's d: (μ1 − μ2)/σ_pooled.
- ANOVA F = MS_between/MS_within.

## Contoh Kasus

Membandingkan dua model klasifikasi pada data yang sama: pakai McNemar test (paired) atau bootstrap CI untuk selisih accuracy. Atau A/B test: uji t dua sampel independen, hitung effect size, laporkan CI untuk lift.

## Implementasi

```python
import numpy as np
from scipy import stats

# one-sample t-test
def t_test_mean(x, mu0=0):
    x = np.asarray(x)
    t, p = stats.ttest_1samp(x, mu0)
    ci = stats.t.interval(0.95, len(x)-1,
                          loc=x.mean(), scale=stats.sem(x))
    return t, p, ci

# two-sample t-test
t, p = stats.ttest_ind(group_a, group_b, equal_var=False)

# chi-square independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# bootstrap CI for difference in means
rng = np.random.default_rng(0)
diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
         for _ in range(10000)]
ci = np.percentile(diffs, [2.5, 97.5])
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A 95% CI for mean accuracy is [0.82, 0.88]. Which statement is correct?",
    "options": [
      "If we repeated the experiment many times, 95% of such CIs would contain the true mean",
      "There is a 95% probability the true mean is in [0.82,0.88]",
      "The true mean is definitely between 0.82 and 0.88",
      "The sample mean is 0.85 with 95% certainty"
    ],
    "answer": 0,
    "explanation": "Frequentist CI interpretation: coverage probability over repeated sampling.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "You test 20 features for correlation with the target at α=0.05 and find 2 'significant'. Expected false positives by chance alone?",
    "options": ["~1", "~0", "~10", "~2 guaranteed real"],
    "answer": 0,
    "explanation": "20 × 0.05 = 1 expected false positive under null; multiple testing inflates error.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `confidence_interval_mean(x, alpha=0.05)` returning (lower, upper) using t-distribution.",
    "starter_code": "import numpy as np\nfrom scipy import stats\n\ndef confidence_interval_mean(x, alpha=0.05):\n    x = np.asarray(x, dtype=float)\n    pass\n",
    "function": "confidence_interval_mean",
    "tests": [
      {"call": "confidence_interval_mean([1,2,3,4,5,6,7,8,9,10])[0] < 5.5", "expected": true},
      {"call": "confidence_interval_mean([1,2,3,4,5,6,7,8,9,10])[1] > 5.5", "expected": true},
      {"call": "confidence_interval_mean([10]*20, 0.05)", "expected": [10.0, 10.0], "tolerance": 0.001}
    ],
    "hints": ["stats.t.interval(1-alpha, df=n-1, loc=mean, scale=sem)", "return tuple of floats"],
    "explanation": "Standard CI for population mean.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `cohens_d(group1, group2)` returning Cohen's d effect size (pooled std).",
    "starter_code": "import numpy as np\n\ndef cohens_d(group1, group2):\n    g1 = np.asarray(group1, dtype=float)\n    g2 = np.asarray(group2, dtype=float)\n    pass\n",
    "function": "cohens_d",
    "tests": [
      {"call": "cohens_d([1,2,3],[4,5,6])", "expected": -3.0, "tolerance": 0.01},
      {"call": "cohens_d([1,2,3],[1,2,3])", "expected": 0.0, "tolerance": 0.001}
    ],
    "hints": ["pooled std: sqrt(((n1-1)s1²+(n2-1)s2²)/(n1+n2-2))", "d = (mean1-mean2)/sp"],
    "explanation": "Effect size independent of sample size.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "A model shows p=0.001 improvement over baseline on test set with n=100,000. Is this practically meaningful? What additional information do you need to decide?",
    "rubric": [
      "Notes statistical significance != practical significance",
      "Requests effect size / absolute improvement",
      "Mentions cost/complexity tradeoff and confidence interval",
      "Possibly notes data leakage or multiple testing concerns"
    ],
    "points": 2
  }
]
```

## Referensi

- OpenStax Introductory Statistics (free)
- scipy.stats documentation
- "All of Statistics" by Larry Wasserman
- ASA statement on p-values (2016)
