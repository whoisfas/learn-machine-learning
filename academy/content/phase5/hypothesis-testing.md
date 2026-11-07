---
{
  "id": "p5-hypothesis-testing",
  "title": "Hypothesis Testing for ML Research",
  "phase": 5,
  "difficulty": "advanced",
  "estimated_hours": 12,
  "objectives": [
    "Formulate null and alternative hypotheses correctly",
    "Choose appropriate statistical tests for ML comparisons",
    "Interpret p-values and confidence intervals correctly",
    "Avoid common pitfalls: p-hacking, HARKing, underpowered tests"
  ],
  "concepts": [
    "Null vs alternative hypothesis",
    "p-value interpretation and limitations",
    "Type I and Type II errors",
    "Parametric vs non-parametric tests",
    "Paired vs independent tests",
    "Multiple comparisons correction",
    "Effect size vs statistical significance",
    "Bootstrap methods for ML metrics"
  ],
  "formulas": [
    {"name": "p-value", "latex": "p = P(\\text{data} \\geq \\text{observed} | H_0 \\text{ true})", "note": "NOT probability H0 is true."},
    {"name": "Type I error", "latex": "\\alpha = P(\\text{reject } H_0 | H_0 \\text{ true})", "note": "False positive rate."},
    {"name": "Type II error", "latex": "\\beta = P(\\text{fail to reject } H_0 | H_1 \\text{ true})", "note": "False negative rate; power = 1-\\beta."},
    {"name": "McNemar test", "latex": "\\chi^2 = \\frac{(b-c)^2}{b+c}", "note": "For comparing two classifiers on same data."}
  ],
  "references": [
    "Wasserstein & Lazar 'The ASA Statement on p-Values'",
    "scipy.stats documentation",
    "Dietterich 'Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms'",
    "'Statistical Comparisons of Classifiers over Multiple Data Sets' — Demšar 2006"
  ]
}
---

## Penjelasan

Hypothesis testing adalah framework statistik untuk memutuskan apakah data memberikan cukup bukti untuk menolak hipotesis null. Di ML research, ini digunakan untuk membandingkan model, menguji apakah improvement signifikan, dan memvalidasi temuan.

**Framework.** H0 (null): tidak ada efek/perbedaan (mis. "model A = model B"). H1 (alternative): ada efek (mis. "model A > model B"). Test statistic dihitung dari data; p-value = probabilitas memperoleh hasil sekurang-kurangnya ekstrem *jika H0 benar*. Tolak H0 jika p < α (biasanya 0.05).

**Kesalahan interpretasi umum.** p-value BUKAN probabilitas H0 benar; BUKAN probabilitas hasil karena kebetulan; BUKAN ukuran besarnya efek. p=0.03 berarti "jika H0 benar, 3% kemungkinan melihat data sekurang-kurangnya seextreme ini" — bukan "97% yakin H1 benar".

**Type I & II errors.** Type I (false positive): tolak H0 padahal benar — α = 0.05 berarti 5% false positive rate. Type II (false negative): gagal tolak H0 padahal H1 benar — β; power = 1-β = probabilitas mendeteksi efek yang ada. Power rendah = underpowered study.

**Test selection untuk ML.**
- *Model comparison pada data sama*: paired tests — McNemar (classification), paired t-test/Wilcoxon (regression scores).
- *Multiple datasets*: Friedman test + post-hoc Nemenyi; atau Wilcoxon signed-rank per dataset.
- *Bootstrap*: resample test set untuk CI pada metrik apapun — paling fleksibel.
- *Permutation test*: acak label untuk null distribution — assumption-free.

**Multiple comparisons.** Uji 20 hipotesis pada α=0.05 -> ~64% peluang minimal satu false positive. Koreksi: Bonferroni (α/k — konservatif), Benjamini-Hochberg FDR (kurang konservatif), atau pre-register primary hypothesis.

## Konsep Penting

1. **p < 0.05 ≠ practically significant** — dengan n besar, efek kecil pun signifikan; laporkan effect size + CI.
2. **Pre-registration** — tentukan hipotesis dan test sebelum melihat data.
3. **Multiple testing inflation** — selalu koreksi atau deklarasikan primary endpoint.
4. **Bootstrap > parametric untuk ML** — metrik ML (F1, AUC) tidak selalu normal; bootstrap CI lebih reliable.
5. **Negative results valid** — "tidak ada perbedaan signifikan" adalah temuan ilmiah yang valid.

## Formula

- p-value = P(statistic ≥ observed | H0).
- McNemar: χ² = (b-c)²/(b+c) untuk discordant pairs.
- Bootstrap CI: percentile method pada resampled statistics.
- Bonferroni: α_corrected = α/k.

## Contoh Kasus

Membandingkan Model A vs B pada 5 datasets: untuk setiap dataset, hitung perbedaan accuracy; gunakan Wilcoxon signed-rank pada 5 differences; atau Friedman test untuk multiple models. Laporkan effect size (mean difference) + 95% CI bootstrap.

## Implementasi

```python
import numpy as np
from scipy import stats

def bootstrap_ci(metric_fn, data, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for any metric."""
    rng = np.random.default_rng(seed)
    n = len(data)
    stats_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats_samples.append(metric_fn(data[idx]))
    lower = np.percentile(stats_samples, (1-ci)/2 * 100)
    upper = np.percentile(stats_samples, (1+ci)/2 * 100)
    return lower, upper

def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar test for comparing two classifiers."""
    # b = A correct, B wrong; c = A wrong, B correct
    b = np.sum((pred_a == y_true) & (pred_b != y_true))
    c = np.sum((pred_a != y_true) & (pred_b == y_true))
    if b + c == 0:
        return 0, 1.0
    chi2 = (abs(b - c) - 1)**2 / (b + c)  # continuity correction
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_value

def permutation_test(metric_fn, data, n_perm=1000, seed=42):
    """Permutation test for significance."""
    rng = np.random.default_rng(seed)
    observed = metric_fn(data)
    perm_stats = []
    for _ in range(n_perm):
        perm_data = data.copy()
        rng.shuffle(perm_data)
        perm_stats.append(metric_fn(perm_data))
    p_value = np.mean(np.array(perm_stats) >= observed)
    return observed, p_value
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A p-value of 0.03 means:",
    "options": [
      "If H0 is true, there's a 3% chance of seeing data this extreme",
      "There's a 97% chance H1 is true",
      "The effect is large",
      "The result is practically significant"
    ],
    "answer": 0,
    "explanation": "p-value is conditional on H0, not probability of hypothesis being true.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Type I error in hypothesis testing is:",
    "options": [
      "Rejecting H0 when it's actually true (false positive)",
      "Failing to reject H0 when H1 is true",
      "Making a type error in code",
      "Using wrong test"
    ],
    "answer": 0,
    "explanation": "Type I = false positive; controlled by alpha.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "McNemar test is appropriate for comparing classifiers because:",
    "options": [
      "It uses only discordant pairs (where classifiers disagree)",
      "It's faster",
      "It handles missing data",
      "It doesn't need labels"
    ],
    "answer": 0,
    "explanation": "McNemar focuses on cases where predictions differ — more powerful for paired comparison.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `bootstrap_mean_ci(data, n_boot=1000, ci=0.95)` returning (lower, upper) bootstrap CI for mean.",
    "starter_code": "import numpy as np\n\ndef bootstrap_mean_ci(data, n_boot=1000, ci=0.95, seed=42):\n    data = np.asarray(data, dtype=float)\n    rng = np.random.default_rng(seed)\n    n = len(data)\n    pass\n",
    "function": "bootstrap_mean_ci",
    "tests": [
      {"call": "bootstrap_mean_ci([1,2,3,4,5], 100, 0.95)[0] < 3.0", "expected": true},
      {"call": "bootstrap_mean_ci([1,2,3,4,5], 100, 0.95)[1] > 3.0", "expected": true}
    ],
    "hints": ["Resample with replacement; compute mean each time; take percentiles"],
    "explanation": "Bootstrap confidence interval.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You test 50 different model variants and report only the best one with p=0.01. Explain why this is problematic and how you'd design a more honest evaluation.",
    "rubric": [
      "Multiple comparisons: 50 tests -> expected 2.5 false positives at alpha=0.05",
          "Report all results or use correction (Bonferroni/FDR)",
      "Pre-register primary hypothesis; treat others as exploratory",
      "Use nested CV or held-out test set for final confirmation",
      "Report effect sizes and confidence intervals, not just p-values"
        ],
    "points": 2
  }
]
```

## Referensi

- Wasserstein & Lazar (2016) 'The ASA Statement on p-Values'
- Dietterich (1998) 'Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms'
- Demšar (2006) 'Statistical Comparisons of Classifiers over Multiple Data Sets'
- scipy.stats documentation
- 'Beyond Power Calculations for Hypothesis Testing' — various
