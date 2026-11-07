---
{
  "id": "p5-statistical-significance",
  "title": "Statistical Significance in ML Research",
  "phase": 5,
  "difficulty": "advanced",
  "estimated_hours": 12,
  "objectives": [
    "Distinguish statistical from practical significance",
    "Choose appropriate tests for ML metric comparisons",
    "Report confidence intervals and effect sizes",
    "Avoid p-hacking and significance chasing"
  ],
  "concepts": [
    "Statistical vs practical significance",
    "Confidence intervals for ML metrics",
    "Effect size measures: Cohen's d, odds ratio, AUC difference",
    "Multiple comparisons problem",
    "Bootstrap confidence intervals",
    "Permutation tests",
    "Bayesian alternatives to p-values",
    "Reproducibility of significance claims"
  ],
  "formulas": [
    {"name": "Confidence interval", "latex": "\\hat{\\theta} \\pm z_{\\alpha/2} \\cdot SE(\\hat{\\theta})", "note": "Range containing true parameter with confidence 1-\\alpha."},
    {"name": "Bootstrap CI", "latex": "[\\hat{\\theta}^*_{(\\alpha/2)}, \\hat{\\theta}^*_{(1-\\alpha/2)}]", "note": "Percentile bootstrap on resampled statistics."},
    {"name": "Effect size", "latex": "d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_{pooled}}", "note": "Standardized mean difference."}
  ],
  "references": [
    "Wasserstein & Lazar 'The ASA Statement on p-Values' (2016)",
    "Cumming 'Understanding The New Statistics'",
    "Dietterich 'Approximate Statistical Tests'",
    "scipy.stats documentation"
  ]
}
---

## Penjelasan

Statistical significance adalah konsep yang sering disalahgunakan di ML research. Memahami perbedaan antara *statistical significance* (apakah perbedaan bisa dijelaskan oleh chance?) dan *practical significance* (apakah perbedaan cukup besar untuk berarti?) adalah skill essential untuk research yang rigorous.

**Statistical significance.** p < α berarti kita menolak hipotesis null bahwa tidak ada perbedaan. Tapi p-value tergantung pada sample size: dengan n sangat besar, perbedaan yang sangat kecil pun signifikan secara statistik.

**Practical significance.** Effect size mengukur *seberapa besar* perbedaan — Cohen's d, odds ratio, absolute difference in accuracy/AUC. Perbedaan 0.1% accuracy mungkin signifikan statistik tapi tidak praktis signifikan (biaya deployment lebih besar dari benefit).

**Confidence intervals.** Lebih informatif dari p-value: [0.85, 0.92] memberi range plausible untuk true accuracy. Lebih baik: report CI + point estimate + effect size.

**Multiple comparisons.** Uji 20 hipotesis -> expected 1 false positive. Koreksi: Bonferroni (α/20), Benjamini-Hochberg (FDR), atau hierarchical testing.

**Bootstrap.** Resample test set 1000+ kali, hitung metrik setiap kali, ambil percentile CI. Paling fleksibel untuk metrik ML apapun.

## Konsep Penting

1. **p < 0.05 ≠ practically significant** — laporkan effect size dan CI.
2. **Pre-registration** — tentukan primary endpoint sebelum eksperimen.
3. **Multiple testing** — selalu koreksi atau deklarasikan eksploratori.
4. **Bootstrap > parametric** — metrik ML tidak selalu normal; bootstrap lebih robust.
5. **Reproducibility** — significance claim harus reproducible; report all seeds/runs.

## Formula

- CI: estimate ± z·SE.
- Bootstrap: percentile CI dari resampled statistics.
- Cohen's d: (μ1-μ2)/σ_pooled.
- Bonferroni: α/k.

## Contoh Kasus

Paper melaporkan "Model A significantly better than B (p=0.03)" tapi tidak report effect size. Investigasi: accuracy difference = 0.3%, n=100k -> p<0.001 tapi effect size Cohen's d=0.02 (negligible). Kesimpulan: statistically significant tapi practically meaningless.

## Implementasi

```python
import numpy as np
from scipy import stats

def bootstrap_ci_metric(metric_fn, y_true, y_pred_fn, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap CI for any ML metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        metric_samples.append(metric_fn(y_true[idx], y_pred_fn(idx)))
    lower = np.percentile(metric_samples, (1-ci)/2 * 100)
    upper = np.percentile(metric_samples, (1+ci)/2 * 100)
    return lower, upper

def effect_size_report(scores_a, scores_b):
    """Report effect size alongside p-value."""
    a, b = np.array(scores_a), np.array(scores_b)
    t_stat, p_val = stats.ttest_rel(a, b)
    diff = a - b
    cohens_d = diff.mean() / diff.std()
    return {
        'mean_diff': diff.mean(),
        'std_diff': diff.std(),
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_val,
        'ci_95': (diff.mean() - 1.96*diff.std()/np.sqrt(len(diff)),
                  diff.mean() + 1.96*diff.std()/np.sqrt(len(diff)))
    }

def bonferroni_correct(p_values, alpha=0.05):
    """Apply Bonferroni correction."""
    k = len(p_values)
    corrected_alpha = alpha / k
    return [p < corrected_alpha for p in p_values]
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Statistical significance means:",
    "options": [
      "The observed difference is unlikely under the null hypothesis",
      "The effect is large",
      "The result is important",
      "The model is good"
    ],
    "answer": 0,
    "explanation": "p<0.05 means <5% chance of this result if no true effect — not magnitude.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Practical significance refers to:",
    "options": [
      "Whether the effect size is large enough to matter in practice",
      "Whether p<0.05",
      "Whether the test was run correctly",
      "Whether the model converged"
    ],
    "answer": 0,
    "explanation": "A 0.01% improvement can be statistically significant but practically meaningless.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "math",
    "question": "Cohen's d for mean difference 2.0 and pooled std 4.0 is:",
    "answer": 0.5,
    "tolerance": 0.001,
    "hints": ["d = 2.0/4.0"],
    "explanation": "Medium effect size.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `report_significance(scores_a, scores_b, alpha=0.05)` returning dict with p-value, effect size, CI, and interpretation.",
    "starter_code": "import numpy as np\nfrom scipy import stats\n\ndef report_significance(scores_a, scores_b, alpha=0.05):\n    a = np.asarray(scores_a, dtype=float)\n    b = np.asarray(scores_b, dtype=float)\n    pass\n",
    "function": "report_significance",
    "tests": [
      {"call": "report_significance([0.8,0.82,0.85,0.87,0.9],[0.75,0.78,0.8,0.82,0.85])['p_value'] < 0.05", "expected": true}
    ],
    "hints": ["ttest_rel; cohens_d; CI = mean_diff ± 1.96*se; return dict"],
    "explanation": "Complete significance report.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "A paper reports 'p=0.049, statistically significant improvement'. The effect size is 0.2% accuracy. Write a paragraph critique and suggest how the authors should have reported this.",
    "rubric": [
      "p=0.049 barely significant; likely wouldn't survive correction",
          "0.2% accuracy is negligible practical significance",
      "Should report effect size + CI, not just p-value",
      "Consider if improvement justifies added complexity/cost"
        ],
    "points": 2
  }
]
```

## Referensi

- Wasserstein & Lazar (2016) 'The ASA Statement on p-Values'
- Cumming (2014) 'The New Statistics'
- Dietterich (1998) 'Approximate Statistical Tests'
- scipy.stats documentation
