---
{
  "id": "p6-result-testing",
  "title": "Testing and Validating Results",
  "phase": 6,
  "difficulty": "advanced",
  "estimated_hours": 10,
  "objectives": [
    "Apply rigorous statistical tests to research results",
    "Validate results across multiple dimensions: seeds, datasets, splits",
    "Detect and report failures honestly",
    "Prepare results for publication-quality reporting"
  ],
  "concepts": [
    "Robustness testing: multiple seeds, data subsets, hyperparameters",
    "Statistical tests: t-test, Wilcoxon, permutation, bootstrap",
    "Confidence intervals and effect sizes",
    "Cross-validation for small datasets",
    "Failure analysis: error analysis, confusion matrices",
    "Publication standards: what reviewers expect",
    "Reproducibility verification"
  ],
  "formulas": [
    {"name": "Bootstrap CI", "latex": "[\\hat{\\theta}^*_{(\\alpha/2)}, \\hat{\\theta}^*_{(1-\\alpha/2)}]", "note": "Percentile bootstrap CI."},
    {"name": "Effect size", "latex": "d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_{pooled}}", "note": "Cohen's d."}
  ],
  "references": [
    "Phase 5: Statistical Significance, Hypothesis Testing",
    "Phase 5: Reproducible Research",
    "NeurIPS/ICML reproducibility guidelines"
  ]
}
---

## Penjelasan

Testing dan validating results adalah tahap di mana Anda membuktikan bahwa temuan Anda robust — bukan kebetulan, bukan overfitting, bukan artifact dari satu run. Ini adalah bedanya antara 'hasil menarik' dan 'kontribusi ilmiah'.

**Robustness testing.** Hasil harus stabil terhadap: (1) *random seeds* — run 5-10 seeds, report mean±std. (2) *data subsets* — bootstrap atau different splits. (3) *hyperparameters* — sensitivity analysis. (4) *implementation details* — different frameworks jika memungkinkan.

**Statistical tests.** Untuk model comparison: paired t-test (normal), Wilcoxon signed-rank (non-normal), permutation test (assumption-free), McNemar (classification). Untuk single metric CI: bootstrap. Selalu report effect size + CI + p-value.

**Cross-validation.** Untuk dataset kecil: k-fold CV untuk robust estimate; repeated CV untuk variance reduction. Untuk time series: walk-forward validation, bukan random split.

**Failure analysis.** Jangan hanya report successes — analisis failures: confusion matrix, error cases, out-of-distribution performance, adversarial robustness. Failures mengajarkan lebih banyak dari successes.

**Publication standards.** Reviewers expect: multiple runs, statistical significance, ablation studies, reproducibility (code+data), honest limitations. Paper tanpa ini akan ditolak atau diminta revision besar.

## Konsep Penting

1. **Single run is anecdote** — multiple seeds minimum.
2. **Statistics > p-values** — effect size + CI lebih informatif.
3. **Failures are findings** — analyze and report them.
4. **Reproducibility is mandatory** — code+data+seeds.
5. **Robustness > peak performance** — stable method lebih valuable dari lucky peak.

## Implementasi

```python
import numpy as np
from scipy import stats

def robustness_test(model_fn, seeds, X_train, y_train, X_test, y_test, metric_fn):
    """Test robustness across seeds."""
    scores = []
    for seed in seeds:
        model = model_fn(seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores.append(metric_fn(y_test, preds))
    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'cv': np.std(scores) / np.mean(scores)
    }

def comprehensive_test(method_scores, baseline_scores, alpha=0.05):
    """Full statistical comparison."""
    diffs = np.array(method_scores) - np.array(baseline_scores)
    t_stat, p_val = stats.ttest_rel(method_scores, baseline_scores)
    return {
        'method_mean': np.mean(method_scores),
        'baseline_mean': np.mean(baseline_scores),
        'mean_diff': diffs.mean(),
        'std_diff': diffs.std(),
        'cohens_d': diffs.mean() / (diffs.std() + 1e-8),
        't_stat': t_stat, 'p_value': p_val,
        'ci_95': (diffs.mean() - 1.96*diffs.std()/np.sqrt(len(diffs)),
                  diffs.mean() + 1.96*diffs.std()/np.sqrt(len(diffs))),
        'significant': p_val < alpha,
        'wins': np.sum(diffs > 0), 'total': len(diffs)
    }
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "The minimum number of seeds for robust ML evaluation is typically:",
    "options": ["3-5", "1", "10-20", "100"],
    "answer": 0,
    "explanation": "3-5 seeds minimum; more for high variance methods.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Failure analysis is important because:",
    "options": [
      "Failures reveal model limitations and guide improvement",
      "It makes papers longer",
      "Reviewers require it",
      "It is easy"
    ],
    "answer": 0,
    "explanation": "Understanding failures is as important as reporting successes.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "For comparing two classifiers on the same test set, the appropriate test is:",
    "options": [
      "McNemar test (uses discordant pairs)",
      "Independent t-test",
      "Chi-square test",
      "ANOVA"
    ],
    "answer": 0,
    "explanation": "Paired data requires paired tests; McNemar is designed for classifier comparison.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `error_analysis(y_true, y_pred, feature_matrix)` identifying top-k features most associated with errors.",
    "starter_code": "import numpy as np\n\ndef error_analysis(y_true, y_pred, feature_matrix):\n    errors = (y_true != y_pred)\n    correct = ~errors\n    # compare feature distributions between errors and correct\n    error_mean = feature_matrix[errors].mean(axis=0)\n    correct_mean = feature_matrix[correct].mean(axis=0)\n    diff = np.abs(error_mean - correct_mean)\n    top_k = np.argsort(-diff)[:5]\n    return {'top_features': top_k, 'error_rate': errors.mean()}\n",
    "function": "error_analysis",
    "tests": [
      {"call": "error_analysis([0,1,0,1],[0,0,1,1],[[1,2],[3,4],[5,6],[7,8]])['error_rate']", "expected": 0.5}
    ],
    "hints": ["Compare feature means between error and correct predictions"],
    "explanation": "Identify features associated with model errors.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your model achieves 95% accuracy on test set but fails catastrophically on a new hospital's data. Describe your validation strategy to catch this before deployment.",
    "rubric": [
      "Cross-dataset validation: test on multiple hospitals/domains",
          "Distribution shift detection: monitor input statistics",
      "Calibration analysis: confidence vs accuracy on new data",
      "Uncertainty quantification: flag low-confidence predictions",
      "Report limitations honestly: 'validated on X hospitals, may not generalize to Y'"
        ],
    "points": 3
  }
]
```

## Referensi

- Phase 5: Statistical Significance, Hypothesis Testing
- NeurIPS/ICML reproducibility guidelines
- 'The Measure and Mismeasure of ML' — Lipton & Steinhardt
