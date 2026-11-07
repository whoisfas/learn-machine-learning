---
{
  "id": "p6-baseline-comparison",
  "title": "Baseline Comparison",
  "phase": 6,
  "difficulty": "intermediate",
  "estimated_hours": 8,
  "objectives": [
    "Select appropriate baselines for fair comparison",
    "Implement baseline models correctly",
    "Analyze when your method beats or loses to baselines",
    "Report comparison results honestly"
  ],
  "concepts": [
    "Baseline types: simple models, prior methods, state-of-art",
    "Fair comparison: same data, splits, metrics, resources",
    "Baseline reproduction vs citation",
    "Statistical comparison of methods",
    "When baselines beat your method: analysis",
    "Ablation as baseline comparison",
    "Reporting: tables, significance tests, effect sizes"
  ],
  "formulas": [],
  "references": [
    "Phase 5: Benchmarking",
    "Phase 5: Statistical Significance",
    "Papers With Code for baseline implementations"
  ]
}
---

## Penjelasan

Baseline comparison adalah cara membuktikan kontribusi Anda meaningful. Tanpa baseline yang fair, improvement claim tidak bisa diverifikasi. Baseline yang baik: reproducible, fair, dan representative.

**Jenis baselines.**
- *Simple baselines*: majority class, mean prediction, linear model — sanity check.
- *Prior methods*: metode yang sudah published untuk masalah sama — literature comparison.
- *State-of-art*: metode terbaik saat ini — competitive comparison.
- *Ablation variants*: model Anda tanpa komponen tertentu — internal comparison.

**Fair comparison principles.** (1) *Same data*: train/test splits identik. (2) *Same preprocessing*: pipeline sama untuk semua methods. (3) *Same metrics*: evaluasi identik. (4) *Same resources*: compute budget sebanding. (5) *Statistical rigor*: multiple runs, significance tests.

**Baseline reproduction vs citation.** Mengutip angka dari paper lain bisa misleading — dataset version, preprocessing, atau protocol mungkin berbeda. Lebih baik reproduce baseline sendiri pada data Anda; jika tidak memungkinkan, jelaskan perbedaan.

**Ketika baseline menang.** Jika baseline outperform method Anda: (1) analisis kenapa — method Anda mungkin overfit, atau baseline lebih suitable untuk problem ini. (2) Investigate conditions: baseline mungkin lebih baik pada subset tertentu. (3) Report honestly — negative result adalah kontribusi.

**Reporting.** Tables: methods sebagai rows, datasets sebagai columns, metrics sebagai cells. Bold best per column. Report mean±std over multiple runs. Significance markers untuk improvements.

## Konsep Penting

1. **Baseline is control** — eksperimen tanpa control tidak valid.
2. **Fair > impressive** — fair comparison lebih penting dari menang di benchmark.
3. **Reproduce before compare** — citation numbers bisa misleading.
4. **Analyze losses** — baseline yang menang mengajarkan sesuatu tentang problem.
5. **Report all results** — jangan cherry-pick; publish semua comparisons.

## Implementasi

```python
import numpy as np
from scipy import stats

def baseline_comparison_table(results):
    """Generate comparison table from results dict."""
    # results: {method: {dataset: {metric: score}}}
    methods = list(results.keys())
    datasets = list(results[methods[0]].keys())
    metrics = list(results[methods[0]][datasets[0]].keys())

    table = {}
    for metric in metrics:
        table[metric] = {}
        for ds in datasets:
            table[metric][ds] = {}
            for method in methods:
                score = results[method][ds][metric]
                table[metric][ds][method] = score
    return table

def significance_vs_baseline(method_scores, baseline_scores, alpha=0.05):
    """Paired t-test comparing method vs baseline across datasets."""
    diffs = np.array(method_scores) - np.array(baseline_scores)
    t_stat, p_val = stats.ttest_rel(method_scores, baseline_scores)
    return {
        'mean_diff': diffs.mean(),
        'cohens_d': diffs.mean() / diffs.std(),
        't_stat': t_stat, 'p_value': p_val,
        'significant': p_val < alpha,
        'wins': np.sum(diffs > 0), 'losses': np.sum(diffs < 0)
    }
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A fair baseline comparison requires:",
    "options": [
      "Same data splits, preprocessing, metrics, and resources for all methods",
      "Using the best baseline only",
      "Running each method once",
      "Different test sets per method"
    ],
    "answer": 0,
    "explanation": "Controlled conditions ensure the comparison is meaningful.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Why reproduce baselines instead of citing published numbers?",
    "options": [
      "Published numbers may use different data versions, preprocessing, or protocols",
      "Reproduction is faster",
      "Citation is always wrong",
      "Baselines don't work"
    ],
    "answer": 0,
    "explanation": "Controlled comparison requires same conditions.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "When your method loses to a baseline, you should:",
    "options": [
      "Analyze why and report honestly — it may reveal problem insights",
      "Hide the result",
      "Tune until you win",
      "Claim baseline is wrong"
    ],
    "answer": 0,
    "explanation": "Negative results are scientifically valuable.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `win_rate(method_scores, baseline_scores)` returning fraction of datasets where method wins.",
    "starter_code": "import numpy as np\n\ndef win_rate(method_scores, baseline_scores):\n    method_scores = np.asarray(method_scores)\n    baseline_scores = np.asarray(baseline_scores)\n    diffs = method_scores - baseline_scores\n    return np.mean(diffs > 0)\n",
    "function": "win_rate",
    "tests": [
      {"call": "win_rate([0.9,0.8,0.7],[0.85,0.82,0.75])", "expected": 0.333, "tolerance": 0.01}
    ],
    "hints": ["Count positive diffs / total"],
    "explanation": "Win rate calculation.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your novel method achieves 92% accuracy but a simple gradient boosting baseline gets 95%. Write a paragraph explaining this result and what you'd investigate next.",
    "rubric": [
      "Acknowledge baseline superiority honestly",
          "Possible causes: method complexity, insufficient training, wrong inductive bias",
      "Investigate: ablation, hyperparameter sensitivity, data regime",
      "Consider if problem suits simpler methods — publish as analysis"
        ],
    "points": 2
  }
]
```

## Referensi

- Phase 5: Benchmarking, Statistical Significance
- Papers With Code for baseline implementations
