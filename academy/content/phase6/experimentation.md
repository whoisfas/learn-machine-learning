---
{
  "id": "p6-experimentation",
  "title": "Running Research Experiments",
  "phase": 6,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Design and execute systematic experiment suites",
    "Log experiments for reproducibility and comparison",
    "Analyze results statistically",
    "Iterate based on experimental findings"
  ],
  "concepts": [
    "Experiment suite design: what to test, in what order",
    "Experiment tracking: parameters, metrics, artifacts",
    "Statistical analysis: significance, effect size, confidence intervals",
    "Ablation studies within experiment suites",
    "Result visualization for papers",
    "Iteration: hypothesis -> experiment -> analysis -> new hypothesis",
    "Handling negative results"
  ],
  "formulas": [],
  "references": [
    "Phase 5: Experimental Design, Hypothesis Testing, Statistical Significance",
    "Phase 5: Reproducible Research",
    "MLflow/W&B documentation"
  ]
}
---

## Penjelasan

Experimentation adalah inti research — di sinilah hipotesis bertemu data. Suite eksperimen yang baik menjawab pertanyaan spesifik secara sistematis, dengan dokumentasi yang memungkinkan reproduksi dan perbandingan.

**Experiment suite design.** Urutan eksperimen: (1) baseline reproduction — verify pipeline works. (2) Main hypothesis test — eksperimen utama. (3) Ablation studies — component contributions. (4) Robustness tests — different seeds, data subsets, hyperparameters. (5) Comparison experiments — vs baselines dan state-of-art.

**Experiment tracking.** Setiap eksperimen dicatat: hypothesis, parameters, data version, code version, metrics, artifacts (plots, models), timestamp, seed. Tools: MLflow, W&B, atau academy progress tracker.

**Statistical analysis.** Setiap comparison: effect size + confidence interval + p-value. Multiple comparisons: correction. Report all results, bukan hanya yang signifikan.

**Visualization untuk papers.** Figures yang informatif: learning curves, ablation bar charts, confusion matrices, ROC curves, attention maps. Setiap figure harus mendukung satu poin dalam paper.

**Iteration cycle.** Eksperimen -> analisis -> insight -> new hypothesis -> new experiment. Negative results memandu direction: "component X tidak berkontribusi" -> simplify atau investigate why.

## Konsep Penting

1. **Systematic > ad-hoc** — rencana eksperimen sebelum eksekusi.
2. **Log everything** — setiap eksperimen, setiap parameter, setiap hasil.
3. **Statistics matter** — single run tidak cukup; multiple seeds, significance tests.
4. **Negative results are data** — failed experiments inform next steps.
5. **Reproducibility** — eksperimen harus bisa diulang oleh orang lain.

## Implementasi

```python
import json, hashlib
from datetime import datetime
import numpy as np
from scipy import stats

def experiment_tracker():
    " ""Simple experiment tracking."""
    experiments = []
    def log(name, hypothesis, params, results, seed=42):
        record = {
            'id': len(experiments),
            'name': name,
            'hypothesis': hypothesis,
            'timestamp': datetime.now().isoformat(),
            'params': params,
            'results': results,
            'seed': seed,
            'code_hash': hashlib.md5(
                json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        }
        experiments.append(record)
        return record
    def compare(exp1_id, exp2_id, metric):
        e1 = experiments[exp1_id]['results'][metric]
        e2 = experiments[exp2_id]['results'][metric]
        return {
            'exp1': e1, 'exp2': e2,
            'diff': e1 - e2,
            'winner': 'exp1' if e1 > e2 else 'exp2'
        }
    return log, compare, experiments

def statistical_comparison(scores_a, scores_b, alpha=0.05):
    """Full statistical comparison report."""
    a, b = np.array(scores_a), np.array(scores_b)
    t_stat, p_val = stats.ttest_rel(a, b)
    diff = a - b
    return {
        'mean_a': a.mean(), 'mean_b': b.mean(),
        'mean_diff': diff.mean(), 'std_diff': diff.std(),
        'cohens_d': diff.mean() / diff.std(),
        't_stat': t_stat, 'p_value': p_val,
        'ci_95': (diff.mean() - 1.96*diff.std()/np.sqrt(len(diff)),
                  diff.mean() + 1.96*diff.std()/np.sqrt(len(diff))),
        'significant': p_val < alpha
    }
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "The first experiment in a research suite should typically be:",
    "options": [
      "Baseline reproduction to verify the pipeline works",
      "The most complex model",
      "The final evaluation",
      "Ablation studies"
    ],
    "answer": 0,
    "explanation": "Verify pipeline correctness before testing hypotheses.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Every experiment should be logged with:",
    "options": [
      "Hypothesis, parameters, data version, metrics, seed, code version",
      "Only the final accuracy",
      "Only the model name",
      "Nothing — just run it"
    ],
    "answer": 0,
    "explanation": "Complete logging enables reproducibility and comparison.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "When comparing two models, you should report:",
    "options": [
      "Effect size, confidence interval, and p-value across multiple runs",
      "Only the best single run",
      "Only training accuracy",
      "Only the winner"
    ],
    "answer": 0,
    "explanation": "Statistical rigor requires multiple runs and complete reporting.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `experiment_summary(results)` aggregating multiple experiment runs into mean±std per configuration.",
    "starter_code": "import numpy as np\nfrom collections import defaultdict\n\ndef experiment_summary(results):\n    # results: list of dicts with 'config' and 'score'\n    grouped = defaultdict(list)\n    for r in results:\n        key = json.dumps(r['config'], sort_keys=True)\n        grouped[key].append(r['score'])\n    summary = {}\n    for k, scores in grouped.items():\n        summary[k] = {'mean': np.mean(scores), 'std': np.std(scores), 'n': len(scores)}\n    return summary\n",
    "function": "experiment_summary",
    "tests": [
      {"call": "experiment_summary([{'config':{'lr':0.01},'score':0.8},{'config':{'lr':0.01},'score':0.82},{'config':{'lr':0.001},'score':0.75}])", "expected": "check"}
    ],
    "hints": ["Group by config; compute mean/std/n per group"],
    "explanation": "Aggregate experiment results.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your main hypothesis is rejected: the new method doesn't outperform baseline. Describe how you'd proceed — do you abandon, pivot, or investigate further? What would you report?",
    "rubric": [
      "Negative results are valid scientific contributions",
          "Investigate WHY: is the method flawed? wrong problem? insufficient data?",
      "Consider boundary conditions: maybe method works for specific cases",
      "Report honestly with analysis of failure modes",
      "Pivot to related question if original is truly dead end"
        ],
    "points": 3
  }
]
```

## Referensi

- Phase 5: Experimental Design, Hypothesis Testing, Statistical Significance
- Phase 5: Reproducible Research
- MLflow, Weights & Biases documentation
