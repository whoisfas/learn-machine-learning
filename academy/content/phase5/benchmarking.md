---
{
  "id": "p5-benchmarking",
  "title": "Benchmarking",
  "phase": 5,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Design fair and informative benchmarks",
    "Understand benchmark datasets and leaderboards",
    "Avoid benchmark overfitting and Goodhart's law",
    "Create custom benchmarks for new problems"
  ],
  "concepts": [
    "Benchmark design: task, dataset, metric, protocol",
    "Standard benchmarks: ImageNet, GLUE, MMLU, etc.",
    "Leaderboard dynamics and saturation",
    "Goodhart's law: 'when a measure becomes a target'",
    "Benchmark overfitting vs genuine progress",
    "Custom benchmark creation",
    "Reproducibility of benchmark results",
    "Statistical comparison on benchmarks"
  ],
  "formulas": [],
  "references": [
    "ImageNet, GLUE, SuperGLUE, MMLU papers",
    "Kaggle competition guidelines",
    "'The Measure and Mismeasure of ML' — various",
    "Papers With Code leaderboards"
  ]
}
---

## Penjelasan

Benchmark adalah standar untuk membandingkan metode — dataset + task + metric + protocol yang tetap. Benchmarks memungkinkan kemajuan terukur dalam ML, tapi juga menciptakan masalah: overfitting ke benchmark, Goodhart's law, dan distorsi riset.

**Komponen benchmark.** (1) *Task*: apa yang harus dilakukan (classify, generate, predict). (2) *Dataset*: data evaluasi yang fixed dan representative. (3) *Metric*: cara mengukur performa (accuracy, BLEU, F1). (4) *Protocol*: train/val/test split, allowed resources, evaluation procedure.

**Benchmark terkenal.** ImageNet (image classification), GLUE/SuperGLUE (NLP understanding), MMLU (massive multitask), HELM (holistic evaluation), COCO (object detection), KITTI (autonomous driving). Setiap benchmark memiliki karakteristik: saturasi (semua model ~sama), bias (data distribution), dan blind spots.

**Goodhart's law.** "When a measure becomes a target, it ceases to be a good measure." Contoh: optimasi ImageNet accuracy -> model yang overfit ke ImageNet quirks; optimasi BLEU -> text generation yang gaming metric tapi tidak natural. Solusi: multiple metrics, human evaluation, diverse benchmarks.

**Benchmark overfitting.** Dengan iterasi cukup, researcher bisa mengoptimasi benchmark secara tidak sengaja — hyperparameter tuning pada test set, data augmentation yang exploit benchmark quirks, atau model yang memorizes benchmark patterns. Indikator: improvement besar di benchmark tapi tidak di real-world deployment.

**Custom benchmarks.** Untuk domain baru, buat benchmark sendiri: pilih dataset representative, definisikan metric yang align dengan use case, dokumentasikan protocol, release publicly. Benchmark yang baik: challenging, fair, diverse, reproducible.

## Konsep Penting

1. **Benchmark != truth** — benchmark adalah proxy; real-world performance bisa berbeda.
2. **Saturation** — ketika semua model ~sama di benchmark, benchmark tidak lagi informatif; butuh yang lebih sulit.
3. **Multiple metrics** — satu metric tidak cukup; laporkan accuracy + robustness + efficiency + fairness.
4. **Hidden test sets** — test set yang tidak public mencegah overfitting; Kaggle model.
5. **Human evaluation** — untuk generation tasks, human judgment sering lebih reliable dari automatic metrics.

## Formula

- Benchmark score: single metric or composite.
- Statistical comparison: paired tests across benchmark instances.

## Contoh Kasus

Benchmark untuk medical imaging: bukan hanya accuracy, tapi sensitivity/specificity per disease, calibration, fairness across demographics, robustness to distribution shift. Custom benchmark dengan 10 hospitals, 5 diseases, 3 scanners — lebih informatif dari ImageNet-style single metric.

## Implementasi

```python
# Custom benchmark harness
def run_benchmark(models, datasets, metric_fn, protocol='standard'):
    """Run models on benchmark datasets and collect results."""
    results = {}
    for model_name, model in models.items():
        results[model_name] = {}
        for dataset_name, (X_train, y_train, X_test, y_test) in datasets.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = metric_fn(y_test, preds)
            results[model_name][dataset_name] = score
    return results

def benchmark_report(results, baseline=None):
    """Generate benchmark comparison report."""
    report = {}
    for model, scores in results.items():
        report[model] = {
            'mean': np.mean(list(scores.values())),
            'std': np.std(list(scores.values())),
            'per_dataset': scores
        }
    if baseline:
        for model in report:
            report[model]['vs_baseline'] = {
                ds: report[model]['per_dataset'][ds] - baseline['per_dataset'][ds]
                for ds in report[model]['per_dataset']
            }
    return report

# statistical comparison across datasets
from scipy import stats
def benchmark_significance(results, model_a, model_b):
    """Paired test across benchmark datasets."""
    a_scores = [results[model_a][ds] for ds in results[model_a]]
    b_scores = [results[model_b][ds] for ds in results[model_b]]
    t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
    return {'t': t_stat, 'p': p_val, 'mean_diff': np.mean(np.array(a_scores)-np.array(b_scores))}
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Goodhart's law in ML benchmarking refers to:",
    "options": [
      "When optimizing for a benchmark metric, the metric stops reflecting true performance",
      "Benchmarks are always accurate",
      "Models always overfit",
      "Leaderboards are unbiased"
    ],
    "answer": 0,
    "explanation": "Optimizing for a measure makes it a poor measure of what we actually care about.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "A benchmark is 'saturated' when:",
    "options": [
      "All models perform similarly, making it uninformative for comparison",
      "It has too much data",
      "It uses deep learning",
      "It is too easy"
    ],
    "answer": 0,
    "explanation": "Saturated benchmarks no longer differentiate methods.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The most important component of a fair benchmark is:",
    "options": [
      "Fixed protocol: same data, splits, metrics, and resources for all methods",
      "Large dataset",
      "Deep learning models",
      "Public leaderboard"
    ],
    "answer": 0,
    "explanation": "Fairness requires controlled comparison conditions.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `benchmark_comparison(results, model_a, model_b)` returning per-dataset differences and overall statistics.",
    "starter_code": "import numpy as np\n\ndef benchmark_comparison(results, model_a, model_b):\n    diffs = {}\n    for ds in results[model_a]:\n        diffs[ds] = results[model_a][ds] - results[model_b][ds]\n    pass\n    return {'per_dataset': diffs, 'mean_diff': np.mean(list(diffs.values())), 'win_rate': np.mean([d>0 for d in diffs.values()])}\n",
    "function": "benchmark_comparison",
    "tests": [
      {"call": "benchmark_comparison({'A':{'d1':0.9,'d2':0.8},'B':{'d1':0.85,'d2':0.82}},'A','B')['mean_diff']", "expected": 0.015, "tolerance": 0.001}
    ],
    "hints": ["Compute per-dataset diff; return mean and win rate"],
    "explanation": "Benchmark comparison analysis.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "A new model achieves 99.5% on ImageNet but fails on real-world deployment photos. Explain possible causes and how you'd design a better benchmark to catch this.",
    "rubric": [
      "Benchmark overfitting: model exploits ImageNet-specific patterns",
          "Distribution shift: deployment photos differ from ImageNet",
      "Better benchmark: multiple datasets, domain shifts, corruption robustness, human evaluation",
      "Consider out-of-distribution detection and uncertainty calibration"
        ],
    "points": 2
  }
]
```

## Referensi

- ImageNet, GLUE, SuperGLUE, MMLU, HELM papers
- Papers With Code leaderboards
- Kaggle competition guidelines
- 'The Measure and Mismeasure of ML' — Lipton & Steinhardt
- 'Benchmark lottery' discussions
