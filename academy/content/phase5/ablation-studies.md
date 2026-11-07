---
{
  "id": "p5-ablation-studies",
  "title": "Ablation Studies",
  "phase": 5,
  "difficulty": "advanced",
  "estimated_hours": 10,
  "objectives": [
    "Design ablation studies that isolate component contributions",
    "Distinguish between necessary and sufficient components",
    "Analyze interaction effects between components",
    "Report ablation results clearly and honestly"
  ],
  "concepts": [
    "Ablation = removing components to measure their contribution",
    "Leave-one-out vs leave-one-in designs",
    "Factorial ablation for interaction effects",
    "Incremental vs decremental ablation",
    "Statistical significance of ablation results",
    "Ablation vs hyperparameter tuning distinction",
    "Common mistakes: insufficient ablation, confounded comparisons"
  ],
  "formulas": [
    {"name": "Component contribution", "latex": "\\Delta_i = M_{full} - M_{\\text{without } i}", "note": "Performance drop when removing component i."},
    {"name": "Interaction effect", "latex": "\\Delta_{ij} = M_{full} - M_{-i} - M_{-j} + M_{-ij}", "note": "Non-additive effects between components."}
  ],
  "references": [
    "Meyes et al. 'Ablation Studies in Artificial Neural Networks' (2020)",
    "NeurIPS/ICML paper guidelines on ablations",
    "'Ablation Interpretation' — various",
    "Transformer ablation studies (Vaswani et al.)"
  ]
}
---

## Penjelasan

Ablation study adalah eksperimen di mana komponen model dihapus atau dimodifikasi untuk mengukur kontribusi masing-masing terhadap performa akhir. Ini adalah cara paling informatif untuk memahami *mengapa* model bekerja — bukan hanya *bahwa* model bekerja.

**Tujuan ablation.** (1) Validasi klaim: apakah komponen X benar-benar diperlukan? (2) Understand mechanism: bagaimana komponen berkontribusi? (3) Simplify model: hapus komponen yang tidak signifikan. (4) Guide future work: di mana improvement bisa dilakukan?

**Desain ablation.**
- *Leave-one-out*: mulai dari model lengkap, hapus satu komponen per eksperimen. Baik untuk model dengan sedikit komponen.
- *Leave-one-in (incremental)*: mulai dari baseline minimal, tambahkan satu komponen per eksperimen. Baik untuk menunjukkan komponen mana yang essential.
- *Factorial*: semua kombinasi on/off untuk setiap komponen — paling informatif tapi mahal (2^k eksperimen untuk k komponen).

**Contoh ablation yang baik.** Paper transformer asli (Vaswani et al. 2017) melakukan ablation: hapus positional encoding -> BLEU turun 2.5; hapus multi-head -> turun 1.0; ganti ReLU dengan ReLU -> turun. Ini menunjukkan komponen mana yang critical.

**Kesalahan umum.**
- *Insufficient ablation*: hanya test satu konfigurasi, tidak cover semua komponen.
- *Confounded comparison*: mengubah komponen sekaligus dengan hyperparameter — tidak tahu mana yang menyebabkan perubahan.
- *Single run*: tidak ada variance estimate; hasil bisa noise.
- *No statistical test*: perbedaan kecil mungkin tidak signifikan.

## Konsep Penting

1. **Ablation ≠ hyperparameter tuning** — ablation menjawab "apakah komponen ini diperlukan?"; tuning menjawab "nilai optimal berapa?".
2. **Interaction effects** — komponen A dan B mungkin hanya bekerja bersama; ablation factorial mengungkap ini.
3. **Negative results valuable** — "komponen X tidak signifikan" adalah temuan yang memandu simplifikasi.
4. **Report all ablations** — jangan cherry-pick; publish semua results termasuk yang tidak signifikan.
5. **Statistical rigor** — multiple seeds, confidence intervals, significance tests.

## Formula

- Contribution: Δi = M_full - M_{-i}.
- Interaction: Δij = M_full - M_{-i} - M_{-j} + M_{-ij}.
- Significance: paired t-test atau Wilcoxon pada per-fold/per-seed differences.

## Contoh Kasus

Ablation untuk model "Graph Attention Network for fraud detection": baseline GCN -> +attention -> +edge features -> +temporal encoding. Hasil: attention +2.3% AUC, edge features +1.1%, temporal +0.5% (tidak signifikan). Kesimpulan: temporal encoding dihapus, model disederhanakan.

## Implementasi

```python
import numpy as np
from scipy import stats

def ablation_study(model_fn, components, base_config, metric_fn, n_seeds=5):
    """Run ablation: full model vs without each component."""
    results = {}
    for comp in ['full'] + [f'-{c}' for c in components]:
        scores = []
        for seed in range(n_seeds):
            config = base_config.copy()
            if comp != 'full':
                config['remove'] = comp[1:]
            model = model_fn(config, seed)
            scores.append(metric_fn(model))
        results[comp] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
    return results

def ablation_significance(full_scores, ablated_scores):
    """Test if ablation significantly reduces performance."""
    t_stat, p_value = stats.ttest_rel(full_scores, ablated_scores)
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_drop': np.mean(full_scores) - np.mean(ablated_scores)
    }

# factorial ablation
def factorial_ablation(model_fn, components, base_config, metric_fn):
    """Test all 2^k combinations."""
    import itertools
    results = {}
    for mask in itertools.product([0,1], repeat=len(components)):
        config = base_config.copy()
        active = [c for c, m in zip(components, mask) if m]
        config['components'] = active
        key = '+'.join(active) if active else 'baseline'
        results[key] = metric_fn(model_fn(config))
    return results
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "An ablation study measures:",
    "options": [
      "The contribution of each component by removing it and measuring performance change",
      "The best hyperparameters",
      "The fastest training time",
      "The model size"
    ],
    "answer": 0,
    "explanation": "Ablation isolates component effects — most informative for understanding model behavior.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Leave-one-out vs leave-one-in ablation differ in:",
    "options": [
      "LOO starts from full model; LOI starts from minimal baseline",
      "LOO is faster",
      "LOI is more accurate",
      "They are identical"
    ],
    "answer": 0,
    "explanation": "LOO = decremental; LOI = incremental. Both valid, different perspectives.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "A common mistake in ablation studies is:",
    "options": [
      "Changing multiple components at once, confounding the results",
      "Using too many seeds",
      "Testing too many components",
      "Reporting negative results"
    ],
    "answer": 0,
    "explanation": "One component per experiment isolates its effect; multiple changes confound attribution.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `component_contribution(full_score, ablated_score)` returning dict with absolute and relative contribution.",
    "starter_code": "def component_contribution(full_score, ablated_score):\n    pass\n",
    "function": "component_contribution",
    "tests": [
      {"call": "component_contribution(0.9, 0.8)", "expected": "check"}
    ],
    "hints": ["absolute = full - ablated; relative = (full-ablated)/full; return dict"],
    "explanation": "Quantify component contribution.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your ablation shows component X improves accuracy by 0.5% but the difference is not statistically significant (p=0.12). Should you keep X? Justify your decision considering model complexity, interpretability, and deployment constraints.",
    "rubric": [
      "Not significant -> could be noise; consider removing for simplicity",
          "But if X is cheap and doesn't hurt, keeping is defensible",
      "Consider Occam's razor: simpler model preferred if performance equal",
      "Report honestly: 'X shows trend but not significant; removed for parsimony'"
        ],
    "points": 2
  }
]
```

## Referensi

- Meyes et al. (2020) 'Ablation Studies in Artificial Neural Networks'
- NeurIPS/ICML reproducibility guidelines on ablations
- 'Ablation Interpretation' — various venues
- Transformer ablation in Vaswani et al. (2017)
