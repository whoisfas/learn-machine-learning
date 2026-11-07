---
{
  "id": "p5-experimental-design",
  "title": "Experimental Design",
  "phase": 5,
  "difficulty": "advanced",
  "estimated_hours": 12,
  "objectives": [
    "Design experiments that isolate causal effects",
    "Control confounding variables and bias",
    "Choose appropriate experimental designs: A/B, factorial, ablation",
    "Determine sample size and statistical power"
  ],
  "concepts": [
    "Independent, dependent, and control variables",
    "Randomization and blocking",
    "A/B testing for model comparison",
    "Factorial designs for multiple factors",
    "Ablation studies for component analysis",
    "Sample size calculation and power analysis",
    "Confounding and Simpson's paradox",
    "Quasi-experiments and observational studies"
  ],
  "formulas": [
    {"name": "Sample size", "latex": "n = \\frac{(z_{\\alpha/2} + z_\\beta)^2 \\sigma^2}{\\delta^2}", "note": "For detecting effect size δ with power 1-β."},
    {"name": "Cohen's d", "latex": "d = \\frac{\\mu_1 - \\mu_2}{\\sigma_{pooled}}", "note": "Standardized effect size."}
  ],
  "references": [
    "Fisher 'The Design of Experiments'",
    "Montgomery 'Design and Analysis of Experiments'",
    "Kohavi 'Trustworthy Online Controlled Experiments'",
    "scipy.stats for power analysis"
  ]
}
---

## Penjelasan

Experimental design adalah jantung scientific research — desain yang baik memisahkan efek yang kita minati dari noise dan confounding. Di ML, eksperimen buruk menghasilkan kesimpulan yang salah meski metrik terlihat bagus.

**Komponen eksperimen.** *Independent variable* = yang kita manipulasi (model architecture, hyperparameter). *Dependent variable* = yang kita ukur (accuracy, loss). *Control variables* = yang kita tahan tetap (dataset, random seed, hardware). *Confounding variables* = yang mempengaruhi keduanya tanpa kita sadari.

**Randomization.** Acak assignment ke treatment/control groups menghilangkan systematic bias. Dalam ML: random train/test split, random seeds, random hyperparameter sampling.

**Blocking.** Jika ada sumber variasi yang diketahui (mis. dataset berbeda, hardware berbeda), block by factor itu — analisis dalam setiap block lalu pool hasil. Mengurangi variance dan mengontrol confounder.

**Desain eksperimen umum di ML.**
- *A/B test*: bandingkan model A vs B pada data yang sama; metrik = perbedaan performa.
- *Factorial*: manipulasi multiple factors (mis. architecture × dataset) untuk melihat interaksi.
- *Ablation*: hapus/tambahkan satu komponen untuk mengukur kontribusinya.
- *Cross-validation*: repeated measures design; setiap fold adalah block.

**Sample size & power.** Berapa banyak data/perulangan dibutuhkan untuk mendeteksi effect size tertentu? Terlalu sedikit = underpowered (miss true effects); terlalu banyak = wasteful. Power analysis: n = (z_α/2 + z_β)²σ²/δ².

## Konsep Penting

1. **Randomization mengontrol unknown confounders** — tidak perlu tahu semua confounders, cukup acak.
2. **Blocking mengontrol known confounders** — lebih efisien dari randomization saja.
3. **Ablation adalah eksperimen paling informatif** — menunjukkan komponen mana yang benar-benar berkontribusi.
4. **Multiple comparisons** — uji banyak hipotesis -> false positives; koreksi (Bonferroni, FDR).
5. **Pre-registration** — desain eksperimen harus ditentukan sebelum melihat results.

## Formula

- Effect size: Cohen's d = (μ1-μ2)/σ_pooled.
- Power: 1-β = P(reject H0 | H1 true).
- Sample size: n = (z_α/2 + z_β)²·2σ²/δ² (two-sample).

## Contoh Kasus

Ablation study untuk model baru: baseline -> +component A -> +component B -> +both. Hasil menunjukkan B memberikan +5% accuracy tapi A tidak signifikan — A dihapus, model disederhanakan. Ini adalah experimental design yang baik: setiap eksperimen menjawab satu pertanyaan.

## Implementasi

```python
from scipy import stats
import numpy as np

def power_analysis(effect_size, alpha=0.05, power=0.8):
    """Calculate required sample size for two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / effect_size) ** 2 * 2
    return int(np.ceil(n))

def ab_test(model_a_scores, model_b_scores, alpha=0.05):
    """Paired t-test for model comparison."""
    diff = np.array(model_a_scores) - np.array(model_b_scores)
    t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
    return {
        'mean_diff': diff.mean(),
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'cohens_d': diff.mean() / diff.std()
    }

def factorial_design(factors):
    """Generate all combinations for factorial experiment."""
    import itertools
    return list(itertools.product(*factors.values()))
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Randomization in experiments primarily serves to:",
    "options": [
      "Control for unknown confounding variables",
      "Speed up experiments",
      "Reduce sample size",
      "Make results prettier"
    ],
    "answer": 0,
    "explanation": "Random assignment balances confounders across groups on average.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "An ablation study aims to:",
    "options": [
      "Measure the contribution of each component by removing it",
      "Test all combinations at once",
      "Find the best hyperparameters",
      "Compare different datasets"
    ],
    "answer": 0,
    "explanation": "Ablation isolates component effects — most informative experiment design.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "math",
    "question": "For a two-sample t-test with effect size d=0.5, alpha=0.05, power=0.8, approximately how many samples per group?",
    "answer": 64,
    "tolerance": 5,
    "hints": ["n = 2*(z_alpha+z_beta)^2/d^2; z_alpha≈1.96, z_beta≈0.84"],
    "explanation": "~64 per group for medium effect size.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `paired_ttest(scores_a, scores_b)` returning t-statistic and p-value for paired samples.",
    "starter_code": "import numpy as np\nfrom scipy import stats\n\ndef paired_ttest(scores_a, scores_b):\n    a = np.asarray(scores_a, dtype=float)\n    b = np.asarray(scores_b, dtype=float)\n    pass\n",
    "function": "paired_ttest",
    "tests": [
      {"call": "paired_ttest([0.8,0.82,0.85,0.87,0.9],[0.75,0.78,0.8,0.82,0.85])[1] < 0.05", "expected": true}
    ],
    "hints": ["stats.ttest_rel(a, b); return (t_stat, p_value)"],
    "explanation": "Paired t-test for model comparison on same data.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You run 20 experiments testing different hyperparameters and find 3 with p<0.05. Your colleague says '3 significant results!'. Explain the multiple comparisons problem and how you'd correct for it.",
    "rubric": [
      "20 tests at alpha=0.05 -> expected 1 false positive by chance",
          "Bonferroni correction: alpha/20 = 0.0025",
      "Or FDR correction (Benjamini-Hochberg) for less conservative",
      "Or pre-register primary hypothesis and treat others as exploratory"
        ],
    "points": 2
  }
]
```

## Referensi

- Fisher, 'The Design of Experiments' (1935)
- Montgomery, 'Design and Analysis of Experiments'
- Kohavi et al., 'Trustworthy Online Controlled Experiments'
- scipy.stats documentation (power analysis, t-tests)
