---
{
  "id": "p6-research-hypothesis",
  "title": "Formulating Research Hypotheses",
  "phase": 6,
  "difficulty": "intermediate",
  "estimated_hours": 6,
  "objectives": [
    "Distinguish hypothesis from research question and prediction",
    "Formulate testable, falsifiable hypotheses",
    "Design experiments to test hypotheses",
    "Handle hypothesis revision during research"
  ],
  "concepts": [
    "Null vs alternative hypothesis in ML context",
    "Directional vs non-directional hypotheses",
    "Hypothesis operationalization: making abstract testable",
    "Pre-registration of hypotheses",
    "Multiple hypotheses and correction",
    "Hypothesis revision vs HARKing",
    "Exploratory vs confirmatory analysis"
  ],
  "formulas": [
    {"name": "Hypothesis test", "latex": "H_0: \\theta = \\theta_0 \\quad vs \\quad H_1: \\theta \\neq \\theta_0", "note": "Null vs alternative."}
  ],
  "references": [
    "Phase 5: Hypothesis Testing",
    "Phase 5: Experimental Design",
    "'Hypothesis Development' — research methods texts"
  ]
}
---

## Penjelasan

Hipotesis adalah prediksi spesifik yang bisa diuji — jembatan antara research question (pertanyaan) dan eksperimen (jawaban). Hipotesis yang baik bisa disangkal oleh data; hipotesis yang buruk selalu "benar" sehingga tidak ilmiah.

**Jenis hipotesis.**
- *Null (H0)*: tidak ada efek/perbedaan — "Model A = Model B".
- *Alternative (H1)*: ada efek — "Model A > Model B" atau "Model A ≠ Model B".
- *Directional*: "Model A > Model B" (one-tailed).
- *Non-directional*: "Model A ≠ Model B" (two-tailed).

**Operationalization.** Hipotesis abstrak harus diterjemahkan ke measurable: "model baru lebih baik" -> "accuracy pada test set > baseline + 3% dengan p<0.05". Setiap istilah harus punya definisi operasional.

**Contoh hipotesis ML.** "Menambahkan physics-informed loss akan mengurangi relative L2 error pada PDE solving minimal 15% dibanding model tanpa physics loss, pada 3 benchmark datasets, dengan significance p<0.05." — specific, measurable, falsifiable.

**Hipotesis vs prediksi.** Hipotesis = testable statement tentang relationship; prediksi = expected outcome. "Model akan akurat" adalah prediksi; "Model A lebih akurat dari B karena mekanisme X" adalah hipotesis.

**Hipotesis revision.** Selama research, hipotesis bisa direvisi berdasarkan exploratory findings — tapi harus transparan: dokumentasikan revisi, bedakan confirmatory (pre-registered) vs exploratory (post-hoc) analysis.

## Konsep Penting

1. **Falsifiability** — jika data apapun bisa diinterpretasikan mendukung hipotesis, itu bukan hipotesis.
2. **Pre-registration** — tulis hipotesis sebelum eksperimen; anti-HARKing.
3. **Operationalization** — setiap konsep abstrak harus punya metrik konkret.
4. **Multiple hypotheses** — jika test banyak hipotesis, koreksi untuk multiple comparisons.
5. **Negative results valid** — menolak hipotesis adalah kontribusi ilmiah.

## Contoh Kasus

Research question: "Does data augmentation improve small-data performance?"
Hipotesis H1: "Rotation + flip augmentation meningkatkan accuracy pada CIFAR-10 subset 1000 images minimal 5% vs no augmentation."
H0: "Augmentation tidak meningkatkan accuracy >2%."
Eksperimen: train dengan/ tanpa augmentation pada 5 seeds; paired t-test; laporkan effect size.

## Implementasi

```python
def hypothesis_template():
    return {
        'research_question': '',
        'null_hypothesis': '',
        'alternative_hypothesis': '',
        'directional': True,
        'operationalization': {
            'metric': '',
            'threshold': '',
            'significance_level': 0.05,
            'test': ''
        },
        'pre_registered': False,
        'revised': False,
        'revision_reason': ''
    }

def hypothesis_testable(hyp):
    """Check if hypothesis is testable."""
    checks = {
        'has_h0': bool(hyp.get('null_hypothesis')),
        'has_h1': bool(hyp.get('alternative_hypothesis')),
        'has_metric': bool(hyp.get('operationalization', {}).get('metric')),
        'has_threshold': bool(hyp.get('operationalization', {}).get('threshold')),
        'has_test': bool(hyp.get('operationalization', {}).get('test')),
        'falsifiable': 'not' not in hyp.get('alternative_hypothesis', '').lower() or '?' in hyp.get('alternative_hypothesis', '')
    }
    return {'testable': all(checks.values()), 'checks': checks}
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A falsifiable hypothesis is one that:",
    "options": [
      "Can be proven wrong by evidence",
      "Is always true",
      "Cannot be tested",
      "Is vague"
    ],
    "answer": 0,
    "explanation": "Scientific hypotheses must be testable and potentially refutable.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Operationalization means:",
    "options": [
      "Translating abstract concepts into measurable variables",
      "Making hypotheses vague",
      "Running experiments",
      "Writing papers"
    ],
    "answer": 0,
    "explanation": "'Better performance' -> 'accuracy > baseline + 5%' is operationalization.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "HARKing refers to:",
    "options": [
      "Presenting post-hoc findings as if they were pre-registered hypotheses",
      "Using too much data",
      "Overfitting",
      "Publishing too fast"
    ],
    "answer": 0,
    "explanation": "Hypothesizing After Results are Known undermines scientific validity.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `test_hypothesis(scores_treatment, scores_control, alpha=0.05)` performing one-tailed t-test and returning decision.",
    "starter_code": "import numpy as np\nfrom scipy import stats\n\ndef test_hypothesis(scores_treatment, scores_control, alpha=0.05):\n    t_stat, p_val = stats.ttest_ind(scores_treatment, scores_control, alternative='greater')\n    return {\n        't_stat': t_stat,\n        'p_value': p_val,\n        'reject_null': p_val < alpha,\n        'effect_size': np.mean(scores_treatment) - np.mean(scores_control)\n    }\n",
    "function": "test_hypothesis",
    "tests": [
      {"call": "test_hypothesis([0.9,0.92,0.95,0.97,0.99],[0.8,0.82,0.85,0.87,0.9])['reject_null']", "expected": true}
    ],
    "hints": ["One-tailed t-test: treatment > control"],
    "explanation": "Hypothesis testing implementation.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your exploratory analysis reveals an unexpected pattern: model performs worse on images with text. You now want to test this formally. Describe how to transition from exploratory finding to confirmatory hypothesis without HARKing.",
    "rubric": [
      "Document exploratory finding transparently",
          "Formulate explicit hypothesis about text-containing images",
      "Pre-register hypothesis and experimental design",
      "Collect new data or use held-out test set to avoid circularity",
      "Report as confirmatory only if pre-registered; otherwise label exploratory"
        ],
    "points": 2
  }
]
```

## Referensi

- Phase 5: Hypothesis Testing, Experimental Design
- 'Hypothesis Development' — research methodology texts
