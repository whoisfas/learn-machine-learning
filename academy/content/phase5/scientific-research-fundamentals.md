---
{
  "id": "p5-scientific-research-fundamentals",
  "title": "Scientific Research Fundamentals",
  "phase": 5,
  "difficulty": "intermediate",
  "estimated_hours": 8,
  "objectives": [
    "Understand the scientific method and its application to ML",
    "Distinguish exploration, hypothesis testing, and confirmatory research",
    "Formulate research questions that are answerable",
    "Recognize threats to validity in ML research"
  ],
  "concepts": [
    "Scientific method: observation -> hypothesis -> experiment -> analysis -> conclusion",
    "Exploratory vs confirmatory research",
    "Research question formulation (PICO-style for ML)",
    "Internal vs external validity",
    "Construct validity: does the metric measure what we claim?",
    "Reproducibility vs replicability vs generalizability",
    "Peer review process",
    "Pre-registration of hypotheses"
  ],
  "formulas": [],
  "references": [
    "Popper 'The Logic of Scientific Discovery'",
    "Ioannidis 'Why Most Published Research Findings Are False'",
    "ACM artifact review guidelines",
    "Pineau et al. 'Improving Reproducibility in ML Research'"
  ]
}
---

## Penjelasan

Scientific research adalah proses sistematis untuk menghasilkan pengetahuan baru melalui observasi, hipotesis, eksperimen, dan analisis. Di ML, research berarti bukan hanya "mencoba model" — melainkan menjawab pertanyaan ilmiah dengan metodologi yang rigorous.

**Scientific method.** (1) Observasi/pertanyaan: "mengapa model X gagal pada data Y?" (2) Hipotesis: "karena distribusi shift" — harus falsifiable. (3) Eksperimen: desain yang bisa memisahkan hipotesis dari alternatif. (4) Analisis: statistik, visualisasi, ablation. (5) Kesimpulan: terima/tolak hipotesis dengan tingkat keyakinan.

**Exploratory vs confirmatory.** Exploratory = mencari pola tanpa hipotesis awal (EDA, visualization). Confirmatory = menguji hipotesis yang sudah dispesifikasikan sebelumnya. Bahaya: *HARKing* (Hypothesizing After Results are Known) — mengarang hipotesis setelah melihat data. Pre-registration (menulis hipotesis sebelum eksperimen) mencegah ini.

**Research question yang baik.** Harus: spesifik (bukan "improve accuracy" tapi "improve accuracy on imbalanced fraud detection by >5%"), measurable (metrik yang jelas), feasible (data dan compute tersedia), relevant (signifikansi ilmiah/praktis), novel (belum dijawab di literature).

**Validity threats.**
- *Internal validity*: apakah eksperimen benar-benar menguji hipotesis? (confounding, leakage).
- *External validity*: apakah hasil generalize ke konteks lain? (dataset bias, domain specificity).
- *Construct validity*: apakah metrik mengukur apa yang kita klaim? (accuracy pada imbalance tidak mengukur "kualitas").

**Reproducibility levels.** Reproducible = same data + code -> same results. Replicable = independent team, same method -> similar results. Generalizable = results hold across contexts. ML research sering gagal di level pertama (tidak ada code/data) — reproducibility crisis.

## Konsep Penting

1. **Falsifiability** — hipotesis yang tidak bisa dibuktikan salah bukan hipotesis ilmiah.
2. **Pre-registration** — tulis hipotesis + metodologi sebelum eksperimen; anti-HARKing.
3. **Baseline is control** — eksperimen ML butuh baseline untuk membuktikan improvement bukan kebetulan.
4. **Negative results matter** — "metode X tidak bekerja untuk Y" adalah kontribusi ilmiah valid.
5. **Reproducibility checklist** — seed, data version, code, environment, hardware — semua harus terdokumentasi.

## Formula

- Statistical power: 1 - β (probability of detecting true effect).
- Effect size: standardized measure of practical significance.

## Contoh Kasus

Research question: "Does adding physics-informed loss improve neural operator accuracy on PDE solving?" Hipotesis: "PINN loss mengurangi error >20% pada advection-dominated problems." Eksperimen: train dengan/ tanpa physics loss pada 3 PDE benchmarks; bandingkan relative L2 error dengan paired t-test; ablation pada bobot physics loss.

## Implementasi

```python
# Reproducibility checklist implementation
def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def experiment_log(name, hypothesis, params, results, notes=""):
    import json, hashlib
    from datetime import datetime
    record = {
        'name': name,
        'hypothesis': hypothesis,
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'results': results,
        'code_hash': hashlib.md5(
            json.dumps(params, sort_keys=True).encode()).hexdigest()[:8],
        'notes': notes
    }
    with open('research_log.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A good research question in ML should be:",
    "options": [
      "Specific, measurable, feasible, relevant, and novel",
      "As broad as possible",
      "Already answered in literature",
      "Impossible to test"
    ],
    "answer": 0,
    "explanation": "Good research questions are focused, testable, and contribute new knowledge.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "HARKing refers to:",
    "options": [
      "Hypothesizing After Results are Known — presenting post-hoc findings as a priori hypotheses",
      "Using too much data",
      "Overfitting to test set",
      "Publishing negative results"
    ],
    "answer": 0,
    "explanation": "HARKing undermines confirmatory research; pre-registration prevents it.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Internal validity in an ML experiment means:",
    "options": [
      "The experiment correctly tests the hypothesis without confounding",
      "Results generalize to all domains",
      "The model is interpretable",
      "The code runs correctly"
    ],
    "answer": 0,
    "explanation": "Internal validity = causal inference within the experiment; external = generalization.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `pre_registration_template()` returning a dict with fields for hypothesis, methodology, metrics, and decision criteria to be filled before experiments.",
    "starter_code": "def pre_registration_template():\n    pass\n",
    "function": "pre_registration_template",
    "tests": [
      {"call": "set(pre_registration_template().keys()) >= {'hypothesis','methodology','metrics','decision_criteria'}", "expected": true}
    ],
    "hints": ["Return dict with keys: hypothesis, methodology, metrics, decision_criteria, timestamp"],
    "explanation": "Pre-registration structure for confirmatory research.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You discover that your model's improvement over baseline is entirely due to a data leakage bug, not the new method. Describe how you'd report this honestly and what you'd do next.",
    "rubric": [
      "Acknowledge the leakage and retract the claim",
          "Document the bug for reproducibility",
      "Fix leakage and re-run experiments",
      "Report negative result if method doesn't actually help",
      "This is a valuable contribution — prevents others from repeating the mistake"
        ],
    "points": 2
  }
]
```

## Referensi

- Popper, 'The Logic of Scientific Discovery'
- Ioannidis (2005) 'Why Most Published Research Findings Are False'
- ACM Artifact Review and Badging
- Pineau et al. (2021) 'Improving Reproducibility in ML Research'
- 'The Seven Reasons ML Results Don't Reproduce' — various
