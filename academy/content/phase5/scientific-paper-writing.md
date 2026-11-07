---
{
  "id": "p5-scientific-paper-writing",
  "title": "Scientific Paper Writing",
  "phase": 5,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Structure a research paper using IMRaD format",
    "Write clear abstracts, introductions, and conclusions",
    "Present figures and tables effectively",
    "Navigate peer review and revision"
  ],
  "concepts": [
    "IMRaD structure: Introduction, Methods, Results, Discussion",
    "Abstract writing: problem, method, results, conclusion in 150-250 words",
    "Introduction: motivation, gap, contribution, roadmap",
    "Methods: reproducibility, sufficient detail for replication",
    "Results: figures, tables, statistical reporting",
    "Discussion: interpretation, limitations, future work",
    "Related work positioning",
    "Peer review process and rebuttal"
  ],
  "formulas": [],
  "references": [
    "'Writing Science' by Schimel",
    "'Science Research Writing' — university guides",
    "NeurIPS/ICML author guidelines",
    "'The Elements of Style' for scientific writing",
    "Papers With Code for reproducibility"
  ]
}
---

## Penjelasan

Scientific paper writing adalah skill yang membedakan research yang baik dari research yang *dikomunikasikan* dengan baik. Paper yang buruk secara tulisan bisa menolak kontribusi ilmiah yang valid; paper yang baik secara tulisan bisa mempercepat adopsi dan dampak.

**IMRaD structure.** *Introduction*: motivation (why this problem matters), gap (what's missing), contribution (what you did), roadmap (paper structure). *Methods*: sufficient detail untuk replication — bukan cookbook, tapi cukup untuk reproduce. *Results*: figures, tables, statistical reporting — present data, bukan interpretasi. *Discussion/Conclusion*: interpretasi, limitations, implications, future work.

**Abstract.** 150-250 kata: problem (1-2 kalimat), method (2-3 kalimat), key results (2-3 kalimat dengan angka), conclusion (1-2 kalimat). Harus standalone — bisa dibaca tanpa paper.

**Introduction yang baik.** Mulai dari big picture -> narrow down ke specific problem -> identify gap -> state contribution -> preview results. Jangan mulai dengan "In this paper we..." — mulai dengan mengapa masalah ini penting.

**Methods section.** Cukup detail untuk replication: dataset description, preprocessing, model architecture, hyperparameters, training procedure, evaluation protocol. Sertakan statistical tests dan ablation studies. Jangan tulis "we used standard techniques" — sebutkan spesifiknya.

**Results presentation.** Figures > tables > text. Setiap figure: caption self-contained, axis labels, error bars, statistical significance markers. Tables: concise, bold best results, report mean±std over multiple runs.

**Discussion.** Interpretasi results dalam konteks literature: apakah results confirm atau contradict prior work? Apa limitations? Apa implications untuk field? Apa future directions?

## Konsep Penting

1. **Clarity > eloquence** — scientific writing harus presisi, bukan indah.
2. **One idea per paragraph** — struktur logis membantu reader follow argument.
3. **Figures are arguments** — setiap figure harus mendukung satu poin; caption harus self-contained.
4. **Report honestly** — negative results, limitations, dan failed experiments adalah bagian dari scientific integrity.
5. **Rebuttal is dialogue** — peer review bukan penolakan; rebuttal yang baik menjawab concerns dengan evidence.

## Formula

- Abstract length: 150-250 words typical.
- Paper structure: IMRaD + related work + conclusion.

## Contoh Kasus

Bad introduction: "In this paper, we propose a new method for image classification." Good introduction: "Image classification accuracy has plateaued at 95% on ImageNet, but real-world deployment shows 20% drop due to distribution shift. We identify that current augmentation strategies don't address this gap. We propose X which improves OOD accuracy by 15% while maintaining in-distribution performance."

## Implementasi

```python
def paper_structure_template():
    return {
        'title': '',
        'abstract': {
            'problem': '',
            'method': '',
            'results': '',
            'conclusion': ''
        },
        'introduction': {
            'motivation': '',
            'gap': '',
            'contribution': '',
            'roadmap': ''
        },
        'methods': {
            'data': '',
            'model': '',
            'training': '',
            'evaluation': ''
        },
        'results': {
            'main_results': '',
            'ablation': '',
            'analysis': ''
        },
        'discussion': {
            'interpretation': '',
            'limitations': '',
            'future_work': ''
        },
        'related_work': '',
        'conclusion': ''
    }

def abstract_word_count(abstract):
    """Check abstract length."""
    return len(abstract.split())

def figure_caption_check(caption):
    """Verify figure caption is self-contained."""
    checks = {
        'has_title': 'Figure' in caption,
        'has_description': len(caption.split()) > 20,
        'mentions_data': any(w in caption.lower() for w in ['data','result','performance','accuracy']),
        'mentions_method': any(w in caption.lower() for w in ['model','method','approach','ours'])
    }
    return checks
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "IMRaD structure stands for:",
    "options": [
      "Introduction, Methods, Results, And Discussion",
      "Introduction, Methods, Results, Analysis",
      "Information, Methods, Results, Discussion",
      "Introduction, Models, Results, Data"
    ],
    "answer": 0,
    "explanation": "Standard scientific paper structure.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "A good abstract should:",
    "options": [
      "Be self-contained: problem, method, results, conclusion in 150-250 words",
      "Be as long as possible",
      "Only describe the method",
      "Avoid mentioning results"
    ],
    "answer": 0,
    "explanation": "Abstract must stand alone — readers may only read abstract.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The Methods section should include:",
    "options": [
      "Sufficient detail for independent replication",
      "Only high-level overview",
      "Marketing language",
      "Results interpretation"
    ],
    "answer": 0,
    "explanation": "Methods = reproducibility blueprint.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `check_abstract(abstract)` validating word count (150-250) and presence of key sections.",
    "starter_code": "def check_abstract(abstract):\n    words = abstract.split()\n    wc = len(words)\n    has_problem = any(w in abstract.lower() for w in ['problem','challenge','issue'])\n    has_method = any(w in abstract.lower() for w in ['we propose','we introduce','method'])\n    has_results = any(w in abstract.lower() for w in ['achieve','result','accuracy','improve'])\n    pass\n",
    "function": "check_abstract",
    "tests": [
      {"call": "check_abstract('We address the problem of X. We propose method Y. It achieves 95% accuracy on benchmark Z.')", "expected": "check"}
    ],
    "hints": ["Return dict with word_count, in_range, has_problem, has_method, has_results, valid"],
    "explanation": "Abstract quality checker.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Reviewer says: 'The paper lacks ablation studies and doesn't discuss limitations.' Write a rebuttal paragraph addressing this concern professionally and explaining what you'll add.",
    "rubric": [
      "Acknowledge the concern respectfully",
          "Commit to adding ablation studies in revision",
      "Add limitations section discussing scope and assumptions",
      "If already have ablations, point to specific section/table"
        ],
    "points": 2
  }
]
```

## Referensi

- Schimel, 'Writing Science'
- NeurIPS/ICML/ICLR author guidelines
- 'The Elements of Style' — Strunk & White (for clarity)
- 'Science Research Writing' — university writing centers
- Papers With Code for reproducibility standards
