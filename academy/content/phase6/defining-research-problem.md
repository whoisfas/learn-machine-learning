---
{
  "id": "p6-defining-research-problem",
  "title": "Defining the Research Problem",
  "phase": 6,
  "difficulty": "intermediate",
  "estimated_hours": 8,
  "objectives": [
    "Translate a vague interest into a concrete research problem",
    "Formulate a falsifiable research question",
    "Scope the problem: what's in, what's out",
    "Assess feasibility before committing"
  ],
  "concepts": [
    "Problem statement structure",
    "Research question vs hypothesis vs objective",
    "Scope definition: boundaries and assumptions",
    "Feasibility assessment: data, compute, expertise, time",
    "Problem decomposition into sub-problems",
    "Success criteria definition",
    "Stakeholder needs and practical relevance"
  ],
  "formulas": [],
  "references": [
    "Phase 5: Scientific Research Fundamentals",
    "Phase 5: Research Gap Identification",
    "'How to Choose a Research Problem' — various guides",
    "Research proposal templates"
  ]
}
---

## Penjelasan

Phase 6 adalah fase *praksis* — Anda tidak lagi mempelajari tentang research, melainkan *melakukan* research. Modul pertama memandu Anda dari ide mentah menjadi research problem yang well-defined dan actionable.

**Dari minat ke masalah.** Mulai dari area yang Anda minati (mis. "saya tertarik pada time series forecasting"). Kemudian persempit: masalah spesifik apa yang belum terjawab? "Bagaimana transformer performanya pada time series dengan missing values di domain healthcare?"

**Struktur problem statement.** (1) *Context*: mengapa masalah ini penting? (2) *Gap*: apa yang belum diketahui/terselesaikan? (3) *Objective*: apa yang ingin Anda capai? (4) *Approach*: bagaimana Anda akan melakukannya? (5) *Success criteria*: bagaimana Anda tahu berhasil?

**Research question yang baik.** Falsifiable (bisa dibuktikan salah), specific (bukan "improve ML"), measurable (metrik jelas), feasible (resources tersedia), relevant (signifikan ilmiah/praktis).

**Scoping.** Tentukan boundary: dataset apa? task apa? constraint apa? Asumsi apa yang Anda buat? Apa yang *tidak* akan Anda cover? Scoping yang baik mencegah scope creep.

**Feasibility check.** Sebelum commit: data tersedia? compute cukup (MacBook Air M3, 16GB — no large GPU training)? expertise ada atau bisa dipelajari? timeline realistis? Jika tidak feasible, adjust scope atau pivot.

## Konsep Penting

1. **Problem yang baik adalah separuh research** — problem definition yang buruk menghasilkan research yang tidak fokus.
2. **Falsifiability** — jika hasil apapun bisa diinterpretasikan sebagai "sukses", maka bukan research.
3. **Scope early, scope often** — revisit scope setelah exploratory analysis; adjust jika perlu.
4. **Success criteria upfront** — definisikan "berhasil" sebelum eksperimen, bukan setelah.
5. **Feasibility > ambition** — research kecil yang selesai lebih baik dari ambition besar yang tidak selesai.

## Contoh Kasus

Minat: "saya ingin menggunakan genetic programming untuk scientific discovery".

Research problem: "Can genetic programming discover physically meaningful equations from noisy astronomical data better than symbolic regression baselines?" — specific, falsifiable, feasible (datasets tersedia via academy dataset manager), measurable (equation accuracy, physical plausibility).

## Implementasi

```python
def research_problem_template():
    return {
        'title': '',
        'context': '',          # why this matters
        'gap': '',              # what's missing
        'research_question': '', # specific, falsifiable question
        'hypothesis': '',       # testable prediction
        'objectives': [],       # concrete goals
        'scope_in': [],         # what's covered
        'scope_out': [],        # what's excluded
        'datasets': [],         # data sources
        'methods': [],          # approaches to try
        'success_criteria': [], # measurable outcomes
        'feasibility': {
            'data_available': False,
            'compute_sufficient': False,
            'expertise_level': '',
            'time_estimate': ''
        }
    }

def feasibility_check(problem):
    """Score feasibility of research problem."""
    score = 0
    checks = [
        ('data', problem['feasibility']['data_available']),
        ('compute', problem['feasibility']['compute_sufficient']),
        ('objectives_clear', len(problem['objectives']) > 0),
        ('success_defined', len(problem['success_criteria']) > 0),
        ('scope_defined', len(problem['scope_in']) > 0 and len(problem['scope_out']) > 0)
    ]
    for name, ok in checks:
        if ok: score += 1
    return {'score': score, 'max': len(checks), 'feasible': score >= 4}
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A good research question should be:",
    "options": [
      "Falsifiable, specific, measurable, feasible, and relevant",
      "As broad as possible",
      "Impossible to answer",
      "Already answered"
    ],
    "answer": 0,
    "explanation": "Well-defined problems produce focused, rigorous research.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Scoping a research problem means:",
    "options": [
      "Defining what's in and out of scope, plus assumptions",
      "Making the problem bigger",
      "Adding more objectives",
      "Choosing a venue"
    ],
    "answer": 0,
    "explanation": "Clear boundaries prevent scope creep and keep research focused.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Before committing to a research problem, the most important feasibility question is:",
    "options": [
      "Can I actually complete this with available data, compute, and expertise?",
      "Is it novel?",
      "Will it get published?",
      "Is it interesting?"
    ],
    "answer": 0,
    "explanation": "Feasibility determines whether research can be completed, not just whether it's worthwhile.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `problem_quality_score(problem)` evaluating a research problem dict against criteria.",
    "starter_code": "def problem_quality_score(problem):\n    criteria = {\n        'has_question': bool(problem.get('research_question')),\n        'has_hypothesis': bool(problem.get('hypothesis')),\n        'has_objectives': len(problem.get('objectives', [])) >= 2,\n        'has_scope': bool(problem.get('scope_in')) and bool(problem.get('scope_out')),\n        'has_criteria': len(problem.get('success_criteria', [])) >= 1,\n        'falsifiable': 'not' not in problem.get('research_question', '').lower() or '?' in problem.get('research_question', '')\n    }\n    score = sum(criteria.values())\n    return {'score': score, 'max': len(criteria), 'criteria': criteria}\n",
    "function": "problem_quality_score",
    "tests": [
      {"call": "problem_quality_score({'research_question':'Does X improve Y?','hypothesis':'X improves Y by 10%','objectives':['o1','o2'],'scope_in':['a'],'scope_out':['b'],'success_criteria':['acc>0.9']})['score']", "expected": 6}
    ],
    "hints": ["Count how many criteria are met; return score dict"],
    "explanation": "Evaluate research problem quality.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You have three research ideas: (A) novel architecture for image classification, (B) apply existing method to new medical dataset, (C) theoretical analysis of why transformers fail on time series. Evaluate each on novelty, feasibility, and potential impact. Which would you choose and why?",
    "rubric": [
      "A: high novelty but high risk; needs GPU resources",
          "B: lower novelty but high feasibility; practical impact",
      "C: high novelty, moderate feasibility; theoretical contribution",
      "Consider: MacBook Air M3 constraint, 16GB RAM, no external GPU"
        ],
    "points": 3
  }
]
```

## Referensi

- Phase 5 modules: Scientific Research Fundamentals, Research Gap Identification
- 'How to Choose a Research Problem' — graduate school guides
- Research proposal templates (NSF, NIH)
