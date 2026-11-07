---
{
  "id": "p6-paper-submission",
  "title": "Paper Writing and Submission",
  "phase": 6,
  "difficulty": "advanced",
  "estimated_hours": 12,
  "objectives": [
    "Write a complete research paper from experiments to submission",
    "Navigate venue selection and formatting requirements",
    "Respond to peer review effectively",
    "Understand publication ethics and open science"
  ],
  "concepts": [
    "Paper structure: title, abstract, intro, methods, results, discussion, conclusion",
    "Venue selection: conference vs journal, impact factor, audience",
    "Formatting: LaTeX templates, page limits, supplementary materials",
    "Peer review process: review, rebuttal, revision, acceptance",
    "Publication ethics: plagiarism, self-plagiarism, authorship, data availability",
    "Open science: preprints, code/data sharing, reproducibility badges",
    "Rebuttal writing: addressing concerns with evidence"
  ],
  "formulas": [],
  "references": [
    "Phase 5: Scientific Paper Writing",
    "NeurIPS/ICML/ICLR author guidelines",
    "ACM publication ethics",
    "arXiv preprint guidelines"
  ]
}
---

## Penjelasan

Paper writing dan submission adalah puncak research journey — di sinilah semua pekerjaan eksperimen dan analisis dikomunikasikan ke community. Paper yang baik secara tulisan, jujur secara hasil, dan professional secara proses.

**Paper structure final.** Title (specific, searchable), Abstract (standalone summary), Introduction (motivation -> gap -> contribution), Related Work (positioning), Methods (reproducible), Experiments (setup, results, ablation), Discussion (interpretation, limitations), Conclusion (contributions, future work). Supplementary: additional experiments, proofs, implementation details.

**Venue selection.** Konferensi (NeurIPS, ICML, ICLR, CVPR, ACL) = fast turnaround, high visibility, proceedings. Journal (JMLR, IEEE TPAMI, Nature MI) = slower, more thorough review, higher impact factor. Pertimbangkan: audience, deadline, page limit, open access policy.

**Formatting.** Gunakan official template (LaTeX untuk ML venues). Page limits ketat — supplementary materials untuk detail. Figures: publication quality (Phase 6 modul sebelumnya). References: complete dan accurate.

**Peer review process.** Submission -> assignment to reviewers -> reviews (accept/reject/revise) -> rebuttal (if revise) -> final decision. Reviewer concerns harus dijawab dengan evidence, bukan argumen. Rebuttal yang baik: acknowledge, clarify, provide additional experiments jika diminta.

**Publication ethics.** Plagiarism (text/data/code), self-plagiarism (reusing own work without citation), authorship criteria (substantial contribution), data availability (share if possible), conflict of interest disclosure.

## Konsep Penting

1. **Write for the reader** — reviewer mungkin bukan expert di subfield Anda.
2. **Rebuttal is dialogue** — bukan pertahanan; jawab concerns dengan evidence.
3. **Honesty > acceptance** — negative results dan limitations adalah scientific integrity.
4. **Open science benefits everyone** — share code/data; reproducibility badges.
5. **Rejection is normal** — most papers rejected; revise and resubmit.

## Implementasi

```python
def paper_checklist():
    """Pre-submission checklist."""
    return {
        'title': {'clear': False, 'specific': False, 'keywords': False},
        'abstract': {'word_count_ok': False, 'self_contained': False},
        'introduction': {'motivation': False, 'gap': False, 'contribution': False},
        'methods': {'reproducible': False, 'detailed': False},
        'experiments': {'setup_clear': False, 'baselines': False, 'ablation': False,
                       'statistics': False, 'multiple_seeds': False},
        'figures': {'quality': False, 'captions': False, 'colorblind': False},
        'references': {'complete': False, 'formatted': False},
        'supplementary': {'available': False, 'organized': False},
        'ethics': {'data_availability': False, 'code_availability': False,
                  'conflicts': False}
    }

def rebuttal_template():
    """Template for rebuttal response."""
    return {
        'reviewer_id': '',
        'concern': '',
        'response_type': '',  # clarify/add_experiment/acknowledge_limit
        'response': '',
        'evidence': '',  # additional results, references, clarifications
        'changes_made': []
    }
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "The most important factor in paper acceptance is:",
    "options": [
      "Novel contribution with rigorous experiments and clear writing",
      "Number of authors",
      "Page count",
      "Topic popularity"
    ],
    "answer": 0,
    "explanation": "Reviewers evaluate novelty, rigor, and clarity.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "A rebuttal should:",
    "options": [
      "Address each concern with evidence and clarity, not defensiveness",
      "Argue with reviewers",
      "Ignore minor concerns",
      "Submit unchanged paper"
    ],
    "answer": 0,
    "explanation": "Rebuttal is professional dialogue; evidence-based responses.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Open science practices include:",
    "options": [
      "Sharing code, data, and preprints for reproducibility",
      "Keeping everything secret",
      "Only publishing in top venues",
      "Avoiding preprints"
    ],
    "answer": 0,
    "explanation": "Transparency accelerates science and enables reproducibility.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `venue_recommendation(paper_stats)` suggesting appropriate venues based on contribution type and target audience.",
    "starter_code": "def venue_recommendation(paper_stats):\n    # paper_stats: dict with 'novelty', 'rigor', 'audience', 'deadline_ok'\n    venues = []\n    if paper_stats.get('novelty', 0) > 0.8:\n        venues.extend(['NeurIPS', 'ICML', 'ICLR'])\n    if paper_stats.get('audience') == 'ml_core':\n        venues.append('JMLR')\n    if paper_stats.get('deadline_ok'):\n        venues.append('arXiv preprint')\n    return {'recommended': list(set(venues)), 'preprint': True}\n",
    "function": "venue_recommendation",
    "tests": [{"call": "venue_recommendation({'novelty':0.9,'audience':'ml_core','deadline_ok':True})", "expected": "check"}],
    "hints": ["Consider novelty level, audience, and timeline"],
    "explanation": "Venue selection helper.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your paper is rejected with reviews saying 'incremental contribution' and 'insufficient experiments'. Write a response strategy: would you rebut, revise, or resubmit elsewhere? Justify your decision.",
    "rubric": [
      "Analyze if rejection is fair or based on misunderstanding",
          "If incremental: strengthen novelty argument or add experiments",
      "If misunderstanding: rebuttal with evidence",
      "Consider resubmission to different venue if scope mismatch",
      "Always incorporate valid reviewer feedback"
        ],
    "points": 3
  }
]
```

## Referensi

- Phase 5: Scientific Paper Writing
- NeurIPS/ICML/ICLR author guidelines
- ACM publication ethics
- arXiv preprint guidelines
