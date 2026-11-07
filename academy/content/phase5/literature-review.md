---
{
  "id": "p5-literature-review",
  "title": "Literature Review",
  "phase": 5,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Conduct systematic literature searches",
    "Synthesize findings across multiple papers",
    "Identify trends, contradictions, and open problems",
    "Write a literature review section for a paper"
  ],
  "concepts": [
    "Search strategy: keywords, databases, snowballing",
    "Inclusion/exclusion criteria",
    "Quality assessment of papers",
    "Thematic synthesis vs chronological",
    "Citation analysis and co-citation",
    "Survey papers and their role",
    "Writing: narrative vs systematic review"
  ],
  "formulas": [],
  "references": [
    "Kitchenham 'Systematic Literature Reviews in Software Engineering'",
    "PRISMA guidelines for systematic reviews",
    "Google Scholar, Semantic Scholar, Connected Papers",
    "'How to Write a Literature Review' — various guides"
  ]
}
---

## Penjelasan

Literature review adalah proses sistematis mengumpulkan, mengevaluasi, dan mensintesis pengetahuan yang sudah ada dalam suatu bidang. Di ML research, literature review bukan sekadar "membaca paper" — melainkan membangun pemahaman terstruktur tentang state-of-the-art, methodology yang digunakan, dan gap yang tersisa.

**Search strategy.** Mulai dengan survey papers dan recent reviews untuk peta area. Gunakan keyword combinations di Google Scholar, Semantic Scholar, arXiv. *Snowballing*: dari paper kunci, telusuri references (backward) dan citing papers (forward). Filter berdasarkan tahun, venue, citation count.

**Inclusion/exclusion criteria.** Tentukan kriteria objektif: publish dalam venue bereputasi (NeurIPS, ICML, ICLR, JMLR, IEEE), peer-reviewed, relevan dengan topik, dalam bahasa yang dipahami, tersedia full-text. Dokumentasikan jumlah paper yang disaring di setiap tahap (PRISMA flow).

**Kualitas assessment.** Evaluasi: apakah methodology sound? apakah experiments rigorous? apakah claims didukung data? apakah ada leakage atau HARKing? Paper dengan kode/data tersedia lebih reliable.

**Synthesis approaches.**
- *Thematic*: kelompokkan paper berdasarkan tema/metode; lebih baik untuk memahami landscape.
- *Chronological*: urutkan perkembangan dari waktu ke waktu; baik untuk historical context.
- *Methodological*: bandingkan pendekatan berbeda untuk masalah sama; baik untuk method comparison.

**Writing literature review.** Struktur: (1) intro — mengapa topik penting; (2) body — synthesize findings per tema, bandingkan approaches, identify gaps; (3) conclusion — apa yang sudah diketahui, apa yang belum, di mana kontribusi Anda masuk.

## Konsep Penting

1. **Systematic > ad-hoc** — dokumentasikan search process; hindari cherry-picking.
2. **Quality over quantity** — 20 paper berkualitas lebih baik dari 200 paper dangkal.
3. **Synthesis bukan summary** — literature review harus *mengintegrasikan* findings, bukan hanya mendaftar.
4. **Identify contradictions** — jika paper A bilang X dan paper B bilang not-X, itu adalah research opportunity.
5. **Position your work** — literature review harus mengarah ke gap yang akan Anda isi.

## Formula

- PRISMA flow: identified -> screened -> eligible -> included.
- Citation impact: citations/year normalized by field.

## Contoh Kasus

Literature review untuk "graph neural networks for molecular property prediction": thematic synthesis — (1) message passing architectures (GCN, GAT, MPNN), (2) datasets (MoleculeNet, ZINC), (3) applications (drug discovery, materials), (4) challenges (interpretability, scalability, generalization). Identifikasi gap: interpretability methods untuk GNN belum mature.

## Implementasi

```python
# Literature search and organization
def organize_papers(papers):
    """Organize papers by theme/method for synthesis."""
    themes = {}
    for p in papers:
        theme = p.get('theme', 'other')
        themes.setdefault(theme, []).append(p)
    return themes

def extract_key_findings(papers):
    """Extract and compare key findings across papers."""
    findings = []
    for p in papers:
        findings.append({
            'paper': p['title'],
            'method': p.get('method', ''),
            'key_result': p.get('key_result', ''),
            'limitation': p.get('limitation', '')
        })
    return findings

def identify_contradictions(findings):
    """Find papers with contradictory claims."""
    contradictions = []
    for i, f1 in enumerate(findings):
        for f2 in findings[i+1:]:
            if f1['key_result'] != f2['key_result'] and \
               f1.get('same_problem') and f2.get('same_problem'):
                contradictions.append((f1['paper'], f2['paper']))
    return contradictions
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "The primary purpose of a literature review in research is to:",
    "options": [
      "Synthesize existing knowledge and identify gaps for your contribution",
      "List all papers on a topic",
      "Prove your method is best",
      "Summarize each paper individually"
    ],
    "answer": 0,
    "explanation": "Literature review integrates findings and positions your work.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Snowballing in literature search refers to:",
    "options": [
      "Tracing references backward and citations forward from key papers",
      "Reading papers randomly",
      "Using only Google Scholar",
      "Filtering by year"
    ],
    "answer": 0,
    "explanation": "Snowballing expands search beyond keyword queries.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "A thematic literature review groups papers by:",
    "options": [
      "Common themes, methods, or research questions",
      "Publication year only",
      "Author name",
      "Journal name"
    ],
    "answer": 0,
    "explanation": "Thematic synthesis reveals patterns across papers.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `prisma_flow(identified, screened, eligible)` returning dict with counts at each stage and exclusion reasons.",
    "starter_code": "def prisma_flow(identified, screened, eligible):\n    # identified: total papers found\n    # screened: papers meeting title/abstract criteria\n    # eligible: papers meeting full-text criteria\n    pass\n",
    "function": "prisma_flow",
    "tests": [
      {"call": "prisma_flow(1000, 200, 50)", "expected": "check"}
    ],
    "hints": ["Return dict with identified, screened, eligible, excluded_title, excluded_fulltext counts"],
    "explanation": "PRISMA flow documentation.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You find 30 papers on 'transformers for time series'. Paper A says transformers beat LSTM by 15%; Paper B says LSTM is better for short sequences. Write a paragraph synthesizing this contradiction and identifying the research gap.",
    "rubric": [
      "Identifies that results depend on sequence length/data regime",
          "Notes methodological differences (datasets, metrics, baselines)",
      "Gap: systematic comparison across sequence lengths and domains",
      "Proposes controlled experiment to resolve contradiction"
        ],
    "points": 2
  }
]
```

## Referensi

- Kitchenham et al., 'Systematic Literature Reviews in Software Engineering'
- PRISMA guidelines (http://www.prisma-statement.org)
- Google Scholar, Semantic Scholar, Connected Papers
- 'How to Write a Literature Review' — university writing guides
