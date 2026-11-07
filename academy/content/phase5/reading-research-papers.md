---
{
  "id": "p5-reading-research-papers",
  "title": "Reading Research Papers",
  "phase": 5,
  "difficulty": "intermediate",
  "estimated_hours": 8,
  "objectives": [
    "Read papers efficiently using structured approaches",
    "Extract key information: problem, method, results, limitations",
    "Critically evaluate paper claims and methodology",
    "Build a personal knowledge management system for papers"
  ],
  "concepts": [
    "Three-pass reading approach",
    "SQ3R method for papers",
    "Key sections: abstract, intro, method, experiments, conclusion",
    "Critical reading: claims vs evidence",
    "Red flags: leakage, HARKing, cherry-picking",
    "Note-taking and knowledge organization",
    "Reproducing results from papers"
  ],
  "formulas": [],
  "references": [
    "Keshav 'How to Read a Paper'",
    "'The Memex and the Machine' — various",
    "Papers with Code (paperswithcode.com)",
    "arXiv Sanity, Semantic Scholar"
  ]
}
---

## Penjelasan

Membaca research paper secara efektif adalah skill yang berbeda dari membaca textbook. Papers padat informasi, asumsi implisit, dan klaim yang perlu diverifikasi. Strategi reading yang terstruktur menghemat waktu dan meningkatkan pemahaman.

**Three-pass approach (Keshav).**
- *Pass 1 (5-10 min)*: baca title, abstract, intro, conclusion, scan figures/tables. Tujuan: pahami big picture — apa masalahnya, apa kontribusinya, apakah relevan. Putuskan: baca detail atau skip.
- *Pass 2 (1 hour)*: baca method, experiments, results dengan detail. Fokus: bagaimana mereka melakukannya, apakah methodology sound, apakah experiments rigorous. Ambil notes pada key points.
- *Pass 3 (beberapa jam)*: reproduksi mental atau actual — coba pahami detail teknis, verifikasi klaim, cari flaws. Ini untuk paper yang akan Anda bangun di atasnya.

**SQ3R untuk papers.** Survey (skim struktur), Question (apa yang saya ingin tahu?), Read (baca aktif), Recite (rangkum dengan kata sendiri), Review (kembali dan evaluasi).

**Critical reading.** Jangan terima klaim mentah-mentah. Pertanyaan: Apakah baseline fair? Apakah data leakage mungkin? Apakah metrics appropriate? Apakah results statistically significant? Apakah code/data available? Apakah claims didukung evidence atau overclaiming?

**Red flags.**
- *Data leakage*: preprocessing dilakukan pada seluruh data sebelum split.
- *Cherry-picking*: hanya report best results, sembunyikan failures.
- *Weak baselines*: bandingkan dengan model lemah, bukan state-of-art.
- *No error bars*: single run tanpa variance estimate.
- *Unrealistic assumptions*: data synthesis yang tidak representatif.

**Knowledge management.** Setiap paper: catat problem, method summary, key results, limitations, relevance to your work, reproduction status. Tools: Zotero, Obsidian, Notion, atau file markdown per paper.

## Konsep Penting

1. **Pass 1 dulu selalu** — jangan baca detail sebelum tahu apakah paper relevan.
2. **Figures > text** — figures dan tables sering lebih informatif dari prose.
3. **Method section adalah inti** — understanding method lebih penting dari results.
4. **Reproduce sebelum percaya** — jika memungkinkan, reimplement atau run code mereka.
5. **Paper adalah argumen** — bukan kebenaran mutlak; evaluate seperti reviewer.

## Formula

- Reading time heuristic: Pass 1 = 10 min, Pass 2 = 1 hour, Pass 3 = half day per important paper.

## Contoh Kasus

Membaca paper "BERT: Pre-training of Deep Bidirectional Transformers": Pass 1 -> "pre-training + fine-tuning paradigm untuk NLP". Pass 2 -> method: masked LM + next sentence prediction; experiments: GLUE, SQuAD benchmarks. Pass 3 -> reproduce fine-tuning pada custom dataset; catat bahwa pre-training data tidak fully documented (limitation).

## Implementasi

```python
# Paper reading notes template
def paper_notes_template():
    return {
        'title': '',
        'authors': '',
        'venue': '',
        'year': '',
        'problem': '',
        'contribution': '',
        'method_summary': '',
        'key_results': '',
        'limitations': '',
        'relevance_to_my_work': '',
        'reproduction_status': '',
        'code_available': False,
        'data_available': False,
        'critical_notes': []
    }

# Quick paper evaluation checklist
def paper_evaluation(paper):
    checks = {
        'baseline_fair': paper.get('baseline_comparison') == 'fair',
        'no_leakage': paper.get('leakage_check') == 'pass',
        'stats_significant': paper.get('error_bars') == True,
        'code_available': paper.get('code') != None,
        'data_available': paper.get('data') != None,
        'reproducible': paper.get('seed_reported') == True
    }
    return checks
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "The three-pass reading approach starts with:",
    "options": [
      "Skimming title, abstract, intro, conclusion to decide if paper is relevant",
      "Reading method section in detail",
      "Implementing the code",
      "Reading all references"
    ],
    "answer": 0,
    "explanation": "Pass 1 is a quick filter — don't invest time before knowing relevance.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "A red flag in ML papers is:",
    "options": [
      "Preprocessing done on full dataset before train/test split (data leakage)",
      "Using cross-validation",
      "Reporting multiple metrics",
      "Having many authors"
    ],
    "answer": 0,
    "explanation": "Leakage invalidates results — preprocessing must be inside CV folds.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "When reading a paper critically, the most important question is:",
    "options": [
      "Do the experiments actually support the claims?",
      "Is the writing clear?",
      "Is the topic popular?",
      "How many citations does it have?"
    ],
    "answer": 0,
    "explanation": "Evaluate evidence vs claims, not just surface quality.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `extract_paper_summary(abstract)` extracting problem, method, and results using simple heuristics (first sentence = problem, etc.).",
    "starter_code": "def extract_paper_summary(abstract):\n    sentences = abstract.split('. ')\n    pass\n",
    "function": "extract_paper_summary",
    "tests": [
      {"call": "extract_paper_summary('We propose X. It achieves 95% accuracy. Results show improvement.')", "expected": "check"}
    ],
    "hints": ["Heuristic: sentence with 'we propose' = method; with 'achieves/results' = results; first sentence = problem"],
    "explanation": "Automated paper summarization heuristic.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "A paper claims 'state-of-the-art' but only compares against 2018 baselines. Describe your critical evaluation and what additional experiments you'd run to verify the claim.",
    "rubric": [
      "Note weak baseline comparison — not truly SOTA",
          "Check if 2023-2024 methods were excluded unfairly",
      "Run comparison against recent methods (transformers, latest architectures)",
      "Consider if task definition matches current benchmarks",
      "Evaluate on multiple datasets, not just the one in paper"
        ],
    "points": 2
  }
]
```

## Referensi

- Keshav, 'How to Read a Paper' (ACM SIGCOMM CCR)
- 'The Memex and the Machine' — various reading guides
- Papers With Code (paperswithcode.com)
- arXiv Sanity, Semantic Scholar, Connected Papers
- 'Ten Simple Rules for Reading a Scientific Paper' — PLOS Computational Biology
