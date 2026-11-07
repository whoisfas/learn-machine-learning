---
{
  "id": "p5-research-gap-identification",
  "title": "Research Gap Identification",
  "phase": 5,
  "difficulty": "advanced",
  "estimated_hours": 10,
  "objectives": [
    "Systematically identify gaps in existing literature",
    "Distinguish between incremental and transformative research",
    "Evaluate whether a gap is worth pursuing",
    "Position your research relative to prior work"
  ],
  "concepts": [
    "Types of research gaps: methodological, empirical, theoretical, practical",
    "Systematic literature review process",
    "Gap analysis frameworks",
    "Novelty assessment: incremental vs transformative",
    "Feasibility and impact evaluation",
    "Positioning: how your work differs from prior art",
    "Research agenda and future work identification"
  ],
  "formulas": [],
  "references": [
    "Kitchenham 'Guidelines for performing Systematic Literature Reviews'",
    "Web of Science / Google Scholar search strategies",
    "'How to Read a Paper' by Keshav",
    "Research gap analysis frameworks"
  ]
}
---

## Penjelasan

Research gap identification adalah proses menemukan area yang belum dieksplorasi atau belum terjawab dalam literature — titik di mana kontribusi baru Anda bisa berarti. Gap yang baik bukan sekadar "belum ada yang melakukan X" tapi "X belum dilakukan *dan* ada alasan kuat mengapa itu penting".

**Jenis research gaps.**
- *Methodological*: metode existing tidak optimal untuk masalah tertentu; ada ruang untuk improvement.
- *Empirical*: klaim existing belum diuji pada data/domain baru; generalisasi belum diverifikasi.
- *Theoretical*: pemahaman fundamental belum lengkap; ada kontradiksi atau inkonsistensi.
- *Practical*: masalah dunia nyata belum punya solusi ML; atau solusi existing tidak deployed.

**Proses identifikasi.** (1) Baca survey papers dan recent reviews untuk peta area. (2) Track citation network — papers yang banyak dikutip = fondasi; yang jarang = frontier. (3) Identifikasi asumsi yang tidak diuji atau kontradiksi antar papers. (4) Perhatikan "future work" sections — sering berisi gaps eksplisit. (5) Evaluasi feasibility: data tersedia? compute cukup? expertise ada?

**Evaluasi gap.** Pertanyaan kunci: Apakah gap ini *signifikan* (impact besar jika diisi)? *Feasible* (bisa dilakukan dengan resources)? *Novel* (belum ada yang mencoba atau mencoba dan gagal — kenapa)? *Timely* (relevan sekarang, bukan 10 tahun lalu)?

**Positioning.** Setelah gap ditemukan, posisikan kontribusi: "Different from X (yang melakukan A pada B), we do C pada D because E". Ini menjadi related work section di paper.

## Konsep Penting

1. **Gap != absence** — "belum ada yang melakukan" bukan cukup; harus ada *alasan* mengapa itu belum dilakukan dan mengapa itu penting.
2. **Incremental vs transformative** — incremental = perbaikan kecil pada metode existing; transformative = pendekatan baru yang mengubah paradigma.
3. **Failed attempts matter** — jika 5 paper mencoba dan gagal, pahami kenapa sebelum mencoba lagi.
4. **Survey papers adalah peta** — mulai dari situ untuk memahami landscape.
5. **Gap bisa muncul dari technology shift** — metode lama mungkin tidak relevan; gap baru muncul dari tools baru (GPU, transformers).

## Formula

- Citation analysis: highly-cited = established; low-cited recent = frontier.
- Research impact proxy: citations/year, venue prestige.

## Contoh Kasus

Gap identification untuk penelitian Anda: "Transformer-based models untuk time series forecasting menunjukkan akurasi superior tapi membutuhkan data besar. Gap: bagaimana performa transformer vs classical methods pada time series pendek (<1000 points)?" — ini methodological + empirical gap yang feasible dan signifikan.

## Implementasi

```python
# Systematic literature search helper
def search_arxiv(query, max_results=50):
    import urllib.request, urllib.parse, re
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&max_results={max_results}"
    with urllib.request.urlopen(url) as f:
        data = f.read().decode()
    titles = re.findall(r'<title>(.*?)</title>', data, re.DOTALL)
    return [t.strip().replace('\n',' ') for t in titles[1:]]  # skip feed title

def citation_network_analysis(seed_papers, depth=2):
    """Trace citations backward/forward to map research landscape."""
    # Implementation would use Semantic Scholar API or OpenAlex
    pass

# Gap analysis template
def gap_analysis_template():
    return {
        'gap_description': '',
        'gap_type': '',  # methodological/empirical/theoretical/practical
        'evidence': [],  # papers showing the gap exists
        'why_important': '',
        'feasibility': '',
        'novelty_assessment': '',
        'positioning': '',
        'risks': []
    }
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A research gap is best described as:",
    "options": [
      "An unanswered question or unexplored area with scientific or practical significance",
      "Any topic not yet published",
      "A mistake in prior work",
      "A new dataset"
    ],
    "answer": 0,
    "explanation": "Good gaps are significant, feasible, novel, and timely.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "The difference between incremental and transformative research is:",
    "options": [
      "Incremental improves existing methods; transformative introduces new paradigms",
      "Incremental uses less data",
      "Transformative is always better",
      "They are the same"
    ],
    "answer": 0,
    "explanation": "Both are valid; transformative has higher risk but higher potential impact.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "When evaluating a research gap, the most important question is:",
    "options": [
      "Why hasn't this been done, and why does it matter?",
      "Is it easy?",
      "Is it popular?",
      "Does it use deep learning?"
    ],
    "answer": 0,
    "explanation": "Understanding why the gap exists prevents repeating failed attempts.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `literature_synthesis(papers)` organizing paper summaries into themes/gaps structure.",
    "starter_code": "def literature_synthesis(papers):\n    # papers: list of dicts with 'title', 'year', 'method', 'findings', 'limitations'\n    pass\n",
    "function": "literature_synthesis",
    "tests": [
      {"call": "literature_synthesis([{'title':'A','year':2020,'method':'CNN','findings':'good','limitations':'slow'}])", "expected": "check"}
    ],
    "hints": ["Group by method/theme; extract common limitations; identify gaps"],
    "explanation": "Systematic synthesis of literature.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You find a paper claiming method X achieves 99% accuracy on your problem, but your replication gets 85%. Describe your investigation process to determine if this is a research gap, a bug, or a misunderstanding.",
    "rubric": [
      "Check for data leakage in original paper",
          "Verify dataset splits and preprocessing match",
      "Check if original used different data version or augmented data",
      "Contact authors if needed; document discrepancies",
      "If replication correct, this discrepancy IS a research finding"
        ],
    "points": 2
  }
]
```

## Referensi

- Kitchenham et al., 'Guidelines for performing Systematic Literature Reviews'
- Web of Science, Google Scholar, Semantic Scholar search
- Keshav, 'How to Read a Paper'
- 'Systematic Reviews in Software Engineering' — Keele guidelines
