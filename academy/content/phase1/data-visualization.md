---
{
  "id": "p1-data-visualization",
  "title": "Data Visualization with Matplotlib & Seaborn",
  "phase": 1,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": [
    "Create accurate, honest plots with matplotlib and seaborn",
    "Choose the right plot type for the data question",
    "Use color, labels, and annotations effectively",
    "Save publication-quality figures and avoid misleading encodings"
  ],
  "concepts": [
    "Figure vs Axes model (pyplot vs OO interface)",
    "Univariate: histogram, KDE, box, violin",
    "Bivariate: scatter, line, heatmap, pairplot",
    "Categorical: bar, countplot, pointplot",
    "seaborn: distplot->hist, catplot, lmplot, FacetGrid",
    "Color scales, log scales, dual axes",
    "Misleading plots: truncated axes, 3D charts, dual y-axis abuse"
  ],
  "formulas": [
    {"name": "Histogram bin width", "latex": "h = \\frac{\\max-\\min}{k}", "note": "Freedman-Diaconis: h = 2·IQR·n^(-1/3) is a principled default."}
  ],
  "references": [
    "matplotlib docs — quick start",
    "seaborn docs — tutorial",
    "Fundamentals of Data Visualization (Wilke)"
  ]
}
---

## Penjelasan

Visualisasi adalah alat utama EDA dan komunikasi hasil eksperimen. Matplotlib menyediakan fondasi (Figure/Axes, granular control); Seaborn membungkusnya dengan plot statistik siap pakai yang secara default lebih informatif.

**Model Figure/Axes.** `plt.figure()` + `fig.add_subplot()` atau `plt.subplots(2,2)` memberi grid axes. Prefer OO interface (`ax.plot`) daripada `plt.plot` untuk kode yang bisa di-reuse dan disimpan sebagai fungsi.

**Memilih plot.**
- Distribusi satu variabel: `hist` (shape, skew), `kde` (smooth density), `box/violin` (banding antar grup + outlier).
- Relasi dua variabel: `scatter` (korelasi, cluster), `line` (time series), `heatmap` (matriks — korelasi, confusion matrix).
- Kategorikal: `bar` (agregat), `countplot` (frekuensi), `pointplot` (tren per kategori).
- Multivariat: `pairplot` (semua pasangan), `jointplot` (marginal + scatter).

**Encoding.** Posisi > warna > ukuran > bentuk dalam hal akurasi baca. Untuk sequential data pakai colormap sequential (`viridis`); diverging untuk data dengan titik tengah bermakna (`coolwarm`); jangan pakai rainbow/jet.

**Kejujuran visual.** Jangan potong sumbu y tanpa indikasi; hindari area/3D untuk perbandingan presisi; satu sumbu y ganda jarang tepat; selalu beri label sumbu + judul + legend. Untuk plot model ML: kurva loss vs epoch, confusion matrix heatmap, feature importance bar, residual plot.

**Seaborn.** `sns.scatterplot(data=df, x=, y=, hue=, size=)` — `hue` untuk kategori, `size` untuk magnitudo. `sns.histplot(kde=True)`, `sns.boxplot`, `sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)`. `sns.pairplot(df, hue='target')`. `sns.lmplot` untuk regresi + confidence interval.

## Konsep Penting

1. **Aspect ratio & readability** — figure default terlalu kecil; `figsize=(10,6)` minimum untuk laporan.
2. **`tight_layout()` / `constrained_layout`** — cegah label terpotong.
3. **Log scale** — `ax.set_xscale('log')` untuk data heavy-tailed (pendapatan, populasi).
4. **Annotate insight** — plot terbaik menjawab satu pertanyaan; judul = pertanyaan + jawaban.
5. **Save reproducible** — `fig.savefig('plot.png', dpi=300, bbox_inches='tight')`.

## Formula

- Freedman-Diaconis bin width: h = 2·IQR·n^(−1/3); jumlah bin ≈ (max−min)/h.
- KDE: f̂(x) = (1/nh)·Σ K((x−xi)/h), K kernel (Gaussian umum), h bandwidth.

## Contoh Kasus

EDA dataset housing: pairplot untuk 8 fitur (sample dulu jika lambat), heatmap korelasi, scatter median_income vs house_value dengan hue ocean_proximity, distribusi target (skewed -> pertimbangkan log-transform). Diagnosis model: plot residual vs predicted — pola = model bias.

## Implementasi

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(df['MedHouseVal'], kde=True, ax=axes[0])
axes[0].set_title('Target distribution (right-skewed)')

sns.scatterplot(data=df, x='MedInc', y='MedHouseVal',
                hue='ocean_proximity', alpha=0.3, ax=axes[1])
axes[1].set_title('Income vs value')

corr = df.select_dtypes('number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[2])
axes[2].set_title('Feature correlations')

plt.tight_layout()
plt.savefig('eda.png', dpi=150, bbox_inches='tight')
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "You want to compare the distribution of house prices across 5 neighborhoods AND see outliers per group. Best single plot type?",
    "options": ["violin plot", "stacked bar chart", "line plot", "pie chart"],
    "answer": 0,
    "explanation": "Violin shows full distribution shape per group plus outliers; boxplot is the simpler alternative.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "A scatterplot of feature A vs feature B colored by class C shows three well-separated clusters. What does this suggest?",
    "options": [
      "Classes are linearly separable in 2D -> simple model may suffice",
      "Data is corrupted",
      "A and B are independent",
      "Need more features"
    ],
    "answer": 0,
    "explanation": "Clear cluster separation by class means low Bayes error; try LDA/logistic regression first.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Write `plot_correlation_heatmap(df, path)` that computes numeric correlations, plots an annotated heatmap with diverging colormap centered at 0, and saves to path.",
    "starter_code": "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\ndef plot_correlation_heatmap(df, path):\n    pass\n",
    "function": "plot_correlation_heatmap",
    "tests": [
      {"call": "import os,tempfile; d=tempfile.mkdtemp(); plot_correlation_heatmap(pd.DataFrame({'a':[1,2,3],'b':[1,2,3]}), os.path.join(d,'c.png')); os.path.exists(os.path.join(d,'c.png'))", "expected": true}
    ],
    "hints": ["df.select_dtypes('number').corr()", "sns.heatmap(..., annot=True, cmap='coolwarm', center=0)"],
    "explanation": "Correlation heatmap is standard EDA for multicollinearity checks.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "open",
    "question": "A colleague plots monthly revenue with the y-axis starting at 90% of the max value, making a 2% change look like 50%. Explain why this misleads, and describe two honest alternatives to show the same data.",
    "rubric": [
      "Explains truncated axis exaggerates differences",
      "Suggests full axis from zero OR clearer alternatives (index/base-100, annotate delta)",
      "Mentions context: when truncation is acceptable (e.g., time series with noted break)"
    ],
    "points": 1
  }
]
```

## Referensi

- matplotlib & seaborn official tutorials
- Claus Wilke, "Fundamentals of Data Visualization" (free online)
- Data-to-Viz: https://www.data-to-viz.com
