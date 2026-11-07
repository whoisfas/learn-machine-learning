---
{
  "id": "p6-visualization-reporting",
  "title": "Visualization and Research Reporting",
  "phase": 6,
  "difficulty": "intermediate",
  "estimated_hours": 8,
  "objectives": [
    "Create publication-quality figures for research papers",
    "Choose appropriate visualization types for different data",
    "Write clear figure captions and result descriptions",
    "Structure research reports and papers"
  ],
  "concepts": [
    "Figure types: learning curves, bar charts, scatter plots, heatmaps, ROC curves",
    "Publication standards: resolution, fonts, colorblind-friendly",
    "Caption writing: self-contained descriptions",
    "Table design: concise, bold best, mean±std",
    "Result narrative: what figures show, not just what they are",
    "Supplementary materials organization",
    "Reproducible figure generation: code + data"
  ],
  "formulas": [],
  "references": [
    "Phase 1: Data Visualization",
    "Nature/Science figure guidelines",
    "'Fundamentals of Data Visualization' — Claus Wilke",
    "matplotlib/seaborn documentation"
  ]
}
---

## Penjelasan

Visualization adalah cara Anda berkomunikasi temuan research. Figure yang buruk bisa membunuh paper yang baik; figure yang baik bisa membuat research accessible. Publication-quality figures membutuhkan attention to detail.

**Figure types untuk ML research.**
- *Learning curves*: train/val loss vs epoch — menunjukkan convergence, overfitting.
- *Bar charts*: model comparison across datasets — bold best, error bars.
- *Scatter plots*: predicted vs actual — calibration, outliers.
- *Heatmaps*: confusion matrices, attention maps, correlation matrices.
- *ROC/PR curves*: classification performance across thresholds.
- *Ablation charts*: component contribution visualization.

**Publication standards.** Resolution: 300+ DPI untuk print. Fonts: readable at print size (8-12pt). Colors: colorblind-friendly palettes (viridis, not jet). Labels: axis labels, units, legend. Captions: self-contained — bisa dipahami tanpa membaca text.

**Caption writing.** Setiap caption harus: (1) describe what figure shows, (2) mention data/source, (3) highlight key takeaway, (4) be self-contained. "Figure 1: Test accuracy comparison across 5 datasets. Our method (blue) outperforms baselines on 4/5 datasets, with largest improvement on Dataset C (+12%)."

**Table design.** Concise: methods as rows, metrics as columns. Bold best per column. Report mean±std. Footnotes untuk statistical significance markers. Don't repeat what's in figures.

**Result narrative.** Text harus *menjelaskan* figures, bukan mendeskripsikan ulang. "As shown in Figure 2, our method achieves 15% higher accuracy on Dataset A (p<0.01), suggesting that component X is critical for this task."

## Konsep Penting

1. **Figures are arguments** — setiap figure mendukung satu poin.
2. **Self-contained captions** — reader harus paham tanpa text.
3. **Colorblind-friendly** — ~8% pria colorblind; gunakan safe palettes.
4. **Reproducible figures** — code + data untuk setiap figure.
5. **Less is more** — satu figure, satu message.

## Implementasi

```python
import matplotlib.pyplot as plt
import numpy as np

def publication_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def learning_curve(train_losses, val_losses, save_path='learning_curve.png'):
    """Plot learning curves."""
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def comparison_bar(results, save_path='comparison.png'):
    """Bar chart for model comparison."""
    methods = list(results.keys())
    scores = [results[m] for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_ylim(0, 1)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "A figure caption should be:",
    "options": [
      "Self-contained: understandable without reading the main text",
      "As short as possible",
      "Only a title",
      "In the appendix only"
    ],
    "answer": 0,
    "explanation": "Captions must describe what, where, and key takeaway.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "For colorblind-friendly visualizations, use:",
    "options": [
      "Viridis, cividis, or colorblind-safe palettes",
      "Jet colormap",
      "Red-green contrasts",
      "Rainbow colors"
    ],
    "answer": 0,
    "explanation": "Jet and red-green are problematic for colorblind users.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "A learning curve plot shows:",
    "options": [
      "Training and validation loss/metrics vs epoch to diagnose convergence and overfitting",
      "Model architecture",
      "Data distribution",
      "Hyperparameter sensitivity"
    ],
    "answer": 0,
    "explanation": "Learning curves reveal training dynamics.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `confusion_matrix_plot(y_true, y_pred, labels)` creating a heatmap visualization.",
    "starter_code": "import matplotlib.pyplot as plt\nimport numpy as np\nfrom sklearn.metrics import confusion_matrix\n\ndef confusion_matrix_plot(y_true, y_pred, labels, save_path='cm.png'):\n    cm = confusion_matrix(y_true, y_pred, labels=labels)\n    fig, ax = plt.subplots(figsize=(5, 4))\n    im = ax.imshow(cm, cmap='Blues')\n    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45)\n    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)\n    ax.set_xlabel('Predicted'); ax.set_ylabel('True')\n    for i in range(len(labels)):\n        for j in range(len(labels)):\n            ax.text(j, i, cm[i, j], ha='center', va='center')\n    fig.savefig(save_path, dpi=300, bbox_inches='tight')\n    plt.close(fig)\n",
    "function": "confusion_matrix_plot",
    "tests": [{"call": "confusion_matrix_plot([0,1,0,1],[0,0,1,1],['neg','pos'])", "expected": "check"}],
    "hints": ["Use sklearn confusion_matrix; imshow heatmap; annotate cells"],
    "explanation": "Confusion matrix visualization.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your paper has 15 figures and reviewers say 'too many figures, unclear narrative'. How would you restructure your visualizations for a NeurIPS submission?",
    "rubric": [
      "Reduce to 5-7 main figures; move rest to appendix",
          "Each figure supports one specific claim",
      "Improve captions: self-contained with key takeaway",
      "Use consistent style across all figures",
      "Create summary table for detailed results"
        ],
    "points": 2
  }
]
```

## Referensi

- Phase 1: Data Visualization module
- Nature/Science figure guidelines
- 'Fundamentals of Data Visualization' — Claus Wilke
- matplotlib/seaborn documentation
