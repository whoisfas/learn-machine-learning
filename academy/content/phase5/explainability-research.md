---
{
  "id": "p5-explainability-research",
  "title": "Explainability in Research Context",
  "phase": 5,
  "difficulty": "advanced",
  "estimated_hours": 10,
  "objectives": [
    "Evaluate explanation methods as research objects",
    "Design experiments to validate explanation faithfulness",
    "Position explainability research within ML literature",
    "Understand current debates and open problems in XAI"
  ],
  "concepts": [
    "Explanation as hypothesis vs explanation as proof",
    "Faithfulness evaluation: perturbation, deletion, sufficiency",
    "Plausibility vs faithfulness distinction",
    "Human evaluation of explanations",
    "XAI for scientific discovery vs XAI for trust",
    "Causal vs correlational explanations",
    "Open problems: stability, completeness, user studies"
  ],
  "formulas": [],
  "references": [
    "Ribeiro & Singh 'Anchors: High-Precision Model-Agnostic Explanations'",
    "'Towards a rigorous science of interpretable ML' — Doshi-Velez & Kim",
    "'Evaluating Explainable AI: Which Algorithm and Metrics to Use?' — Nauta et al.",
    "ACM FAT* conference proceedings"
  ]
}
---

## Penjelasan

Modul ini membahas explainability dari perspektif *research* — bukan hanya menggunakan LIME/SHAP, melainkan meneliti apakah explanations itu sendiri valid, reliable, dan berguna. Ini adalah frontier research aktif di ML.

**Explanation sebagai objek research.** Pertanyaan ilmiah: "Apakah explanation X mencerminkan reasoning model Y?" — ini berbeda dari "apakah explanation X masuk akal?" Explanation bisa plausible (terlihat masuk akal) tapi tidak faithful (tidak mencerminkan model sebenarnya).

**Faithfulness evaluation.** Bagaimana mengukur apakah explanation faithful? Metode: (1) *Perturbation*: ubah fitur yang ditandai explanation sebagai penting -> prediksi harus berubah signifikan. (2) *Deletion*: hapus fitur penting -> performance harus drop. (3) *Sufficiency*: hanya gunakan fitur yang ditandai -> harus bisa rekonstruksi prediksi. (4) *Comprehensiveness*: seberapa banyak information explanation capture.

**Plausibility vs faithfulness.** Plausibility = apakah explanation masuk akal bagi manusia (subjective). Faithfulness = apakah explanation mencerminkan model (objective). Kedua bisa berbeda: explanation bisa plausible tapi misleading (model sebenarnya pakai shortcut lain).

**Human evaluation.** User studies: apakah explanation membantu pengguna membuat keputusan yang lebih baik? Apakah meningkatkan trust secara kalibrasi (bukan over-trust)? Apakah explanation mengungkap bias?

**XAI untuk science vs trust.** Dua motivasi berbeda: (1) *Trust/safety*: regulatory compliance, debugging, fairness — explanation untuk stakeholder. (2) *Science*: memahami apa yang dipelajari model untuk insight ilmiah — explanation sebagai research tool.

## Konsep Penting

1. **Explanation != ground truth** — explanation adalah aproksimasi; model mungkin punya reasoning berbeda.
2. **Stability problem** — penjelasan untuk input yang sama bisa berbeda antar run; masalah untuk reliability.
3. **Causality vs correlation** — explanation menunjukkan korelasi fitur-prediksi, bukan kausalitas.
4. **User studies essential** — explanation yang "benar" secara teknis bisa tidak berguna bagi user.
5. **Open problem: completeness** — bagaimana tahu explanation sudah mencakup semua alasan model memutuskan?

## Formula

- Faithfulness metrics: comprehensiveness, sufficiency, deletion AUC.
- Stability: variance of explanations across runs/perturbations.

## Contoh Kasus

Research question: "Apakah SHAP values faithful terhadap gradient boosting models?" Eksperimen: bandingkan SHAP feature importance dengan actual permutation importance; uji pada synthetic data dengan known ground truth; ukur correlation antara explanation ranking dan true feature importance.

## Implementasi

```python
import numpy as np
from sklearn.inspection import permutation_importance

def faithfulness_test(model, X, y, explainer_fn, metric_fn):
    """Test if explanation-identified features are actually important."""
    # get baseline performance
    baseline = metric_fn(y, model.predict(X))
    # get explanation-based feature ranking
    explanations = explainer_fn(model, X)
    feature_ranking = np.argsort(-np.abs(explanations))
    # perturb top-k features and measure performance drop
    results = []
    for k in [1, 3, 5, 10]:
        X_perturbed = X.copy()
        # shuffle top-k features
        for i in range(k):
            idx = feature_ranking[i]
            X_perturbed[:, idx] = np.random.permutation(X_perturbed[:, idx])
        perturbed_score = metric_fn(y, model.predict(X_perturbed))
        drop = baseline - perturbed_score
        results.append({'k': k, 'performance_drop': drop})
    return results

def stability_test(explainer_fn, X, n_runs=10):
    """Measure explanation stability across runs."""
    explanations = []
    for _ in range(n_runs):
        explanations.append(explainer_fn(X))
    # compute variance across runs
    return np.var(explanations, axis=0)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Faithfulness of an explanation means:",
    "options": [
      "The explanation accurately reflects the model's actual reasoning",
      "The explanation is plausible to humans",
      "The explanation is simple",
      "The explanation is fast"
    ],
    "answer": 0,
    "explanation": "Faithfulness = explanation matches model behavior; plausibility = looks reasonable.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "A perturbation test for explanation faithfulness works by:",
    "options": [
      "Modifying features identified as important and checking if predictions change",
      "Adding random noise",
      "Removing the model",
      "Changing the dataset"
    ],
    "answer": 0,
    "explanation": "If explanation is faithful, perturbing 'important' features should degrade performance.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The main open problem in XAI research is:",
    "options": [
      "Ensuring explanations are both faithful and useful to humans",
      "Making explanations faster",
      "Using fewer features",
      "Removing neural networks"
    ],
    "answer": 0,
    "explanation": "Bridging the gap between technical faithfulness and human utility.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `explanation_stability(explanations)` computing variance of explanations across multiple runs.",
    "starter_code": "import numpy as np\n\ndef explanation_stability(explanations):\n    # explanations: list of explanation arrays (one per run)\n    explanations = np.array(explanations)\n    pass\n    return np.mean(np.var(explanations, axis=0))\n",
    "function": "explanation_stability",
    "tests": [
      {"call": "explanation_stability([np.array([1,2]),np.array([1,2]),np.array([1,2])])", "expected": 0.0},
      {"call": "explanation_stability([np.array([1,0]),np.array([0,1])]) > 0", "expected": true}
    ],
    "hints": ["np.var(explanations, axis=0).mean()"],
    "explanation": "Measure explanation consistency.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Design an experiment to test whether SHAP explanations help doctors make better diagnostic decisions than showing raw model confidence scores. Describe your study design, metrics, and potential confounds.",
    "rubric": [
      "Randomized controlled trial: doctors with SHAP vs confidence only",
          "Metrics: diagnostic accuracy, time to decision, confidence calibration",
      "Control for doctor experience, case difficulty",
      "Potential confounds: learning effects, Hawthorne effect, explanation complexity"
        ],
    "points": 3
  }
]
```

## Referensi

- Doshi-Velez & Kim (2017) 'Towards a Rigorous Science of Interpretable ML'
- Nauta et al. 'Evaluating Explainable AI: Which Algorithm and Metrics to Use?'
- Ribeiro & Singh (2018) 'Anchors: High-Precision Model-Agnostic Explanations'
- ACM FAT* conference proceedings
- 'Interpretable Machine Learning' — Molnar (free book)
