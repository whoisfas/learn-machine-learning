---
{
  "id": "p4-explainable-ai",
  "title": "Explainable AI (XAI)",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Understand why explainability matters in ML",
    "Apply post-hoc explanation methods: LIME, SHAP",
    "Interpret model internals: attention, gradients, feature importance",
    "Evaluate explanation quality and faithfulness"
  ],
  "concepts": [
    "Interpretability vs explainability vs transparency",
    "Post-hoc methods: LIME, SHAP (KernelSHAP, TreeSHAP)",
    "Intrinsic interpretability: linear models, decision trees, GAMs",
    "Attention visualization (limitations)",
    "Gradient-based methods: saliency, integrated gradients",
    "Counterfactual explanations",
    "Evaluation of explanations: faithfulness, stability, plausibility",
    "XAI in high-stakes domains: healthcare, finance, criminal justice"
  ],
  "formulas": [
    {"name": "SHAP value", "latex": "\\phi_i = \\sum_{S \\subseteq N\\setminus\\{i\\}} \\frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S\\cup\\{i\\}) - f(S)]", "note": "Shapley value from game theory."},
    {"name": "LIME", "latex": "\\xi(x) = \\arg\\min_{g \\in G} L(f, g, \\pi_x) + \\Omega(g)", "note": "Local interpretable surrogate."}
  ],
  "references": [
    "Ribeiro et al. 2016 'Why Should I Trust You?' (LIME)",
    "Lundberg & Lee 2017 'A Unified Approach to Interpreting Model Predictions' (SHAP)",
    "Molnar 'Interpretable Machine Learning' (free book)",
    "SHAP documentation"
  ]
}
---

## Penjelasan

Explainable AI (XAI) adalah bidang yang membuat keputusan model ML bisa dipahami manusia. Penting untuk trust, debugging, fairness, dan compliance — terutama di domain high-stakes (medis, finansial, hukum).

**Tingkat interpretabilitas.**
- *Transparent/intrinsic*: model itu sendiri bisa diinterpretasi — linear regression (koefisien), decision tree (rules), logistic regression (odds ratios).
- *Post-hoc*: model black-box, tapi penjelasan dibuat setelah prediksi — LIME, SHAP, saliency maps.
- *Black-box*: tidak ada penjelasan (deep neural network mentah).

**LIME (Local Interpretable Model-agnostic Explanations).** Untuk satu prediksi, sample data di sekitar titik itu (perturbasi), beri weight berdasarkan kedekatan, fit model linear sederhana pada sample itu — koefisien linear menunjukkan fitur mana yang paling berpengaruh *secara lokal*. Kelebihan: model-agnostic, intuitif. Kekurangan: bisa tidak stabil, hanya lokal.

**SHAP (SHapley Additive exPlanations).** Berbasis game theory: Shapley value mengukur kontribusi setiap fitur terhadap prediksi dengan mempertimbangkan semua subset fitur. KernelSHAP = aproksimasi LIME dengan jaminan teoretis; TreeSHAP = efisien untuk tree-based models. SHAP values additive: base_value + Σ SHAP_i = prediction.

**Gradient-based.** Saliency maps (gradient input terhadap output), integrated gradients (integral gradient sepanjang path dari baseline), Grad-CAM (untuk CNN — visualisasi region penting di gambar).

**Counterfactual explanations.** "Apa perubahan minimal pada input yang akan mengubah prediksi?" — mis. "Jika income naik $5k, loan akan di-approve". Powerful untuk actionable explanations.

## Konsep Penting

1. **Explanation != truth** — LIME/SHAP adalah aproksimasi; bisa tidak faithful terhadap model sebenarnya.
2. **Local vs global** — LIME/SHAP lokal per prediksi; global importance butuh agregasi atau metode lain.
3. **Attention is not explanation** — attention weights di transformer tidak reliable sebagai explanation (Bibal et al. 2020).
4. **Evaluation matters** — faithfulness (apakah explanation mencerminkan model?), stability (apakah konsisten?), plausibility (apakah masuk akal domain?).
5. **Interpretability-performance tradeoff** — model interpretable (linear, tree) sering kurang akurat dari black-box; SHAP bridging gap.

## Formula

- SHAP: φi = Σ_S [|S|!(|N|-|S|-1)!/|N|!] [f(S∪{i}) - f(S)].
- LIME: local surrogate with proximity weighting.
- Integrated gradients: IG_i = (x_i - x'_i) ∫_0^1 ∂F(x' + α(x-x'))/∂x_i dα.

## Contoh Kasus

Loan approval model (gradient boosting): SHAP values menunjukkan income paling berpengaruh, diikuti debt_ratio dan credit_score. Untuk applicant yang ditolak, counterfactual: "Naikkan income $8k atau turunkan debt_ratio 0.1 untuk approval". SHAP summary plot menunjukkan distribusi impact per fitur.

## Implementasi

```python
import shap
import matplotlib.pyplot as plt

# TreeSHAP for tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# summary plot: global importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# force plot for single prediction
shap.force_plot(explainer.expected_value, shap_values[0],
                X_test[0], feature_names=feature_names)

# KernelSHAP for any model
explainer_ks = shap.KernelExplainer(model.predict_proba, X_background)
ks_values = explainer_ks.shap_values(X_test[:10])

# LIME
from lime import lime_tabular
explainer_lime = lime_tabular.LimeTabularExplainer(
    X_train, feature_names=feature_names,
    class_names=['reject','approve'], mode='classification')
exp = explainer_lime.explain_instance(X_test[0], model.predict_proba)
exp.show_in_notebook()

# counterfactual with DiCE
import dice_ml
# dice_data = dice_ml.Data(...)
# dice_model = dice_ml.Model(model=model, backend='sklearn')
# exp = dice_gen.generate_counterfactuals(...)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "SHAP values are based on:",
    "options": [
      "Shapley values from cooperative game theory",
      "Gradient descent",
      "Random perturbations",
      "Decision trees"
    ],
    "answer": 0,
    "explanation": "SHAP = SHapley Additive exPlanations; Shapley values fairly distribute prediction among features.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "LIME explains predictions by:",
    "options": [
      "Fitting a simple interpretable model locally around the prediction",
      "Using gradients",
      "Perturbing all features equally",
      "Analyzing attention weights"
    ],
    "answer": 0,
    "explanation": "LIME fits local linear surrogate weighted by proximity to the instance.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Why can't attention weights be trusted as explanations?",
    "options": [
      "Research shows attention doesn't reliably correlate with feature importance",
      "They are always zero",
      "They are too complex",
      "They only work for images"
    ],
    "answer": 0,
    "explanation": "Bibal et al. 2020: attention weights don't reliably indicate which inputs drive predictions.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `shap_kernel_approx(f, x, background, S_samples)` estimating Shapley value for feature i using kernel SHAP approximation (simplified: use only singleton and empty coalitions).",
    "starter_code": "import numpy as np\nimport itertools\n\ndef shap_kernel_approx(f, x, background, feature_idx):\n    # f: callable(x) -> prediction\n    # background: array of reference samples\n    # feature_idx: which feature to explain\n    pass\n",
    "function": "shap_kernel_approx",
    "tests": [
      {"call": "shap_kernel_approx(lambda x: float(x[0]+2*x[1]), [1.0,1.0], [[0,0]], 0)", "expected": "check"}
    ],
    "hints": ["phi_i = sum over coalitions S not containing i of weight * (f(S+i) - f(S))", "simplified: use empty and full coalitions"],
    "explanation": "SHAP value estimation via coalition sampling.",
    "points": 3
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "A bank uses a deep neural network for loan approval. Regulators require 'meaningful explanations' for rejected applicants. Compare three XAI approaches (LIME, SHAP, counterfactual) for this use case — which would you deploy and why?",
    "rubric": [
      "SHAP: consistent, theoretically grounded, global+local explanations",
          "Counterfactual: actionable ('what to change to get approved')",
      "LIME: fast but unstable, less reliable for regulatory use",
      "Consideration: stability, faithfulness, auditability requirements"
        ],
    "points": 3
  }
]
```

## Referensi

- Ribeiro, Singh, Guestrin (2016) 'Why Should I Trust You? Explaining Predictions of Any Classifier' (LIME)
- Lundberg & Lee (2017) 'A Unified Approach to Interpreting Model Predictions' (SHAP)
- Molnar, 'Interpretable Machine Learning' (free book)
- SHAP documentation (https://shap.readthedocs.io)
- Bibal et al. (2020) 'Is Attention Explanation? An Introduction to the Debate'
