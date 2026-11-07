---
{
  "id": "p3-loss-functions",
  "title": "Loss Functions",
  "phase": 3,
  "difficulty": "intermediate",
  "estimated_hours": 10,
  "objectives": [
    "Understand common loss functions for regression and classification",
    "Derive gradients for MSE, MAE, cross-entropy, hinge",
    "Choose appropriate loss for task and label distribution",
    "Handle numerical stability in loss computation"
  ],
  "concepts": [
    "MSE, MAE, Huber loss for regression",
    "Cross-entropy, binary cross-entropy, focal loss for classification",
    "Hinge loss for SVM-like objectives",
    "KL divergence and distribution matching",
    "Custom loss functions and multi-task learning",
    "Loss weighting and class imbalance handling",
    "Numerical stability (log-sum-exp trick)"
  ],
  "formulas": [
    {"name": "MSE", "latex": "L = \\frac{1}{n}\\sum_i (y_i - \\hat{y}_i)^2", "note": "Sensitive to outliers; smooth gradient."},
    {"name": "MAE", "latex": "L = \\frac{1}{n}\\sum_i |y_i - \\hat{y}_i|", "note": "Robust; non-differentiable at 0."},
    {"name": "Huber", "latex": "L_\\delta = \\begin{cases} \\frac{1}{2}(y-\\hat{y})^2 & |y-\\hat{y}|\\le\\delta \\\\ \\delta(|y-\\hat{y}|-\\frac{\\delta}{2}) & \\text{else} \\end{cases}", "note": "Quadratic near 0, linear for outliers."},
    {"name": "Cross-entropy", "latex": "L = -\\sum_c y_c \\log \\hat{p}_c", "note": "Standard classification loss."},
    {"name": "Focal loss", "latex": "L = -\\alpha(1-p_t)^\\gamma \\log p_t", "note": "Down-weights easy examples; for imbalance."}
  ],
  "references": [
    "Goodfellow et al. ch. 6.2",
    "Lin et al. 2017 'Focal Loss for Dense Object Detection'",
    "PyTorch nn loss documentation"
  ]
}
---

## Penjelasan

Loss function mengukur seberapa salah prediksi model untuk satu contoh; training = meminimalkan rata-rata loss. Pilihan loss sangat memengaruhi apa yang dipelajari model — loss adalah "objective function" yang dioptimasi.

**Regression losses.**
- *MSE*: kuadrat error; smooth, convex; sensitif terhadap outlier (error besar dihukum berat). Gradient: 2(y−ŷ)/n.
- *MAE*: absolute error; robust terhadap outlier; non-differentiable di 0 (subgradient dipakai). Gradient konstan magnitude.
- *Huber*: quadratic untuk |error|<δ, linear untuk >δ — kompromi MSE/MAE; baik untuk data dengan outlier sporadis.
- *Quantile loss*: untuk interval prediction; asymmetric (pinball loss).

**Classification losses.**
- *Cross-entropy (log loss)*: −Σ y·log(p). Standar untuk softmax klasifikasi. Gradient elegant: p − y. Convex untuk linear models.
- *Binary cross-entropy*: untuk binary/multi-label; −[y·log(p) + (1−y)·log(1−p)].
- *Hinge loss*: max(0, 1−y·f(x)) — untuk SVM-like margin maximization; linear penalty untuk margin violation.
- *Focal loss*: modifikasi CE dengan factor (1−pt)^γ — down-weight easy examples, fokus pada hard examples; sangat efektif untuk extreme class imbalance (object detection).

**Numerical stability.** Softmax + cross-entropy bisa digabung: log_softmax + NLLLoss lebih stabil dari softmax -> log terpisah. log-sum-exp trick: log Σexp(xi) = max(x) + log Σexp(xi − max(x)) — mencegah overflow.

## Konsep Penting

1. **Loss ≠ metric** — loss dioptimasi; metric dievaluasi; keduanya terkait tapi berbeda tujuan.
2. **Outlier sensitivity** — MSE dihukum outlier; MAE/Huber lebih robust.
3. **Class imbalance** — CE unbiased tapi bisa didominasi kelas mayoritas; focal loss atau class weights membantu.
4. **Smoothness matters** — loss harus differentiable (atau subdifferentiable) untuk gradient descent.
5. **Custom losses** — domain knowledge bisa di-encode ke loss (mis. physics-informed loss menambah penalty pelanggaran hukum fisika).

## Formula

- CE gradient (softmax): ∂L/∂z = p − y.
- BCE: −[y log p + (1−y) log(1−p)].
- Huber δ: quadratic if |r|≤δ else linear.
- Focal: −α(1−pt)^γ log(pt), pt = p if y=1 else 1−p.

## Contoh Kasus

Object detection dengan extreme imbalance (99.9% background): focal loss γ=2 down-weight easy background examples, fokus pada rare objects. Bandingkan dengan weighted CE — focal lebih adaptif karena weight tergantung confidence.

## Implementasi

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# standard losses
ce = nn.CrossEntropyLoss()          # logits -> loss
bce = nn.BCEWithLogitsLoss()        # stable binary CE
mse = nn.MSELoss()
huber = nn.HuberLoss(delta=1.0)

# focal loss implementation
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1-pt)**gamma * ce).mean()

# numerical stability: log_softmax + NLL
loss = F.nll_loss(F.log_softmax(logits, dim=1), targets)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Why is cross-entropy preferred over MSE for classification?",
    "options": [
      "It provides stronger gradients for confident wrong predictions and works with probabilities",
      "It's faster",
      "MSE doesn't work for classification",
      "It's more interpretable"
    ],
    "answer": 0,
    "explanation": "CE gradient p-y is large when confident wrong; MSE saturates. CE is proper scoring rule for probabilities.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Huber loss is preferred over MSE when:",
    "options": [
      "Data contains outliers that shouldn't dominate training",
      "You need maximum accuracy",
      "Data is perfectly clean",
      "You want sparse gradients"
    ],
    "answer": 0,
    "explanation": "Huber is quadratic near zero (efficient) but linear for large errors (robust).",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "math",
    "question": "For binary cross-entropy with y=1, p=0.9, compute the loss.",
    "answer": 0.10536051565782628,
    "tolerance": 0.001,
    "hints": ["-log(0.9)"],
    "explanation": "-ln(0.9) ≈ 0.105.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `focal_loss(logits, targets, alpha=0.25, gamma=2.0)` for multiclass classification returning mean focal loss.",
    "starter_code": "import torch\nimport torch.nn.functional as F\n\ndef focal_loss(logits, targets, alpha=0.25, gamma=2.0):\n    ce = F.cross_entropy(logits, targets, reduction='none')\n    pass\n",
    "function": "focal_loss",
    "tests": [
      {"call": "focal_loss(torch.tensor([[2.0,0.0]]), torch.tensor([0]), 0.25, 2.0).item() > 0", "expected": true}
    ],
    "hints": ["pt = exp(-ce); loss = alpha*(1-pt)^gamma*ce; return mean"],
    "explanation": "Focal loss down-weights easy examples.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Your regression model uses MSE but test predictions are pulled toward outliers. Explain why and compare MAE vs Huber loss as alternatives — including their gradient properties.",
    "rubric": [
      "MSE gradient proportional to error -> outliers dominate",
          "MAE gradient constant magnitude -> robust but non-smooth at 0",
      "Huber combines: quadratic near 0 (smooth), linear for outliers (robust)",
      "Mentions gradient clipping or data cleaning as complementary"
        ],
    "points": 2
  }
]
```

## Referensi

- Goodfellow et al. ch. 6.2 (loss functions)
- Lin et al. (2017) 'Focal Loss for Dense Object Detection'
- PyTorch nn loss documentation
- 'A Survey of Loss Functions' — various
