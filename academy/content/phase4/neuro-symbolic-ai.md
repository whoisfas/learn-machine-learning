---
{
  "id": "p4-neuro-symbolic-ai",
  "title": "Neuro-Symbolic AI",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 14,
  "objectives": [
    "Understand the integration of neural and symbolic AI",
    "Recognize limitations of pure neural and pure symbolic approaches",
    "Explore key neuro-symbolic architectures",
    "Apply neuro-symbolic methods to reasoning tasks"
  ],
  "concepts": [
    "Symbolic AI: logic, rules, knowledge representation",
    "Neural AI: pattern recognition, learning from data",
    "Neuro-symbolic integration approaches",
    "Logic Tensor Networks (LTN)",
    "Differentiable inductive logic programming",
    "Neural theorem provers",
    "Knowledge graph embeddings + reasoning",
    "Common sense reasoning challenges"
  ],
  "formulas": [],
  "references": [
    "Garcez & Lamb 'Neurosymbolic AI: The 3rd Wave'",
    "d'Avila Garcez et al. 'Neural-Symbolic Learning and Reasoning'",
    "LTN documentation",
    "DeepProbLog documentation"
  ]
}
---

## Penjelasan

Neuro-Symbolic AI menggabungkan kekuatan neural networks (belajar dari data, pattern recognition) dengan symbolic AI (logic, reasoning, knowledge representation). Tujuannya: sistem yang bisa belajar dari data *dan* bernalar secara eksplisit — mengatasi keterbatasan masing-masing pendekatan.

**Keterbatasan pendekatan murni.**
- *Pure neural*: black-box, tidak bisa explain reasoning, butuh banyak data, kesulitan dengan compositional reasoning.
- *Pure symbolic*: butuh knowledge engineering manual, tidak bisa belajar dari data, brittle terhadap noise.

**Pendekatan integrasi.**
- *Neural-symbolic*: neural network menghasilkan representasi yang bisa diproses symbolic engine; atau symbolic rules dipelajari oleh neural network.
- *Logic Tensor Networks (LTN)*: real-valued logic — predicates dan functions direpresentasikan sebagai neural networks; logical constraints di-differentiate dan dioptimasi bersama data loss.
- *Differentiable ILP*: inductive logic programming yang differentiable — belajar rules logika dari data.
- *Neural theorem provers*: neural network memandu proof search dalam theorem prover.
- *Knowledge graph + GNN*: graph neural networks untuk reasoning di atas knowledge graphs.

**Contoh tasks.** Visual question answering dengan reasoning ("berapa banyak objek yang lebih besar dari X?"), commonsense reasoning, mathematical reasoning, program synthesis dengan constraints.

## Konsep Penting

1. **Complementary strengths** — neural untuk perception/learning; symbolic untuk reasoning/interpretability.
2. **Differentiable logic** — kunci integrasi: logical operators (AND, OR, NOT) di-relax ke continuous functions.
3. **Knowledge injection** — symbolic knowledge (rules, ontologies) bisa di-encode sebagai constraints atau priors.
4. **Explainability** — neuro-symbolic systems bisa memberikan reasoning trace, bukan hanya prediction.
5. **Scalability challenge** — symbolic reasoning bisa combinatorial; neural approximation membantu.

## Formula

- Fuzzy logic: AND(a,b) = a·b; OR(a,b) = a+b−a·b; NOT(a) = 1−a.
- LTN: truth values in [0,1]; logical constraints as loss terms.

## Contoh Kasus

Visual reasoning: "Apakah ada dua mobil yang lebih besar dari truk?" — object detection (neural) -> symbolic reasoning over detected objects -> answer. Atau mathematical reasoning: neural network memprediksi langkah proof, symbolic engine memverifikasi.

## Implementasi

```python
import torch
import torch.nn as nn

# Simple differentiable logic for neuro-symbolic demo
class LogicLayer(nn.Module):
    """Fuzzy logic operations as neural layers."""
    def __init__(self):
        super().__init__()

    def fuzzy_and(self, a, b):
        return a * b

    def fuzzy_or(self, a, b):
        return a + b - a * b

    def fuzzy_not(self, a):
        return 1 - a

    def implies(self, a, b):
        # a -> b = NOT a OR b
        return self.fuzzy_or(self.fuzzy_not(a), b)

# Example: learn to satisfy logical constraints
class NeuroSymbolicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(),
            nn.Linear(32, 2))  # outputs truth values for P, Q
        self.logic = LogicLayer()

    def forward(self, x):
        # returns truth values for predicates P(x), Q(x)
        return torch.sigmoid(self.encoder(x))

    def constraint_loss(self, x):
        """Loss enforcing logical constraints like forall x: P(x) -> Q(x)."""
        p, q = self.forward(x).unbind(dim=1)
        # constraint: P(x) -> Q(x) should be true
        implication = self.logic.implies(p, q)
        return (1 - implication).mean()  # minimize violation
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Neuro-symbolic AI combines:",
    "options": [
      "Neural learning capabilities with symbolic reasoning",
      "Two neural networks",
      "Two symbolic systems",
      "Rule-based and statistical separately"
    ],
    "answer": 0,
    "explanation": "Neuro-symbolic = neural networks + symbolic logic/reasoning integration.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "Logic Tensor Networks (LTN) represent logical predicates as:",
    "options": [
      "Neural networks outputting truth values in [0,1]",
      "Boolean variables",
      "First-order logic formulas",
      "Decision trees"
    ],
    "answer": 0,
    "explanation": "LTN relaxes logic to real-valued [0,1] — differentiable for gradient descent.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "The main advantage of neuro-symbolic over pure neural is:",
    "options": [
      "Explicit reasoning and interpretability alongside learning",
      "Faster training",
      "Less data needed always",
      "Better accuracy always"
    ],
    "answer": 0,
    "explanation": "Symbolic component provides reasoning trace and compositional generalization.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `fuzzy_logic_and(a, b)`, `fuzzy_logic_or(a, b)`, `fuzzy_logic_not(a)` for tensors in [0,1].",
    "starter_code": "import torch\n\ndef fuzzy_and(a, b): return a * b\ndef fuzzy_or(a, b): return a + b - a * b\ndef fuzzy_not(a): return 1 - a\n\ndef fuzzy_implies(a, b):\n    pass\n",
    "function": "fuzzy_implies",
    "tests": [
      {"call": "fuzzy_implies(torch.tensor([1.0]), torch.tensor([0.5])).item()", "expected": 0.5},
      {"call": "fuzzy_implies(torch.tensor([0.0]), torch.tensor([0.0])).item()", "expected": 1.0}
    ],
    "hints": ["a -> b = NOT a OR b = (1-a) + b - (1-a)*b = 1 - a + a*b"],
    "explanation": "Fuzzy implication: a->b = ~a OR b.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Design a neuro-symbolic system for medical diagnosis: neural network processes symptoms/images, symbolic engine applies medical rules. Describe the architecture, how they interact, failure modes, and validation strategy.",
    "rubric": [
      "Neural extracts features/probabilities; symbolic applies diagnostic rules",
          "Logical constraints ensure consistency (e.g., contraindications)",
      "Failure modes: conflicting neural/symbolic outputs, incomplete rules",
      "Validation: clinical trials, expert review, ablation of components"
        ],
    "points": 3
  }
]
```

## Referensi

- Garcez & Lamb (2020) 'Neurosymbolic AI: The 3rd Wave'
- d'Avila Garcez et al., 'Neural-Symbolic Learning and Reasoning'
- Logic Tensor Networks documentation
- DeepProbLog documentation
- 'The Third Wave of AI' — Special issue, AI Magazine
