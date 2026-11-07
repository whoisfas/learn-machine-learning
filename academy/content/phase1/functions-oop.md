---
{
  "id": "p1-functions-oop",
  "title": "Functions and Object-Oriented Programming",
  "phase": 1,
  "difficulty": "beginner",
  "estimated_hours": 10,
  "objectives": [
    "Design functions with clear contracts (args, returns, docstrings)",
    "Use *args, **kwargs, keyword-only args correctly",
    "Build classes with encapsulation, inheritance, and polymorphism",
    "Understand dunder methods and composition vs inheritance"
  ],
  "concepts": [
    "First-class functions, closures, higher-order functions",
    "Default mutable argument pitfall",
    "Keyword-only arguments (`*`)",
    "Classes: `__init__`, `self`, attributes, methods",
    "Inheritance, `super()`, MRO",
    "Composition over inheritance",
    "Dunder methods: `__repr__`, `__eq__`, `__len__`, `__call__`",
    "Abstract base classes (abc)"
  ],
  "formulas": [],
  "references": [
    "Python docs — data model: https://docs.python.org/3/reference/datamodel.html",
    "Real Python OOP tutorials"
  ]
}
---

## Penjelasan

Fungsi yang baik adalah kontrak: input jelas, output jelas, error terdefinisi. Di ML, hampir semua library adalah fungsi-fungsi murni (`fit`, `predict`, `transform`) — memahami kontrak fungsi membuat Anda bisa membaca source code sklearn secara efektif.

**Fungsi lanjutan.** Fungsi adalah objek first-class: bisa disimpan di variabel, dikirim sebagai argumen (`sorted(xs, key=len)`), dikembalikan. *Closure* adalah fungsi yang mengingat variabel dari scope tempat ia didefinisikan — dasar dari decorator dan factory function. *Higher-order function* menerima/mengembalikan fungsi.

**Keyword-only args.** `def train(model, *, epochs=10, lr=0.01)` memaksa caller menulis nama argumen — mencegah bug positional seperti `train(m, 10, 0.01)` yang ambigu.

**OOP untuk ML.** Hampir semua estimator sklearn mengikuti pola yang sama: class dengan `fit(X, y)`, `predict(X)`, `transform(X)`. Memahami pola ini membuat custom estimator (mis. wrapper untuk preprocessing khusus) mudah ditulis. Konsep inti:

- **Encapsulation**: data + perilaku dalam satu objek; atribut "private" dengan konvensi `_attr`.
- **Inheritance**: subclass mewarisi perilaku; gunakan `super().__init__()`.
- **Polymorphism**: objek berbeda bisa dipakai di tempat yang sama jika interface-nya sama (`for clf in [lr, rf, svm]: clf.fit(...)`).
- **Composition**: "has-a" lebih fleksibel dari "is-a". Pipeline sklearn = komposisi transformer+estimator.

**Dunder methods** membuat objek Pythonic: `__repr__` untuk debugging, `__eq__` untuk perbandingan, `__call__` membuat instance bisa dipanggil seperti fungsi.

## Konsep Penting

1. **`self` adalah instance** — method tanpa `self` tidak bisa akses atribut instance.
2. **Mutable default arg** — `def f(x=[])` berbagi list yang sama antar panggilan. Selalu pakai `None`.
3. **Composition > inheritance** — untuk perilaku yang bervariasi, injeksikan objek, jangan warisi.
4. **Duck typing** — "if it walks like a duck..."; interface implisit lebih Pythonic dari hierarki dalam.
5. **`@property`** — method yang diakses seperti atribut; berguna untuk nilai turunan (mis. `accuracy` dari confusion matrix).

## Formula

- Method resolution order (MRO): C3 linearization — `ClassName.__mro__` menunjukkan urutan lookup.

## Contoh Kasus

Membangun `MetricsTracker` class untuk eksperimen: simpan history loss/accuracy per epoch, `__repr__` menampilkan ringkasan, `__call__` untuk update. Atau custom sklearn-compatible transformer class dengan `fit/transform/predict` agar bisa masuk `Pipeline`.

## Implementasi

```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0

    def __call__(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

# closure example
def make_scaler(factor=1.0):
    def scale(x):
        return x * factor
    return scale
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "What does this print? — code: def f(x=[]):\n    x.append(1)\n    return x\nprint(f(), f())",
    "options": ["[1] [1]", "[1] [1, 1]", "[1, 1] [1, 1]", "Error"],
    "answer": 1,
    "explanation": "The default list is created once at function definition; both calls share it.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "coding",
    "question": "Implement class `Stack` with methods `push(x)`, `pop()`, `peek()`, `is_empty()`, `__len__`, and `__repr__` showing contents.",
    "starter_code": "class Stack:\n    def __init__(self):\n        self._items = []",
    "function": "Stack",
    "tests": [
      {"call": "s=Stack(); s.push(1); s.push(2); s.peek()", "expected": 2},
      {"call": "s=Stack(); s.push(1); s.pop()", "expected": 1},
      {"call": "len(Stack())", "expected": 0},
      {"call": "Stack().is_empty()", "expected": true}
    ],
    "hints": ["peek returns last item without removing", "pop removes and returns"],
    "explanation": "LIFO stack using a list.",
    "points": 2
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `compose(f, g)` returning a function h such that h(x) = f(g(x)).",
    "starter_code": "def compose(f, g):\n    pass",
    "function": "compose",
    "tests": [
      {"call": "compose(lambda x: x+1, lambda x: x*2)(3)", "expected": 7},
      {"call": "compose(str, abs)(-5)", "expected": "5"}
    ],
    "hints": ["Return a lambda or inner function", "Order matters: g first"],
    "explanation": "Function composition — fundamental for pipelines.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "open",
    "question": "Design a class `Polynomial` (degree-n with coefficients list) supporting `__call__(x)`, `degree` property, and `__add__`. Sketch the design decisions (attributes, validation) before coding.",
    "rubric": [
      "Coefficients stored in consistent order (documented)",
      "Input validation for coefficient types/lengths",
      "degree property handles trailing zeros correctly",
      "__add__ handles different degrees"
    ],
    "points": 2
  }
]
```

## Referensi

- Python docs — data model & classes
- "Fluent Python" ch. 8-9 (objects & protocols)
- sklearn developer docs — how to write custom estimators
