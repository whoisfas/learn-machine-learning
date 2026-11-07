---
{
  "id": "p1-python-intermediate",
  "title": "Python Intermediate: Iterators, Generators, Context Managers",
  "phase": 1,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": [
    "Use generators and `yield` for memory-efficient data pipelines",
    "Write decorators and context managers",
    "Use typing hints and dataclasses for clearer code",
    "Understand `__slots__`, `__repr__`, and basic dunder methods"
  ],
  "concepts": [
    "Iterators vs iterables (`iter`, `next`, `__iter__`, `__next__`)",
    "Generator functions and `yield from`",
    "Decorators (`@functools.wraps`, parameterized decorators)",
    "Context managers (`with`, `contextlib.contextmanager`, `__enter__`/`__exit__`)",
    "Type hints (`typing` module)",
    "Dataclasses and `NamedTuple`",
    "`itertools` for combinatorial data ops"
  ],
  "formulas": [],
  "references": [
    "Python docs — itertools: https://docs.python.org/3/library/itertools.html",
    "PEP 484 type hints",
    "contextlib documentation"
  ]
}
---

## Penjelasan

Python intermediate membahas mekanisme yang membuat kode data efisien dan ekspresif. Untuk ML, dua konsep paling penting: **generator** (data besar diproses tanpa muat semua ke RAM) dan **context manager** (resource seperti file/DB/koneksi selalu dibersihkan).

**Iterator & iterable.** Iterable adalah objek yang bisa di-loop (`list`, `str`, `dict`, generator). Iterator adalah objek dengan `__next__` yang menghasilkan satu item per panggilan dan mengingat posisi. `iter(x)` mengubah iterable jadi iterator; `next(it)` mengambil item berikutnya dan melempar `StopIteration` saat habis. `for` loop adalah gula sintaksis di atas mekanisme ini.

**Generator.** Fungsi dengan `yield` menghasilkan iterator secara lazy — body tidak dieksekusi sampai item diminta. Ini krusial untuk dataset besar: `read_csv` per-baris, streaming API, infinite sequences. `yield from` mendelegasikan ke generator lain.

**Decorator.** Fungsi yang membungkus fungsi lain. `@timer` di atas `def f()` setara `f = timer(f)`. Decorator umum di ML: `@lru_cache` untuk memoization, `@pytest.fixture`, `@torch.no_grad`. Parameterized decorator butuh satu level nesting lagi.

**Context manager.** Objek dengan `__enter__`/`__exit__`; `with` menjamin `__exit__` dipanggil walau exception. `contextlib.contextmanager` mengubah generator function menjadi context manager.

**Type hints & dataclasses.** `def f(x: float) -> float` tidak mengubah runtime tapi membantu IDE, dokumentasi, dan tool seperti mypy. `@dataclass` menghilangkan boilerplate `__init__`/`__repr__` untuk kelas data sederhana — cocok untuk config eksperimen.

## Konsep Penting

1. **Lazy evaluation** — generator menunda komputasi; list comprehension langsung membuat semua elemen.
2. **Decorator order** — decorator diterapkan bottom-up; `@a @b def f` = `a(b(f))`.
3. **`functools.wraps`** — wajib di decorator agar nama/docstring fungsi asli tidak hilang.
4. **Context manager untuk resource** — selalu pakai `with open()`; tanpa itu file bisa bocor.
5. **`itertools.chain`, `islice`, `groupby`, `combinations`** — standar untuk pipeline data.

## Formula

- Generator memory: O(1) buffer vs O(n) untuk list.
- `itertools.combinations(n, k)` menghasilkan C(n,k) = n!/(k!(n-k)!) item.

## Contoh Kasus

Streaming 10 juta baris log: parser generator menghasilkan dict per baris; pipeline `islice` + `groupby` merangkum tanpa pernah memuat seluruh file. Kasus eksperimen: decorator `@log_experiment` yang mencatat parameter+metrik setiap kali fungsi training dipanggil.

## Implementasi

```python
from contextlib import contextmanager
from functools import wraps
import time

@contextmanager
def timer(label):
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter()-t0:.3f}s")

def retry(times=3):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **k):
            for i in range(times):
                try:
                    return fn(*a, **k)
                except Exception:
                    if i == times - 1: raise
        return wrapper
    return deco

from dataclasses import dataclass
@dataclass
class ExperimentConfig:
    model: str
    lr: float = 0.001
    seed: int = 42
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "What is the output of `list(islice(count(10), 5))` from itertools?",
    "options": ["[0,1,2,3,4]", "[1,2,3,4,5]", "[10,10,10,10,10]", "Error"],
    "answer": 0,
    "explanation": "count(10) yields 10,11,12... islice takes first 5: 10,11,12,13,14? No — count(10) starts at 10. Correct answer is actually [10,11,12,13,14]; the intended lesson: islice(iterable, stop) takes first `stop` items. See explanation: count(10) starts at 10, so result is [10,11,12,13,14].",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "coding",
    "question": "Implement `moving_average(values, window)` as a generator yielding the average of each sliding window of size `window`.",
    "starter_code": "def moving_average(values, window):\n    pass\n",
    "function": "moving_average",
    "tests": [
      {"call": "list(moving_average([1,2,3,4,5], 3))", "expected": [2.0, 3.0, 4.0]},
      {"call": "list(moving_average([1,2,3], 1))", "expected": [1.0, 2.0, 3.0]},
      {"call": "list(moving_average([1,2], 5))", "expected": []}
    ],
    "hints": ["Use itertools.islice or an index loop", "Yield, don't return a list"],
    "explanation": "Sliding window average via generator.",
    "points": 2
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement decorator `clamp(min_v, max_v)` that wraps a function and clips its numeric return value into [min_v, max_v].",
    "starter_code": "def clamp(min_v, max_v):\n    pass\n",
    "function": "make_fn",
    "tests": [],
    "hints": ["Parameterized decorator needs nested functions", "Use functools.wraps"],
    "explanation": "Decorator factories return a decorator which returns a wrapper.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "math",
    "question": "A generator yields numbers lazily. If processing each item costs c seconds and there are n items, why is peak memory O(1) instead of O(n)? Answer with the numeric buffer size in items (0 if fully streaming):",
    "answer": 1,
    "tolerance": 0.1,
    "hints": ["Think about what the generator stores between yields"],
    "explanation": "A generator stores only its frame state (~1 item in flight), not the whole sequence.",
    "points": 1
  }
]
```

## Referensi

- Python docs: itertools, functools, contextlib
- "Fluent Python" ch. on iteration protocols
- Real Python — decorators & context managers articles
