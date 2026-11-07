---
{
  "id": "p1-python-fundamentals",
  "title": "Python Fundamentals for Machine Learning",
  "phase": 1,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": [
    "Write correct Python programs using variables, types, control flow, and functions",
    "Use lists, dicts, tuples, sets, and comprehensions idiomatically",
    "Read and write files, handle exceptions, and use modules",
    "Prepare data structures that NumPy and Pandas can consume"
  ],
  "concepts": [
    "Dynamic typing and duck typing",
    "Mutability: lists/dicts vs tuples/strings",
    "List/dict/set comprehensions",
    "Functions, default args, *args/**kwargs, lambda",
    "File I/O and context managers",
    "Exception handling (try/except/else/finally)",
    "Modules, packages, pip, virtual environments"
  ],
  "formulas": [
    {"name": "Big-O intuition", "latex": "T(n) \\propto n^k", "note": "Nested loops over n items cost O(n^2); prefer comprehensions/builtins."}
  ],
  "references": [
    "Python official tutorial: https://docs.python.org/3/tutorial/",
    "Real Python tutorials: https://realpython.com/",
    "PEP 8 style guide: https://peps.python.org/pep-0008/"
  ]
}
---

## Penjelasan

Python adalah bahasa utama ekosistem Machine Learning. Berbeda dengan bahasa lain, Python menekankan keterbacaan: indentasi menentukan blok kode, bukan kurung kurawal. Untuk ML, Anda akan menulis banyak kode data: membaca file, membersihkan nilai, mengubah struktur data, dan memanggil library seperti NumPy dan Pandas.

**Tipe data dasar.** `int`, `float`, `str`, `bool`, `None`. Python menggunakan *dynamic typing*: tipe melekat pada objek, bukan variabel. `x = 5` lalu `x = "lima"` valid, tapi akan merusak kode downstream. Untuk ML, kebiasaan baik adalah konsisten: kolom numerik -> float/int, kategorikal -> str.

**Struktur data.**
- `list`: urutan mutable, indexing `[0]`, slicing `[1:4]`, method `.append()`, `.pop()`, `.sort()`.
- `tuple`: urutan immutable, cocok untuk pasangan (key, value) sederhana.
- `dict`: key-value, lookup O(1) rata-rata, `.get(key, default)` aman dari KeyError.
- `set`: himpunan unik, operasi irisan/gabung cepat.

**Struktur kontrol.** `if/elif/else`, `for` (iterasi langsung, bukan index), `while`, dan comprehensions: `[x*x for x in xs if x > 0]`. Hindari loop untuk transformasi besar — NumPy/Pandas jauh lebih cepat.

**Fungsi.** `def f(a, b=2): ...`, return value, docstring. `lambda` untuk fungsi satu baris. `*args` menangkap argumen posisi variabel; `**kwargs` menangkap keyword. Hindari mutable default argument (`def f(x=[])` adalah bug klasik — gunakan `None`).

**File & error.** `with open(...) as f` menjamin file tertutup. `try/except Exception as e` untuk penanganan error. Di ML, hampir semua error adalah `KeyError`, `ValueError`, `IndexError` — baca pesan errornya.

**Modul & environment.** `import numpy as np`, `from sklearn.model_selection import train_test_split`. Gunakan virtual environment per proyek (`python -m venv .venv`) agar versi library tidak bentrok.

## Konsep Penting

1. **Mutable vs immutable** — list/dict bisa diubah di tempat; int/str/tuple tidak. Ini menentukan kapan copy diperlukan (`a.copy()`, `list(a)`).
2. **Iterasi vs indexing** — `for row in df.itertuples()` bukan `for i in range(len(df))`.
3. **Comprehension scope** — `[x for x in xs]` tidak membocorkan `x` ke luar (Python 3).
4. **Truthy/falsy** — `0`, `""`, `[]`, `{}`, `None` bernilai False. Berguna untuk `if row:` tapi berbahaya jika data mengandung 0 yang valid.
5. **`is` vs `==`** — `is` membandingkan identitas objek; untuk angka/string gunakan `==`.

## Formula

- Kompleksitas: loop tunggal O(n), nested O(n²), dict lookup O(1) rata-rata.
- Memory list of ints ~ 28 byte/elemen di CPython 64-bit; NumPy array jauh lebih hemat (8 byte/float64).

## Contoh Kasus

Membersihkan data mentah: membaca CSV baris per baris, mengonversi string ke float dengan penanganan error, mengelompokkan nilai ke dalam dict statistik. Kasus nyata: parsing log file eksperimen dan merangkum metrik per run.

## Implementasi

```python
def mean_std(values):
    """Return (mean, population std) of a list of numbers."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5

# parsing with error handling
scores = []
for line in open("scores.txt"):
    try:
        scores.append(float(line.strip()))
    except ValueError:
        continue  # skip malformed lines
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "What is printed? — code: a = [1, 2, 3]\nb = a\nb.append(4)\nprint(len(a))",
    "options": ["3", "4", "Error", "1"],
    "answer": 1,
    "explanation": "b references the same list object; append mutates it, so a also has 4 elements.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "coding",
    "question": "Implement `word_count(text)` returning a dict mapping each lowercase word to its frequency. Words are split on whitespace; strip punctuation .,!?;:",
    "starter_code": "import re\n\ndef word_count(text):\n    pass",
    "function": "word_count",
    "tests": [
      {"call": "word_count('Hello, world! Hello.')", "expected": {"hello": 2, "world": 1}},
      {"call": "word_count('a b a c b a')", "expected": {"a": 3, "b": 2, "c": 1}},
      {"call": "word_count('')", "expected": {}}
    ],
    "hints": ["Use re.findall(r\"[a-zA-Z]+\", text.lower())", "Use dict.get or collections.Counter"],
    "explanation": "Count tokens after normalizing case and removing punctuation.",
    "points": 2
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `safe_divide(a, b)` returning a/b, or None if b == 0 or inputs are not numbers.",
    "starter_code": "def safe_divide(a, b):\n    pass",
    "function": "safe_divide",
    "tests": [
      {"call": "safe_divide(10, 2)", "expected": 5.0},
      {"call": "safe_divide(10, 0)", "expected": null},
      {"call": "safe_divide('a', 2)", "expected": null}
    ],
    "hints": ["try/except (ZeroDivisionError, TypeError)", "Return None on failure"],
    "explanation": "Graceful error handling instead of crashing.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "open",
    "question": "Explain in 3-5 sentences why Python's `for x in list` is preferred over `for i in range(len(list))` when you only need the elements. Relate it to readability and common bugs.",
    "rubric": [
      "Mentions readability/intent clarity",
      "Mentions avoiding index errors or off-by-one bugs",
      "Connects to iterating over dicts/sets where index doesn't exist"
    ],
    "points": 1
  }
]
```

## Referensi

- Official Python tutorial (docs.python.org/3/tutorial)
- Real Python — practical tutorials
- "Fluent Python" by Luciano Ramalho (optional reading)
