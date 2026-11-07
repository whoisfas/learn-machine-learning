"""Tests for the exercise grading engine."""
import numpy as np

from academy.exercises.engine import (
    grade_concept, grade_math, grade_coding, grade_open, grade_exercise,
)
from academy.content.loader import load_all_modules, get_module


def test_grade_concept_correct():
    ex = {"id": "e1", "type": "concept", "question": "q",
          "options": ["a", "b"], "answer": 1, "points": 1}
    r = grade_concept(ex, "1")
    assert r.passed and r.score == 1.0


def test_grade_concept_wrong():
    ex = {"id": "e1", "type": "concept", "question": "q",
          "options": ["a", "b"], "answer": 1, "points": 1}
    assert not grade_concept(ex, "0").passed


def test_grade_concept_text_answer():
    ex = {"id": "e1", "type": "concept", "question": "q",
          "answer": "Gradient Descent", "points": 2}
    assert grade_concept(ex, "gradient   descent").passed


def test_grade_math_with_tolerance():
    ex = {"id": "e1", "type": "math", "question": "q",
          "answer": 0.5, "tolerance": 0.01, "points": 1}
    assert grade_math(ex, "0.505").passed
    assert not grade_math(ex, "0.6").passed


def test_grade_math_percent():
    ex = {"id": "e1", "type": "math", "question": "q",
          "answer": 0.25, "tolerance": 1e-6, "points": 1}
    assert grade_math(ex, "25%").passed


def test_grade_coding_pass():
    ex = {"id": "e1", "type": "coding", "question": "q",
          "function": "add", "tests": [
              {"call": "add(2, 3)", "expected": 5},
              {"call": "add(-1, 1)", "expected": 0},
          ], "points": 2}
    code = "def add(a, b):\n    return a + b\n"
    r = grade_coding(ex, code)
    assert r.passed and r.score == 2.0


def test_grade_coding_partial():
    ex = {"id": "e1", "type": "coding", "question": "q",
          "function": "add", "tests": [
              {"call": "add(2, 3)", "expected": 5},
              {"call": "add(-1, 1)", "expected": 0},
          ], "points": 2}
    code = "def add(a, b):\n    return a + b if a > 0 else a - b\n"
    r = grade_coding(ex, code)
    assert not r.passed
    assert 0 < r.score < 2.0  # one test passes, one fails -> partial


def test_grade_coding_syntax_error():
    ex = {"id": "e1", "type": "coding", "question": "q",
          "function": "f", "tests": [{"call": "f()", "expected": 1}], "points": 1}
    r = grade_coding(ex, "def f(:\n")
    assert not r.passed and "failed to execute" in r.detail


def test_grade_coding_missing_function():
    ex = {"id": "e1", "type": "coding", "question": "q",
          "function": "f", "tests": [{"call": "f()", "expected": 1}], "points": 1}
    r = grade_coding(ex, "x = 1\n")
    assert not r.passed and "not found" in r.detail


def test_grade_coding_with_numpy_pandas():
    ex = {"id": "e1", "type": "coding", "question": "q",
          "function": "total", "tests": [
              {"call": "total(np.array([1,2,3]))", "expected": 6},
          ], "points": 1}
    code = "import numpy as np\n\ndef total(a):\n    return int(np.sum(a))\n"
    assert grade_coding(ex, code).passed


def test_grade_open_returns_rubric_not_score():
    ex = {"id": "e1", "type": "open", "question": "q",
          "rubric": ["covers X", "covers Y"], "points": 2}
    r = grade_open(ex, "my answer")
    assert not r.passed and r.score == 0.0
    assert "OPEN_ENDED" in r.detail and "covers X" in r.detail


def test_grade_exercise_dispatch():
    ex = {"id": "e1", "type": "math", "question": "q",
          "answer": 2, "points": 1}
    assert grade_exercise(ex, "2").passed


# ---- integration: real exercises from real modules ----

def _get(module_id, ex_id):
    m = get_module(module_id, load_all_modules())
    for ex in m.exercises:
        if ex["id"] == ex_id:
            return ex
    raise KeyError(ex_id)


def test_real_coding_exercise_word_count():
    ex = _get("p1-python-fundamentals", "ex2")
    good = ("import re\n\ndef word_count(text):\n"
            "    words = re.findall(r'[a-zA-Z]+', text.lower())\n"
            "    d = {}\n    for w in words:\n        d[w] = d.get(w, 0) + 1\n    return d\n")
    assert grade_exercise(ex, good).passed
    assert not grade_exercise(ex, "def word_count(t):\n    return {}").passed


def test_real_math_exercise():
    # find any math exercise and check grading works
    for m in load_all_modules():
        for ex in m.exercises:
            if ex["type"] == "math":
                r = grade_exercise(ex, str(ex["answer"]))
                assert r.passed, f"{m.id}/{ex['id']} math self-grade failed"
                return
    pytest.skip("no math exercises found")


def test_coding_exercises_have_valid_test_calls():
    """Every coding exercise's starter_code must at least parse."""
    import ast
    n = n_coding = 0
    for m in load_all_modules():
        for ex in m.exercises:
            if ex["type"] == "coding":
                n_coding += 1
                if ex.get("starter_code"):
                    ast.parse(ex["starter_code"])
                    n += 1
    assert n_coding > 50
    assert n > 30
