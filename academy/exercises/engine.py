"""Exercise engine: loads exercises from modules and auto-grades them.

Exercise schema (JSON inside module markdown ```json block):

Concept (multiple choice / short answer):
  {"id","type":"concept","question","options":[...],"answer":0|"text",
   "explanation","points":1}

Coding:
  {"id","type":"coding","question","starter_code","function":"fn_name",
   "signature":"fn_name(a, b)","tests":[{"call":"fn_name(2,3)","expected":5}],
   "hints":[...],"explanation","points":1}

Math:
  {"id","type":"math","question","answer":0.5,"tolerance":1e-6,
   "hints":[...],"explanation","points":1}

Open-ended (stored + rubric, manual grading):
  {"id","type":"open","question","rubric":[...],"points":1}
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

def _import_pandas():
    try:
        import pandas as pd
        return pd
    except Exception:  # noqa: BLE001
        return None


SAFE_IMPORTS = {
    "np": np,
    "numpy": np,
    "math": math,
    "json": json,
    "re": re,
    "pd": _import_pandas(),
    "pandas": _import_pandas(),
}


@dataclass
class GradeResult:
    exercise_id: str
    exercise_type: str
    score: float
    max_score: float
    passed: bool
    detail: str = ""

    def to_dict(self):
        return {
            "exercise_id": self.exercise_id,
            "exercise_type": self.exercise_type,
            "score": self.score,
            "max_score": self.max_score,
            "passed": self.passed,
            "detail": self.detail,
        }


# ---------------- graders ----------------

def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def grade_concept(ex: dict, answer: str) -> GradeResult:
    pts = ex.get("points", 1)
    correct = ex.get("answer")
    if isinstance(correct, int) and "options" in ex:
        try:
            idx = int(str(answer).strip())
        except ValueError:
            idx = -1
        ok = idx == correct
        detail = ex.get("explanation", "")
    else:
        ok = _norm(answer) == _norm(correct)
        detail = ex.get("explanation", "")
    return GradeResult(ex["id"], "concept", pts if ok else 0.0, pts, ok, detail)


def grade_math(ex: dict, answer: str) -> GradeResult:
    pts = ex.get("points", 1)
    tol = ex.get("tolerance", 1e-6)
    try:
        val = float(str(answer).strip().rstrip("%"))
        if str(answer).strip().endswith("%"):
            val /= 100.0
    except ValueError:
        return GradeResult(ex["id"], "math", 0.0, pts, False, "Answer must be a number")
    expected = float(ex["answer"])
    ok = math.isclose(val, expected, rel_tol=tol, abs_tol=tol)
    return GradeResult(ex["id"], "math", pts if ok else 0.0, pts, ok,
                       ex.get("explanation", ""))


def _exec_user_code(code: str) -> dict:
    """Execute user code in a namespace with safe-ish imports."""
    ns = dict(SAFE_IMPORTS)
    ns["__name__"] = "__submission__"
    exec(compile(code, "<submission>", "exec"), ns)  # noqa: S102
    return ns


def grade_coding(ex: dict, code: str) -> GradeResult:
    pts = ex.get("points", 1)
    fname = ex.get("function")
    tests = ex.get("tests", [])
    try:
        ns = _exec_user_code(code)
    except Exception as e:  # noqa: BLE001
        return GradeResult(ex["id"], "coding", 0.0, pts, False,
                           f"Code failed to execute: {type(e).__name__}: {e}")
    if fname and fname not in ns:
        return GradeResult(ex["id"], "coding", 0.0, pts, False,
                           f"Function `{fname}` not found in submission")

    passed = 0
    errors = []
    for t in tests:
        call = t["call"]
        try:
            result = eval(call, {**ns, fname: ns[fname]})  # noqa: S307
            expected = t["expected"]
            if isinstance(expected, (int, float)) and isinstance(result, (int, float, np.number)):
                ok = math.isclose(float(result), float(expected),
                                  rel_tol=t.get("tolerance", 1e-6),
                                  abs_tol=t.get("tolerance", 1e-6))
            else:
                ok = result == expected or _norm(result) == _norm(expected)
            if ok:
                passed += 1
            else:
                errors.append(f"{call} -> {result!r} (expected {expected!r})")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{call} raised {type(e).__name__}: {e}")

    frac = passed / len(tests) if tests else 0.0
    score = pts * frac
    detail = ex.get("explanation", "")
    if errors:
        detail = (detail + " | " if detail else "") + "; ".join(errors[:5])
    return GradeResult(ex["id"], "coding", round(score, 4), pts,
                       score >= pts * 0.999, detail)


def grade_open(ex: dict, answer: str) -> GradeResult:
    """Open exercises are stored for manual review with a rubric."""
    pts = ex.get("points", 1)
    return GradeResult(
        ex["id"], "open", 0.0, pts, False,
        "OPEN_ENDED: stored for manual grading. Rubric: "
        + "; ".join(ex.get("rubric", [])),
    )


GRADERS = {
    "concept": grade_concept,
    "math": grade_math,
    "coding": grade_coding,
    "open": grade_open,
}


def grade_exercise(ex: dict, answer: str) -> GradeResult:
    grader = GRADERS.get(ex.get("type"))
    if grader is None:
        return GradeResult(ex.get("id", "?"), "unknown", 0.0, 0.0, False,
                           f"Unknown exercise type: {ex.get('type')}")
    return grader(ex, answer)


def get_exercise(module, exercise_id: str) -> dict:
    for ex in module.exercises:
        if ex["id"] == exercise_id:
            return ex
    raise KeyError(f"Exercise {exercise_id} not found in module {module.id}")
