"""Load learning modules from markdown files.

Module file format:

---
{
  "id": "p1-python-fundamentals",
  "title": "Python Fundamentals",
  "phase": 1,
  "difficulty": "beginner",
  "estimated_hours": 8,
  "objectives": ["...", "..."],
  "concepts": ["..."],
  "formulas": [{"name": "...", "latex": "...", "note": "..."}],
  "references": ["..."]
}
---

## Penjelasan
...markdown...

## Contoh Kasus
...

## Implementasi
```python
...
```

## Latihan
```json
[
  {"id": "ex1", "type": "concept|coding|math|open", "question": "...", ...}
]
```
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CONTENT_DIR

SECTION_NAMES = [
    "Penjelasan",
    "Konsep Penting",
    "Formula",
    "Contoh Kasus",
    "Implementasi",
    "Latihan",
    "Referensi",
]


@dataclass
class Module:
    id: str
    title: str
    phase: int
    difficulty: str
    estimated_hours: float
    objectives: list
    concepts: list
    formulas: list
    references: list
    sections: dict = field(default_factory=dict)  # name -> markdown text
    exercises: list = field(default_factory=list)
    path: Path = None

    @property
    def exercises_file(self):
        return None


def _parse_frontmatter(text: str):
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = json.loads(text[3:end])
    body = text[end + 4 :]
    return meta, body


def _parse_sections(body: str) -> dict:
    sections = {}
    # split on "## " headings
    parts = re.split(r"^##\s+(.+?)\s*$", body, flags=re.M)
    # parts: [prelude, name1, content1, name2, content2, ...]
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].strip()] = parts[i + 1].strip()
    return sections


def _parse_exercises(sections: dict) -> list:
    text = sections.get("Latihan", "")
    m = re.search(r"```json\s*\n(.*?)```", text, flags=re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid exercise JSON: {e}")


def load_module(path: Path) -> Module:
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    sections = _parse_sections(body)
    exercises = _parse_exercises(sections)
    return Module(
        id=meta["id"],
        title=meta["title"],
        phase=meta["phase"],
        difficulty=meta.get("difficulty", "beginner"),
        estimated_hours=meta.get("estimated_hours", 1.0),
        objectives=meta.get("objectives", []),
        concepts=meta.get("concepts", []),
        formulas=meta.get("formulas", []),
        references=meta.get("references", []),
        sections=sections,
        exercises=exercises,
        path=path,
    )


def load_all_modules(content_dir: Path = CONTENT_DIR) -> list[Module]:
    modules = []
    for path in sorted(content_dir.glob("phase*/**/*.md")):
        try:
            modules.append(load_module(path))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to load {path}: {e}")
    return modules


def get_module(module_id: str, modules: list[Module] | None = None) -> Module:
    modules = modules if modules is not None else load_all_modules()
    for m in modules:
        if m.id == module_id:
            return m
    raise KeyError(f"Module not found: {module_id}")
