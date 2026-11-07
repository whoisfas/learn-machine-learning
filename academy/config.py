"""Central configuration for the ML Learning & Research Academy."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONTENT_DIR = ROOT / "academy" / "content"
DATASETS_DIR = ROOT / "data"
BUNDLED_DIR = DATASETS_DIR / "bundled"
DOWNLOADED_DIR = DATASETS_DIR / "downloaded"
PROJECTS_DIR = ROOT / "academy" / "projects"
DB_PATH = DATASETS_DIR / "academy.db"

for d in (DATASETS_DIR, BUNDLED_DIR, DOWNLOADED_DIR):
    d.mkdir(parents=True, exist_ok=True)
