"""Progress tracking helpers on top of SQLAlchemy models."""
from __future__ import annotations

from datetime import datetime, timezone

from .db import session, ModuleProgress, ExerciseAttempt, Note, Experiment


def mark_started(module_id: str, phase: int):
    s = session()
    mp = s.get(ModuleProgress, module_id)
    if mp is None:
        mp = ModuleProgress(module_id=module_id, phase=phase, status="in_progress")
        s.add(mp)
    elif mp.status == "not_started":
        mp.status = "in_progress"
    s.commit()
    s.close()


def mark_completed(module_id: str, phase: int, score: float = 0.0):
    s = session()
    mp = s.get(ModuleProgress, module_id)
    if mp is None:
        mp = ModuleProgress(module_id=module_id, phase=phase)
        s.add(mp)
    mp.status = "completed"
    mp.completed_at = datetime.now(timezone.utc)
    mp.best_score = max(mp.best_score or 0.0, score)
    s.commit()
    s.close()


def record_attempt(module_id: str, ex_id: str, ex_type: str, answer: str,
                   score: float, max_score: float, passed: bool, detail: str = ""):
    s = session()
    s.add(ExerciseAttempt(
        module_id=module_id, exercise_id=ex_id, exercise_type=ex_type,
        answer=(answer or "")[:4000], score=score, max_score=max_score,
        passed=1 if passed else 0, detail=(detail or "")[:2000],
    ))
    s.commit()
    s.close()


def get_progress(module_id: str):
    s = session()
    mp = s.get(ModuleProgress, module_id)
    s.close()
    return mp


def all_progress():
    s = session()
    rows = s.query(ModuleProgress).all()
    s.close()
    return rows


def module_score(module_id: str):
    """Average score across attempts for a module."""
    s = session()
    rows = s.query(ExerciseAttempt).filter_by(module_id=module_id).all()
    s.close()
    if not rows:
        return 0.0
    return sum(r.score for r in rows) / sum(r.max_score for r in rows if r.max_score) * 100 \
        if any(r.max_score for r in rows) else 0.0


def add_note(module_id: str, text: str):
    s = session()
    s.add(Note(module_id=module_id, text=text))
    s.commit()
    s.close()


def get_notes(module_id: str | None = None):
    s = session()
    q = s.query(Note)
    if module_id:
        q = q.filter_by(module_id=module_id)
    rows = q.order_by(Note.created_at).all()
    s.close()
    return rows


def log_experiment(project_id: str, name: str, params: dict, metrics: dict,
                   artifacts: list | None = None, notes: str = ""):
    s = session()
    s.add(Experiment(project_id=project_id, name=name, params=params,
                     metrics=metrics, artifacts=artifacts or [], notes=notes))
    s.commit()
    s.close()


def list_experiments(project_id: str | None = None):
    s = session()
    q = s.query(Experiment)
    if project_id:
        q = q.filter_by(project_id=project_id)
    rows = q.order_by(Experiment.created_at.desc()).all()
    s.close()
    return rows
