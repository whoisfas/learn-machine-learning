"""SQLAlchemy models and session management for progress tracking."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine

from .config import DB_PATH

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModuleProgress(Base):
    __tablename__ = "module_progress"

    module_id = Column(String(64), primary_key=True)
    phase = Column(Integer, nullable=False)
    status = Column(String(16), default="not_started")  # not_started|in_progress|completed
    started_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)
    best_score = Column(Float, default=0.0)


class ExerciseAttempt(Base):
    __tablename__ = "exercise_attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_id = Column(String(64), nullable=False, index=True)
    exercise_id = Column(String(64), nullable=False)
    exercise_type = Column(String(16), nullable=False)
    answer = Column(Text, nullable=True)
    score = Column(Float, default=0.0)
    max_score = Column(Float, default=0.0)
    passed = Column(Integer, default=0)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_id = Column(String(64), nullable=False, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=_utcnow)


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(64), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    params = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    artifacts = Column(JSON, default=list)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=_utcnow)


class DatasetFetch(Base):
    """Track which external datasets have been downloaded."""

    __tablename__ = "dataset_fetches"

    name = Column(String(64), primary_key=True)
    path = Column(String(512), nullable=False)
    fetched_at = Column(DateTime, default=_utcnow)


def get_engine(db_path=None):
    return create_engine(f"sqlite:///{db_path or DB_PATH}", future=True)


_engine = None
_Session = None


def init_db(db_path=None):
    global _engine, _Session
    _engine = get_engine(db_path)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, future=True)
    return _engine


def session():
    if _Session is None:
        init_db()
    return _Session()
