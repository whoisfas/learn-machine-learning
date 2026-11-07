"""Tests for progress tracking (SQLite)."""
from academy import db, progress


def test_init_db_creates_tables(tmp_path):
    engine = db.init_db(tmp_path / "fresh.db")
    tables = set(db.Base.metadata.tables)
    assert engine is not None
    assert {"module_progress", "exercise_attempts", "notes",
            "experiments", "dataset_fetches"} <= tables


def test_module_lifecycle():
    progress.mark_started("p1-test", 1)
    mp = progress.get_progress("p1-test")
    assert mp.status == "in_progress"

    progress.record_attempt("p1-test", "ex1", "concept", "0", 1.0, 1.0, True)
    progress.record_attempt("p1-test", "ex2", "concept", "1", 0.0, 1.0, False)
    score = progress.module_score("p1-test")
    assert abs(score - 50.0) < 0.01

    progress.mark_completed("p1-test", 1, score)
    mp = progress.get_progress("p1-test")
    assert mp.status == "completed"
    assert mp.completed_at is not None
    assert mp.best_score == 50.0


def test_notes():
    progress.add_note("p1-test", "first note")
    progress.add_note("p1-test", "second note")
    notes = progress.get_notes("p1-test")
    assert len(notes) == 2
    assert notes[0].text == "first note"
    assert len(progress.get_notes()) >= 2


def test_experiments():
    progress.log_experiment("proj-x", "baseline", {"lr": 0.01},
                            {"acc": 0.9}, notes="first run")
    progress.log_experiment("proj-x", "tuned", {"lr": 0.001},
                            {"acc": 0.93})
    exps = progress.list_experiments("proj-x")
    assert len(exps) == 2
    assert exps[0].name == "tuned"  # newest first
    assert exps[0].metrics["acc"] == 0.93


def test_all_progress_listing():
    progress.mark_started("p2-test", 2)
    ids = {r.module_id for r in progress.all_progress()}
    assert {"p1-test", "p2-test"} <= ids
