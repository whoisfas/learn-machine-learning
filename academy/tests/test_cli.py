"""Smoke tests for the CLI."""
import json

import pytest

from academy.cli import main


def test_list_modules(capsys):
    main(["list"])
    out = capsys.readouterr().out
    assert "PHASE 1" in out and "modules" in out


def test_list_phase_filter(capsys):
    main(["list", "--phase", "4"])
    out = capsys.readouterr().out
    assert "PHASE 4" in out
    assert "PHASE 1" not in out.split("PHASE 4")[0]


def test_show_module(capsys):
    main(["show", "p1-python-fundamentals"])
    out = capsys.readouterr().out
    assert "Python Fundamentals" in out
    assert "Objectives" in out


def test_show_missing_module(capsys):
    with pytest.raises(SystemExit):
        main(["show", "does-not-exist"])


def test_dataset_list(capsys):
    main(["dataset", "list"])
    out = capsys.readouterr().out
    assert "iris" in out and "nasa_exoplanet_archive" in out


def test_dataset_generate(capsys):
    main(["dataset", "generate", "synthetic_gp",
          "--kwargs", json.dumps({"n": 20, "seed": 1})])
    out = capsys.readouterr().out
    assert "shape" in out


def test_project_list_and_show(capsys):
    main(["project", "list"])
    out = capsys.readouterr().out
    assert "House Price" in out
    main(["project", "show", "p01-house-price"])
    out = capsys.readouterr().out
    assert "Brief" in out and "Acceptance criteria" in out


def test_search(capsys):
    main(["search", "genetic programming"])
    out = capsys.readouterr().out
    assert "genetic" in out.lower()


def test_experiment_log_and_list(capsys):
    main(["experiment", "log", "p01", "test-run",
          "--params", '{"lr":0.01}', "--metrics", '{"acc":0.9}'])
    assert "logged" in capsys.readouterr().out.lower()
    main(["experiment", "list", "p01"])
    out = capsys.readouterr().out
    assert "test-run" in out


def test_progress_command(capsys):
    main(["progress"])
    out = capsys.readouterr().out
    assert "Phase 1" in out and "completed" in out


def test_submit_coding_exercise(tmp_path, capsys):
    sol = tmp_path / "sol.py"
    sol.write_text("import re\n\ndef word_count(text):\n"
                   "    words = re.findall(r'[a-zA-Z]+', text.lower())\n"
                   "    d = {}\n    for w in words:\n"
                   "        d[w] = d.get(w, 0) + 1\n    return d\n")
    main(["submit", "p1-python-fundamentals", "ex2", "--file", str(sol)])
    out = capsys.readouterr().out
    assert "PASS" in out
