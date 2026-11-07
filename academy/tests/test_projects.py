"""Tests for project specs."""
from academy.projects.specs import list_projects, get_project, PROJECTS


def test_project_count():
    assert len(list_projects()) >= 15


def test_project_stages():
    stages = {p.stage for p in list_projects()}
    assert {"beginner", "intermediate", "advanced"} <= stages


def test_required_advanced_projects():
    ids = {p.id for p in list_projects()}
    for frag in ["genetic", "symbolic", "automated"]:
        assert any(frag in i for i in ids), f"missing project with '{frag}'"


def test_project_fields_complete():
    for p in list_projects():
        assert p.title and p.brief and p.dataset
        assert len(p.objectives) >= 2
        assert len(p.task_breakdown) >= 3
        assert p.starter_code.strip()
        assert len(p.acceptance_criteria) >= 3
        assert p.metrics and p.deliverables
        assert p.research_extension


def test_get_project():
    p = get_project(PROJECTS[0].id)
    assert p.id == PROJECTS[0].id


def test_get_project_missing():
    import pytest
    with pytest.raises(KeyError):
        get_project("nonexistent-project")
