"""Tests for the markdown content loader."""
from academy.content.loader import load_all_modules, get_module


def test_module_count_meets_target():
    mods = load_all_modules()
    assert len(mods) >= 60, f"only {len(mods)} modules loaded"


def test_all_six_phases_present():
    mods = load_all_modules()
    phases = {m.phase for m in mods}
    assert phases == {1, 2, 3, 4, 5, 6}


def test_every_module_has_required_fields():
    for m in load_all_modules():
        assert m.id and m.title
        assert m.objectives, f"{m.id} missing objectives"
        assert m.concepts, f"{m.id} missing concepts"
        assert m.references, f"{m.id} missing references"
        assert m.estimated_hours > 0
        assert "Penjelasan" in m.sections, f"{m.id} missing Penjelasan"
        assert m.exercises, f"{m.id} has no exercises"


def test_exercises_have_required_fields():
    for m in load_all_modules():
        for ex in m.exercises:
            assert ex.get("id") and ex.get("type") and ex.get("question")
            assert ex.get("points", 1) > 0
            if ex["type"] == "concept":
                assert "options" in ex and "answer" in ex
            elif ex["type"] == "coding":
                assert "tests" in ex and "function" in ex
            elif ex["type"] == "math":
                assert "answer" in ex
            elif ex["type"] == "open":
                assert "rubric" in ex and len(ex["rubric"]) >= 2


def test_get_module_lookup():
    mods = load_all_modules()
    m = get_module(mods[0].id, mods)
    assert m.id == mods[0].id


def test_special_phase4_modules_exist():
    ids = {m.id for m in load_all_modules()}
    for expected in ["p4-genetic-programming", "p4-symbolic-regression",
                     "p4-genetic-algorithms"]:
        assert expected in ids
