import pytest

from academy import db


@pytest.fixture(scope="session")
def session_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("db") / "test_academy.db"


@pytest.fixture(autouse=True)
def test_db(session_db_path):
    """Rebind the global session to a throwaway SQLite DB before each test."""
    db.init_db(session_db_path)
    yield session_db_path
