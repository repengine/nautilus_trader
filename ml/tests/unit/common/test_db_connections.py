from __future__ import annotations

from typing import Iterator

import pytest

from ml.common.db_connections import ConnectionRole
from ml.common.db_connections import collect_postgres_candidates


@pytest.fixture(autouse=True)
def _clear_db_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    Ensure connection resolver tests start with a clean environment.
    """

    variables = (
        "NAUTILUS_DB",
        "DATABASE_URL",
        "ML_DB_CONNECTION",
        "NAUTILUS_DB_CONNECTION",
        "DB_CONNECTION",
        "NAUTILUS_REGISTRY_DB_URL",
        "ML_DB_HOST",
        "POSTGRES_HOST",
        "NAUTILUS_DB_HOST",
        "ML_DB_PORT",
        "POSTGRES_HOST_PORT",
        "NAUTILUS_DB_PORT",
    )
    for name in variables:
        monkeypatch.delenv(name, raising=False)
    yield


def test_collect_candidates_prefers_compose_port() -> None:
    candidates = collect_postgres_candidates(ConnectionRole.PRIMARY)
    assert candidates.urls
    first = candidates.urls[0]
    assert first.startswith("postgresql://postgres:postgres@localhost:5433/nautilus")
    assert any(":5432/" in url for url in candidates.urls)


def test_collect_candidates_respects_explicit_override() -> None:
    override = "postgresql://user:secret@db.example.com:15432/custom"
    candidates = collect_postgres_candidates(ConnectionRole.PRIMARY, explicit=override)
    assert candidates.urls[0] == override
    assert override in candidates.urls
