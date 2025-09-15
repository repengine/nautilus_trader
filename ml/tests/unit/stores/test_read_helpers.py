"""
ReadQueryMixin tests for table qualification behavior.

Ensures that for sqlite dialect, _qualified_table returns the base name.
"""

from __future__ import annotations

from types import SimpleNamespace

from ml.stores.mixins import ReadQueryMixin


class _Helper(ReadQueryMixin):
    pass


def test_qualified_table_sqlite_returns_unqualified() -> None:
    h = _Helper()
    # Fake engine with sqlite dialect
    h.engine = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))  # type: ignore[attr-defined]
    assert h._qualified_table("ml_model_predictions") == "ml_model_predictions"


def test_qualified_table_postgres_returns_public_qualified() -> None:
    h = _Helper()
    h.engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))  # type: ignore[attr-defined]
    assert h._qualified_table("ml_model_predictions") == "public.ml_model_predictions"
