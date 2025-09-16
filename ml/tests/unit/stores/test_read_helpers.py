"""
ReadQueryMixin tests for table qualification behavior.

Ensures that for sqlite dialect, _qualified_table returns the base name.

"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from ml.stores.mixins import ReadQueryMixin


class _Helper(ReadQueryMixin):
    pass


def test_qualified_table_sqlite_returns_unqualified() -> None:
    h = _Helper()
    # Fake engine with sqlite dialect
    cast(Any, h).engine = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    assert h._qualified_table("ml_model_predictions") == "ml_model_predictions"


def test_qualified_table_postgres_returns_public_qualified() -> None:
    h = _Helper()
    cast(Any, h).engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    assert h._qualified_table("ml_model_predictions") == "public.ml_model_predictions"
