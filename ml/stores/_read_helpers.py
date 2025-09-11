"""
Helpers for common read patterns in SQL-backed stores.

Provides utilities to qualify table names depending on the SQL dialect and a lightweight
wrapper for issuing read-only queries via the engine.

"""

from __future__ import annotations

from typing import Any


class ReadQueryMixin:
    """
    Mixin offering helpers for read-side queries.
    """

    engine: Any  # SQLAlchemy Engine at runtime

    def _qualified_table(self, base: str) -> str:
        """
        Return a fully-qualified table name for the current dialect.

        Uses schema-qualified names for PostgreSQL and raw names for SQLite.

        """
        name: str | None = None
        try:
            eng = getattr(self, "engine", None)
            if eng is not None:
                dialect = getattr(eng, "dialect", None)
                if dialect is not None:
                    name = getattr(dialect, "name", None)
        except Exception:
            name = None

        if name == "sqlite":
            return base
        return f"public.{base}"
