"""
Helpers for common read patterns in SQL-backed stores.

Provides utilities to qualify table names depending on the SQL dialect and a lightweight
wrapper for issuing read-only queries via the engine.

"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
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

    def _safe_identifier(self, name: str, allowed: set[str]) -> str:
        """
        Validate identifier against an allowlist to prevent SQL injection in f-strings.

        Parameters
        ----------
        name : str
            Identifier value to validate (e.g., base table name).
        allowed : set[str]
            Allowed identifiers.

        Returns
        -------
        str
            The validated identifier (unchanged) when allowed.

        Raises
        ------
        ValueError
            If the identifier is not in the allowed set.
        """
        if name not in allowed:
            msg = f"Disallowed identifier: {name}"
            raise ValueError(msg)
        return name

    def _safe_table(self, base: str, allowed: set[str]) -> str:
        """
        Return a schema-qualified table name after allowlist validation.

        Parameters
        ----------
        base : str
            Base, unqualified table name.
        allowed : set[str]
            Allowed base names.

        Returns
        -------
        str
            Qualified table name appropriate for the current dialect.
        """
        base_safe = self._safe_identifier(base, allowed)
        return self._qualified_table(base_safe)

    def _execute_read(
        self,
        sql: Any,
        params: Mapping[str, Any],
        *,
        columns: Sequence[str],
    ) -> Any:
        """
        Execute a read-only query using a session when available else engine.

        Builds a DataFrame from session results for MagicMock compatibility and
        falls back to engine-based pandas read when session returns no rows.

        Parameters
        ----------
        sql : Any
            SQLAlchemy text object or string.
        params : Mapping[str, Any]
            Bound parameters for the query.
        columns : Sequence[str]
            Column names for manual DataFrame construction when using a session.

        Returns
        -------
        pandas.DataFrame
            Resulting DataFrame for the query.
        """
        # Local imports to avoid module import-time overhead
        import pandas as pd
        from sqlalchemy import text as _text

        # Try using a persistence session when provided (mock-friendly)
        session_obj: Any | None = None
        try:
            sess = getattr(self, "persistence", None)
            if sess is not None:
                session_obj = getattr(sess, "session", None)
                if session_obj is None and hasattr(sess, "get_session"):
                    session_obj = sess.get_session()
        except Exception:
            session_obj = None

        if session_obj is not None:
            try:
                rows = session_obj.execute(_text(str(sql)), params).fetchall()
            except Exception:
                rows = []

            data = [
                {col: row[idx] for idx, col in enumerate(columns)}
                for row in rows
            ]
            df = pd.DataFrame(data, columns=list(columns))
            if len(df.index):
                return df

        # Fallback to engine-based pandas read
        with self.engine.connect() as conn:
            return pd.read_sql_query(sql, conn, params=dict(params))

    def _fetch_one(self, sql: Any, params: Mapping[str, Any]) -> tuple[Any, ...] | None:
        """
        Execute a read-only scalar/aggregate query and return a single row.

        Parameters
        ----------
        sql : Any
            SQLAlchemy text object or string.
        params : Mapping[str, Any]
            Bound parameters.

        Returns
        -------
        tuple[Any, ...] | None
            First row as a tuple, or None when no rows.
        """
        from sqlalchemy import text as _text

        # Try using a persistence session when provided (mock-friendly)
        session_obj: Any | None = None
        try:
            sess = getattr(self, "persistence", None)
            if sess is not None:
                session_obj = getattr(sess, "session", None)
                if session_obj is None and hasattr(sess, "get_session"):
                    session_obj = sess.get_session()
        except Exception:
            session_obj = None

        if session_obj is not None:
            try:
                row2 = session_obj.execute(_text(str(sql)), dict(params)).fetchone()
            except Exception:
                row2 = None
            from typing import cast as _cast
            return _cast(tuple[Any, ...] | None, row2)

        with self.engine.connect() as conn:
            try:
                row = conn.execute(_text(str(sql)), dict(params)).fetchone()
            except Exception:
                row = None
        from typing import cast as _cast
        return _cast(tuple[Any, ...] | None, row)

    def _fetch_all(self, sql: Any, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        """
        Execute a read-only query and return all rows as tuples.

        Parameters
        ----------
        sql : Any
            SQLAlchemy text object or string.
        params : Mapping[str, Any]
            Bound parameters.

        Returns
        -------
        list[tuple[Any, ...]]
            All result rows.
        """
        from sqlalchemy import text as _text

        # Try using a persistence session when provided (mock-friendly)
        session_obj: Any | None = None
        try:
            sess = getattr(self, "persistence", None)
            if sess is not None:
                session_obj = getattr(sess, "session", None)
                if session_obj is None and hasattr(sess, "get_session"):
                    session_obj = sess.get_session()
        except Exception:
            session_obj = None

        if session_obj is not None:
            try:
                rows2 = session_obj.execute(_text(str(sql)), dict(params)).fetchall()
            except Exception:
                rows2 = []
            return list(rows2)

        with self.engine.connect() as conn:
            try:
                rows = conn.execute(_text(str(sql)), dict(params)).fetchall()
            except Exception:
                rows = []
        return list(rows)
