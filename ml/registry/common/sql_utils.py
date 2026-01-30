"""
SQL helpers for registry instrumentation tables.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


def set_instrumentation_search_path(session: Session) -> None:
    """
    Ensure instrumentation tables resolve in the public schema.

    Args:
        session: SQLAlchemy session used for registry instrumentation queries.
    """
    session.execute(text("SET LOCAL search_path TO public, pg_catalog"))
