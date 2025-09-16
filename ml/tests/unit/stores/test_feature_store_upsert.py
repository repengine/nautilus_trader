from __future__ import annotations

from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert
from typing import Any


def test_upsert_uses_bracket_style_for_values() -> None:
    """
    Ensure upsert compiles when referencing EXCLUDED["values"].

    Using attribute access can break for reserved column names like "values".

    """
    md = MetaData()
    tbl = Table(
        "ml_feature_values",
        md,
        Column("feature_set_id"),
        Column("instrument_id"),
        Column("ts_event"),
        Column("ts_init"),
        Column("values", JSON),
    )

    stmt: Any = insert(tbl).values(
        {
            "feature_set_id": "f",
            "instrument_id": "i",
            "ts_event": 1,
            "ts_init": 1,
            "values": {},
        },
    )

    from typing import cast

    stmt = cast(
        Any,
        stmt.on_conflict_do_update(
            index_elements=["feature_set_id", "instrument_id", "ts_event"],
            set_={
                "values": stmt.excluded["values"],
                "ts_init": stmt.excluded.ts_init,
            },
        ),
    )

    # Compilation to PostgreSQL dialect should not raise
    compiled: Any = stmt.compile(dialect=postgresql.dialect())
    _ = str(compiled)
