from __future__ import annotations

import pytest
from sqlalchemy import text

from ml.core.db_engine import EngineManager
from ml.stores.schema_audit import SchemaAuditor
from ml.stores.schema_audit import TableExpectation


@pytest.mark.database
def test_schema_auditor_passes_on_partitioned_table(test_database) -> None:
    engine = EngineManager.get_engine(test_database.connection_string)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audit_partitioned (
                    id BIGINT NOT NULL,
                    ts_event BIGINT NOT NULL,
                    payload INTEGER,
                    PRIMARY KEY (id, ts_event)
                ) PARTITION BY RANGE (ts_event);
                """,
            ),
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audit_partitioned_default
                PARTITION OF audit_partitioned DEFAULT;
                """,
            ),
        )
    try:
        expectation = TableExpectation(
            table="audit_partitioned",
            partition_columns=("ts_event",),
            required_columns=("id", "ts_event", "payload"),
            require_primary_key=("id", "ts_event"),
        )
        auditor = SchemaAuditor(
            db_url=test_database.connection_string,
            expectations=(expectation,),
            function_expectations=(),
        )
        report = auditor.inspect()
        assert report.healthy
        assert report.tables[0].issues == ()
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS audit_partitioned CASCADE"))

@pytest.mark.database
def test_schema_auditor_flags_heap_table(test_database) -> None:
    engine = EngineManager.get_engine(test_database.connection_string)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audit_heap (
                    id BIGINT,
                    ts_event BIGINT NOT NULL
                );
                """,
            ),
        )
    try:
        expectation = TableExpectation(
            table="audit_heap",
            required_columns=("ts_event",),
        )
        auditor = SchemaAuditor(
            db_url=test_database.connection_string,
            expectations=(expectation,),
            function_expectations=(),
        )
        report = auditor.inspect()
        assert not report.healthy
        assert "table not partitioned" in report.tables[0].issues
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS audit_heap CASCADE"))
