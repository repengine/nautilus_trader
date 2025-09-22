from __future__ import annotations

from ml.cli.apply_migrations import split_statements


def test_split_statements_handles_dollar_quoted_blocks() -> None:
    sql = (
        "CREATE TABLE t(a int);\n"
        "CREATE FUNCTION f() RETURNS void AS $$ BEGIN RAISE NOTICE 'x'; END $$ LANGUAGE plpgsql;\n"
        "CREATE INDEX i ON t(a);"
    )
    stmts = list(split_statements(sql))
    assert len(stmts) == 3
    assert "FUNCTION f()" in stmts[1]
