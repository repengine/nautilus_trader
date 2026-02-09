from __future__ import annotations

from ml.cli.apply_migrations import apply_files
from ml.cli.apply_migrations import build_plan
from ml.cli.apply_migrations import split_statements
from ml.stores.common.sql_splitter import split_sql_statements
from ml.stores.migrations_runner import apply_migration_files
from ml.stores.migrations_runner import build_migration_plan


def test_cli_migration_helpers_point_to_canonical_owners() -> None:
    assert build_plan is build_migration_plan
    assert apply_files is apply_migration_files
    assert split_statements is split_sql_statements


def test_split_statements_handles_dollar_quoted_blocks() -> None:
    sql = (
        "CREATE TABLE t(a int);\n"
        "CREATE FUNCTION f() RETURNS void AS $$ BEGIN RAISE NOTICE 'x'; END $$ LANGUAGE plpgsql;\n"
        "CREATE INDEX i ON t(a);"
    )
    stmts = list(split_statements(sql))
    assert len(stmts) == 3
    assert "FUNCTION f()" in stmts[1]
