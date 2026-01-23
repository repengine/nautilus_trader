"""
Schema inspection utilities for validating Nautilus ML database layouts.

The helpers verify that critical tables (feature/model/strategy stores and
market data class tables) satisfy the partitioning and column requirements
expected by the consolidated migrations.  Results can be consumed programmatically or via
the CLI::

    poetry run python -m ml.stores.schema_audit inspect --db-url postgresql://...
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager


@dataclass(frozen=True, slots=True)
class TableExpectation:
    """
    Declarative requirements for a SQL table.
    """

    table: str
    schema: str = "public"
    required_columns: tuple[str, ...] = ()
    partition_columns: tuple[str, ...] = ("ts_event",)
    require_partitioned: bool = True
    require_default_partition: bool = True
    require_primary_key: tuple[str, ...] | None = None
    require_generated_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FunctionExpectation:
    """
    Required PostgreSQL functions (identified via ``to_regprocedure``).
    """

    signature: str  # e.g. "create_monthly_partitions(text,date,integer)"


@dataclass(frozen=True, slots=True)
class TableInspection:
    """
    Result of validating a single table against its expectation.
    """

    table: str
    issues: tuple[str, ...]
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class FunctionInspection:
    """
    Result of validating a required function.
    """

    signature: str
    present: bool

    @property
    def healthy(self) -> bool:
        return self.present


@dataclass(frozen=True, slots=True)
class SchemaAuditReport:
    """
    Aggregate schema inspection result.
    """

    tables: tuple[TableInspection, ...]
    functions: tuple[FunctionInspection, ...]

    @property
    def healthy(self) -> bool:
        return all(table.healthy for table in self.tables) and all(
            fn.healthy for fn in self.functions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "tables": [
                {"table": t.table, "issues": list(t.issues), "details": dict(t.details)}
                for t in self.tables
            ],
            "functions": [
                {"signature": f.signature, "present": f.present} for f in self.functions
            ],
        }


class SchemaAuditor:
    """
    Executes schema inspections against a PostgreSQL database.
    """

    def __init__(
        self,
        *,
        db_url: str,
        expectations: Sequence[TableExpectation] | None = None,
        function_expectations: Sequence[FunctionExpectation] | None = None,
    ) -> None:
        self._db_url = db_url
        self._engine: Engine | None = None
        self._expectations = tuple(expectations or _DEFAULT_TABLE_EXPECTATIONS)
        self._function_expectations = tuple(function_expectations or _DEFAULT_FUNCTIONS)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = EngineManager.get_engine(self._db_url)
        return self._engine

    def inspect(self) -> SchemaAuditReport:
        """
        Run the configured inspections.
        """
        inspections: list[TableInspection] = []
        with self.engine.connect() as connection:
            for expectation in self._expectations:
                inspections.append(self._inspect_table(connection, expectation))
            fn_results = tuple(
                self._inspect_function(connection, fn_expectation)
                for fn_expectation in self._function_expectations
            )
        return SchemaAuditReport(tables=tuple(inspections), functions=fn_results)

    def _inspect_table(self, connection: Any, expectation: TableExpectation) -> TableInspection:
        qualified = f"{expectation.schema}.{expectation.table}"
        regclass = connection.execute(
            text("SELECT to_regclass(:name)::oid"),
            {"name": qualified},
        ).scalar_one_or_none()
        issues: list[str] = []
        details: dict[str, Any] = {}
        if regclass is None:
            issues.append("table missing")
            return TableInspection(
                table=qualified,
                issues=tuple(issues),
                details={"exists": False},
            )

        details["exists"] = True
        table_oid = regclass
        details["oid"] = table_oid

        partition_info = connection.execute(
            text(
                """
                SELECT partstrat, partattrs
                FROM pg_partitioned_table
                WHERE partrelid = :oid
                """,
            ),
            {"oid": table_oid},
        ).mappings().first()

        is_partitioned = partition_info is not None
        details["partitioned"] = is_partitioned
        if expectation.require_partitioned and not is_partitioned:
            issues.append("table not partitioned")

        if is_partitioned:
            attrs = partition_info["partattrs"]
            partition_columns = self._partition_columns(connection, table_oid, attrs or [])
            details["partition_columns"] = partition_columns
            if expectation.partition_columns and tuple(partition_columns) != expectation.partition_columns:
                issues.append(
                    f"partition columns mismatch (expected {expectation.partition_columns}, "
                    f"found {tuple(partition_columns)})",
                )
            if expectation.require_default_partition and not self._has_default_partition(
                connection,
                table_oid,
            ):
                issues.append("default partition missing")

        column_map = self._column_map(connection, expectation.schema, expectation.table)
        missing_columns = [
            column for column in expectation.required_columns if column not in column_map
        ]
        if missing_columns:
            issues.append(f"missing columns: {', '.join(missing_columns)}")

        generated_columns = {
            column: info["is_generated"] for column, info in column_map.items()
        }
        for column in expectation.require_generated_columns:
            flag = generated_columns.get(column)
            if flag != "ALWAYS":
                issues.append(f"column '{column}' is not GENERATED ALWAYS (value={flag!r})")

        if expectation.require_primary_key:
            pk_columns = self._primary_key_columns(connection, table_oid)
            details["primary_key"] = pk_columns
            if tuple(pk_columns) != expectation.require_primary_key:
                issues.append(
                    f"primary key mismatch (expected {expectation.require_primary_key}, "
                    f"found {tuple(pk_columns)})",
                )

        return TableInspection(
            table=qualified,
            issues=tuple(issues),
            details=details,
        )

    def _inspect_function(
        self,
        connection: Any,
        expectation: FunctionExpectation,
    ) -> FunctionInspection:
        present = connection.execute(
            text("SELECT to_regprocedure(:signature)"),
            {"signature": expectation.signature},
        ).scalar_one_or_none() is not None
        return FunctionInspection(signature=expectation.signature, present=present)

    @staticmethod
    def _partition_columns(
        connection: Any,
        oid: Any,
        attnums: Sequence[int],
    ) -> list[str]:
        if not attnums:
            return []
        att_numbers = [int(num) for num in attnums]
        rows = connection.execute(
            text(
                """
                SELECT a.attname
                FROM pg_attribute a
                WHERE a.attrelid = :oid
                  AND a.attnum = ANY(CAST(:nums AS int2[]))
                ORDER BY array_position(CAST(:nums AS int2[]), a.attnum)
                """,
            ),
            {"oid": oid, "nums": att_numbers},
        )
        return [row[0] for row in rows]

    @staticmethod
    def _column_map(connection: Any, schema: str, table: str) -> dict[str, Mapping[str, Any]]:
        rows = connection.execute(
            text(
                """
                SELECT column_name, data_type, is_generated
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table
                """,
            ),
            {"schema": schema, "table": table},
        )
        return {row[0]: row._mapping for row in rows}

    @staticmethod
    def _primary_key_columns(connection: Any, oid: Any) -> list[str]:
        rows = connection.execute(
            text(
                """
                SELECT a.attname
                FROM pg_index i
                JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS cols(attnum, ord) ON true
                JOIN pg_attribute a
                  ON a.attrelid = i.indrelid
                 AND a.attnum = cols.attnum
                WHERE i.indrelid = :oid
                  AND i.indisprimary
                ORDER BY cols.ord
                """,
            ),
            {"oid": oid},
        )
        return [row[0] for row in rows]

    @staticmethod
    def _has_default_partition(connection: Any, oid: Any) -> bool:
        result = connection.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_inherits inh
                    JOIN pg_class c ON c.oid = inh.inhrelid
                    WHERE inh.inhparent = :oid
                      AND pg_get_expr(c.relpartbound, c.oid) = 'DEFAULT'
                )
                """,
            ),
            {"oid": oid},
        ).scalar_one()
        return bool(result)


def _default_table_expectations() -> tuple[TableExpectation, ...]:
    return (
        TableExpectation(
            table="ml_feature_values",
            required_columns=("feature_set_id", "instrument_id", "ts_event", "ts_init", "values"),
            require_primary_key=("feature_set_id", "instrument_id", "ts_event"),
        ),
        TableExpectation(
            table="ml_model_predictions",
            required_columns=("model_id", "instrument_id", "ts_event", "prediction"),
            require_primary_key=("model_id", "instrument_id", "ts_event"),
        ),
        TableExpectation(
            table="ml_strategy_signals",
            required_columns=("strategy_id", "instrument_id", "ts_event", "signal_type"),
            require_primary_key=("strategy_id", "instrument_id", "ts_event"),
        ),
        TableExpectation(
            table="market_data_bar",
            required_columns=(
                "instrument_id",
                "ts_event",
                "ts_init",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ),
            partition_columns=("ts_event",),
            require_primary_key=("instrument_id", "ts_event"),
        ),
        TableExpectation(
            table="market_data_quote_tick",
            required_columns=("instrument_id", "ts_event", "ts_init", "bid", "ask", "bid_size", "ask_size"),
            partition_columns=("ts_event",),
            require_generated_columns=("spread", "mid_price"),
            require_primary_key=("instrument_id", "ts_event"),
        ),
        TableExpectation(
            table="market_data_tbbo",
            required_columns=("instrument_id", "ts_event", "ts_init", "bid", "ask", "bid_size", "ask_size"),
            partition_columns=("ts_event",),
            require_generated_columns=("spread", "mid_price"),
            require_primary_key=("instrument_id", "ts_event"),
        ),
        TableExpectation(
            table="market_data_mbp1",
            required_columns=("instrument_id", "ts_event", "ts_init", "bid", "ask", "bid_size", "ask_size"),
            partition_columns=("ts_event",),
            require_generated_columns=("spread", "mid_price"),
            require_primary_key=("instrument_id", "ts_event"),
        ),
        TableExpectation(
            table="market_data_trade_tick",
            required_columns=("instrument_id", "ts_event", "ts_init", "last"),
            partition_columns=("ts_event",),
            require_primary_key=("instrument_id", "ts_event"),
        ),
    )


def _default_function_expectations() -> tuple[FunctionExpectation, ...]:
    return (
        FunctionExpectation("create_monthly_partitions(text,date,integer)"),
    )


_DEFAULT_TABLE_EXPECTATIONS = _default_table_expectations()
_DEFAULT_FUNCTIONS = _default_function_expectations()


def default_table_expectations() -> tuple[TableExpectation, ...]:
    """
    Return the default table expectations used by the schema audit.

    Returns:
        Tuple of TableExpectation instances.
    """
    return _DEFAULT_TABLE_EXPECTATIONS


def default_function_expectations() -> tuple[FunctionExpectation, ...]:
    """
    Return the default function expectations used by the schema audit.

    Returns:
        Tuple of FunctionExpectation instances.
    """
    return _DEFAULT_FUNCTIONS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the Nautilus ML database schema.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Run the schema audit")
    inspect_parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help="PostgreSQL connection string (defaults to DB_CONNECTION/DATABASE_URL)",
    )
    inspect_parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit the result as JSON instead of human-readable text",
    )
    return parser


def _default_db_url() -> str | None:
    import os

    for key in ("DB_CONNECTION", "DATABASE_URL", "NAUTILUS_DB"):
        value = os.getenv(key)
        if value:
            return value
    return None


def _cmd_inspect(db_url: str, emit_json: bool) -> int:
    auditor = SchemaAuditor(db_url=db_url)
    report = auditor.inspect()
    if emit_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Schema healthy: {report.healthy}")
        for entry in report.tables:
            status = "ok" if entry.healthy else "issues"
            print(f"- Table {entry.table}: {status}")
            if entry.issues:
                for issue in entry.issues:
                    print(f"    * {issue}")
        for fn in report.functions:
            status = "present" if fn.present else "missing"
            print(f"- Function {fn.signature}: {status}")
    return 0 if report.healthy else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        db_url = args.db_url or _default_db_url()
        if not db_url:
            parser.error("Set --db-url or export DB_CONNECTION/DATABASE_URL")
        return _cmd_inspect(db_url, emit_json=args.emit_json)
    parser.error("Unsupported command")


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "FunctionExpectation",
    "FunctionInspection",
    "SchemaAuditReport",
    "SchemaAuditor",
    "TableExpectation",
    "TableInspection",
    "default_function_expectations",
    "default_table_expectations",
    "main",
]
