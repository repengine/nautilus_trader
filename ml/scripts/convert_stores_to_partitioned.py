"""
Convert ML store tables to partitioned tables safely.

This script checks whether the three ML store tables are partitioned. If a table
is non-partitioned, it creates a new partitioned parent, pre-creates monthly
partitions covering existing data, copies rows, swaps names, and recreates
standard indexes.

Tables covered:
- ml_feature_values
- ml_model_predictions
- ml_strategy_signals

Usage:
  uv run --active --no-sync python -m ml.scripts.convert_stores_to_partitioned \
      --db-url postgresql://postgres:postgres@localhost:5432/nautilus

Options:
  --tables ml_feature_values,ml_model_predictions  # subset
  --ahead 3    # months ahead to pre-create
  --dry-run    # preview actions without executing writes

Notes:
- Designed for dev/test DBs. For production, schedule a maintenance window.
- If tables already partitioned, script skips them.

"""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager


@dataclass(frozen=True)
class TableSpec:
    name: str
    create_parent_sql: str
    index_sql: Iterable[str]


FEATURES = TableSpec(
    name="ml_feature_values",
    create_parent_sql=(
        """
CREATE TABLE {name} (
    id BIGSERIAL,
    feature_set_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    values JSONB NOT NULL,
    is_live BOOLEAN DEFAULT FALSE,
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, ts_event)
) PARTITION BY RANGE (ts_event)
        """
    ),
    index_sql=(
        "CREATE INDEX IF NOT EXISTS idx_ml_feature_values_lookup ON {name} (feature_set_id, instrument_id, ts_event)",
        "CREATE INDEX IF NOT EXISTS idx_ml_feature_values_live ON {name} (is_live) WHERE is_live = TRUE",
    ),
)


PREDICTIONS = TableSpec(
    name="ml_model_predictions",
    create_parent_sql=(
        """
CREATE TABLE {name} (
    model_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    prediction FLOAT NOT NULL,
    confidence FLOAT,
    features_used JSONB,
    inference_time_ms FLOAT,
    is_live BOOLEAN DEFAULT FALSE,
    created_at BIGINT,
    PRIMARY KEY (model_id, instrument_id, ts_event)
) PARTITION BY RANGE (ts_event)
        """
    ),
    index_sql=(
        "CREATE INDEX IF NOT EXISTS idx_ml_model_predictions_lookup ON {name} (model_id, instrument_id, ts_event)",
        "CREATE INDEX IF NOT EXISTS idx_ml_model_predictions_live ON {name} (is_live) WHERE is_live = TRUE",
    ),
)


SIGNALS = TableSpec(
    name="ml_strategy_signals",
    create_parent_sql=(
        """
CREATE TABLE {name} (
    strategy_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    strength FLOAT NOT NULL,
    model_predictions JSONB,
    risk_metrics JSONB,
    execution_params JSONB,
    is_live BOOLEAN DEFAULT FALSE,
    created_at BIGINT,
    PRIMARY KEY (strategy_id, instrument_id, ts_event)
) PARTITION BY RANGE (ts_event)
        """
    ),
    index_sql=(
        "CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_lookup ON {name} (strategy_id, instrument_id, ts_event)",
        "CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_type ON {name} (signal_type)",
        "CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_live ON {name} (is_live) WHERE is_live = TRUE",
    ),
)


def _is_partitioned(engine: Engine, table: str) -> bool:
    with engine.connect() as conn:
        r = conn.execute(
            text(
                """
SELECT EXISTS (
  SELECT 1 FROM pg_class c
  JOIN pg_partitioned_table p ON p.partrelid = c.oid
 WHERE c.relname = :t
)
                """,
            ),
            {"t": table},
        )
        return bool(r.scalar())


def _ts_month_floor(ns: int) -> dt.date:
    d = dt.datetime.utcfromtimestamp(ns / 1_000_000_000).date().replace(day=1)
    return d


def _iter_months(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = dt.date(start.year, start.month, 1)
    last = dt.date(end.year, end.month, 1)
    while cur <= last:
        yield cur
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)


def _create_partition(engine: Engine, parent: str, base: str, month: dt.date) -> None:
    # Name partitions canonically using base table name (not parent alias)
    name = f"{base}_{month.year:04d}_{month.month:02d}"
    start = int(dt.datetime(month.year, month.month, 1).timestamp() * 1e9)
    if month.month == 12:
        end_d = dt.date(month.year + 1, 1, 1)
    else:
        end_d = dt.date(month.year, month.month + 1, 1)
    end = int(dt.datetime(end_d.year, end_d.month, 1).timestamp() * 1e9)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
CREATE TABLE IF NOT EXISTS {name}
PARTITION OF {parent}
FOR VALUES FROM ({start}) TO ({end})
                """,
            ),
        )


def _copy_rows(engine: Engine, src: str, dst: str) -> int:
    # Build intersection column list in destination order; skip created_at to avoid type mismatches
    with engine.connect() as conn:
        src_cols = [
            r[0]
            for r in conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns WHERE table_name=:t ORDER BY ordinal_position",
                ),
                {"t": src},
            )
        ]
        dst_cols = [
            r[0]
            for r in conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns WHERE table_name=:t ORDER BY ordinal_position",
                ),
                {"t": dst},
            )
        ]
    cols = [c for c in dst_cols if c in src_cols and c != "created_at"]
    col_list = ", ".join(cols)
    with engine.begin() as conn:
        res = conn.execute(text(f"INSERT INTO {dst} ({col_list}) SELECT {col_list} FROM {src}"))
        try:
            return int(res.rowcount or 0)
        except Exception:
            return 0


def convert_one(engine: Engine, spec: TableSpec, months_ahead: int, dry_run: bool = False) -> None:
    if _is_partitioned(engine, spec.name):
        print(f"{spec.name}: already partitioned; skipping")
        return

    backup = f"{spec.name}_legacy"
    newp = f"{spec.name}_p"
    print(f"{spec.name}: converting to partitioned parent -> {newp}")
    if dry_run:
        return

    with engine.begin() as conn:
        # Clean up stale new parent if previous attempt failed mid-flight
        conn.execute(text(f"DROP TABLE IF EXISTS {newp} CASCADE"))
        # Create new partitioned parent
        conn.execute(text(spec.create_parent_sql.format(name=newp)))

    # Create partitions covering existing data + ahead
    with engine.connect() as conn:
        r = conn.execute(text(f"SELECT MIN(ts_event), MAX(ts_event) FROM {spec.name}"))
        row = r.fetchone()
        min_ns, max_ns = (row[0], row[1]) if row else (None, None)

    today = dt.date.today()
    end_month = dt.date(today.year, today.month, 1)
    # If there is data, extend range to include it
    if max_ns is not None:
        end_month = max(end_month, _ts_month_floor(int(max_ns)))
    # add months ahead window
    for _ in range(months_ahead):
        if end_month.month == 12:
            end_month = dt.date(end_month.year + 1, 1, 1)
        else:
            end_month = dt.date(end_month.year, end_month.month + 1, 1)

    if min_ns is None:
        # no data; ensure current and ahead only
        start_month = dt.date.today().replace(day=1)
    else:
        start_month = _ts_month_floor(int(min_ns))

    for m in _iter_months(start_month, end_month):
        _create_partition(engine, newp, spec.name, m)

    # Copy rows to partitioned parent
    copied = _copy_rows(engine, spec.name, newp)
    print(f"{spec.name}: copied {copied} rows")

    # Swap names
    with engine.begin() as conn:
        # Drop old backup if exists
        conn.execute(text(f"DROP TABLE IF EXISTS {backup} CASCADE"))
        conn.execute(text(f"ALTER TABLE {spec.name} RENAME TO {backup}"))
        conn.execute(text(f"ALTER TABLE {newp} RENAME TO {spec.name}"))

        # Recreate standard indexes on new parent
        for idx in spec.index_sql:
            conn.execute(text(idx.format(name=spec.name)))

    print(f"{spec.name}: conversion complete; old table -> {backup}")

    # Post-swap: rename any child partitions that were created with the temporary parent prefix
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename LIKE :p",
            ),
            {"p": f"{spec.name}_p_%"},
        ).fetchall()
        for (old_name,) in rows:
            # Expect suffix YYYY_MM at the end
            parts = old_name.rsplit("_", 2)
            if len(parts) >= 3:
                yyyy, mm = parts[-2], parts[-1]
                new_name = f"{spec.name}_{yyyy}_{mm}"
            else:
                continue
            # Only rename if target doesn't exist
            exists = conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename=:n)",
                ),
                {"n": new_name},
            ).scalar()
            if not exists:
                conn.execute(text(f"ALTER TABLE {old_name} RENAME TO {new_name}"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert ML store tables to partitioned tables")
    p.add_argument("--db-url", dest="db_url", default=None)
    p.add_argument("--tables", default=",".join([FEATURES.name, PREDICTIONS.name, SIGNALS.name]))
    p.add_argument("--ahead", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    import os

    db_url = (
        args.db_url
        or os.getenv("DATABASE_URL")
        or "postgresql://postgres:postgres@localhost:5432/nautilus"
    )
    engine = EngineManager.get_engine(db_url)

    specs: dict[str, TableSpec] = {s.name: s for s in (FEATURES, PREDICTIONS, SIGNALS)}
    selected = [t.strip() for t in args.tables.split(",") if t.strip()]
    for t in selected:
        spec = specs.get(t)
        if not spec:
            print(f"Unknown table {t}; skipping")
            continue
        convert_one(engine, spec, months_ahead=args.ahead, dry_run=bool(args.dry_run))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
