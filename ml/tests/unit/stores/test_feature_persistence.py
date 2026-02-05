from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self
from sqlalchemy import BIGINT
from sqlalchemy import BOOLEAN
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy import select

from ml.stores.feature_persistence import FeaturePersistence


@dataclass
class CircuitBreakerStub:
    """
    Minimal circuit breaker stub for persistence tests.
    """

    can_execute_result: bool
    successes: int = 0
    failures: int = 0
    raise_on_success: bool = False
    raise_on_failure: bool = False

    def can_execute(self) -> bool:
        return self.can_execute_result

    def record_success(self) -> None:
        if self.raise_on_success:
            raise RuntimeError("record_success failed")
        self.successes += 1

    def record_failure(self) -> None:
        if self.raise_on_failure:
            raise RuntimeError("record_failure failed")
        self.failures += 1


def _build_feature_values_table(engine: Any) -> Table:
    metadata = MetaData()
    table = Table(
        "ml_feature_values",
        metadata,
        Column("feature_set_id", String(255), primary_key=True),
        Column("instrument_id", String(100), primary_key=True),
        Column("ts_event", BIGINT, primary_key=True),
        Column("ts_init", BIGINT),
        Column("values", JSON, nullable=False),
        Column("is_live", BOOLEAN, default=False),
        Column("source", String(50)),
    )
    metadata.create_all(engine)
    return table


def test_execute_write_skips_when_circuit_breaker_open(
    isolated_prometheus_registry: object,
) -> None:
    """
    Circuit breaker should short-circuit writes without touching the engine.
    """
    engine = create_engine("sqlite:///:memory:")
    table = _build_feature_values_table(engine)
    breaker = CircuitBreakerStub(can_execute_result=False)

    persistence = FeaturePersistence(engine, table, circuit_breaker=breaker)

    result = persistence._execute_write(
        {
            "feature_set_id": "fs_demo",
            "instrument_id": "EUR/USD.SIM",
            "ts_event": 1,
            "ts_init": 1,
            "values": {"f1": 1.0},
            "is_live": False,
            "source": "computed",
        },
    )

    assert result is False
    assert breaker.successes == 0
    assert breaker.failures == 0

    with engine.connect() as conn:
        rows = conn.execute(select(table.c.feature_set_id)).fetchall()
    assert rows == []


def test_execute_write_inserts_row_and_records_success(
    isolated_prometheus_registry: object,
) -> None:
    """
    Successful writes should insert rows and record breaker success.
    """
    engine = create_engine("sqlite:///:memory:")
    table = _build_feature_values_table(engine)
    breaker = CircuitBreakerStub(can_execute_result=True)

    persistence = FeaturePersistence(engine, table, circuit_breaker=breaker)

    result = persistence._execute_write(
        {
            "feature_set_id": "fs_demo",
            "instrument_id": "EUR/USD.SIM",
            "ts_event": 10,
            "ts_init": 10,
            "values": {"f1": 1.0},
            "is_live": False,
            "source": "computed",
        },
    )

    assert result is True
    assert breaker.successes == 1
    assert breaker.failures == 0

    with engine.connect() as conn:
        rows = conn.execute(select(table.c.instrument_id)).fetchall()

    assert rows == [("EUR/USD.SIM",)]


def test_execute_write_ignores_success_hook_failures(
    isolated_prometheus_registry: object,
) -> None:
    """
    Exceptions in record_success should be swallowed to preserve writes.
    """
    engine = create_engine("sqlite:///:memory:")
    table = _build_feature_values_table(engine)
    breaker = CircuitBreakerStub(can_execute_result=True, raise_on_success=True)

    persistence = FeaturePersistence(engine, table, circuit_breaker=breaker)

    result = persistence._execute_write(
        {
            "feature_set_id": "fs_demo",
            "instrument_id": "EUR/USD.SIM",
            "ts_event": 20,
            "ts_init": 20,
            "values": {"f1": 1.0},
            "is_live": False,
            "source": "computed",
        },
    )

    assert result is True
    assert breaker.successes == 0
    assert breaker.failures == 0


def test_execute_write_records_failure_on_exception(
    isolated_prometheus_registry: object,
) -> None:
    """
    Exceptions during execution should return False and record failures.
    """

    class FailingEngine:
        def begin(self) -> Any:
            class _Ctx:
                def __enter__(self) -> Self:
                    return self

                def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> bool:
                    return False

                def execute(self, _stmt: Any) -> None:
                    raise RuntimeError("boom")

            return _Ctx()

    engine = FailingEngine()
    table = _build_feature_values_table(create_engine("sqlite:///:memory:"))
    breaker = CircuitBreakerStub(can_execute_result=True)

    persistence = FeaturePersistence(engine, table, circuit_breaker=breaker)

    result = persistence._execute_write(
        {
            "feature_set_id": "fs_demo",
            "instrument_id": "EUR/USD.SIM",
            "ts_event": 30,
            "ts_init": 30,
            "values": {"f1": 1.0},
            "is_live": False,
            "source": "computed",
        },
    )

    assert result is False
    assert breaker.successes == 0
    assert breaker.failures == 1


def test_write_batch_continues_after_item_failure(
    isolated_prometheus_registry: object,
) -> None:
    """
    Batch writes should continue processing after a failing item.
    """

    class StubPersistence(FeaturePersistence):
        def __init__(self) -> None:
            engine = create_engine("sqlite:///:memory:")
            table = _build_feature_values_table(engine)
            super().__init__(engine, table)
            self.rows: list[dict[str, Any]] = []

        def _execute_write(self, row: dict[str, Any]) -> bool:
            self.rows.append(row)
            if row.get("feature_set_id") == "bad":
                raise RuntimeError("boom")
            return True

    class Item:
        def __init__(self, feature_set_id: str) -> None:
            self.feature_set_id = feature_set_id
            self.instrument_id = "EUR/USD.SIM"
            self.ts_event = 1
            self.ts_init = 1
            self.feature_values = {"f1": 1.0}

    persistence = StubPersistence()

    persistence.write_batch([Item("good"), Item("bad")])

    assert len(persistence.rows) == 2
