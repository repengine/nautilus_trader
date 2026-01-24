from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from ml.data.coverage.manager import BucketStatus
from ml.data.coverage.manager import CoverageBucketMode
from ml.data.coverage.manager import CoverageManager
from ml.data.coverage.manager import CoverageManagerConfig
from ml.data.coverage.manager import DatasetCoverageConfig
from ml.data.coverage.manager import _parse_dataset_arg
from ml.data.coverage.types import DAY_NS
from ml.stores.protocols import CoverageProviderProtocol

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")

class _FakeCoverageProvider(CoverageProviderProtocol):
    def __init__(self, buckets: dict[str, set[int]] | None = None) -> None:
        self._buckets = buckets or {}

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
        entity_field: str = "instrument_id",
    ) -> set[int]:
        key = f"{dataset_id}:{schema}:{instrument_id}"
        return set(self._buckets.get(key, set()))

class _HealthyAuditor:
    def inspect(self):
        @dataclass
        class _Report:
            healthy: bool = True

            def to_dict(self) -> dict[str, object]:
                return {"healthy": True}

        return _Report()

class _FailingAuditor:
    def inspect(self):
        @dataclass
        class _Report:
            healthy: bool = False

            def to_dict(self) -> dict[str, object]:
                return {"healthy": False, "tables": []}

        return _Report()

def test_classify_buckets_detects_catalog_and_source_gaps() -> None:
    dataset = DatasetCoverageConfig(
        dataset_id="EQUS.MINI",
        schema="tbbo",
        instruments=("AAPL.XNAS",),
    )
    config = CoverageManagerConfig(datasets=(dataset,), lookback_days=2)
    bucket_today_ns = int(datetime(2024, 1, 11, tzinfo=UTC).timestamp() * 1_000_000_000)
    bucket_yesterday_ns = bucket_today_ns - DAY_NS
    sql_provider = _FakeCoverageProvider(
        {"EQUS.MINI:tbbo:AAPL.XNAS": {bucket_today_ns // DAY_NS}},
    )
    catalog_provider = _FakeCoverageProvider(
        {"EQUS.MINI:tbbo:AAPL.XNAS": {bucket_yesterday_ns // DAY_NS}},
    )
    manager = CoverageManager(
        config=config,
        sql_provider=sql_provider,
        catalog_provider=catalog_provider,
    )
    classifications = manager.classify_buckets(reference_time=datetime(2024, 1, 11, tzinfo=UTC))
    assert len(classifications) == 2
    statuses = {c.status for c in classifications}
    assert BucketStatus.HEALTHY in statuses
    assert BucketStatus.RESTORE_FROM_CATALOG in statuses


def test_classify_buckets_catalog_mode_uses_catalog_union() -> None:
    dataset = DatasetCoverageConfig(
        dataset_id="ml.events_calendar",
        schema="events_calendar",
        instruments=("__GLOBAL__",),
        bucket_mode=CoverageBucketMode.CATALOG,
    )
    config = CoverageManagerConfig(datasets=(dataset,), lookback_days=3)
    bucket_ns = int(datetime(2024, 1, 11, tzinfo=UTC).timestamp() * 1_000_000_000)
    bucket_idx = bucket_ns // DAY_NS
    sql_provider = _FakeCoverageProvider(
        {"ml.events_calendar:events_calendar:__GLOBAL__": {bucket_idx}},
    )
    manager = CoverageManager(
        config=config,
        sql_provider=sql_provider,
        catalog_provider=_FakeCoverageProvider(),
    )
    classifications = manager.classify_buckets(reference_time=datetime(2024, 1, 11, tzinfo=UTC))
    assert len(classifications) == 1
    assert classifications[0].status is BucketStatus.HEALTHY

def test_restore_all_raises_when_schema_audit_fails() -> None:
    dataset = DatasetCoverageConfig(
        dataset_id="EQUS.MINI",
        schema="tbbo",
        instruments=("AAPL.XNAS",),
    )
    config = CoverageManagerConfig(datasets=(dataset,), lookback_days=1)
    manager = CoverageManager(
        config=config,
        sql_provider=_FakeCoverageProvider(),
        catalog_provider=_FakeCoverageProvider(),
        schema_auditor=_FailingAuditor(),
    )
    with pytest.raises(RuntimeError):
        manager.restore_all()

def test_parse_dataset_arg_handles_symbols() -> None:
    cfg = _parse_dataset_arg("EQUS.MINI:tbbo:AAPL.XNAS,MSFT.XNAS")
    assert cfg.dataset_id == "EQUS.MINI"
    assert cfg.schema == "tbbo"
    assert cfg.instruments == ("AAPL.XNAS", "MSFT.XNAS")

def test_generate_bucket_specs_includes_entity_field() -> None:
    dataset = DatasetCoverageConfig(
        dataset_id="ml.earnings_actuals",
        schema="earnings",
        instruments=("AAPL",),
        entity_field="ticker",
    )
    manager = CoverageManager(
        config=CoverageManagerConfig(datasets=(dataset,), lookback_days=1),
        sql_provider=_FakeCoverageProvider(),
        catalog_provider=_FakeCoverageProvider(),
    )
    specs = manager.generate_bucket_specs(reference_time=datetime(2024, 1, 1, tzinfo=UTC))
    assert specs
    assert specs[0].entity_field == "ticker"
