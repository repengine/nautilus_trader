from __future__ import annotations

from pathlib import Path

import pytest

from ml.config.dataset_coverage import CoverageDatasetEntry
from ml.config.dataset_coverage import load_dataset_coverage_entries


def _write_config(tmp_path: Path, payload: str) -> Path:
    path = tmp_path / "coverage.toml"
    path.write_text(payload, encoding="utf-8")
    return path


def test_load_dataset_coverage_entries_parses_aliases(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [[datasets]]
        dataset_id = "ml.earnings_actuals"
        schema = "earnings"
        entities = "@tier1_core"
        strip_venue = true
        entity_field = "ticker"

        [datasets.sql]
        table = "earnings_actuals"
        schema = "ml"
        ts_field = "ts_event"
        entity_field = "ticker"

        [datasets.parquet]
        path = "../../data/features/earnings_raw/earnings_actuals"
        partition_field = "ticker"
        timestamp_field = "ts_event"
        """,
    )
    entries = load_dataset_coverage_entries(path)
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, CoverageDatasetEntry)
    assert entry.dataset.dataset_id == "ml.earnings_actuals"
    assert entry.dataset.entity_field == "ticker"
    assert all("." not in item for item in entry.dataset.instruments)
    assert entry.sql_override is not None
    assert entry.sql_override.table_name == "earnings_actuals"
    assert entry.parquet_spec is not None
    assert entry.parquet_spec.partition_field == "ticker"


def test_load_dataset_coverage_entries_requires_entities(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [[datasets]]
        dataset_id = "ml.bad"
        schema = "x"
        """,
    )
    with pytest.raises(ValueError):
        load_dataset_coverage_entries(path)


def test_load_dataset_coverage_entries_preserves_blank_partition_template(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [[datasets]]
        dataset_id = "ml.events_calendar"
        schema = "events_calendar"
        entities = "AAPL"

        [datasets.parquet]
        path = "../../data/features/events/events.parquet"
        partition_template = ""
        """,
    )
    entries = load_dataset_coverage_entries(path)
    assert len(entries) == 1
    spec = entries[0].parquet_spec
    assert spec is not None
    assert spec.partition_template == ""


def test_load_dataset_coverage_entries_parses_entities_file(tmp_path: Path) -> None:
    entities_path = tmp_path / "entities.txt"
    entities_path.write_text("SPY.EQUS\nQQQ.EQUS\n", encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
        [[datasets]]
        dataset_id = "features"
        schema = "feature_values"
        entities_file = "entities.txt"
        entity_field = "instrument_id"
        """,
    )
    entries = load_dataset_coverage_entries(path)
    assert len(entries) == 1
    assert entries[0].dataset.instruments == ("SPY.EQUS", "QQQ.EQUS")


def test_load_dataset_coverage_entries_parses_bucket_mode(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [[datasets]]
        dataset_id = "ml.events_calendar"
        schema = "events_calendar"
        entities = "__GLOBAL__"
        bucket_mode = "catalog"
        """,
    )
    entries = load_dataset_coverage_entries(path)
    assert len(entries) == 1
    assert entries[0].dataset.bucket_mode.value == "catalog"
