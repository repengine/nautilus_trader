from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ml.cli import ingest_dbn_archive as cli
from ml.data.ingest.dbn_archive import DBNArchiveIngestionResult
from ml.data.ingest.dbn_archive import InstrumentIngestionSummary


def test_catalog_only_requires_catalog_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CATALOG_PATH", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["dummy", "--catalog-only"])


def test_catalog_only_uses_parquet_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ingest_calls: dict[str, object] = {}
    catalog_root = tmp_path / "catalog"

    class DummyWriter:
        def __init__(
            self,
            catalog: object,
            replace_on_overlap: bool = False,
            dataset_type_identifier_templates: object | None = None,
        ) -> None:
            ingest_calls["catalog"] = catalog
            ingest_calls["replace_on_overlap"] = replace_on_overlap
            ingest_calls[
                "dataset_type_identifier_templates"
            ] = dataset_type_identifier_templates

        def write(self, *, dataset_id: str, schema: str, instrument_id: str, df: object) -> int:
            _ = (dataset_id, schema, instrument_id, df)
            return 0

    class DummyCatalog:
        def __init__(self, path: str) -> None:
            ingest_calls["catalog_path"] = path

    class DummyIngestor:
        def __init__(self, writer: object, mirror_writer: object) -> None:
            _ = mirror_writer
            ingest_calls["writer"] = writer

        def ingest_archive(self, config: object) -> DBNArchiveIngestionResult:
            ingest_calls["config"] = config
            return DBNArchiveIngestionResult(
                dataset="EQUS.MINI",
                schema="ohlcv-1m",
                source_dataset="EQUS.MINI",
                instruments=(InstrumentIngestionSummary("SPY.EQUS", 1, 1),),
                total_frames=1,
                total_rows=1,
            )

    stub_parquet = types.ModuleType("nautilus_trader.persistence.catalog.parquet")
    stub_parquet.ParquetDataCatalog = DummyCatalog
    stub_catalog = types.ModuleType("nautilus_trader.persistence.catalog")
    stub_catalog.parquet = stub_parquet
    stub_persistence = types.ModuleType("nautilus_trader.persistence")
    stub_persistence.catalog = stub_catalog
    stub_nautilus = types.ModuleType("nautilus_trader")
    stub_nautilus.persistence = stub_persistence

    monkeypatch.setitem(sys.modules, "nautilus_trader", stub_nautilus)
    monkeypatch.setitem(sys.modules, "nautilus_trader.persistence", stub_persistence)
    monkeypatch.setitem(sys.modules, "nautilus_trader.persistence.catalog", stub_catalog)
    monkeypatch.setitem(sys.modules, "nautilus_trader.persistence.catalog.parquet", stub_parquet)
    monkeypatch.setattr(cli, "ParquetCatalogRawMarketDataWriter", DummyWriter)
    monkeypatch.setattr(cli, "DBNArchiveIngestor", DummyIngestor)
    monkeypatch.setattr(cli, "_resolve_archives", lambda path: [catalog_root / "dummy.zip"])

    result = cli.main(
        [
            str(tmp_path),
            "--catalog-only",
            "--catalog-path",
            str(catalog_root),
            "--dataset",
            "EQUS.MINI",
            "--schema",
            "ohlcv-1m",
        ],
    )

    assert result == 0
    assert isinstance(ingest_calls["writer"], DummyWriter)
    assert ingest_calls["catalog_path"] == str(catalog_root)
    assert getattr(ingest_calls["config"], "dataset") == "EQUS.MINI"
