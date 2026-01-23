#!/usr/bin/env python3

"""
Raw dataset IO protocols and parquet-backed adapters.

Consolidates the raw IO Protocols and the ParquetDataCatalog reader/writer.

Migrated from:
- raw_io.py
- raw_io_parquet.py

Existing modules re-export these symbols with deprecation warnings.

"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import structlog

from ml.data.catalog_utils import bars_to_dataframe
from ml.data.catalog_utils import quotes_to_dataframe
from ml.data.catalog_utils import trades_to_dataframe
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetType
from ml.schema import default_identifier_template_for_dataset_type
from ml.schema import validate_dataset_type_templates
from ml.schema import validate_identifier_template


logger = structlog.get_logger(__name__)


@runtime_checkable
class RawIngestionWriterProtocol(Protocol):
    """
    Protocol for writing raw datasets (bars/quotes/trades/mbp1/tbbo).
    """

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int: ...


@runtime_checkable
class RawReaderProtocol(Protocol):
    """
    Protocol for reading raw datasets over a time range.
    """

    def read_range(
        self,
        *,
        dataset_type: DatasetType,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> DataFrameLike: ...


class ParquetCatalogRawReader(RawReaderProtocol):
    """
    Raw reader backed by Nautilus ParquetDataCatalog.
    """

    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog

    def read_range(
        self,
        *,
        dataset_type: DatasetType,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> DataFrameLike:
        start: Any = start_ns
        end: Any = end_ns
        if dataset_type == DatasetType.BARS:
            return cast(
                DataFrameLike,
                bars_to_dataframe(self._catalog, [instrument_id], start, end),
            )
        if dataset_type in (DatasetType.QUOTES, DatasetType.TBBO):
            return cast(
                DataFrameLike,
                quotes_to_dataframe(self._catalog, [instrument_id], start, end),
            )
        if dataset_type == DatasetType.TRADES:
            return cast(
                DataFrameLike,
                trades_to_dataframe(self._catalog, [instrument_id], start, end),
            )
        try:
            from ml._imports import HAS_POLARS
            from ml._imports import pl as _pl

            if HAS_POLARS and _pl is not None:
                return cast(DataFrameLike, _pl.DataFrame({}))
        except Exception as exc:
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "Fallback to empty DataFrame failed in ParquetCatalogRawReader: %s",
                exc,
                exc_info=True,
            )
        return cast(DataFrameLike, [])


class ParquetCatalogRawWriter(RawIngestionWriterProtocol):
    """
    Pass-through writer to ParquetDataCatalog for domain objects.
    """

    def __init__(
        self,
        catalog: Any,
        dataset_type_identifier_templates: Mapping[DatasetType, str] | None = None,
        replace_on_overlap: bool = False,
    ) -> None:
        self._catalog = catalog
        templates = validate_dataset_type_templates(dataset_type_identifier_templates)
        self._dataset_type_templates: dict[DatasetType, str] = templates or {}
        if DatasetType.BARS not in self._dataset_type_templates:
            self._dataset_type_templates[DatasetType.BARS] = (
                default_identifier_template_for_dataset_type(DatasetType.BARS)
            )
        self._replace_on_overlap = replace_on_overlap

    def _identifier_template_for(self, dataset_type: DatasetType) -> str:
        return self._dataset_type_templates.get(
            dataset_type,
            default_identifier_template_for_dataset_type(dataset_type),
        )

    def _resolve_identifier(self, *, dataset_type: DatasetType, instrument_id: str) -> str:
        template = self._identifier_template_for(dataset_type)
        return template.format(instrument_id=instrument_id, schema=dataset_type.value)

    def _catalog_directory(
        self,
        data_cls: type[object],
        identifier: str | None,
    ) -> Path:
        from nautilus_trader.persistence.catalog import parquet as _parquet
        from nautilus_trader.persistence.funcs import urisafe_identifier

        directory = (
            Path(self._catalog.path).expanduser()
            / "data"
            / _parquet.class_to_filename(data_cls)  # type: ignore[attr-defined]
        )
        if identifier is not None:
            directory = directory / urisafe_identifier(identifier)
        return directory

    def _prune_overlaps(
        self,
        *,
        data_cls: type[object],
        identifier: str | None,
        start_ns: int,
        end_ns: int,
    ) -> None:
        if not self._replace_on_overlap:
            return
        try:
            from nautilus_trader.persistence.catalog.parquet import _interval_overlaps
            from nautilus_trader.persistence.catalog.parquet import _timestamps_to_filename
        except Exception as exc:  # pragma: no cover - defensive import
            logger.debug("catalog.overlap.import_failed", exc_info=True, error=str(exc))
            return

        try:
            existing = self._catalog.get_intervals(data_cls, identifier)
        except Exception as exc:  # pragma: no cover - catalog may not expose intervals
            logger.debug(
                "catalog.overlap.intervals_failed",
                exc_info=True,
                error=str(exc),
                data_cls=getattr(data_cls, "__name__", str(data_cls)),
                identifier=identifier,
            )
            return
        candidate = (start_ns, end_ns)
        if not existing or not _interval_overlaps(existing, candidate):
            return
        directory = self._catalog_directory(data_cls, identifier)
        deleted = 0
        for interval_start, interval_end in existing:
            if interval_start <= end_ns and start_ns <= interval_end:
                filename = _timestamps_to_filename(interval_start, interval_end)
                path = directory / filename
                try:
                    # Use the underlying filesystem to preserve protocol semantics.
                    self._catalog.fs.delete(str(path), recursive=False)
                    deleted += 1
                except Exception as exc:  # pragma: no cover - filesystem failures are rare
                    logger.warning(
                        "catalog.overlap.delete_failed",
                        exc_info=True,
                        error=str(exc),
                        path=str(path),
                        data_cls=getattr(data_cls, "__name__", str(data_cls)),
                        identifier=identifier,
                    )
        if deleted:
            logger.info(
                "catalog.overlap.pruned",
                data_cls=getattr(data_cls, "__name__", str(data_cls)),
                identifier=identifier,
                start_ns=start_ns,
                end_ns=end_ns,
                deleted=deleted,
            )

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int:
        """
        Write raw items to the Parquet catalog.

        Accepts either a list of domain objects (fast path) or tabular data for
        bars which will be converted to domain Bars using a default bar template.
        """
        # Fast path: already domain objects
        if isinstance(data, list) and data and not isinstance(data[0], dict):
            items = cast(Iterable[object], data)
            # Leave fast path unchanged unless explicitly replacing overlaps via domain objects.
            if self._replace_on_overlap:
                logger.debug(
                    "catalog.overlap.fast_path_not_supported",
                    data_cls=type(items).__name__,
                )
            self._catalog.write_data(items)
            return len(data)

        # Conversion path for bars/quotes/trades/mbp: DataFrame-like or list[dict]
        if dataset_type in (
            DatasetType.BARS,
            DatasetType.QUOTES,
            DatasetType.TRADES,
            DatasetType.TBBO,
            DatasetType.MBP1,
        ):
            try:
                from nautilus_trader.model.data import Bar as _Bar
                from nautilus_trader.model.data import BarType as _BarType
                from nautilus_trader.model.data import QuoteTick as _QuoteTick
                from nautilus_trader.model.data import TradeTick as _TradeTick
                from nautilus_trader.model.identifiers import InstrumentId as _InstrumentId
                from nautilus_trader.model.identifiers import TradeId as _TradeId
                from nautilus_trader.model.objects import Price as _Price
                from nautilus_trader.model.objects import Quantity as _Quantity

                from nautilus_trader.model.enums import AggressorSide as _AggressorSide
            except Exception as exc:  # pragma: no cover - import-time defensive
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "Unable to import Nautilus types for bar conversion: %s",
                    exc,
                )
                return 0

            # Support pandas/polars by duck-typing iteration
            from typing import Any as _Any
            rows: Iterable[dict[str, _Any]]
            if hasattr(data, "iter_rows"):
                df_any2 = cast(_Any, data)
                rows = cast(Iterable[dict[str, _Any]], df_any2.iter_rows(named=True))
            elif hasattr(data, "iterrows"):
                # Build a concrete list of dicts for strict typing
                rows_list: list[dict[str, _Any]] = []
                df_any = cast(_Any, data)
                for _, row in df_any.iterrows():
                    try:
                        rows_list.append(cast(dict[str, _Any], row.to_dict()))
                    except Exception:
                        rows_list.append({})
                rows = rows_list
            else:
                rows = cast(Iterable[dict[str, _Any]], data)

            if dataset_type == DatasetType.BARS:
                template = self._identifier_template_for(DatasetType.BARS)
                validated_template = validate_identifier_template(
                    template,
                    label="bar identifier template",
                )
                bars: list[_Bar] = []
                for row in rows:
                    inst = str(row.get("instrument_id", "UNKNOWN"))
                    bt = _BarType.from_str(validated_template.format(instrument_id=inst))
                    try:
                        bar_ts_init = int(row.get("ts_init", row["ts_event"]))
                        bars.append(
                            _Bar(
                                bar_type=bt,
                                open=_Price(float(row["open"]), precision=6),
                                high=_Price(float(row["high"]), precision=6),
                                low=_Price(float(row["low"]), precision=6),
                                close=_Price(float(row["close"]), precision=6),
                                volume=_Quantity(float(row.get("volume", 0.0)), precision=0),
                                ts_event=int(row["ts_event"]),
                                ts_init=bar_ts_init,
                            ),
                        )
                    except Exception:
                        continue
                if not bars:
                    return 0
                start_ns = min(bar.ts_init for bar in bars)
                end_ns = max(bar.ts_init for bar in bars)
                identifier = str(bars[0].bar_type)
                self._prune_overlaps(
                    data_cls=_Bar,
                    identifier=identifier,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
                if hasattr(self._catalog, "_write_chunk"):
                    self._catalog._write_chunk(
                        data=bars,
                        data_cls=_Bar,
                        identifier=identifier,
                        start=start_ns,
                        end=end_ns,
                    )
                else:
                    self._catalog.write_data(bars)
                return len(bars)

            def _first_present(row: dict[str, _Any], keys: tuple[str, ...]) -> _Any | None:
                for key in keys:
                    if key in row and row[key] is not None:
                        return row[key]
                return None

            if dataset_type in (DatasetType.QUOTES, DatasetType.TBBO, DatasetType.MBP1):
                quotes: list[_QuoteTick] = []
                for row in rows:
                    try:
                        inst = _InstrumentId.from_str(str(row.get("instrument_id", "UNKNOWN")))
                        bid_val = _first_present(
                            row,
                            (
                                "bid",
                                "bid_px",
                                "bid_price",
                                "bid_px_0",
                                "bid_px_1",
                                "bid_px_00",
                                "bid_px_01",
                            ),
                        )
                        ask_val = _first_present(
                            row,
                            (
                                "ask",
                                "ask_px",
                                "ask_price",
                                "ask_px_0",
                                "ask_px_1",
                                "ask_px_00",
                                "ask_px_01",
                            ),
                        )
                        if bid_val is None or ask_val is None:
                            continue
                        bid = _Price(float(bid_val), precision=6)
                        ask = _Price(float(ask_val), precision=6)
                        bsz = _Quantity(
                            float(
                                _first_present(
                                    row,
                                    (
                                        "bid_size",
                                        "bid_sz",
                                        "bid_sz_0",
                                        "bid_sz_1",
                                        "bid_sz_00",
                                        "bid_sz_01",
                                    ),
                                )
                                or 0.0,
                            ),
                            precision=0,
                        )
                        asz = _Quantity(
                            float(
                                _first_present(
                                    row,
                                    (
                                        "ask_size",
                                        "ask_sz",
                                        "ask_sz_0",
                                        "ask_sz_1",
                                        "ask_sz_00",
                                        "ask_sz_01",
                                    ),
                                )
                                or 0.0,
                            ),
                            precision=0,
                        )
                        quotes.append(
                            _QuoteTick(
                                instrument_id=inst,
                                bid_price=bid,
                                ask_price=ask,
                                bid_size=bsz,
                                ask_size=asz,
                                ts_event=int(row["ts_event"]),
                                ts_init=int(row.get("ts_init", row["ts_event"])),
                            ),
                        )
                    except Exception:
                        continue
                if not quotes:
                    return 0
                start_ns = min(quote.ts_init for quote in quotes)
                end_ns = max(quote.ts_init for quote in quotes)
                identifier = self._resolve_identifier(
                    dataset_type=dataset_type,
                    instrument_id=quotes[0].instrument_id.value,
                )
                self._prune_overlaps(
                    data_cls=_QuoteTick,
                    identifier=identifier,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
                if hasattr(self._catalog, "_write_chunk"):
                    self._catalog._write_chunk(
                        data=quotes,
                        data_cls=_QuoteTick,
                        identifier=identifier,
                        start=start_ns,
                        end=end_ns,
                    )
                else:
                    self._catalog.write_data(quotes)
                return len(quotes)

            if dataset_type == DatasetType.TRADES:
                trades: list[_TradeTick] = []
                for row in rows:
                    try:
                        inst = _InstrumentId.from_str(str(row.get("instrument_id", "UNKNOWN")))
                        price = _Price(float(row["price"]), precision=6)
                        size = _Quantity(float(row.get("size", 0.0)), precision=0)
                        side_str = str(row.get("aggressor_side", "BUYER")).upper()
                        if side_str not in {"BUYER", "SELLER"}:
                            side_str = "BUYER"
                        side = getattr(_AggressorSide, side_str)
                        trade_id_val = str(row.get("trade_id", f"auto_{row.get('ts_event', 0)}"))
                        trade_id_obj = _TradeId(trade_id_val)
                        trades.append(
                            _TradeTick(
                                instrument_id=inst,
                                price=price,
                                size=size,
                                aggressor_side=side,
                                trade_id=trade_id_obj,
                                ts_event=int(row["ts_event"]),
                                ts_init=int(row.get("ts_init", row["ts_event"])),
                            ),
                        )
                    except Exception:
                        continue
                if not trades:
                    return 0
                start_ns = min(trade.ts_init for trade in trades)
                end_ns = max(trade.ts_init for trade in trades)
                identifier = self._resolve_identifier(
                    dataset_type=dataset_type,
                    instrument_id=trades[0].instrument_id.value,
                )
                self._prune_overlaps(
                    data_cls=_TradeTick,
                    identifier=identifier,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
                if hasattr(self._catalog, "_write_chunk"):
                    self._catalog._write_chunk(
                        data=trades,
                        data_cls=_TradeTick,
                        identifier=identifier,
                        start=start_ns,
                        end=end_ns,
                    )
                else:
                    self._catalog.write_data(trades)
                return len(trades)

        # Unsupported conversion for other raw types
        return 0


class FilteredRawWriter(RawIngestionWriterProtocol):
    """
    Raw writer wrapper that skips writes for disabled dataset types.
    """

    def __init__(
        self,
        writer: RawIngestionWriterProtocol,
        enabled: Mapping[DatasetType, bool],
    ) -> None:
        self._writer = writer
        self._enabled = dict(enabled)

    def is_enabled(self, dataset_type: DatasetType) -> bool:
        """
        Return whether dual-write is enabled for the dataset type.
        """
        return self._enabled.get(dataset_type, True)

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int:
        if not self.is_enabled(dataset_type):
            return 0
        return self._writer.write(dataset_type=dataset_type, data=data)


__all__ = [
    "FilteredRawWriter",
    "ParquetCatalogRawReader",
    "ParquetCatalogRawWriter",
    "RawIngestionWriterProtocol",
    "RawReaderProtocol",
]
