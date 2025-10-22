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

import logging
from collections.abc import Iterable
from typing import Any, cast

from ml.data.catalog_utils import bars_to_dataframe
from ml.data.catalog_utils import quotes_to_dataframe
from ml.data.catalog_utils import trades_to_dataframe
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetType
from ml.stores.raw_protocols import RawIngestionWriterProtocol
from ml.stores.raw_protocols import RawReaderProtocol


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

    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog

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
            self._catalog.write_data(items)
            return len(data)

        # Conversion path for bars/quotes/trades: DataFrame-like or list[dict]
        if dataset_type in (DatasetType.BARS, DatasetType.QUOTES, DatasetType.TRADES):
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
                bars: list[_Bar] = []
                for row in rows:
                    inst = str(row.get("instrument_id", "UNKNOWN"))
                    bt = _BarType.from_str(f"{inst}-1-MINUTE-LAST-EXTERNAL")
                    try:
                        bars.append(
                            _Bar(
                                bar_type=bt,
                                open=_Price(float(row["open"]), precision=6),
                                high=_Price(float(row["high"]), precision=6),
                                low=_Price(float(row["low"]), precision=6),
                                close=_Price(float(row["close"]), precision=6),
                                volume=_Quantity(float(row.get("volume", 0.0)), precision=0),
                                ts_event=int(row["ts_event"]),
                                ts_init=int(row.get("ts_init", row["ts_event"])),
                            ),
                        )
                    except Exception as exc:
                        logger.debug(
                            "raw_writer.bar_parse_failed instrument=%s error=%s",
                            inst,
                            exc,
                            exc_info=True,
                        )
                        continue
                if not bars:
                    return 0
                self._catalog.write_data(bars)
                return len(bars)

            if dataset_type == DatasetType.QUOTES:
                quotes: list[_QuoteTick] = []
                for row in rows:
                    try:
                        inst = _InstrumentId.from_str(str(row.get("instrument_id", "UNKNOWN")))
                        bid = _Price(float(row["bid"]), precision=6)
                        ask = _Price(float(row["ask"]), precision=6)
                        bsz = _Quantity(float(row.get("bid_size", 0.0)), precision=0)
                        asz = _Quantity(float(row.get("ask_size", 0.0)), precision=0)
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
                    except Exception as exc:
                        logger.debug(
                            "raw_writer.quote_parse_failed row=%s error=%s",
                            row,
                            exc,
                            exc_info=True,
                        )
                        continue
                if not quotes:
                    return 0
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
                    except Exception as exc:
                        logger.debug(
                            "raw_writer.trade_parse_failed row=%s error=%s",
                            row,
                            exc,
                            exc_info=True,
                        )
                        continue
                if not trades:
                    return 0
                self._catalog.write_data(trades)
                return len(trades)

        # Unsupported conversion for other raw types
        return 0


__all__ = [
    "ParquetCatalogRawReader",
    "ParquetCatalogRawWriter",
    "RawIngestionWriterProtocol",
    "RawReaderProtocol",
]
logger = logging.getLogger(__name__)
