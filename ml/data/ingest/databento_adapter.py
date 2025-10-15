"""
Databento API adapter implementing the DatabentoLikeClient protocol.

This adapter provides a minimal bridge to the `databento` Python client for
historical window fetches, converting results to a normalized pandas DataFrame
with canonical columns for the SQL writer.

Notes:
- Focused on OHLCV bars (`schema='bars'` or equivalent). Extend as needed for quotes/trades.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from ml.data.ingest.policy import DatabentoCoveragePolicy
from ml.data.ingest.resume import DatabentoLikeClient


if TYPE_CHECKING:  # pragma: no cover - type hints only
    from ml.data.ingest.service import DatabentoMetadataClient
    from ml.data.ingest.symbology import DatabentoSymbologyClient


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DatabentoAPIClient(DatabentoLikeClient):
    """
    Minimal Databento client adapter implementing DatabentoLikeClient.
    """

    api_key: str

    def __post_init__(self) -> None:
        try:
            # Lazy import to keep package optional at import time
            import databento as _databento

            self._db = _databento.Historical(self.api_key)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Failed to initialize Databento historical client; install 'databento' and set API key.",
            ) from exc

    def get_data(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        # Apply policy guard (opt-in via env; no-ops if unset)
        policy = DatabentoCoveragePolicy.from_env()
        # Validate dataset/schema against allowlists (if configured)
        policy.validate_dataset_schema(dataset=dataset, schema=schema)
        # Filter symbols and enforce max count
        symbols = policy.filter_symbols(symbols)
        if not symbols:
            return pd.DataFrame()
        symbol = symbols[0]
        # Normalize start/end to aware UTC datetimes for clamping
        s_dt = datetime.fromisoformat(start) if isinstance(start, str) else start
        e_dt = datetime.fromisoformat(end) if isinstance(end, str) else end
        if s_dt.tzinfo is None:
            s_dt = s_dt.replace(tzinfo=UTC)
        if e_dt.tzinfo is None:
            e_dt = e_dt.replace(tzinfo=UTC)
        s_dt, e_dt = policy.clamp_range(s_dt, e_dt, dataset=dataset, schema=schema)
        # Pass original types through to Databento (ISO strings are fine)
        start = s_dt.isoformat()
        end = e_dt.isoformat()
        # Fetch as DataFrame; adjust parameters as needed per API version
        result: Any = self._db.timeseries.get_range(
            dataset=dataset,
            symbols=symbol,
            schema=schema,
            start=start,
            end=end,
            **kwargs,
        )
        # Convert to pandas if the client returned a DBN store object
        df: pd.DataFrame
        if hasattr(result, "to_df"):
            df = result.to_df()
        else:
            df = pd.DataFrame(result)
        # Normalize timestamps to 'ts_event' in ns if present under another column
        if "ts_event" not in df.columns:
            if "ts" in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df["ts"]):
                    df["ts_event"] = df["ts"].astype("int64")
                else:
                    try:
                        df["ts_event"] = pd.to_datetime(df["ts"], utc=True).astype("int64")
                    except Exception as conversion_exc:  # pragma: no cover - defensive
                        logger.debug(
                            "Failed to parse Databento 'ts' column",
                            exc_info=True,
                            extra={"error": repr(conversion_exc)},
                        )
        # Map schema-specific columns to canonical writer columns where possible
        s = schema.lower()
        if "tbbo" in s or "quote" in s:
            # Quotes: bid/ask and sizes
            if "bid" not in df.columns:
                for col in ("bid_px", "bid_price"):
                    if col in df.columns:
                        df["bid"] = df[col]
                        break
            if "ask" not in df.columns:
                for col in ("ask_px", "ask_price"):
                    if col in df.columns:
                        df["ask"] = df[col]
                        break
            if "bid_size" not in df.columns:
                for col in ("bid_sz", "bid_size"):
                    if col in df.columns:
                        df["bid_size"] = df[col]
                        break
            if "ask_size" not in df.columns:
                for col in ("ask_sz", "ask_size"):
                    if col in df.columns:
                        df["ask_size"] = df[col]
                        break
        if "trade" in s:
            # Trades: last/trade_count/vwap
            if "last" not in df.columns:
                for col in ("price", "trade_px", "last_price"):
                    if col in df.columns:
                        df["last"] = df[col]
                        break
            if "trade_count" not in df.columns:
                # Each row represents one trade if not provided
                df["trade_count"] = 1
        if "ts_event" in df.columns:
            event_series = df["ts_event"]
            if not pd.api.types.is_integer_dtype(event_series):
                converted = pd.to_datetime(event_series, utc=True, errors="coerce")
                if pd.api.types.is_datetime64_any_dtype(converted) and converted.notna().all():
                    df["ts_event"] = converted.astype("int64")
                else:
                    try:
                        df["ts_event"] = pd.to_numeric(event_series, errors="raise").astype("int64")
                    except Exception as numeric_exc:  # pragma: no cover - defensive
                        logger.debug(
                            "Failed to coerce Databento 'ts_event' column",
                            exc_info=True,
                            extra={"error": repr(numeric_exc)},
                        )
        if "ts_init" in df.columns:
            init_series = df["ts_init"]
            if not pd.api.types.is_integer_dtype(init_series):
                converted_init = pd.to_datetime(init_series, utc=True, errors="coerce")
                if pd.api.types.is_datetime64_any_dtype(converted_init) and converted_init.notna().all():
                    df["ts_init"] = converted_init.astype("int64")
                else:
                    try:
                        df["ts_init"] = pd.to_numeric(init_series, errors="raise").astype("int64")
                    except Exception as init_numeric_exc:  # pragma: no cover - defensive
                        logger.debug(
                            "Failed to coerce Databento 'ts_init' column",
                            exc_info=True,
                            extra={"error": repr(init_numeric_exc)},
                        )
        elif "ts_event" in df.columns:
            df["ts_init"] = df["ts_event"]
        return df

    @property
    def metadata_client(self) -> DatabentoMetadataClient:
        """Expose the underlying metadata client for discovery helpers."""
        return cast("DatabentoMetadataClient", self._db.metadata)

    @property
    def symbology_client(self) -> DatabentoSymbologyClient | None:
        """Expose the underlying symbology client when available."""
        return cast("DatabentoSymbologyClient | None", getattr(self._db, "symbology", None))
