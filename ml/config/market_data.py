"""
Market data feed descriptors and binding inputs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import msgspec

from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.schema import map_schema_to_dataset_type


DEFAULT_FEED_DESCRIPTOR_PATH = Path(__file__).with_name("market_feed_descriptors.json")


class MarketFeedDescriptor(msgspec.Struct, kw_only=True, frozen=True):
    """
    Declarative feed descriptor describing a raw market data source.
    """

    descriptor_id: str
    dataset_id: str
    provider_dataset_id: str | None = None
    provider_schema: str | None = None
    storage_kind: StorageKind
    schema: str
    symbol_patterns: tuple[str, ...]
    instrument_id_templates: tuple[str, ...]
    license_start: str | None = None
    license_end: str | None = None


class MarketDatasetInput(msgspec.Struct, kw_only=True, frozen=False):
    """
    Dataset build configuration entry referencing a feed descriptor.
    """

    descriptor_id: str | None = None
    dataset_id: str | None = None
    provider_dataset_id: str | None = None
    provider_schema: str | None = None
    symbols: tuple[str, ...] | None = None
    schema: str | None = None
    schema_override: str | None = None
    storage_kind_override: StorageKind | None = None
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if not self.descriptor_id and not self.dataset_id:
            msg = "MarketDatasetInput requires descriptor_id or dataset_id"
            raise ValueError(msg)
        if self.schema is not None and self.schema_override is None:
            self.schema_override = self.schema


@dataclass(slots=True, frozen=True)
class MarketFeedDescriptorSet:
    """
    Wrapper for serialized feed descriptors.
    """

    descriptors: tuple[MarketFeedDescriptor, ...]

    def as_mapping(self) -> Mapping[str, MarketFeedDescriptor]:
        return {descriptor.descriptor_id: descriptor for descriptor in self.descriptors}


def load_market_feed_descriptors(path: Path | None = None) -> MarketFeedDescriptorSet:
    """
    Load feed descriptors from JSON file into typed structures.
    """
    descriptor_path = path or DEFAULT_FEED_DESCRIPTOR_PATH
    raw = json.loads(descriptor_path.read_text(encoding="utf-8"))
    items = raw.get("descriptors", [])
    descriptors = tuple(
        msgspec.convert(
            item,
            type=MarketFeedDescriptor,
            strict=True,
        )
        for item in items
    )
    return MarketFeedDescriptorSet(descriptors=descriptors)


def coerce_storage_kind(value: StorageKind | str | None) -> StorageKind | None:
    """
    Normalize a storage kind value into the :class:`StorageKind` enum.

    Accepts enum members, enum names, or enum values in any case. Strings with
    the ``StorageKind.`` prefix (e.g. ``"StorageKind.POSTGRES"``) are also
    supported to account for serialized enum representations.

    """
    if value is None:
        return None

    if isinstance(value, StorageKind):
        return value

    token = str(value).strip()
    if not token:
        raise ValueError("storage kind token cannot be empty")

    lowered = token.lower()
    if lowered.startswith("storagekind."):
        lowered = lowered.split(".", maxsplit=1)[1]

    try:
        return StorageKind(lowered)
    except ValueError:
        try:
            return StorageKind[lowered.upper()]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown storage kind: {value!r}") from exc


class MarketDataTableProfile(str, Enum):
    """
    Routing profile for SQL market data tables.
    """

    AUTO = "auto"
    LEGACY = "legacy"
    CLASS_TABLES = "class_tables"

    @classmethod
    def from_env(cls, value: str | None) -> MarketDataTableProfile:
        """
        Parse a routing profile from an environment value.

        Args:
            value: Environment value (e.g., "auto", "legacy", "class_tables").

        Returns:
            Resolved MarketDataTableProfile.

        Raises:
            ValueError: If the value is not recognized.
        """
        if value is None:
            return cls.AUTO
        token = value.strip().lower()
        if token in {"auto"}:
            return cls.AUTO
        if token in {"legacy", "monolithic", "market_data"}:
            return cls.LEGACY
        if token in {"class_tables", "class", "per_class", "per-class"}:
            return cls.CLASS_TABLES
        raise ValueError(f"Unknown market data profile: {value!r}")


@dataclass(frozen=True, slots=True)
class MarketDataTableConfig:
    """
    Configuration for SQL market data table routing.

    Attributes:
        profile: Routing profile (auto, legacy, class_tables).
        legacy_table: Legacy monolithic table name.
        bar_table: Per-class table for bars.
        quote_tick_table: Per-class table for quote ticks.
        tbbo_table: Per-class table for TBBO data.
        mbp1_table: Per-class table for MBP-1 data.
        trade_tick_table: Per-class table for trade ticks.
        quote_sentinel_price: Price sentinel to treat as null (e.g., vendor missing-price marker).
    """

    profile: MarketDataTableProfile = MarketDataTableProfile.AUTO
    legacy_table: str = "market_data"
    bar_table: str = "market_data_bar"
    quote_tick_table: str = "market_data_quote_tick"
    tbbo_table: str = "market_data_tbbo"
    mbp1_table: str = "market_data_mbp1"
    trade_tick_table: str = "market_data_trade_tick"
    write_batch_size: int = 10000
    quote_sentinel_price: float | None = 9_223_372_036.85

    def __post_init__(self) -> None:
        for label, value in (
            ("legacy_table", self.legacy_table),
            ("bar_table", self.bar_table),
            ("quote_tick_table", self.quote_tick_table),
            ("tbbo_table", self.tbbo_table),
            ("mbp1_table", self.mbp1_table),
            ("trade_tick_table", self.trade_tick_table),
        ):
            if not value or not value.strip():
                raise ValueError(f"{label} must be non-empty")
        if self.write_batch_size < 1:
            raise ValueError("write_batch_size must be >= 1")

    def table_for_dataset_type(self, dataset_type: DatasetType) -> str:
        """
        Resolve the table name for a dataset type.

        Args:
            dataset_type: Dataset type enum.

        Returns:
            Table name for the dataset type.
        """
        if dataset_type is DatasetType.BARS:
            return self.bar_table
        if dataset_type is DatasetType.TRADES:
            return self.trade_tick_table
        if dataset_type is DatasetType.QUOTES:
            return self.quote_tick_table
        if dataset_type is DatasetType.TBBO:
            return self.tbbo_table
        if dataset_type is DatasetType.MBP1:
            return self.mbp1_table
        return self.legacy_table

    def table_for_schema(self, schema: str) -> str:
        """
        Resolve the table name for a schema token.

        Args:
            schema: Schema identifier (e.g., "ohlcv-1m", "tbbo", "trades").

        Returns:
            Table name for the schema (falls back to legacy table when unknown).
        """
        normalized = schema.strip().lower()
        if not normalized:
            return self.legacy_table
        if "tbbo" in normalized or "bbo" in normalized:
            return self.tbbo_table
        if "quote" in normalized:
            return self.quote_tick_table
        if "trade" in normalized:
            return self.trade_tick_table
        if "mbp" in normalized or "mbo" in normalized or normalized.startswith("l2"):
            return self.mbp1_table
        if "ohlcv" in normalized or "bar" in normalized:
            return self.bar_table
        try:
            dataset_type = map_schema_to_dataset_type(schema)
        except ValueError:
            return self.legacy_table
        return self.table_for_dataset_type(dataset_type)

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        legacy_table: str | None = None,
    ) -> MarketDataTableConfig:
        """
        Build configuration from environment variables.

        Environment overrides:
            ML_MARKET_DATA_PROFILE
            ML_MARKET_DATA_TABLE_LEGACY
            ML_MARKET_DATA_TABLE_BARS
            ML_MARKET_DATA_TABLE_QUOTES
            ML_MARKET_DATA_TABLE_TBBO
            ML_MARKET_DATA_TABLE_MBP1
            ML_MARKET_DATA_TABLE_TRADES
            ML_MARKET_DATA_WRITE_BATCH_SIZE
            ML_MARKET_DATA_QUOTE_SENTINEL_PRICE

        Args:
            env: Optional environment mapping (defaults to os.environ).
            legacy_table: Optional legacy table override when env var is unset.

        Returns:
            MarketDataTableConfig populated from environment values.
        """
        source = env or os.environ
        profile = MarketDataTableProfile.from_env(source.get("ML_MARKET_DATA_PROFILE"))
        defaults = cls()
        batch_size_raw = source.get("ML_MARKET_DATA_WRITE_BATCH_SIZE")
        write_batch_size = defaults.write_batch_size
        if batch_size_raw is not None and batch_size_raw.strip():
            try:
                write_batch_size = int(batch_size_raw)
            except ValueError:
                raise ValueError(
                    "ML_MARKET_DATA_WRITE_BATCH_SIZE must be an integer",
                ) from None
        quote_sentinel_raw = source.get("ML_MARKET_DATA_QUOTE_SENTINEL_PRICE")
        quote_sentinel_price = defaults.quote_sentinel_price
        if quote_sentinel_raw is not None:
            token = quote_sentinel_raw.strip()
            if not token:
                quote_sentinel_price = None
            else:
                try:
                    quote_sentinel_price = float(token)
                except ValueError:
                    raise ValueError(
                        "ML_MARKET_DATA_QUOTE_SENTINEL_PRICE must be a float",
                    ) from None
        return cls(
            profile=profile,
            legacy_table=source.get("ML_MARKET_DATA_TABLE_LEGACY")
            or legacy_table
            or defaults.legacy_table,
            bar_table=source.get("ML_MARKET_DATA_TABLE_BARS") or defaults.bar_table,
            quote_tick_table=source.get("ML_MARKET_DATA_TABLE_QUOTES") or defaults.quote_tick_table,
            tbbo_table=source.get("ML_MARKET_DATA_TABLE_TBBO") or defaults.tbbo_table,
            mbp1_table=source.get("ML_MARKET_DATA_TABLE_MBP1") or defaults.mbp1_table,
            trade_tick_table=source.get("ML_MARKET_DATA_TABLE_TRADES") or defaults.trade_tick_table,
            write_batch_size=write_batch_size,
            quote_sentinel_price=quote_sentinel_price,
        )


__all__ = [
    "DEFAULT_FEED_DESCRIPTOR_PATH",
    "MarketDataTableConfig",
    "MarketDataTableProfile",
    "MarketDatasetInput",
    "MarketFeedDescriptor",
    "MarketFeedDescriptorSet",
    "coerce_storage_kind",
    "load_market_feed_descriptors",
]
