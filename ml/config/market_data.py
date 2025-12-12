"""Market data feed descriptors and binding inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import msgspec

from ml.registry.dataclasses import StorageKind


DEFAULT_FEED_DESCRIPTOR_PATH = Path(__file__).with_name("market_feed_descriptors.json")


class MarketFeedDescriptor(msgspec.Struct, kw_only=True, frozen=True):
    """Declarative feed descriptor describing a raw market data source."""

    descriptor_id: str
    dataset_id: str
    storage_kind: StorageKind
    schema: str
    symbol_patterns: tuple[str, ...]
    instrument_id_templates: tuple[str, ...]
    license_start: str | None = None
    license_end: str | None = None


class MarketDatasetInput(msgspec.Struct, kw_only=True, frozen=False):
    """Dataset build configuration entry referencing a feed descriptor."""

    descriptor_id: str | None = None
    dataset_id: str | None = None
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
    """Wrapper for serialized feed descriptors."""

    descriptors: tuple[MarketFeedDescriptor, ...]

    def as_mapping(self) -> Mapping[str, MarketFeedDescriptor]:
        return {descriptor.descriptor_id: descriptor for descriptor in self.descriptors}


def load_market_feed_descriptors(path: Path | None = None) -> MarketFeedDescriptorSet:
    """Load feed descriptors from JSON file into typed structures."""
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


__all__ = [
    "DEFAULT_FEED_DESCRIPTOR_PATH",
    "MarketDatasetInput",
    "MarketFeedDescriptor",
    "MarketFeedDescriptorSet",
    "coerce_storage_kind",
    "load_market_feed_descriptors",
]
