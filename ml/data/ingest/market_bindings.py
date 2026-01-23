"""Helpers for resolving market data bindings across feeds."""

from __future__ import annotations

import fnmatch
import itertools
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field

from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import MarketFeedDescriptor
from ml.registry.dataclasses import StorageKind


@dataclass(frozen=True)
class ResolvedMarketBinding:
    """Binding between symbols/instruments and a concrete dataset."""

    binding_id: str
    symbol: str
    instrument_ids: tuple[str, ...]
    dataset_id: str
    descriptor_id: str | None
    schema: str | None
    storage_kind: StorageKind | None
    license_start: str | None
    license_end: str | None
    start: str | None
    end: str | None
    source: str
    provider_dataset_id: str | None = None


@dataclass(slots=True)
class MarketBindingStats:
    """Mutable statistics collected during dataset preparation."""

    binding_id: str
    dataset_id: str
    descriptor_id: str | None
    symbol: str
    instrument_ids: tuple[str, ...]
    schema: str | None
    storage_kind: StorageKind | None
    source: str
    license_start: str | None
    license_end: str | None
    provider_dataset_id: str | None = None
    rows_from_store: int = 0
    rows_from_catalog: int = 0
    ts_event_start_ns: int | None = None
    ts_event_end_ns: int | None = None
    source_datasets: set[str] = field(default_factory=set)

    def record(
        self,
        *,
        source: str,
        row_count: int,
        ts_min_ns: int | None,
        ts_max_ns: int | None,
        source_dataset: str | None = None,
    ) -> None:
        if row_count <= 0:
            return
        if source == "store":
            self.rows_from_store += row_count
        else:
            self.rows_from_catalog += row_count
        self._update_bounds(ts_min_ns, ts_max_ns)
        if source_dataset:
            self.source_datasets.add(source_dataset)

    def _update_bounds(self, ts_min_ns: int | None, ts_max_ns: int | None) -> None:
        if ts_min_ns is not None:
            if self.ts_event_start_ns is None or ts_min_ns < self.ts_event_start_ns:
                self.ts_event_start_ns = ts_min_ns
        if ts_max_ns is not None:
            if self.ts_event_end_ns is None or ts_max_ns > self.ts_event_end_ns:
                self.ts_event_end_ns = ts_max_ns


def resolve_market_dataset_bindings(
    *,
    symbols: Sequence[str],
    instrument_ids: Sequence[str] | None,
    market_dataset_id: str | None,
    market_inputs: Sequence[MarketDatasetInput] | None,
    descriptors: Mapping[str, MarketFeedDescriptor],
) -> tuple[ResolvedMarketBinding, ...]:
    """Resolve configured feeds into concrete dataset bindings."""
    normalized_symbols = _normalize_symbols(symbols)
    instrument_lookup = _build_instrument_lookup(instrument_ids)

    bindings: list[ResolvedMarketBinding] = []
    counter = itertools.count(1)
    assigned: dict[str, list[ResolvedMarketBinding]] = {symbol: [] for symbol in normalized_symbols}

    for raw in market_inputs or ():
        matched_symbols = _symbols_for_input(raw, normalized_symbols, instrument_lookup, descriptors)
        if not matched_symbols:
            continue
        descriptor = descriptors.get(raw.descriptor_id or "") if raw.descriptor_id else None
        dataset_id = raw.dataset_id or (descriptor.dataset_id if descriptor else None)
        if dataset_id is None:
            msg = "MarketDatasetInput requires dataset_id when descriptor lacks dataset mapping"
            raise ValueError(msg)
        provider_dataset_id = (
            raw.provider_dataset_id
            or (descriptor.provider_dataset_id if descriptor else None)
            or (descriptor.dataset_id if descriptor else None)
            or dataset_id
        )
        schema = raw.schema_override or (descriptor.schema if descriptor else None)
        storage_kind = raw.storage_kind_override or (descriptor.storage_kind if descriptor else None)
        for symbol in matched_symbols:
            instruments = _resolve_instruments(symbol, instrument_lookup, descriptor)
            binding_id = _binding_id(next(counter), dataset_id, symbol, raw.descriptor_id)
            binding = ResolvedMarketBinding(
                binding_id=binding_id,
                symbol=symbol,
                instrument_ids=instruments,
                dataset_id=dataset_id,
                descriptor_id=raw.descriptor_id,
                schema=schema,
                storage_kind=storage_kind,
                license_start=descriptor.license_start if descriptor else None,
                license_end=descriptor.license_end if descriptor else None,
                start=raw.start,
                end=raw.end,
                source="descriptor",
                provider_dataset_id=provider_dataset_id,
            )
            assigned.setdefault(symbol, []).append(binding)

    if market_dataset_id:
        for symbol in normalized_symbols:
            if assigned.get(symbol):
                continue
            binding_id = _binding_id(next(counter), market_dataset_id, symbol, None)
            instruments = _resolve_instruments(symbol, instrument_lookup, None)
            binding = ResolvedMarketBinding(
                binding_id=binding_id,
                symbol=symbol,
                instrument_ids=instruments,
                dataset_id=market_dataset_id,
                descriptor_id=None,
                schema=None,
                storage_kind=None,
                license_start=None,
                license_end=None,
                start=None,
                end=None,
                source="legacy",
                provider_dataset_id=market_dataset_id,
            )
            assigned.setdefault(symbol, []).append(binding)

    for symbol in normalized_symbols:
        bindings.extend(assigned.get(symbol, []))

    return tuple(bindings)


def _binding_id(counter_value: int, dataset_id: str, symbol: str, descriptor_id: str | None) -> str:
    if descriptor_id:
        return f"{dataset_id}:{symbol}:{descriptor_id}:{counter_value:03d}"
    return f"{dataset_id}:{symbol}:{counter_value:03d}"


def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for symbol in symbols:
        stripped = symbol.strip().upper()
        if not stripped:
            continue
        normalized.append(stripped)
    return tuple(dict.fromkeys(normalized))  # preserve order, deduplicate


def _build_instrument_lookup(instrument_ids: Sequence[str] | None) -> Mapping[str, tuple[str, ...]]:
    if not instrument_ids:
        return {}
    mapping: dict[str, set[str]] = {}
    for item in instrument_ids:
        token = item.strip()
        if not token:
            continue
        upper = token.upper()
        mapping.setdefault(upper, set()).add(upper)
        base = upper.split(".")[0]
        mapping.setdefault(base, set()).add(upper)
    return {key: tuple(sorted(value)) for key, value in mapping.items()}


def _symbols_for_input(
    raw: MarketDatasetInput,
    symbols: Sequence[str],
    instrument_lookup: Mapping[str, tuple[str, ...]],
    descriptors: Mapping[str, MarketFeedDescriptor],
) -> tuple[str, ...]:
    if raw.symbols:
        requested = {symbol.strip().upper() for symbol in raw.symbols if symbol.strip()}
        return tuple(symbol for symbol in symbols if symbol in requested)
    descriptor = descriptors.get(raw.descriptor_id or "") if raw.descriptor_id else None
    if descriptor is None:
        return tuple(symbols)
    matched: list[str] = []
    for symbol in symbols:
        if _matches_descriptor(symbol, descriptor, instrument_lookup):
            matched.append(symbol)
    return tuple(matched)


def _matches_descriptor(
    symbol: str,
    descriptor: MarketFeedDescriptor,
    instrument_lookup: Mapping[str, tuple[str, ...]],
) -> bool:
    base = symbol.split(".")[0]
    candidates = instrument_lookup.get(symbol, ()) + instrument_lookup.get(base, ())
    for pattern in descriptor.symbol_patterns:
        # Instrument-id pattern
        if "." in pattern:
            for candidate in candidates:
                if fnmatch.fnmatchcase(candidate, pattern.upper()):
                    return True
        else:
            if fnmatch.fnmatchcase(base, pattern.upper()):
                return True
    return not descriptor.symbol_patterns


def _resolve_instruments(
    symbol: str,
    instrument_lookup: Mapping[str, tuple[str, ...]],
    descriptor: MarketFeedDescriptor | None,
) -> tuple[str, ...]:
    base = symbol.split(".")[0]
    instruments = set(instrument_lookup.get(symbol, ()))
    instruments.update(instrument_lookup.get(base, ()))

    templates: Iterable[str] = descriptor.instrument_id_templates if descriptor else ()
    for template in templates:
        template_value = template.format(symbol=base)
        instruments.add(template_value.upper())

    if not instruments and "." in symbol:
        instruments.add(symbol.upper())
    if not instruments:
        for suffix in ("XNAS", "XNYS", "ARCX", "NASDAQ", "NYSE"):
            instruments.add(f"{base}.{suffix}")
    return tuple(sorted(instruments))


def resolve_instrument_ids_for_symbols(
    *,
    symbols: Sequence[str],
    descriptor: MarketFeedDescriptor | None,
    instrument_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """
    Resolve instrument IDs for a symbol list using descriptor templates.

    Args:
        symbols: Input symbols (raw or instrument-style).
        descriptor: Optional feed descriptor providing instrument templates.
        instrument_ids: Optional known instrument IDs to seed lookups.

    Returns:
        Tuple of unique instrument IDs (uppercased) in resolution order.

    Example:
        >>> descriptor = MarketFeedDescriptor(
        ...     descriptor_id="EQUS.MINI",
        ...     dataset_id="EQUS.MINI",
        ...     provider_dataset_id="EQUS.MINI",
        ...     storage_kind="postgres",
        ...     schema="ohlcv-1m",
        ...     symbol_patterns=("*",),
        ...     instrument_id_templates=("{symbol}.EQUS",),
        ... )
        >>> resolve_instrument_ids_for_symbols(
        ...     symbols=("AAPL", "MSFT"),
        ...     descriptor=descriptor,
        ... )
        ('AAPL.EQUS', 'MSFT.EQUS')
    """
    normalized = _normalize_symbols(symbols)
    if descriptor is None:
        lookup = _build_instrument_lookup(instrument_ids or normalized)
    else:
        lookup = _build_instrument_lookup(instrument_ids)
    resolved: list[str] = []
    for symbol in normalized:
        resolved.extend(_resolve_instruments(symbol, lookup, descriptor))
    return tuple(dict.fromkeys(resolved))


__all__ = [
    "MarketBindingStats",
    "ResolvedMarketBinding",
    "resolve_instrument_ids_for_symbols",
    "resolve_market_dataset_bindings",
]
