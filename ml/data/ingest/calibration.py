"""Calibration artefacts for EQUS fallback normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class SymbolCalibration:
    """Per-symbol calibration inputs for fallback normalization."""

    sale_condition_allowlist: frozenset[str]
    volume_scale_by_minute: Mapping[int, float]
    price_scaling_by_minute: Mapping[int, float]
    split_events: Mapping[str, float]
    exclude_auction_minutes: frozenset[int]

    def scale_for_minute(self, minute_of_day: int) -> float:
        """Return the volume scale factor for the given minute-of-day (0-1439)."""
        return self.volume_scale_by_minute.get(minute_of_day, 1.0)

    def price_scale_for_minute(self, minute_of_day: int) -> float:
        """Return the price scale factor for a minute-of-day."""
        return self.price_scaling_by_minute.get(minute_of_day, 1.0)


@dataclass(frozen=True)
class CalibrationBundle:
    """Container for all symbol calibrations."""

    generated_at: datetime
    symbols: Mapping[str, SymbolCalibration]

    def for_symbol(self, symbol: str) -> SymbolCalibration | None:
        """Fetch calibration for a symbol (case-insensitive, strip venue suffix)."""
        if not symbol:
            return None
        base_symbol = symbol.split(".")[0].upper()
        return self.symbols.get(base_symbol)


def load_calibration_bundle(path: Path) -> CalibrationBundle:
    """Load calibration artefacts from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    generated_at_raw = payload.get("generated_at")
    if not isinstance(generated_at_raw, str):
        raise ValueError("calibration JSON missing 'generated_at' timestamp")
    generated_at = datetime.fromisoformat(generated_at_raw)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)

    symbols_raw = payload.get("symbols")
    if not isinstance(symbols_raw, Mapping):
        raise ValueError("calibration JSON must contain a 'symbols' object")

    symbols: dict[str, SymbolCalibration] = {}
    for key, value in symbols_raw.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise ValueError("invalid symbol calibration entry")
        allowlist_raw = value.get("sale_condition_allowlist", [])
        if allowlist_raw is None:
            allowlist = frozenset[str]()
        else:
            if not isinstance(allowlist_raw, list) or not all(isinstance(item, str) for item in allowlist_raw):
                raise ValueError("sale_condition_allowlist must be a list of strings")
            allowlist = frozenset(item.strip() for item in allowlist_raw if item.strip())

        volume_scale_raw = value.get("volume_scale_by_minute", {})
        if not isinstance(volume_scale_raw, Mapping):
            raise ValueError("volume_scale_by_minute must be an object mapping minute -> factor")
        volume_scale: dict[int, float] = {}
        for minute_key, factor in volume_scale_raw.items():
            if not isinstance(minute_key, str):
                raise ValueError("minute keys must be strings")
            try:
                minute_int = int(minute_key)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"invalid minute key: {minute_key}") from exc
            if not isinstance(factor, (int, float)):
                raise ValueError("volume scale factors must be numeric")
            if minute_int < 0 or minute_int >= 1440:
                raise ValueError("minute keys must be within 0..1439")
            volume_scale[minute_int] = float(factor)

        price_scale_raw = value.get("price_scale_by_minute", {})
        if not isinstance(price_scale_raw, Mapping):
            raise ValueError("price_scale_by_minute must be an object mapping minute -> factor")
        price_scale: dict[int, float] = {}
        for minute_key, factor in price_scale_raw.items():
            if not isinstance(minute_key, str):
                raise ValueError("price minute keys must be strings")
            try:
                minute_int = int(minute_key)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"invalid price minute key: {minute_key}") from exc
            if not isinstance(factor, (int, float)):
                raise ValueError("price scale factors must be numeric")
            if minute_int < 0 or minute_int >= 1440:
                raise ValueError("price minute keys must be within 0..1439")
            price_scale[minute_int] = float(factor)

        split_events_raw = value.get("split_events", {})
        if not isinstance(split_events_raw, Mapping):
            raise ValueError("split_events must be an object mapping ISO date -> factor")
        split_events: dict[str, float] = {}
        for date_key, factor in split_events_raw.items():
            if not isinstance(date_key, str) or not date_key:
                raise ValueError("split event keys must be ISO date strings")
            if not isinstance(factor, (int, float)):
                raise ValueError("split factors must be numeric")
            split_events[date_key] = float(factor)

        auction_raw = value.get("exclude_auction_minutes", [])
        if auction_raw is None:
            auction_minutes = frozenset[int]()
        else:
            if not isinstance(auction_raw, list) or not all(isinstance(item, int) for item in auction_raw):
                raise ValueError("exclude_auction_minutes must be a list of integers")
            auction_minutes = frozenset(item for item in auction_raw if 0 <= item < 1440)

        symbols[key.upper()] = SymbolCalibration(
            sale_condition_allowlist=allowlist,
            volume_scale_by_minute=volume_scale,
            price_scaling_by_minute=price_scale,
            split_events=split_events,
            exclude_auction_minutes=auction_minutes,
        )

    return CalibrationBundle(generated_at=generated_at, symbols=symbols)


def symbol_calibration_to_mapping(calibration: SymbolCalibration) -> dict[str, object]:
    """Serialise a :class:`SymbolCalibration` to a JSON-compatible mapping."""
    return {
        "sale_condition_allowlist": sorted(calibration.sale_condition_allowlist),
        "volume_scale_by_minute": {
            str(key): float(value)
            for key, value in calibration.volume_scale_by_minute.items()
        },
        "price_scale_by_minute": {
            str(key): float(value)
            for key, value in calibration.price_scaling_by_minute.items()
        },
        "split_events": {
            key: float(value)
            for key, value in calibration.split_events.items()
        },
        "exclude_auction_minutes": sorted(calibration.exclude_auction_minutes),
    }


def calibration_bundle_to_mapping(bundle: CalibrationBundle) -> dict[str, object]:
    """Convert a :class:`CalibrationBundle` into a JSON-serialisable mapping."""
    payload: MutableMapping[str, object] = {
        "generated_at": bundle.generated_at.astimezone(UTC).isoformat(),
        "symbols": {},
    }
    symbols_mapping = cast(MutableMapping[str, object], payload["symbols"])
    for symbol, calibration in bundle.symbols.items():
        symbols_mapping[symbol] = symbol_calibration_to_mapping(calibration)
    return dict(payload)


def dump_calibration_bundle(bundle: CalibrationBundle, path: Path) -> None:
    """Persist a calibration bundle to ``path`` as JSON."""
    payload = calibration_bundle_to_mapping(bundle)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "CalibrationBundle",
    "SymbolCalibration",
    "calibration_bundle_to_mapping",
    "dump_calibration_bundle",
    "load_calibration_bundle",
    "symbol_calibration_to_mapping",
]
