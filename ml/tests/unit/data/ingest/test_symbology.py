from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from databento.common.error import BentoServerError

from ml.data.ingest.symbology import DatabentoSymbologyResolver
from ml.data.ingest.symbology import SymbologyResolutionError


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def _metric_sample_value(
    isolated_prometheus_registry: Any,
    metric_name: str,
    labels: dict[str, str],
) -> float:
    sample = isolated_prometheus_registry.registry.get_sample_value(
        metric_name,
        labels=labels,
    )
    if sample is None:
        return 0.0
    return float(sample)


class _AliasAwareClient:
    def __init__(self, successes: dict[str, str]) -> None:
        self._successes = {symbol.upper(): inst for symbol, inst in successes.items()}
        self.calls: list[tuple[str, str]] = []

    def resolve(
        self,
        *,
        symbols: list[str] | tuple[str, ...],
        dataset: str,
        stype_in: str,
        stype_out: str,
        start_date: str,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        del stype_in, stype_out, start_date, end_date
        symbol = symbols[0].upper()
        self.calls.append((dataset, symbol))
        if symbol not in self._successes:
            raise SymbologyResolutionError(f"Symbol {symbol} not found")
        instrument_id = self._successes[symbol]
        return {
            "result": {
                symbol: (
                    {
                        "s": instrument_id,
                    },
                ),
            },
        }


def test_resolver_uses_alias_for_brk(isolated_prometheus_registry: Any) -> None:
    client = _AliasAwareClient({"BRK.B": "991"})
    resolver = DatabentoSymbologyResolver(client=client)
    labels = {"dataset": "EQUS.MINI"}
    before = _metric_sample_value(
        isolated_prometheus_registry,
        "nautilus_ml_symbology_alias_hits_total",
        labels,
    )
    result = resolver.resolve(dataset="EQUS.MINI", symbol="BRK")
    assert result.preferred == "BRK.B"
    assert result.instrument_id == "991"
    assert client.calls == [("EQUS.MINI", "BRK"), ("EQUS.MINI", "BRK.B")]
    after = _metric_sample_value(
        isolated_prometheus_registry,
        "nautilus_ml_symbology_alias_hits_total",
        labels,
    )
    assert after - before == 1.0


def test_resolver_alias_fallback_propagates_when_missing() -> None:
    client = _AliasAwareClient({})
    resolver = DatabentoSymbologyResolver(client=client)
    with pytest.raises(SymbologyResolutionError):
        resolver.resolve(
            dataset="EQUS.MINI",
            symbol="BRK",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 1, 2, tzinfo=UTC),
        )


class _FlakyServerClient:
    def __init__(self, *, fail_attempts: int, instrument_id: str) -> None:
        self._remaining_failures = fail_attempts
        self._instrument_id = instrument_id
        self.calls = 0

    def resolve(
        self,
        *,
        symbols: list[str] | tuple[str, ...],
        dataset: str,
        stype_in: str,
        stype_out: str,
        start_date: str,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        del dataset, stype_in, stype_out, start_date, end_date
        self.calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise BentoServerError(http_status=502, message="<temporary>")
        symbol = symbols[0].upper()
        return {
            "result": {
                symbol: (
                    {
                        "s": self._instrument_id,
                    },
                ),
            },
        }


def test_resolver_retries_server_error_then_succeeds(isolated_prometheus_registry: Any) -> None:
    client = _FlakyServerClient(fail_attempts=1, instrument_id="123")
    resolver = DatabentoSymbologyResolver(
        client=client,
        retry_attempts=3,
        retry_backoff_seconds=0.0,
    )
    labels = {"dataset": "EQUS.MINI", "status": "502"}
    before = _metric_sample_value(
        isolated_prometheus_registry,
        "nautilus_ml_symbology_retry_total",
        labels,
    )
    result = resolver.resolve(dataset="EQUS.MINI", symbol="DIS")
    assert result.preferred == "DIS"
    assert result.instrument_id == "123"
    assert client.calls == 2
    after = _metric_sample_value(
        isolated_prometheus_registry,
        "nautilus_ml_symbology_retry_total",
        labels,
    )
    assert after - before == 1.0


def test_resolver_raises_after_retry_budget_exhausted(isolated_prometheus_registry: Any) -> None:
    client = _FlakyServerClient(fail_attempts=3, instrument_id="123")
    resolver = DatabentoSymbologyResolver(
        client=client,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    )
    labels = {"dataset": "EQUS.MINI", "status": "502"}
    before = _metric_sample_value(
        isolated_prometheus_registry,
        "nautilus_ml_symbology_retry_total",
        labels,
    )
    with pytest.raises(SymbologyResolutionError):
        resolver.resolve(dataset="EQUS.MINI", symbol="DIS")
    assert client.calls == 2
    after = _metric_sample_value(
        isolated_prometheus_registry,
        "nautilus_ml_symbology_retry_total",
        labels,
    )
    assert after - before == 1.0
