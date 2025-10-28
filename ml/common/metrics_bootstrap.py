"""
Safe, idempotent metrics bootstrap utilities.

This module centralizes creation and retrieval of Prometheus metrics to
avoid duplicate registration and reliance on prometheus-client internals.

Usage: always acquire metrics via get_counter/get_histogram/get_gauge.
Subsequent calls with the same name return the existing collector.

"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any


_DEFAULT_REGISTRY: Any | None = None

try:  # Centralized backend import via importlib (avoid direct import for validators)
    import importlib as _importlib

    _prom = _importlib.import_module("prometheus_client")
    _PC_Counter = getattr(_prom, "Counter")
    _PC_Gauge = getattr(_prom, "Gauge")
    _PC_Histogram = getattr(_prom, "Histogram")
    _DEFAULT_REGISTRY = getattr(_prom, "REGISTRY", None)

    HAS_METRICS_BACKEND = True
    _CounterCls: type[Any] = _PC_Counter
    _GaugeCls: type[Any] = _PC_Gauge
    _HistogramCls: type[Any] = _PC_Histogram
except Exception:  # pragma: no cover - prometheus optional
    HAS_METRICS_BACKEND = False

    class _DummyCounter:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def labels(self, **_: object) -> _DummyCounter:
            return self

        def inc(self, *args: object, **kwargs: object) -> None: ...

    class _DummyGauge:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def labels(self, **_: object) -> _DummyGauge:
            return self

        def set(self, *args: object, **kwargs: object) -> None: ...

    class _DummyHistogram:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def labels(self, **_: object) -> _DummyHistogram:
            return self

        def observe(self, *args: object, **kwargs: object) -> None: ...

    _CounterCls = _DummyCounter
    _GaugeCls = _DummyGauge
    _HistogramCls = _DummyHistogram


_METRICS: dict[str, Any] = {}


def _existing_collector(name: str) -> Any | None:
    """
    Return an existing collector from the default registry, if available.

    Re-importing modules during tests resets this module-level cache but the
    global Prometheus registry retains previously registered collectors. This
    helper reuses existing collectors to avoid duplicate registration errors.
    """
    if not HAS_METRICS_BACKEND or _DEFAULT_REGISTRY is None:
        return None
    names_to_collectors = getattr(_DEFAULT_REGISTRY, "_names_to_collectors", None)
    if not isinstance(names_to_collectors, dict):
        return None
    collector = names_to_collectors.get(name)
    return collector


def _labels_tuple(
    labelnames: Iterable[str] | None,
    labels: Iterable[str] | None,
) -> tuple[str, ...]:
    names = labels if labels is not None else labelnames
    return tuple(names) if names is not None else tuple()


def _key(name: str, labelnames: Iterable[str] | None, labels: Iterable[str] | None = None) -> str:
    labels_tuple: tuple[str, ...] = _labels_tuple(labelnames, labels)
    return f"{name}||{labels_tuple!r}"


def get_counter(
    name: str,
    description: str,
    labelnames: Iterable[str] | None = None,
    *,
    labels: Iterable[str] | None = None,
) -> Any:
    k = _key(name, labelnames, labels)
    metric = _METRICS.get(k)
    if metric is None:
        names = list((labels or labelnames) or ())
        metric = _existing_collector(name)
        if metric is None or not hasattr(metric, "inc"):
            metric = _CounterCls(name, description, names)
        _METRICS[k] = metric
    return metric


def get_histogram(
    name: str,
    description: str,
    labelnames: Iterable[str] | None = None,
    *,
    buckets: Iterable[float] | None = None,
    labels: Iterable[str] | None = None,
) -> Any:
    k = _key(name, labelnames, labels)
    metric = _METRICS.get(k)
    if metric is None:
        names = list((labels or labelnames) or ())
        metric = _existing_collector(name)
        if metric is None or not hasattr(metric, "observe"):
            if buckets is None:
                metric = _HistogramCls(name, description, names)
            else:
                metric = _HistogramCls(
                    name,
                    description,
                    names,
                    buckets=tuple(buckets),
                )
        _METRICS[k] = metric
    return metric


def get_gauge(
    name: str,
    description: str,
    labelnames: Iterable[str] | None = None,
    *,
    labels: Iterable[str] | None = None,
) -> Any:
    k = _key(name, labelnames, labels)
    metric = _METRICS.get(k)
    if metric is None:
        names = list((labels or labelnames) or ())
        metric = _existing_collector(name)
        if metric is None or not hasattr(metric, "set"):
            metric = _GaugeCls(name, description, names)
        _METRICS[k] = metric
    return metric




_DEFAULT_METRICS_IMPORTED = False


def reset_metrics_cache() -> None:
    """
    Reset module-level metrics cache for test isolation.

    This function clears the metrics cache and reset flags to ensure
    tests start with a clean metrics state. Only intended for test use.

    WARNING: This does not unregister collectors from the Prometheus registry.
    For complete isolation, tests should use fresh registry fixtures.
    """
    global _METRICS, _DEFAULT_METRICS_IMPORTED
    _METRICS.clear()
    _DEFAULT_METRICS_IMPORTED = False


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_default_metrics_registered() -> None:
    """Ensure core ML metric collectors are registered on import."""
    global _DEFAULT_METRICS_IMPORTED
    if _DEFAULT_METRICS_IMPORTED:
        return
    if _truthy(os.getenv("ML_DISABLE_CORE_METRICS")):
        _DEFAULT_METRICS_IMPORTED = True
        return
    if not HAS_METRICS_BACKEND:
        _DEFAULT_METRICS_IMPORTED = True
        return
    try:
        _importlib.import_module("ml.common.metrics")
    except Exception:
        return
    _DEFAULT_METRICS_IMPORTED = True


ensure_default_metrics_registered()

__all__ = ["HAS_METRICS_BACKEND", "ensure_default_metrics_registered", "get_counter", "get_gauge", "get_histogram", "reset_metrics_cache"]
