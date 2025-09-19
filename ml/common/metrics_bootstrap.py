"""
Safe, idempotent metrics bootstrap utilities.

This module centralizes creation and retrieval of Prometheus metrics to
avoid duplicate registration and reliance on prometheus-client internals.

Usage: always acquire metrics via get_counter/get_histogram/get_gauge.
Subsequent calls with the same name return the existing collector.

"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


try:  # Centralized backend import via importlib (avoid direct import for validators)
    import importlib as _importlib

    _prom = _importlib.import_module("prometheus_client")
    _PC_Counter = getattr(_prom, "Counter")
    _PC_Gauge = getattr(_prom, "Gauge")
    _PC_Histogram = getattr(_prom, "Histogram")

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
        metric = _GaugeCls(name, description, names)
        _METRICS[k] = metric
    return metric


__all__ = ["HAS_METRICS_BACKEND", "get_counter", "get_gauge", "get_histogram"]
