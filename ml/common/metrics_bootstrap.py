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

from ml.common.metrics import Counter
from ml.common.metrics import Gauge
from ml.common.metrics import Histogram


_METRICS: dict[str, Any] = {}


def _key(name: str, labelnames: Iterable[str] | None) -> str:
    labels_tuple: tuple[str, ...] = tuple(labelnames) if labelnames is not None else tuple()
    return f"{name}||{labels_tuple!r}"


def get_counter(name: str, description: str, labelnames: Iterable[str] | None = None) -> Counter:
    k = _key(name, labelnames)
    metric = _METRICS.get(k)
    if metric is None:
        metric = Counter(name, description, list(labelnames or ()))
        _METRICS[k] = metric
    return metric


def get_histogram(
    name: str,
    description: str,
    labelnames: Iterable[str] | None = None,
    *,
    buckets: Iterable[float] | None = None,
) -> Histogram:
    k = _key(name, labelnames)
    metric = _METRICS.get(k)
    if metric is None:
        if buckets is None:
            metric = Histogram(name, description, list(labelnames or ()))
        else:
            metric = Histogram(name, description, list(labelnames or ()), buckets=tuple(buckets))
        _METRICS[k] = metric
    return metric


def get_gauge(name: str, description: str, labelnames: Iterable[str] | None = None) -> Gauge:
    k = _key(name, labelnames)
    metric = _METRICS.get(k)
    if metric is None:
        metric = Gauge(name, description, list(labelnames or ()))
        _METRICS[k] = metric
    return metric


__all__ = ["get_counter", "get_gauge", "get_histogram"]
