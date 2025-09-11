"""
MetricsManager: typed, centralized facade over metrics bootstrap.

This wrapper standardizes how components acquire and use metrics without ever
touching prometheus_client directly. Under the hood it delegates to
`ml.common.metrics_bootstrap` which ensures idempotent collector creation.

Usage
-----
- Prefer `MetricsManager.default()` to reuse a process-wide instance.
- Acquire or use convenience helpers:
  - `mm.counter(name, desc, labels=[...])`
  - `mm.gauge(name, desc, labels=[...])`
  - `mm.histogram(name, desc, labels=[...], buckets=[...])`
  - `mm.inc(name, desc, labels={...}, amount=1.0)`
  - `mm.set_gauge(name, desc, value, labels={...})`
  - `mm.observe(name, desc, value, labels={...})`

Notes
-----
- Keep metric creation off the hot path where possible; create at import time
  or during initialization, and only call `.inc()/.observe()/.set()` in the loop.
- Do not import prometheus_client directly; this module and metrics_bootstrap
  encapsulate collector creation.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, ClassVar, Protocol

from ml.common.metrics_bootstrap import get_counter as _get_counter
from ml.common.metrics_bootstrap import get_gauge as _get_gauge
from ml.common.metrics_bootstrap import get_histogram as _get_histogram


class _CounterLike(Protocol):
    def labels(self, **kwargs: object) -> _CounterLike: ...

    def inc(self, amount: float = 1.0) -> None: ...


class _HistogramLike(Protocol):
    def labels(self, **kwargs: object) -> _HistogramLike: ...

    def observe(self, amount: float) -> None: ...


class _GaugeLike(Protocol):
    def labels(self, **kwargs: object) -> _GaugeLike: ...

    def set(self, value: float) -> None: ...


def _key(name: str, labels: Iterable[str] | None, kind: str) -> str:
    return f"{kind}::{name}::{tuple(labels or ())!r}"


@dataclass(slots=True)
class MetricsManager:
    """
    Small facade to standardize metric access and common operations.
    """

    _cache: dict[str, Any] = field(default_factory=dict)

    # --------- Metric acquisition ---------
    def counter(self, name: str, description: str, labels: Iterable[str] | None = None) -> _CounterLike:
        k = _key(name, labels, "ctr")
        metric = self._cache.get(k)
        if metric is None:
            metric = _get_counter(name, description, labels)
            self._cache[k] = metric
        return metric

    def histogram(
        self,
        name: str,
        description: str,
        labels: Iterable[str] | None = None,
        *,
        buckets: Iterable[float] | None = None,
    ) -> _HistogramLike:
        k = _key(name, labels, "hist")
        metric = self._cache.get(k)
        if metric is None:
            metric = _get_histogram(name, description, labels, buckets=buckets)
            self._cache[k] = metric
        return metric

    def gauge(self, name: str, description: str, labels: Iterable[str] | None = None) -> _GaugeLike:
        k = _key(name, labels, "gauge")
        metric = self._cache.get(k)
        if metric is None:
            metric = _get_gauge(name, description, labels)
            self._cache[k] = metric
        return metric

    # --------- Convenience helpers ---------
    def inc(
        self,
        name: str,
        description: str,
        *,
        labels: Mapping[str, object] | None = None,
        amount: float = 1.0,
        labelnames: Iterable[str] | None = None,
    ) -> None:
        ctr = self.counter(name, description, list(labelnames or (labels or {}).keys()))
        ctr.labels(**(dict(labels or {}))).inc(float(amount))

    def observe(
        self,
        name: str,
        description: str,
        value: float,
        *,
        labels: Mapping[str, object] | None = None,
        labelnames: Iterable[str] | None = None,
        buckets: Iterable[float] | None = None,
    ) -> None:
        hist = self.histogram(name, description, list(labelnames or (labels or {}).keys()), buckets=buckets)
        hist.labels(**(dict(labels or {}))).observe(float(value))

    def set_gauge(
        self,
        name: str,
        description: str,
        value: float,
        *,
        labels: Mapping[str, object] | None = None,
        labelnames: Iterable[str] | None = None,
    ) -> None:
        g = self.gauge(name, description, list(labelnames or (labels or {}).keys()))
        g.labels(**(dict(labels or {}))).set(float(value))

    # --------- Singleton accessor ---------
    _DEFAULT: ClassVar[MetricsManager | None] = None

    @classmethod
    def default(cls) -> MetricsManager:
        if cls._DEFAULT is None:
            cls._DEFAULT = cls()
        return cls._DEFAULT


__all__ = ["MetricsManager"]
