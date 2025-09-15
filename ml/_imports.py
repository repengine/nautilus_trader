"""
Centralized import management for ML optional dependencies.

This module handles all optional ML dependency imports with proper error handling and
provides flags for feature availability.

"""

from __future__ import annotations

import os
from types import ModuleType
from typing import TYPE_CHECKING, Any


# Detect Prometheus backend without importing ml.common (avoid cycles)
try:  # pragma: no cover - optional dependency
    import importlib as _importlib

    _importlib.import_module("prometheus_client")
    HAS_PROMETHEUS = True
except Exception:
    HAS_PROMETHEUS = False


# Early placeholders to avoid attribute errors during partial initialization
pl: ModuleType | None = None
pd: ModuleType | None = None
ort: ModuleType | None = None
HAS_PANDAS = False


PROMETHEUS_IMPORT_ERROR: Exception | None = None


# Type checking imports (always available, no runtime cost)
if TYPE_CHECKING:
    import lightgbm as lgb
    import mlflow
    import onnx
    import onnxmltools
    import onnxruntime as _ort  # noqa: F401
    import optuna
    import pandas as _pd  # noqa: F401
    import pandas_market_calendars as mcal
    import polars as _pl  # noqa: F401
    import skl2onnx
    import sklearn
    import torch
    import xgboost as xgb


# ONNX Runtime
try:
    import onnxruntime as _ort_runtime

    ort = _ort_runtime
    HAS_ONNX = True
    ONNX_IMPORT_ERROR = None
except ImportError as e:
    HAS_ONNX = False
    ONNX_IMPORT_ERROR = e
    ort = None


# ONNX core (model IO/helpers)
try:
    import onnx

    HAS_ONNX_CORE = True
    ONNX_CORE_IMPORT_ERROR = None
except ImportError as e:
    HAS_ONNX_CORE = False
    ONNX_CORE_IMPORT_ERROR = e
    onnx = None  # type: ignore[assignment,unused-ignore]


# Polars
try:
    import polars as _pl_runtime

    pl = _pl_runtime
    HAS_POLARS = True
    POLARS_IMPORT_ERROR = None
except ImportError as e:
    HAS_POLARS = False
    POLARS_IMPORT_ERROR = e
    pl = None  # type: ignore[assignment,unused-ignore]


# XGBoost
try:
    import xgboost as xgb

    HAS_XGBOOST = True
    XGBOOST_IMPORT_ERROR = None
except ImportError as e:
    HAS_XGBOOST = False
    XGBOOST_IMPORT_ERROR = e
    xgb = None  # type: ignore[assignment,unused-ignore]


# LightGBM
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
    LIGHTGBM_IMPORT_ERROR = None
except ImportError as e:
    HAS_LIGHTGBM = False
    LIGHTGBM_IMPORT_ERROR = e
    lgb = None  # type: ignore[assignment,unused-ignore]


# Optuna
try:
    import optuna

    HAS_OPTUNA = True
    OPTUNA_IMPORT_ERROR = None
except ImportError as e:
    HAS_OPTUNA = False
    OPTUNA_IMPORT_ERROR = e
    optuna = None  # type: ignore[assignment,unused-ignore]


# PyTorch
try:
    import torch

    HAS_TORCH = True
    TORCH_IMPORT_ERROR = None
except ImportError as e:
    HAS_TORCH = False
    TORCH_IMPORT_ERROR = e
    torch = None  # type: ignore[assignment,unused-ignore]


# Joblib (used only in explicitly-guarded test contexts)
try:
    import joblib as joblib

    HAS_JOBLIB = True
    JOBLIB_IMPORT_ERROR = None
except ImportError as e:
    HAS_JOBLIB = False
    JOBLIB_IMPORT_ERROR = e
    joblib = None  # type: ignore[assignment,unused-ignore]


# MLflow (DEPRECATED - use ModelRegistry instead)
# Removed direct imports to prevent telemetry activation
HAS_MLFLOW = False
MLFLOW_IMPORT_ERROR = ImportError("MLflow deprecated - use ModelRegistry")
mlflow = None  # type: ignore[assignment,unused-ignore]


# Scikit-learn
try:
    import sklearn

    HAS_SKLEARN = True
    SKLEARN_IMPORT_ERROR = None
except ImportError as e:
    HAS_SKLEARN = False
    SKLEARN_IMPORT_ERROR = e
    sklearn = None  # type: ignore[assignment,unused-ignore]


# ONNX export tools
try:
    import onnxmltools
    import skl2onnx

    HAS_ONNX_EXPORT = True
    ONNX_EXPORT_IMPORT_ERROR = None
except ImportError as e:
    HAS_ONNX_EXPORT = False
    ONNX_EXPORT_IMPORT_ERROR = e
    onnxmltools = None  # type: ignore[assignment,unused-ignore]
    skl2onnx = None  # type: ignore[assignment,unused-ignore]


# Pandas (used in some ML modules)
try:
    import pandas as _pd_runtime

    pd = _pd_runtime
    HAS_PANDAS = True
    PANDAS_IMPORT_ERROR = None
except ImportError as e:
    HAS_PANDAS = False
    PANDAS_IMPORT_ERROR = e
    pd = None  # type: ignore[assignment,unused-ignore]


# FRED API (fredapi) optional dependency
try:
    import fredapi as fredapi

    HAS_FREDAPI = True
    FREDAPI_IMPORT_ERROR = None
except ImportError as e:
    HAS_FREDAPI = False
    FREDAPI_IMPORT_ERROR = e
    fredapi = None  # type: ignore[assignment,unused-ignore]


# Databento (data collection)
# To keep tests and sandboxed runs stable (no sockets/network), only attempt to
# import Databento when explicitly enabled via environment or when an API key is
# present. Otherwise, mark as unavailable and defer actual use to callers.

_ENABLE_DATABENTO = os.environ.get("ML_ENABLE_DATABENTO", "").lower() in {"1", "true", "yes"}
_HAS_API_KEY = bool(os.environ.get("DATABENTO_API_KEY"))

if _ENABLE_DATABENTO or _HAS_API_KEY:
    try:
        import databento as db

        HAS_DATABENTO = True
        DATABENTO_IMPORT_ERROR = None
    except ImportError as e:  # pragma: no cover - env dependent
        HAS_DATABENTO = False
        DATABENTO_IMPORT_ERROR = e
        db = None  # type: ignore[assignment,unused-ignore]
else:
    HAS_DATABENTO = False
    DATABENTO_IMPORT_ERROR = None
    db = None  # type: ignore[assignment,unused-ignore]


# Pandas Market Calendars (market schedules)
try:
    import pandas_market_calendars as mcal

    HAS_PANDAS_MARKET_CALENDARS = True
    PANDAS_MARKET_CALENDARS_IMPORT_ERROR = None
except ImportError as e:
    HAS_PANDAS_MARKET_CALENDARS = False
    PANDAS_MARKET_CALENDARS_IMPORT_ERROR = e
    mcal = None  # type: ignore[assignment,unused-ignore]


# OpenTelemetry (distributed tracing - optional, off by default)
try:
    from opentelemetry import context as otel_context
    from opentelemetry import propagate as otel_propagate
    from opentelemetry import trace as otel_trace

    HAS_OPENTELEMETRY = True
    OPENTELEMETRY_IMPORT_ERROR = None
except ImportError as e:
    HAS_OPENTELEMETRY = False
    OPENTELEMETRY_IMPORT_ERROR = e
    otel_trace = None  # type: ignore[assignment,unused-ignore]
    otel_context = None  # type: ignore[assignment,unused-ignore]
    otel_propagate = None  # type: ignore[assignment,unused-ignore]


"""Prometheus wrappers: avoid direct prometheus_client imports here.

We expose light adapters that delegate to the centralized metrics bootstrap to
create collectors. This satisfies the "no direct prometheus_client" policy and keeps
existing imports working in modules/tests which patch these names.
"""

class Counter:
    def __init__(self, name: str, description: str, labels: list[str] | None = None) -> None:
        from ml.common.metrics_bootstrap import get_counter as _get

        self._collector = _get(name, description, labels)

    def labels(self, **kwargs: object) -> Any:
        return self._collector.labels(**kwargs)

    def inc(self, *args: object, **kwargs: object) -> None:
        # Allow inc on base collector for compatibility
        try:
            self._collector.inc(*args, **kwargs)
        except Exception:
            pass


class Gauge:
    def __init__(self, name: str, description: str, labels: list[str] | None = None) -> None:
        from ml.common.metrics_bootstrap import get_gauge as _get

        self._collector = _get(name, description, labels)

    def labels(self, **kwargs: object) -> Any:
        return self._collector.labels(**kwargs)

    def set(self, value: float) -> None:
        try:
            self._collector.set(float(value))
        except Exception:
            pass


class Histogram:
    def __init__(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
        *,
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        from ml.common.metrics_bootstrap import get_histogram as _get

        self._collector = _get(name, description, labels, buckets=buckets)

    def labels(self, **kwargs: object) -> Any:
        return self._collector.labels(**kwargs)

    def observe(self, value: float) -> None:
        try:
            self._collector.observe(float(value))
        except Exception:
            pass

_PROM_REGISTRY = object()  # placeholder for compatibility

def _generate_latest_dummy(registry: object | None = None) -> bytes:
    from ml.common.metrics_export import generate_latest as _gen

    return _gen()

_GENERATE_LATEST = _generate_latest_dummy


# Public, unified names with stable signatures
def generate_latest(registry: object | None = None) -> bytes:
    return _GENERATE_LATEST(registry)


REGISTRY: object = _PROM_REGISTRY


def check_ml_dependencies(required: list[str]) -> None:
    """
    Check if required ML dependencies are available.

    Parameters
    ----------
    required : list[str]
        Supported keys: onnx, polars, xgboost, lightgbm, sklearn,
        optuna, mlflow, prometheus, onnx_export, pandas, databento,
        pandas_market_calendars

    Raises
    ------
    ImportError
        If any required dependency is not available.

    """
    checks: dict[str, tuple[bool, str]] = {
        "onnx": (HAS_ONNX, f"ONNX Runtime required. Original error: {ONNX_IMPORT_ERROR}"),
        "polars": (HAS_POLARS, f"Polars required. Original error: {POLARS_IMPORT_ERROR}"),
        "xgboost": (HAS_XGBOOST, f"XGBoost required. Original error: {XGBOOST_IMPORT_ERROR}"),
        "lightgbm": (
            HAS_LIGHTGBM,
            f"LightGBM required. Original error: {LIGHTGBM_IMPORT_ERROR}",
        ),
        "optuna": (HAS_OPTUNA, f"Optuna required. Original error: {OPTUNA_IMPORT_ERROR}"),
        "mlflow": (HAS_MLFLOW, f"MLflow required. Original error: {MLFLOW_IMPORT_ERROR}"),
        "sklearn": (HAS_SKLEARN, f"Scikit-learn required. Original error: {SKLEARN_IMPORT_ERROR}"),
        "prometheus": (
            HAS_PROMETHEUS,
            f"Prometheus Client required. Original error: {PROMETHEUS_IMPORT_ERROR}",
        ),
        "onnx_export": (
            HAS_ONNX_EXPORT,
            f"ONNX export tools (onnxmltools, skl2onnx) required. Original error: {ONNX_EXPORT_IMPORT_ERROR}",
        ),
        "pandas": (HAS_PANDAS, f"Pandas required. Original error: {PANDAS_IMPORT_ERROR}"),
        "databento": (
            HAS_DATABENTO,
            f"Databento required. Original error: {DATABENTO_IMPORT_ERROR}",
        ),
        "pandas_market_calendars": (
            HAS_PANDAS_MARKET_CALENDARS,
            f"pandas_market_calendars required. Original error: {PANDAS_MARKET_CALENDARS_IMPORT_ERROR}",
        ),
        "opentelemetry": (
            HAS_OPENTELEMETRY,
            f"OpenTelemetry required. Original error: {OPENTELEMETRY_IMPORT_ERROR}",
        ),
    }

    errors: list[str] = []
    for key in required:
        ok, msg = checks.get(key, (True, ""))
        if not ok:
            hint = (
                "Install with: pip install 'nautilus-trader[ml]'"
                if key != "pandas"
                else "Install with: pip install pandas"
            )
            errors.append(f"{msg}\n{hint}")

    if errors:
        raise ImportError("\n\n".join(errors))


__all__ = [
    "DATABENTO_IMPORT_ERROR",
    "FREDAPI_IMPORT_ERROR",
    "HAS_DATABENTO",
    "HAS_FREDAPI",
    "HAS_JOBLIB",
    "HAS_LIGHTGBM",
    "HAS_MLFLOW",
    "HAS_ONNX",
    "HAS_ONNX_CORE",
    "HAS_ONNX_EXPORT",
    "HAS_OPENTELEMETRY",
    "HAS_OPTUNA",
    "HAS_PANDAS",
    "HAS_PANDAS_MARKET_CALENDARS",
    "HAS_POLARS",
    "HAS_PROMETHEUS",
    "HAS_SKLEARN",
    "HAS_TORCH",
    "HAS_XGBOOST",
    "LIGHTGBM_IMPORT_ERROR",
    "MLFLOW_IMPORT_ERROR",
    "ONNX_CORE_IMPORT_ERROR",
    "ONNX_EXPORT_IMPORT_ERROR",
    "ONNX_IMPORT_ERROR",
    "OPENTELEMETRY_IMPORT_ERROR",
    "OPTUNA_IMPORT_ERROR",
    "PANDAS_IMPORT_ERROR",
    "PANDAS_MARKET_CALENDARS_IMPORT_ERROR",
    "POLARS_IMPORT_ERROR",
    "PROMETHEUS_IMPORT_ERROR",
    "REGISTRY",
    "SKLEARN_IMPORT_ERROR",
    "TORCH_IMPORT_ERROR",
    "XGBOOST_IMPORT_ERROR",
    "Counter",
    "Gauge",
    "Histogram",
    "check_ml_dependencies",
    "db",
    "fredapi",
    "generate_latest",
    "joblib",
    "lgb",
    "mcal",
    "mlflow",
    "onnx",
    "onnxmltools",
    "optuna",
    "ort",
    "otel_context",
    "otel_propagate",
    "otel_trace",
    "pd",
    "pl",
    "skl2onnx",
    "sklearn",
    "torch",
    "xgb",
]
