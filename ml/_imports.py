"""
Centralized import management for ML optional dependencies.

This module handles all optional ML dependency imports with proper error handling and
provides flags for feature availability.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


# Type checking imports (always available, no runtime cost)
if TYPE_CHECKING:
    import lightgbm as lgb
    import mlflow
    import onnx
    import onnxmltools
    import onnxruntime as ort
    import optuna
    import pandas as pd
    import polars as pl
    import skl2onnx
    import sklearn
    import xgboost as xgb
    from prometheus_client import Counter
    from prometheus_client import Gauge
    from prometheus_client import Histogram


# ONNX Runtime
try:
    import onnxruntime as ort

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
    import polars as pl

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


# MLflow
try:
    import mlflow
    import mlflow.lightgbm
    import mlflow.xgboost

    HAS_MLFLOW = True
    MLFLOW_IMPORT_ERROR = None
except ImportError as e:
    HAS_MLFLOW = False
    MLFLOW_IMPORT_ERROR = e
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
    import pandas as pd

    HAS_PANDAS = True
    PANDAS_IMPORT_ERROR = None
except ImportError as e:
    HAS_PANDAS = False
    PANDAS_IMPORT_ERROR = e
    pd = None  # type: ignore[assignment,unused-ignore]


# Prometheus Client (already handled in metrics.py, included for completeness)
from collections.abc import Callable


_PROM_REGISTRY: Any
_GENERATE_LATEST: Callable[[Any], bytes]

try:
    from prometheus_client import Counter
    from prometheus_client import Gauge
    from prometheus_client import Histogram

    HAS_PROMETHEUS = True
    PROMETHEUS_IMPORT_ERROR = None
except ImportError as e:
    HAS_PROMETHEUS = False
    PROMETHEUS_IMPORT_ERROR = e

    # Dummy implementations
    class Counter:  # type: ignore[no-redef]
        """
        Dummy Counter when prometheus-client is not available.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """
            Initialize mock Counter.

            Parameters
            ----------
            *args : Any
                Positional arguments (ignored).
            **kwargs : Any
                Keyword arguments (ignored).

            """

        def inc(self, *args: Any, **kwargs: Any) -> None:
            """
            Increment counter (no-op).

            Parameters
            ----------
            *args : Any
                Positional arguments (ignored).
            **kwargs : Any
                Keyword arguments (ignored).

            """

        def labels(self, *args: Any, **kwargs: Any) -> Any:
            """
            Get labeled counter (returns self for chaining).

            Parameters
            ----------
            *args : Any
                Positional arguments (ignored).
            **kwargs : Any
                Keyword arguments (ignored).

            Returns
            -------
            Any
                Self for method chaining.

            """
            return self

    class Gauge:  # type: ignore[no-redef]
        """
        Dummy Gauge when prometheus-client is not available.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """
            Initialize mock Gauge.

            Parameters
            ----------
            *args : Any
                Positional arguments (ignored).
            **kwargs : Any
                Keyword arguments (ignored).

            """

        def set(self, *args: Any, **kwargs: Any) -> None:
            """
            Set gauge value (no-op).

            Parameters
            ----------
            *args : Any
                Positional arguments (ignored).
            **kwargs : Any
                Keyword arguments (ignored).

            """

        def labels(self, *args: Any, **kwargs: Any) -> Any:
            """
            Get labeled gauge (returns self for chaining).

            Parameters
            ----------
            *args : Any
                Positional arguments (ignored).
            **kwargs : Any
                Keyword arguments (ignored).

            Returns
            -------
            Any
                Self for method chaining.

            """
            return self

    class Histogram:  # type: ignore[no-redef]
        """
        Dummy Histogram when prometheus-client is not available.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """
            Initialize mock Histogram.

            Parameters
            ----------
            *args : Any
                Positional arguments (ignored).
            **kwargs : Any
                Keyword arguments (ignored).

            """

        def observe(self, *args: Any, **kwargs: Any) -> None:
            """
            Observe value (no-op).

            Parameters
            ----------
            *args : Any
                Positional arguments (ignored).
            **kwargs : Any
                Keyword arguments (ignored).

            """

        def labels(self, *args: Any, **kwargs: Any) -> Any:
            """
            Get labeled histogram (returns self for chaining).

            Parameters
            ----------
            *args : Any
                Positional arguments (ignored).
            **kwargs : Any
                Keyword arguments (ignored).

            Returns
            -------
            Any
                Self for method chaining.

            """
            return self

    class _DummyRegistry:
        """
        Minimal dummy of prometheus_client.REGISTRY used for name lookups.
        """

        def __init__(self) -> None:
            self._names_to_collectors: dict[str, Any] = {}

    # Provide underlying symbols, then expose unified names below
    _PROM_REGISTRY = _DummyRegistry()

    def _generate_latest_dummy(registry: Any = None) -> bytes:
        return b""

    _GENERATE_LATEST = _generate_latest_dummy
else:
    # When Prometheus is available, normalize symbol names for consistent exports
    from prometheus_client import REGISTRY as _REAL_REGISTRY
    from prometheus_client import generate_latest as _REAL_GENERATE_LATEST

    _PROM_REGISTRY = _REAL_REGISTRY
    _GENERATE_LATEST = _REAL_GENERATE_LATEST


# Public, unified names with stable signatures
def generate_latest(registry: Any = None) -> bytes:
    return _GENERATE_LATEST(registry)


REGISTRY: Any = _PROM_REGISTRY


def check_ml_dependencies(required: list[str]) -> None:
    """
    Check if required ML dependencies are available.

    Parameters
    ----------
    required : list[str]
        List of required dependencies: ['onnx', 'polars', 'xgboost', 'lightgbm', 'sklearn',
        'optuna', 'mlflow', 'prometheus', 'onnx_export', 'pandas']

    Raises
    ------
    ImportError
        If any required dependency is not available.

    """
    errors = []

    if "onnx" in required and not HAS_ONNX:
        errors.append(
            f"ONNX Runtime required but not installed. Install with: pip install 'nautilus-trader[ml]'\nOriginal error: {ONNX_IMPORT_ERROR}",
        )

    if "polars" in required and not HAS_POLARS:
        errors.append(
            f"Polars required but not installed. Install with: pip install 'nautilus-trader[ml]'\nOriginal error: {POLARS_IMPORT_ERROR}",
        )

    if "xgboost" in required and not HAS_XGBOOST:
        errors.append(
            f"XGBoost required but not installed. Install with: pip install 'nautilus-trader[ml]'\nOriginal error: {XGBOOST_IMPORT_ERROR}",
        )

    if "lightgbm" in required and not HAS_LIGHTGBM:
        errors.append(
            f"LightGBM required but not installed. Install with: pip install 'nautilus-trader[ml]'\nOriginal error: {LIGHTGBM_IMPORT_ERROR}",
        )

    if "optuna" in required and not HAS_OPTUNA:
        errors.append(
            f"Optuna required but not installed. Install with: pip install 'nautilus-trader[ml]'\nOriginal error: {OPTUNA_IMPORT_ERROR}",
        )

    if "mlflow" in required and not HAS_MLFLOW:
        errors.append(
            f"MLflow required but not installed. Install with: pip install 'nautilus-trader[ml]'\nOriginal error: {MLFLOW_IMPORT_ERROR}",
        )

    if "sklearn" in required and not HAS_SKLEARN:
        errors.append(
            f"Scikit-learn required but not installed. Install with: pip install 'nautilus-trader[ml]'\nOriginal error: {SKLEARN_IMPORT_ERROR}",
        )

    if "prometheus" in required and not HAS_PROMETHEUS:
        errors.append(
            f"Prometheus Client required but not installed. "
            f"Install with: pip install 'nautilus-trader[ml]'\n"
            f"Original error: {PROMETHEUS_IMPORT_ERROR}",
        )

    if "onnx_export" in required and not HAS_ONNX_EXPORT:
        errors.append(
            f"ONNX export tools (onnxmltools, skl2onnx) required but not installed. "
            f"Install with: pip install 'nautilus-trader[ml]'\n"
            f"Original error: {ONNX_EXPORT_IMPORT_ERROR}",
        )

    if "pandas" in required and not HAS_PANDAS:
        errors.append(
            f"Pandas required but not installed. Install with: pip install pandas\nOriginal error: {PANDAS_IMPORT_ERROR}",
        )

    if errors:
        raise ImportError("\n\n".join(errors))


__all__ = [
    "HAS_LIGHTGBM",
    "HAS_MLFLOW",
    "HAS_ONNX",
    "HAS_ONNX_CORE",
    "HAS_ONNX_EXPORT",
    "HAS_OPTUNA",
    "HAS_PANDAS",
    "HAS_POLARS",
    "HAS_PROMETHEUS",
    "HAS_SKLEARN",
    "HAS_XGBOOST",
    "LIGHTGBM_IMPORT_ERROR",
    "MLFLOW_IMPORT_ERROR",
    "ONNX_CORE_IMPORT_ERROR",
    "ONNX_EXPORT_IMPORT_ERROR",
    "ONNX_IMPORT_ERROR",
    "OPTUNA_IMPORT_ERROR",
    "PANDAS_IMPORT_ERROR",
    "POLARS_IMPORT_ERROR",
    "PROMETHEUS_IMPORT_ERROR",
    "REGISTRY",
    "SKLEARN_IMPORT_ERROR",
    "XGBOOST_IMPORT_ERROR",
    "Counter",
    "Gauge",
    "Histogram",
    "check_ml_dependencies",
    "generate_latest",
    "lgb",
    "mlflow",
    "onnx",
    "onnxmltools",
    "optuna",
    "ort",
    "pd",
    "pl",
    "skl2onnx",
    "sklearn",
    "xgb",
]
