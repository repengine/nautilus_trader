# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Centralized import management for ML optional dependencies.

This module handles all optional ML dependency imports with proper error handling and
provides flags for feature availability.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


# Type checking imports (always available, no runtime cost)
if TYPE_CHECKING:
    import onnxruntime as ort
    import polars as pl
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


# Prometheus Client (already handled in metrics.py, included for completeness)
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


def check_ml_dependencies(required: list[str]) -> None:
    """
    Check if required ML dependencies are available.

    Parameters
    ----------
    required : list[str]
        List of required dependencies: ['onnx', 'polars', 'xgboost', 'prometheus']

    Raises
    ------
    ImportError
        If any required dependency is not available.

    """
    errors = []

    if "onnx" in required and not HAS_ONNX:
        errors.append(
            f"ONNX Runtime required but not installed. "
            f"Install with: pip install 'nautilus-trader[ml]'\n"
            f"Original error: {ONNX_IMPORT_ERROR}",
        )

    if "polars" in required and not HAS_POLARS:
        errors.append(
            f"Polars required but not installed. "
            f"Install with: pip install 'nautilus-trader[ml]'\n"
            f"Original error: {POLARS_IMPORT_ERROR}",
        )

    if "xgboost" in required and not HAS_XGBOOST:
        errors.append(
            f"XGBoost required but not installed. "
            f"Install with: pip install 'nautilus-trader[ml]'\n"
            f"Original error: {XGBOOST_IMPORT_ERROR}",
        )

    if "prometheus" in required and not HAS_PROMETHEUS:
        errors.append(
            f"Prometheus Client required but not installed. "
            f"Install with: pip install 'nautilus-trader[ml]'\n"
            f"Original error: {PROMETHEUS_IMPORT_ERROR}",
        )

    if errors:
        raise ImportError("\n\n".join(errors))


__all__ = [
    # Availability flags
    "HAS_ONNX",
    "HAS_POLARS",
    "HAS_PROMETHEUS",
    "HAS_XGBOOST",
    # Import errors
    "ONNX_IMPORT_ERROR",
    "POLARS_IMPORT_ERROR",
    "PROMETHEUS_IMPORT_ERROR",
    "XGBOOST_IMPORT_ERROR",
    "Counter",
    "Gauge",
    "Histogram",
    # Utility function
    "check_ml_dependencies",
    # Imported modules (may be None)
    "ort",
    "pl",
    "xgb",
]
