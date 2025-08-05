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
Common metrics utilities for ML components.

This module provides a consistent interface for metrics collection across ML actors and
strategies, with optional Prometheus support.

"""

from __future__ import annotations

from typing import Any


# Optional prometheus_client for metrics
try:
    from prometheus_client import Counter
    from prometheus_client import Histogram

    HAS_PROMETHEUS = True
except ImportError:
    # Create dummy classes for when prometheus is not available
    class Counter:  # type: ignore[no-redef]
        """
        Dummy Counter class when prometheus is not available.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """
            Initialize dummy Counter.

            Parameters
            ----------
            *args : Any
                Ignored arguments.
            **kwargs : Any
                Ignored keyword arguments.

            """

        def inc(self, *args: Any, **kwargs: Any) -> None:
            """
            Increment counter (no-op in dummy implementation).

            Parameters
            ----------
            *args : Any
                Ignored arguments.
            **kwargs : Any
                Ignored keyword arguments.

            """

    class Histogram:  # type: ignore[no-redef]
        """
        Dummy Histogram class when prometheus is not available.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """
            Initialize dummy Histogram.

            Parameters
            ----------
            *args : Any
                Ignored arguments.
            **kwargs : Any
                Ignored keyword arguments.

            """

        def observe(self, *args: Any, **kwargs: Any) -> None:
            """
            Observe value (no-op in dummy implementation).

            Parameters
            ----------
            *args : Any
                Ignored arguments.
            **kwargs : Any
                Ignored keyword arguments.

            """

    HAS_PROMETHEUS = False


__all__ = ["HAS_PROMETHEUS", "Counter", "Histogram"]
