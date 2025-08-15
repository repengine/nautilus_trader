"""
Common metrics utilities for ML components.

This module provides a consistent interface for metrics collection across ML actors and
strategies, with optional Prometheus support.

"""

from __future__ import annotations

# Import ML dependencies with centralized management
from ml._imports import HAS_PROMETHEUS
from ml._imports import Counter
from ml._imports import Histogram


__all__ = ["HAS_PROMETHEUS", "Counter", "Histogram"]
