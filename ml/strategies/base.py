"""
Base ML strategy compatibility shim.

This module re-exports the component-based facade as the canonical strategy base.

"""

from __future__ import annotations

from ml.strategies.base_facade import BaseMLStrategyFacade as BaseMLStrategy
from ml.strategies.base_facade import SimpleMLStrategyFacade as SimpleMLStrategy


__all__ = [
    "BaseMLStrategy",
    "SimpleMLStrategy",
]
