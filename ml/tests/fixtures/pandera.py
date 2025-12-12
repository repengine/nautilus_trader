#!/usr/bin/env python3
"""
Pandera typing helpers for the ML test suite.

Contract and schema tests traditionally imported ``Series`` directly from
``pandera.typing`` via the legacy ``ml.tests.conftest`` module.  This shim keeps
those annotations centralized and provides a deterministic skip helper so
pytest shards behave consistently when the optional dependency is absent.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import pytest

from ml._imports import HAS_PANDERA
from ml._imports import check_ml_dependencies

# Some pandas builds ship without a __class_getitem__ on Series, which breaks
# Pandera annotations like Series[int]. Patch in a no-op implementation so
# schemas remain subscriptable in tests regardless of pandas version.
try:
    import pandas as _pd

    if not hasattr(_pd.Series, "__class_getitem__"):
        _pd.Series.__class_getitem__ = classmethod(lambda cls, _item: cls)  # type: ignore[misc,assignment]
except Exception:  # pragma: no cover - defensive; absence just leaves types unpatched
    _pd = cast("ModuleType | None", None)

if TYPE_CHECKING:
    import pandera as _pandera_module
    from pandera.typing import DataFrame as PanderaDataFrame
    from pandera.typing import Series as PanderaSeries
else:  # pragma: no cover - TYPE_CHECKING guard
    _pandera_module = cast("ModuleType | None", None)
    PanderaDataFrame = Any  # type: ignore[assignment]
    PanderaSeries = Any  # type: ignore[assignment]

_T = TypeVar("_T")

if HAS_PANDERA:
    import pandera as _pandera_runtime
    from pandera.typing import DataFrame as _PanderaDataFrame
    from pandera.typing import Series as _PanderaSeries

    pa: ModuleType = _pandera_runtime
    DataFrame = _PanderaDataFrame
    Series = _PanderaSeries
else:
    pa = cast("ModuleType", None)

    class _SeriesPlaceholder(Generic[_T]):
        """Subscriptable placeholder used when pandera is unavailable."""

        def __class_getitem__(cls, _item: Any) -> type[_SeriesPlaceholder[Any]]:
            return cls

    class _DataFramePlaceholder(Generic[_T]):
        """Subscriptable placeholder used when pandera is unavailable."""

        def __class_getitem__(cls, _item: Any) -> type[_DataFramePlaceholder[Any]]:
            return cls

    Series = _SeriesPlaceholder  # type: ignore[assignment]
    DataFrame = _DataFramePlaceholder  # type: ignore[assignment]


def ensure_pandera_available(*, strict: bool = False) -> ModuleType:
    """
    Ensure pandera is installed, skipping the current test otherwise.

    Parameters
    ----------
    strict : bool, optional
        When ``True``, raise a ``RuntimeError`` if pandera is missing to surface
        an actionable dependency message.  When ``False`` (default) the caller's
        test is skipped so shards that opt out of pandera remain green.

    Returns
    -------
    ModuleType
        The ``pandera`` module for convenience in call sites.
    """

    if HAS_PANDERA and pa is not None:
        return pa

    if strict:
        check_ml_dependencies(["pandera"])

    pytest.skip(
        "Pandera is required for ML schema contracts. "
        "Install the 'ml-tests' extra or ensure pandera>=0.26.1 is available.",
    )


def require_pandera() -> ModuleType:
    """
    Require pandera and raise if it is missing.

    This helper mirrors ``ensure_pandera_available(strict=True)`` but keeps
    legacy factories readable where skipping is not desirable (e.g. fixtures
    that build schemas during module import).
    """

    return ensure_pandera_available(strict=True)


__all__ = [
    "DataFrame",
    "Series",
    "ensure_pandera_available",
    "pa",
    "require_pandera",
]
