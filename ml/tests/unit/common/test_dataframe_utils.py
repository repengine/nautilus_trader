#!/usr/bin/env python3
from __future__ import annotations

import types

import pytest

from ml.common.dataframe_utils import (
    column_nulls,
    has_columns,
    is_monotonic_non_decreasing,
    total_nulls,
)


def test_total_nulls_and_column_nulls_pandas_like() -> None:
    # Minimal pandas-like stand-in to avoid heavy deps
    class Series:
        def __init__(self, data: list[object]) -> None:
            self._data = data

        def isnull(self):  # noqa: D401
            class _Sum:
                def __init__(self, data: list[object]) -> None:
                    self._data = data

                def sum(self) -> int:
                    return sum(1 for x in self._data if x is None)

            return _Sum(self._data)

    class Frame:
        def __init__(self, data: dict[str, list[object]]) -> None:
            self._data = {k: Series(v) for k, v in data.items()}
            self.columns = list(data.keys())

        def __getitem__(self, key: str) -> Series:
            return self._data[key]

        def isnull(self):  # noqa: D401
            class _Sum:
                def __init__(self, cols: dict[str, Series]) -> None:
                    self._cols = cols

                def sum(self):  # type: ignore[override]
                    class _Sum2:
                        def __init__(self, cols: dict[str, Series]) -> None:
                            self._cols = cols

                        def sum(self) -> int:
                            return sum(col.isnull().sum() for col in self._cols.values())

                    return _Sum2(self._cols)

            return _Sum(self._data)

    df = Frame({"a": [1, None, 3], "b": [None, None, 2]})
    assert total_nulls(df) == 3
    assert column_nulls(df, "a") == 1
    assert column_nulls(df, "b") == 2


def test_has_columns_and_missing() -> None:
    class F:
        def __init__(self) -> None:
            self.columns = ["x", "y"]

    ok, missing = has_columns(F(), {"x", "y"})
    assert ok and missing == set()

    ok2, missing2 = has_columns(F(), {"x", "y", "z"})
    assert not ok2 and missing2 == {"z"}


@pytest.mark.parametrize(
    "seq,expected",
    [
        ([], True),
        ([1], True),
        ([1, 1, 2, 3], True),
        ([1, 0], False),
    ],
)
def test_is_monotonic_non_decreasing(seq: list[int], expected: bool) -> None:
    assert is_monotonic_non_decreasing(seq) is expected

