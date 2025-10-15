#!/usr/bin/env python3

"""
Shared helpers for coercing loosely typed registry payloads into precise types.

These utilities centralize the runtime validation used by strict mypy runs to keep
the persistence layer resilient against malformed or legacy data structures.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any


__all__ = [
    "expect_bool",
    "expect_dict",
    "expect_dict_list",
    "expect_float",
    "expect_float_dict",
    "expect_optional_str",
    "expect_str",
    "expect_str_dict",
    "expect_str_list",
]


def expect_str(value: Any, field: str) -> str:
    """
    Ensure a value is a string.
    """
    if isinstance(value, str):
        return value
    raise TypeError(f"{field} must be a string, received {type(value).__name__}")


def expect_optional_str(value: Any, field: str) -> str | None:
    """
    Coerce an optional string field.
    """
    if value is None:
        return None
    return expect_str(value, field)


def expect_float(value: Any, field: str, default: float | None = None) -> float:
    """
    Coerce numeric values to float with an optional default.
    """
    if value is None:
        if default is not None:
            return default
        raise TypeError(f"{field} must be numeric")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{field} must be numeric, received {type(value).__name__}")


def expect_bool(value: Any, field: str, default: bool | None = None) -> bool:
    """
    Coerce truthy values to bool, accepting common string forms.
    """
    if value is None:
        if default is None:
            raise TypeError(f"{field} must be boolean")
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise TypeError(f"{field} must be boolean-compatible, received {type(value).__name__}")


def expect_str_list(value: Any, field: str) -> list[str]:
    """
    Ensure a value is a sequence of strings.
    """
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[str] = []
        for index, item in enumerate(value):
            result.append(expect_str(item, f"{field}[{index}]"))
        return result
    raise TypeError(f"{field} must be a sequence of strings")


def expect_dict(value: Any, field: str) -> dict[str, Any]:
    """
    Ensure a value is a mapping with string keys.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result[expect_str(key, f"{field} key")] = item
        return result
    raise TypeError(f"{field} must be a mapping")


def expect_float_dict(value: Any, field: str) -> dict[str, float]:
    """
    Ensure a mapping with float-compatible values.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for key, item in value.items():
            if isinstance(item, (int, float)):
                result[expect_str(key, f"{field} key")] = float(item)
                continue
            raise TypeError(
                f"{field} values must be numeric, received {type(item).__name__}",
            )
        return result
    raise TypeError(f"{field} must be a mapping of floats")


def expect_str_dict(value: Any, field: str) -> dict[str, str]:
    """
    Ensure a mapping with string values.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result: dict[str, str] = {}
        for key, item in value.items():
            result[expect_str(key, f"{field} key")] = expect_str(item, f"{field} value")
        return result
    raise TypeError(f"{field} must be a mapping of strings")


def expect_dict_list(value: Any, field: str) -> list[dict[str, Any]]:
    """
    Ensure a value is a sequence of mappings.
    """
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            result.append(
                expect_dict(
                    item,
                    f"{field}[{index}]",
                ),
            )
        return result
    raise TypeError(f"{field} must be a sequence of mappings")
