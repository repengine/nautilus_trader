"""Utility helpers for Databento configuration in deployment entrypoints."""

from __future__ import annotations


def is_valid_databento_key(value: str | None) -> bool:
    """
    Return True when a Databento API key looks structurally valid.

    The production service issues 32-character secrets. We accept keys greater
    or equal to that length so tests can still inject deterministic placeholders.

    Args:
        value: Raw value read from environment variables.

    Returns:
        ``True`` if the key should be treated as valid.

    Example:
        >>> is_valid_databento_key("a" * 32)
        True
        >>> is_valid_databento_key("short")
        False
    """
    if value is None:
        return False
    stripped = value.strip()
    return len(stripped) >= 32
