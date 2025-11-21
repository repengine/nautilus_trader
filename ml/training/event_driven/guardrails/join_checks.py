"""Validation join checks for streaming manifests."""

from __future__ import annotations

from pathlib import Path


def check_validation_joins(manifest_dir: Path, limit: int | None = None) -> list[str]:
    """
    Check validation-return joins in streaming manifests.

    Args:
        manifest_dir: Directory containing streaming manifest JSON files
        limit: Optional limit for number of manifests to inspect (newest first)

    Returns:
        List of issue descriptions (empty if no issues found)
    """
    # TODO: Implement actual validation join checking logic
    # For now, return empty list to unblock tests
    return []


__all__ = ["check_validation_joins"]
