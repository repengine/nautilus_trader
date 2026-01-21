#!/usr/bin/env python3
"""
Configuration for SEC EDGAR smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


DEFAULT_EDGAR_SMOKE_CIK: Final[str] = "0000320193"


@dataclass(frozen=True)
class EdgarSmokeTestConfig:
    """
    Configuration for EDGAR API smoke tests.

    Attributes
    ----------
    cik:
        10-digit CIK used for the submissions endpoint.
    timeout_seconds:
        Timeout in seconds for the HTTP request.
    """

    cik: str = DEFAULT_EDGAR_SMOKE_CIK
    timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.cik or not self.cik.isdigit():
            raise ValueError("cik must be a 10-digit numeric string")
        if len(self.cik) != 10:
            raise ValueError("cik must be a 10-digit numeric string")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


__all__: tuple[str, ...] = (
    "DEFAULT_EDGAR_SMOKE_CIK",
    "EdgarSmokeTestConfig",
)
