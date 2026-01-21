"""
Watermark-based ingestion window configuration defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WatermarkWindowConfig:
    """
    Configuration for deriving ingestion windows from registry watermarks.

    Attributes
    ----------
    use_watermark : bool
        Whether to derive start windows from registry watermarks.
    lookback_days : int
        Days to subtract from the watermark to guard against late arrivals.
    max_window_days : int | None
        Optional maximum window size (days) to cap backfill slices.
    fallback_start_days : int | None
        Optional window size (days) to use when no watermark is available.

    """

    use_watermark: bool = True
    lookback_days: int = 30
    max_window_days: int | None = None
    fallback_start_days: int | None = None

    def __post_init__(self) -> None:
        """
        Validate watermark window configuration.
        """
        if self.lookback_days < 0:
            raise ValueError("lookback_days must be >= 0")
        if self.max_window_days is not None and self.max_window_days < 1:
            raise ValueError("max_window_days must be >= 1")
        if self.fallback_start_days is not None and self.fallback_start_days < 1:
            raise ValueError("fallback_start_days must be >= 1")


def macro_window_defaults() -> WatermarkWindowConfig:
    """
    Default watermark settings for macro ingestion windows.

    Returns
    -------
    WatermarkWindowConfig
        Macro ingestion watermark defaults.
    """
    return WatermarkWindowConfig(
        use_watermark=True,
        lookback_days=365,
        max_window_days=730,
        fallback_start_days=730,
    )


def events_window_defaults() -> WatermarkWindowConfig:
    """
    Default watermark settings for events ingestion windows.

    Returns
    -------
    WatermarkWindowConfig
        Events ingestion watermark defaults.
    """
    return WatermarkWindowConfig(
        use_watermark=True,
        lookback_days=30,
        max_window_days=365,
        fallback_start_days=365,
    )


def earnings_window_defaults() -> WatermarkWindowConfig:
    """
    Default watermark settings for earnings ingestion windows.

    Returns
    -------
    WatermarkWindowConfig
        Earnings ingestion watermark defaults.
    """
    return WatermarkWindowConfig(
        use_watermark=True,
        lookback_days=365,
        max_window_days=730,
        fallback_start_days=730,
    )


def micro_window_defaults() -> WatermarkWindowConfig:
    """
    Default watermark settings for microstructure ingestion windows.

    Returns
    -------
    WatermarkWindowConfig
        Microstructure ingestion watermark defaults.
    """
    return WatermarkWindowConfig(
        use_watermark=True,
        lookback_days=7,
        max_window_days=30,
        fallback_start_days=30,
    )


def l2_window_defaults() -> WatermarkWindowConfig:
    """
    Default watermark settings for L2 ingestion windows.

    Returns
    -------
    WatermarkWindowConfig
        L2 ingestion watermark defaults.
    """
    return WatermarkWindowConfig(
        use_watermark=True,
        lookback_days=7,
        max_window_days=30,
        fallback_start_days=30,
    )


__all__ = [
    "WatermarkWindowConfig",
    "earnings_window_defaults",
    "events_window_defaults",
    "l2_window_defaults",
    "macro_window_defaults",
    "micro_window_defaults",
]
