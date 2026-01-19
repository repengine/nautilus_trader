"""
Configuration for live-paced replay harnesses.
"""

from __future__ import annotations

from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


class LiveReplayConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for live-paced replay execution.

    Parameters
    ----------
    speed_multiplier : PositiveFloat, default 1.0
        Speed multiplier for replay pacing (1.0 = real-time, 2.0 = 2x).
    max_events : PositiveInt | None, default None
        Optional maximum number of events to replay.
    timestamp_field : str, default "ts_min"
        Payload field used as the primary event timestamp for pacing.
    """

    speed_multiplier: PositiveFloat = 1.0
    max_events: PositiveInt | None = None
    timestamp_field: str = "ts_min"

    def __post_init__(self) -> None:
        if not self.timestamp_field.strip():
            raise ValueError("timestamp_field must be non-empty")
