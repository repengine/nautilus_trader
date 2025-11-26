"""Integration tests for TFT-specific dataset format.

This module tests the DatasetBuilder component's ability to build datasets
in TFT (Temporal Fusion Transformer) format with static/time-varying features.

Phase 2.2.2 Status: STRUCTURAL PHASE
- All tests marked @pytest.mark.skip + @pytest.mark.integration
- Tests document expected behavior for Phase 2.2.8
- Full implementation in Phase 2.2.8
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Structural phase - requires full implementation in Phase 2.2.8")
@pytest.mark.integration
def test_build_tft_specific_dataset() -> None:
    """
    Build TFT (Temporal Fusion Transformer) dataset with specific structure.

    Phase 2.2.2: Test skipped (structural phase).
    Phase 2.2.8: Will build TFT-specific dataset.

    Expected Behavior (Phase 2.2.8):
    - Dataset has static features (instrument metadata, day_of_week, month)
    - Dataset has time-varying features (OHLCV, SMA, RSI)
    - Dataset has proper windowing for sequential prediction
    - Dataset has columns for known inputs, unknown inputs, targets
    """


@pytest.mark.skip(reason="Structural phase - requires full implementation in Phase 2.2.8")
@pytest.mark.integration
def test_tft_static_vs_time_varying_features() -> None:
    """
    Verify TFT dataset separates static vs time-varying features correctly.

    Phase 2.2.2: Test skipped (structural phase).
    Phase 2.2.8: Will verify static vs time-varying separation.

    Expected Behavior (Phase 2.2.8):
    - Static features: instrument_id, day_of_week, month (constant per sequence)
    - Time-varying known: hour, minute, is_market_open (known at prediction time)
    - Time-varying unknown: close, volume, sma_20, rsi_14 (predicted targets)
    """


@pytest.mark.skip(reason="Structural phase - requires full implementation in Phase 2.2.8")
@pytest.mark.integration
def test_tft_windowing_correct() -> None:
    """
    Verify TFT windowing for sequential prediction is correct.

    Phase 2.2.2: Test skipped (structural phase).
    Phase 2.2.8: Will verify windowing logic.

    Expected Behavior (Phase 2.2.8):
    - Each sample has 100 historical bars (input)
    - Each sample has 10 future bars (target)
    - Windows slide by 1 bar (or configurable stride)
    - Total samples = (total_bars - window_size - prediction_horizon) / stride
    """
