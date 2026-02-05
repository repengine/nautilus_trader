"""
Property-based tests for prediction surface decision mapping.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.common import decision_from_probability
from ml.common import neutral_band_bounds


@pytest.mark.property
@given(
    probability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    neutral_band=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=2000)
def test_decision_mapping_respects_neutral_band(
    probability: float,
    neutral_band: float,
) -> None:
    """
    Property: neutral band yields HOLD inside bounds and BUY/SELL outside.
    """
    lower, upper = neutral_band_bounds(neutral_band)
    decision = decision_from_probability(probability, neutral_band=neutral_band)

    if probability >= upper:
        assert decision == "BUY"
    elif probability <= lower:
        assert decision == "SELL"
    else:
        assert decision == "HOLD"
