"""
Tests for MultiChannelRingBuffer invariants and chronological extraction.

Contract:
- O(1) append; count <= capacity; index wraps mod capacity
- get_last_row equals last appended values
- get_channel_view returns a view (no allocation / does not own data)
- get_channel_chronological equals a naive chronological reference
- Property: chronological extraction matches Python list rotation reference
"""

from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.core.cache import MultiChannelRingBuffer


@pytest.mark.unit
def test_append_wrap_and_views() -> None:
    cap = 4
    ch = 2
    rb = MultiChannelRingBuffer(size=cap, channels=ch)

    rows: list[list[float]] = [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0],
    ]
    for r in rows:
        rb.append(r)

    # Invariants
    assert rb.count <= cap
    assert rb.count == cap
    assert rb.index == (len(rows) % cap)

    # Last row equals last append
    np.testing.assert_allclose(rb.get_last_row(), np.array(rows[-1], dtype=np.float32))

    # Channel view is a view (does not own its memory)
    view0 = rb.get_channel_view(0)
    assert view0.flags["OWNDATA"] is False

    # Chronological channel equals naive rotation reference
    # Keep only the last 'cap' rows in chronological order (oldest -> newest)
    tail = rows[-cap:]
    naive_ch0 = np.array([r[0] for r in tail], dtype=np.float32)
    got_ch0 = rb.get_channel_chronological(0)
    np.testing.assert_allclose(got_ch0, naive_ch0)

    # Wrap correctness for channel 1 as well
    naive_ch1 = np.array([r[1] for r in tail], dtype=np.float32)
    got_ch1 = rb.get_channel_chronological(1)
    np.testing.assert_allclose(got_ch1, naive_ch1)


@pytest.mark.unit
@given(
    seq=st.lists(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=3, max_size=3), min_size=1, max_size=50),
)
def test_chronological_property(seq: list[list[float]]) -> None:
    """
    For random sequences of 3-channel rows, chronological extraction on each channel
    matches a simple list rotation reference for ring buffers.
    """
    cap = 8
    ch = 3
    rb = MultiChannelRingBuffer(size=cap, channels=ch)
    for row in seq:
        # Coerce to correct width (truncate/pad)
        vals = (row + [0.0, 0.0, 0.0])[:ch]
        rb.append(vals)

    # Build Python reference for the last min(len(seq), cap) items
    tail = seq[-cap:]
    ref0 = np.array([float(r[0]) for r in tail], dtype=np.float32)
    ref1 = np.array([float(r[1]) for r in tail], dtype=np.float32)
    ref2 = np.array([float(r[2]) for r in tail], dtype=np.float32)

    np.testing.assert_allclose(rb.get_channel_chronological(0), ref0)
    np.testing.assert_allclose(rb.get_channel_chronological(1), ref1)
    np.testing.assert_allclose(rb.get_channel_chronological(2), ref2)

