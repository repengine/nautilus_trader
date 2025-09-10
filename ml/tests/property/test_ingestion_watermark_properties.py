from __future__ import annotations

import random
from typing import List, Tuple

import pandas as pd
from hypothesis import given, strategies as st

from ml.data.fixtures import make_tbbo_fixture


def _chunk_indices(n: int, parts: int) -> List[Tuple[int, int]]:
    # Split range(0, n) into roughly equal contiguous chunks
    base = n // parts
    rem = n % parts
    idx = 0
    out: List[Tuple[int, int]] = []
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        out.append((idx, idx + size))
        idx += size
    return out


@given(parts=st.integers(min_value=1, max_value=7))
def test_watermark_progression_non_decreasing(parts: int) -> None:
    df, _ = make_tbbo_fixture(rows=60)
    # Permute to simulate out-of-order arrival, then sort within each chunk
    perm = list(range(len(df)))
    random.Random(123).shuffle(perm)
    df_perm = df.iloc[perm].reset_index(drop=True)

    chunks = _chunk_indices(len(df_perm), max(parts, 1))
    watermarks: List[int] = []
    last_wm = -1
    for lo, hi in chunks:
        batch = df_perm.iloc[lo:hi].sort_values("ts_event")
        wm_batch = int(batch["ts_event"].max()) if not batch.empty else last_wm
        # Watermark progression is the max seen so far across all processed events
        next_wm = max(last_wm, wm_batch)
        watermarks.append(next_wm)
        assert next_wm >= last_wm
        last_wm = next_wm
    # final watermark equals max ts_event
    assert last_wm == int(df["ts_event"].max())
