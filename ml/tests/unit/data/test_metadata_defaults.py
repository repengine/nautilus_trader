from __future__ import annotations

import polars as pl

from ml.data.providers.metadata import InstrumentMetadataProvider
from ml.data.sources.metadata import MockMetadataSource, default_metadata


def test_metadata_defaults_parity() -> None:
    symbols = ["XYZ", "ABC"]
    provider = InstrumentMetadataProvider(source=MockMetadataSource(seed=1))

    # Force empty/unknown paths by asking provider to validate defaults
    df = provider._empty_metadata_frame(symbols)
    assert isinstance(df, pl.DataFrame)
    # Compare row-by-row to canonical default_metadata
    expected = [default_metadata(sym) for sym in symbols]
    for i, sym in enumerate(symbols):
        row = df.row(i, named=True)
        for k, v in expected[i].items():
            assert row[k] == v
