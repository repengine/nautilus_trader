from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import pytest


try:
    from hypothesis import HealthCheck
    from hypothesis import given
    from hypothesis import settings
    from hypothesis import strategies as st
except Exception:  # pragma: no cover
    pytest.skip("hypothesis not available", allow_module_level=True)

from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.tests.utils.targets import build_default_target_semantics
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


@settings(max_examples=6, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n=st.integers(min_value=5, max_value=30),
    step=st.integers(min_value=60_000_000_000, max_value=120_000_000_000),  # 1-2 minutes in ns
)
def test_builder_time_index_monotonic(
    tmp_path: Path,
    patch_bars_to_dataframe,
    n: int,
    step: int,
    sample_bar_series_config_factory,
) -> None:
    freq_minutes = max(1, min(2, round(step / 60_000_000_000)))
    patch_bars_to_dataframe(
        "ml.data.tft_dataset_builder",
        sample_bar_series_config_factory(rows=n, freq_minutes=int(freq_minutes)),
    )
    builder = TFTDatasetBuilder(ParquetDataCatalog(path=str(tmp_path)), symbols=["SPY"])
    target_semantics = build_default_target_semantics(
        horizon_minutes=1,
        threshold=0.001,
        legacy_aliases=True,
    )
    df_raw = builder.build_training_dataset(
        target_semantics=target_semantics,
        use_polars=True,
        lookback_periods=0,
    )
    df = cast(pl.DataFrame, df_raw)
    assert not df.is_empty()
    ti: pl.Series = df.get_column("time_index")
    # time_index must be monotonic increasing 0..n-1
    assert list(ti.to_list()) == list(range(len(ti)))
