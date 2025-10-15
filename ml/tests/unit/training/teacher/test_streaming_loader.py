from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from ml._imports import HAS_PANDAS
from ml._imports import HAS_TORCH
from ml._imports import check_ml_dependencies
from ml._imports import pd
import ml.common.metrics_bootstrap as metrics_bootstrap
from ml.training.teacher.streaming_loader import (
    TFTStreamingConfig,
    TFTStreamingDataModule,
    TFTStreamingDataset,
    TFTStreamingSummary,
    build_streaming_dataloader,
    collect_streaming_metadata,
    filter_metadata_by_instruments,
    instrument_row_counts,
    is_within_shard_budget,
    split_metadata_by_row_fraction,
    split_metadata_by_time,
    summarize_metadata,
)

try:  # optional dependency for parity comparison
    from pytorch_forecasting import TimeSeriesDataSet

    HAS_PYTORCH_FORECASTING = True
except Exception:  # pragma: no cover - dependency missing
    TimeSeriesDataSet = None  # type: ignore[assignment]
    HAS_PYTORCH_FORECASTING = False


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_collect_streaming_metadata_counts_and_vocab(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    frame = pd.DataFrame(
        {
            "time_index": np.arange(6, dtype=np.int64),
            "instrument_id": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "feature_a": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
            "feature_b": np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
            "category_x": ["x", "y", "x", "y", "z", "z"],
        },
    )
    parquet_path = tmp_path / "dataset.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature_a", "feature_b", "category_x"),
        categorical_columns=("category_x", "instrument_id"),
        numeric_columns=("feature_a", "feature_b"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        shard_row_budget=3,
    )

    summary = summarize_metadata(metadata)
    assert summary.total_shards == len(metadata.shard_indices)
    assert summary.total_rows == 6
    assert summary.max_shard_rows <= 3
    assert is_within_shard_budget(summary, shard_row_budget=3)

    counts = instrument_row_counts(metadata)
    assert counts == {"AAPL": 3, "MSFT": 3}
    assert metadata.categorical_vocab["category_x"] == ("x", "y", "z")
    assert metadata.categorical_vocab["instrument_id"] == ("AAPL", "MSFT")

    stats = metadata.numeric_stats["feature_a"]
    assert stats.count == 6
    assert pytest.approx(stats.mean, rel=1e-6) == float(frame["feature_a"].mean())
    assert pytest.approx(stats.variance, rel=1e-6) == float(frame["feature_a"].var(ddof=1))

    train_meta, val_meta = split_metadata_by_time(metadata, cutoff_time=2)
    assert len(train_meta.shard_indices) >= 1
    assert len(val_meta.shard_indices) >= 1

    frac_train, frac_val = split_metadata_by_row_fraction(metadata, train_fraction=0.5)
    assert frac_train.shard_indices
    assert frac_val.shard_indices

    filtered = filter_metadata_by_instruments(metadata, ["AAPL"])
    assert all(shard.instrument_id == "AAPL" for shard in filtered.shard_indices)

    metrics_keys = metrics_bootstrap._METRICS.keys()
    assert "ml_tft_streaming_metadata_shards_total||()" in metrics_keys


@pytest.mark.skipif(not (HAS_PANDAS and HAS_TORCH), reason="pandas and torch required")
def test_streaming_dataloader_emits_expected_batch(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    frame = pd.DataFrame(
        {
            "time_index": np.arange(8, dtype=np.int64),
            "instrument_id": ["AAPL"] * 8,
            "y": np.linspace(0.0, 7.0, num=8, dtype=np.float32),
            "feature": np.linspace(10.0, 80.0, num=8, dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "stream.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        shard_row_budget=32,
    )

    config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=3,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=0,
        num_workers=0,
    )

    frac_train, frac_val = split_metadata_by_row_fraction(metadata, train_fraction=0.5)
    assert frac_train.shard_indices
    assert frac_val.shard_indices
    dataloader = build_streaming_dataloader(parquet_path, metadata, config)
    batch_inputs, (targets, weights) = next(iter(dataloader))

    assert weights is None
    assert batch_inputs["encoder_cont"].shape == (2, 3, 1)
    assert batch_inputs["decoder_cont"].shape == (2, 1, 1)
    assert batch_inputs["encoder_cat"].shape == (2, 3, 1)
    assert batch_inputs["decoder_cat"].shape == (2, 1, 1)
    assert batch_inputs["encoder_target"].shape == (2, 3)
    assert batch_inputs["decoder_target"].shape == (2, 1)
    assert batch_inputs["decoder_time_idx"].shape == (2, 1)
    assert batch_inputs["target_scale"].shape == (2, 2)
    assert batch_inputs["groups"].shape == (2, 1)
    assert targets.shape == (2, 1)

    data_module = TFTStreamingDataModule(
        parquet_path,
        config=config,
        train_metadata=metadata,
        val_metadata=frac_val,
        shuffle_train=False,
    )
    data_module.setup(stage="fit")
    module_train_loader = data_module.train_dataloader()
    module_val_loader = data_module.val_dataloader()

    module_batch_inputs, (module_targets, _) = next(iter(module_train_loader))
    assert module_batch_inputs["encoder_cont"].shape == (2, 3, 1)
    assert module_targets.shape == (2, 1)

    if module_val_loader is not None:
        module_val_batch_inputs, (module_val_targets, _) = next(iter(module_val_loader))
        assert module_val_batch_inputs["decoder_cont"].shape[0] >= 1
        assert module_val_targets.ndim == 2

    metric_keys = metrics_bootstrap._METRICS.keys()
    assert "ml_tft_streaming_iterated_shards_total||()" in metric_keys


@pytest.mark.skipif(not (HAS_PANDAS and HAS_TORCH), reason="pandas and torch required")
def test_streaming_dataset_shard_partitioning(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    instruments: Sequence[str] = ("AAPL", "MSFT")
    rows: list[int] = list(range(12))
    frame = pd.DataFrame(
        {
            "time_index": np.arange(len(rows), dtype=np.int64),
            "instrument_id": np.array([instruments[i // 6] for i in rows]),
            "y": np.linspace(0.0, 11.0, num=len(rows), dtype=np.float32),
            "feature": np.linspace(1.0, 12.0, num=len(rows), dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "shards.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        shard_row_budget=2,
    )

    config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=3,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=True,
        seed=11,
        num_workers=0,
    )

    dataset = build_streaming_dataloader(parquet_path, metadata, config).dataset  # type: ignore[attr-defined]
    assert isinstance(dataset, TFTStreamingDataset)

    base_seed = config.seed or 0
    shard_order = np.random.default_rng(base_seed).permutation(len(metadata.shard_indices))
    worker0 = dataset._resolve_shards(worker_id=0, num_workers=2, order=shard_order)
    worker1 = dataset._resolve_shards(worker_id=1, num_workers=2, order=shard_order)

    all_ids = {shard.shard_id for shard in metadata.shard_indices}
    worker0_ids = {shard.shard_id for shard in worker0}
    worker1_ids = {shard.shard_id for shard in worker1}

    assert worker0_ids.isdisjoint(worker1_ids)
    assert worker0_ids.union(worker1_ids) == all_ids


def test_is_within_shard_budget_guard() -> None:
    summary = TFTStreamingSummary(total_shards=2, total_rows=10, max_shard_rows=12)
    assert is_within_shard_budget(summary, shard_row_budget=12)
    assert not is_within_shard_budget(summary, shard_row_budget=10, tolerance_pct=0.0)


@pytest.mark.skipif(
    not (HAS_PANDAS and HAS_TORCH and HAS_PYTORCH_FORECASTING),
    reason="pandas, torch, and pytorch-forecasting required",
)
def test_streaming_matches_pytorch_forecasting_batch(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    frame = pd.DataFrame(
        {
            "time_index": np.arange(12, dtype=np.int64),
            "instrument_id": ["AAPL"] * 6 + ["MSFT"] * 6,
            "y": np.linspace(0.0, 11.0, num=12, dtype=np.float32),
            "feature": np.linspace(1.0, 12.0, num=12, dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "regression.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        shard_row_budget=32,
    )

    config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=4,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=0,
        num_workers=0,
    )

    stream_loader = build_streaming_dataloader(parquet_path, metadata, config)
    stream_inputs, (stream_targets, _) = next(iter(stream_loader))

    training = TimeSeriesDataSet(
        frame,
        time_idx="time_index",
        target="y",
        group_ids=["instrument_id"],
        max_encoder_length=config.max_encoder_length,
        max_prediction_length=config.max_prediction_length,
        static_categoricals=list(config.static_categoricals),
        static_reals=list(config.static_reals),
        time_varying_known_reals=list(config.time_varying_known_reals),
        time_varying_unknown_reals=list(config.time_varying_unknown_reals),
    )
    pf_loader = training.to_dataloader(train=True, batch_size=2, num_workers=0, shuffle=False)
    pf_inputs, pf_targets = next(iter(pf_loader))

    np.testing.assert_allclose(
        stream_inputs["encoder_cont"].detach().cpu().numpy(),
        pf_inputs["encoder_cont"].detach().cpu().numpy(),
        atol=1e-1,
    )
    np.testing.assert_allclose(
        stream_inputs["decoder_cont"].detach().cpu().numpy(),
        pf_inputs["decoder_cont"].detach().cpu().numpy(),
        atol=1e-1,
    )
    np.testing.assert_allclose(
        stream_inputs["encoder_target"].detach().cpu().numpy(),
        pf_inputs["encoder_target"].detach().cpu().numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        stream_inputs["decoder_target"].detach().cpu().numpy(),
        pf_inputs["decoder_target"].detach().cpu().numpy(),
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        stream_inputs["encoder_cat"].detach().cpu().numpy(),
        pf_inputs["encoder_cat"].detach().cpu().numpy(),
    )
    np.testing.assert_array_equal(
        stream_inputs["decoder_cat"].detach().cpu().numpy(),
        pf_inputs["decoder_cat"].detach().cpu().numpy(),
    )
    np.testing.assert_array_equal(
        stream_inputs["groups"].detach().cpu().numpy(),
        pf_inputs["groups"].detach().cpu().numpy(),
    )
    np.testing.assert_allclose(
        stream_inputs["target_scale"].detach().cpu().numpy(),
        pf_inputs["target_scale"].detach().cpu().numpy(),
        atol=2e-1,
    )
    np.testing.assert_allclose(
        stream_targets.detach().cpu().numpy(),
        pf_targets[0].detach().cpu().numpy(),
        atol=1e-5,
    )
try:  # optional dependency for regression parity test
    from pytorch_forecasting import TimeSeriesDataSet

    HAS_PYTORCH_FORECASTING = True
except Exception:  # pragma: no cover - optional dependency guard
    TimeSeriesDataSet = None  # type: ignore[assignment]
    HAS_PYTORCH_FORECASTING = False
