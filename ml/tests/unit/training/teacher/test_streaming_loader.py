from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from ml._imports import HAS_PANDAS
from ml._imports import HAS_TORCH
from ml._imports import check_ml_dependencies
from ml._imports import pd
import ml.common.metrics_bootstrap as metrics_bootstrap
import ml.training.teacher.streaming_loader as streaming_loader_module
from ml.training.teacher.streaming_loader import (
    StreamingLimitSummary,
    TFTStreamingConfig,
    TFTStreamingDataModule,
    TFTStreamingDataset,
    TFTStreamingSummary,
    TFTStreamingMetadata,
    TFTShardIndex,
    apply_streaming_limits,
    build_streaming_dataloader,
    collect_streaming_metadata,
    count_sequences,
    filter_metadata_by_instruments,
    filter_metadata_by_shard_ids,
    instrument_row_counts,
    is_within_shard_budget,
    materialize_streaming_frame,
    resolve_shard_order,
    split_metadata_by_row_fraction,
    split_metadata_by_time,
    summarize_metadata,
)
from ml.training.datasets.time_series_formatter import TimeSeriesFormatter

try:  # optional dependency for parity comparison
    from pytorch_forecasting import TimeSeriesDataSet

    HAS_PYTORCH_FORECASTING = True
except Exception:  # pragma: no cover - dependency missing
    TimeSeriesDataSet = None  # type: ignore[assignment]
    HAS_PYTORCH_FORECASTING = False

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def _to_pandas(frame: Any) -> Any:
    if pd is None:
        raise RuntimeError("pandas dependency required for conversion")
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return frame


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
        target_col="feature_a",
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
    assert metadata.instrument_target_stats["AAPL"].count == 3
    assert metadata.instrument_target_stats["MSFT"].count == 3
    assert pytest.approx(metadata.instrument_target_stats["AAPL"].mean, rel=1e-6) == float(
        frame.loc[frame["instrument_id"] == "AAPL", "feature_a"].mean(),
    )
    assert pytest.approx(metadata.instrument_target_stats["MSFT"].mean, rel=1e-6) == float(
        frame.loc[frame["instrument_id"] == "MSFT", "feature_a"].mean(),
    )

    train_meta, val_meta = split_metadata_by_time(metadata, cutoff_time=2)
    assert len(train_meta.shard_indices) >= 1
    assert len(val_meta.shard_indices) >= 1

    frac_train, frac_val = split_metadata_by_row_fraction(metadata, train_fraction=0.5)
    assert frac_train.shard_indices
    assert frac_val.shard_indices
    assert {shard.instrument_id for shard in frac_train.shard_indices} == {"AAPL", "MSFT"}
    assert {shard.instrument_id for shard in frac_val.shard_indices} == {"AAPL", "MSFT"}

    filtered = filter_metadata_by_instruments(metadata, ["AAPL"])
    assert all(shard.instrument_id == "AAPL" for shard in filtered.shard_indices)

    shard_counter = getattr(streaming_loader_module, "_METADATA_SHARDS_COUNTER", None)
    counter_value = getattr(getattr(shard_counter, "_value", None), "get", lambda: 0)()
    assert counter_value >= summary.total_shards


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_collect_streaming_metadata_includes_target_stats_when_missing_from_features(
    tmp_path: Path,
) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    frame = pd.DataFrame(
        {
            "time_index": np.arange(4, dtype=np.int64),
            "instrument_id": ["AAPL"] * 4,
            "feature": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "y": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "dataset.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature",),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
        shard_row_budget=10,
    )

    target_stats = metadata.numeric_stats["y"]
    assert target_stats.count == 4
    assert pytest.approx(target_stats.mean, rel=1e-6) == float(frame["y"].mean())
    assert metadata.instrument_target_stats["AAPL"].count == 4


@pytest.mark.skipif(not HAS_PANDAS or not HAS_TORCH, reason="pandas/torch dependency required")
def test_streaming_loader_rss_metrics_when_iterating_emits_scan_and_batch_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_prometheus_registry: Any,
) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")
    if not getattr(streaming_loader_module, "HAS_PYARROW", False):
        pytest.skip("pyarrow dependency required")

    monkeypatch.setattr(streaming_loader_module, "current_rss_mb", lambda: 128.0)

    frame = pd.DataFrame(
        {
            "time_index": np.arange(6, dtype=np.int64),
            "instrument_id": ["AAPL"] * 6,
            "feature_a": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
            "y": np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.0], dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "dataset.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature_a",),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature_a", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
        shard_row_budget=10,
    )

    config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=(),
        static_reals=(),
        time_varying_known_reals=("feature_a",),
        time_varying_unknown_reals=(),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=1,
        num_workers=0,
    )

    loader = build_streaming_dataloader(parquet_path, metadata, config)
    next(iter(loader))

    registry = isolated_prometheus_registry.registry
    scan_value = registry.get_sample_value(
        "ml_tft_streaming_rss_mb",
        labels={"stage": "scan"},
    )
    assembly_value = registry.get_sample_value(
        "ml_tft_streaming_rss_mb",
        labels={"stage": "batch_assembly"},
    )
    assert scan_value == pytest.approx(128.0)
    assert assembly_value == pytest.approx(128.0)


def test_filter_metadata_by_shard_ids_recomputes_counts() -> None:
    shards = (
        TFTShardIndex(
            shard_id="A-1",
            instrument_id="AAPL",
            row_start=0,
            row_end=9,
            row_count=10,
            time_start=0,
            time_end=9,
        ),
        TFTShardIndex(
            shard_id="A-2",
            instrument_id="AAPL",
            row_start=10,
            row_end=19,
            row_count=10,
            time_start=10,
            time_end=19,
        ),
        TFTShardIndex(
            shard_id="B-1",
            instrument_id="MSFT",
            row_start=0,
            row_end=4,
            row_count=5,
            time_start=0,
            time_end=4,
        ),
    )
    metadata = TFTStreamingMetadata(
        shard_indices=shards,
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts={"AAPL": 20, "MSFT": 5},
        instrument_target_stats={
            "AAPL": streaming_loader_module.RunningStats(count=20, mean=1.0, m2=2.0),
            "MSFT": streaming_loader_module.RunningStats(count=5, mean=2.0, m2=1.0),
        },
    )

    filtered = filter_metadata_by_shard_ids(metadata, ["A-2", "B-1"])

    assert {shard.shard_id for shard in filtered.shard_indices} == {"A-2", "B-1"}
    assert filtered.instrument_row_counts == {"AAPL": 10, "MSFT": 5}
    assert set(filtered.instrument_target_stats.keys()) == {"AAPL", "MSFT"}


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
        target_col="y",
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
    decoder_times = batch_inputs["decoder_time_idx"].detach().cpu().numpy()
    assert decoder_times.dtype == np.int64
    assert batch_inputs["target_scale"].shape == (2, 2)
    assert batch_inputs["groups"].shape == (2, 1)
    decoder_group_ids = batch_inputs["decoder_group_ids"].detach().cpu().numpy()
    group_ids = batch_inputs["groups"].detach().cpu().numpy()
    assert decoder_group_ids.shape == (2, 1)
    assert np.array_equal(decoder_group_ids[:, 0], group_ids[:, 0])
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
    decoder_group_ids_train = module_batch_inputs["decoder_group_ids"].detach().cpu().numpy()
    group_ids_train = module_batch_inputs["groups"].detach().cpu().numpy()
    assert decoder_group_ids_train.shape[0] == group_ids_train.shape[0]
    assert np.array_equal(decoder_group_ids_train[:, 0], group_ids_train[:, 0])
    assert module_targets.shape == (2, 1)

    if module_val_loader is not None:
        module_val_batch_inputs, (module_val_targets, _) = next(iter(module_val_loader))
        assert module_val_batch_inputs["decoder_cont"].shape[0] >= 1
        decoder_group_ids_val = module_val_batch_inputs["decoder_group_ids"].detach().cpu().numpy()
        group_ids_val = module_val_batch_inputs["groups"].detach().cpu().numpy()
        assert decoder_group_ids_val.shape[0] == group_ids_val.shape[0]
        assert np.array_equal(decoder_group_ids_val[:, 0], group_ids_val[:, 0])
    assert module_val_targets.ndim == 2

    iteration_counter = getattr(streaming_loader_module, "_ITERATION_SHARDS_COUNTER", None)
    iter_value = getattr(getattr(iteration_counter, "_value", None), "get", lambda: 0)()
    assert iter_value >= 1


@pytest.mark.skipif(not (HAS_PANDAS and HAS_TORCH), reason="pandas and torch required")
def test_streaming_dataset_preserves_large_time_indices(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    base_time = 23_668_320
    total_rows = 12
    frame = pd.DataFrame(
        {
            "time_index": np.arange(base_time, base_time + total_rows, dtype=np.int64),
            "instrument_id": np.array(["WFC"] * total_rows, dtype=object),
            "y": np.linspace(0.0, 1.0, num=total_rows, dtype=np.float32),
            "feature": np.linspace(1.0, 2.0, num=total_rows, dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "large_times.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
        shard_row_budget=64,
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
        max_prediction_length=2,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=0,
        num_workers=0,
    )
    loader = build_streaming_dataloader(parquet_path, metadata, config)
    observed_times: list[int] = []
    for batch_inputs, _ in loader:
        decoder_times = batch_inputs["decoder_time_idx"].detach().cpu().numpy()
        assert decoder_times.dtype == np.int64
        observed_times.extend(decoder_times.reshape(-1).astype(int, copy=False).tolist())
    expected_times = frame["time_index"].to_numpy(dtype=np.int64)
    expected_window = set(expected_times[config.max_encoder_length :].tolist())
    assert set(observed_times) == expected_window


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
        target_col="y",
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
        max_encoder_length=2,
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


def test_split_metadata_by_row_fraction_preserves_instrument_coverage_manual() -> None:
    shards = (
        TFTShardIndex(
            shard_id="aapl-0",
            instrument_id="AAPL",
            row_start=0,
            row_end=10,
            row_count=10,
            time_start=100,
            time_end=109,
        ),
        TFTShardIndex(
            shard_id="aapl-1",
            instrument_id="AAPL",
            row_start=10,
            row_end=20,
            row_count=10,
            time_start=110,
            time_end=119,
        ),
        TFTShardIndex(
            shard_id="msft-0",
            instrument_id="MSFT",
            row_start=0,
            row_end=12,
            row_count=12,
            time_start=200,
            time_end=211,
        ),
        TFTShardIndex(
            shard_id="msft-1",
            instrument_id="MSFT",
            row_start=12,
            row_end=24,
            row_count=12,
            time_start=212,
            time_end=223,
        ),
    )
    metadata = TFTStreamingMetadata(
        shard_indices=shards,
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts={"AAPL": 20, "MSFT": 24},
    )
    train_meta, val_meta = split_metadata_by_row_fraction(metadata, train_fraction=0.3)
    train_instruments = {shard.instrument_id for shard in train_meta.shard_indices}
    val_instruments = {shard.instrument_id for shard in val_meta.shard_indices}
    assert train_instruments == {"AAPL", "MSFT"}
    assert val_instruments == {"AAPL", "MSFT"}


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_build_streaming_dataloader_respects_limits(tmp_path: Path) -> None:
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
    parquet_path = tmp_path / "cap_limits.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
        shard_row_budget=6,
    )

    base_config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=0,
        num_workers=0,
    )

    limited_shards_config = replace(base_config, max_shards=1)
    loader = build_streaming_dataloader(parquet_path, metadata, limited_shards_config)
    dataset = loader.dataset  # type: ignore[attr-defined]
    assert isinstance(dataset, TFTStreamingDataset)
    assert len(dataset._metadata.shard_indices) == 1  # type: ignore[attr-defined]

    limited_sequences_config = replace(base_config, max_total_sequences=4)
    loader_seq = build_streaming_dataloader(parquet_path, metadata, limited_sequences_config)
    dataset_seq = loader_seq.dataset  # type: ignore[attr-defined]
    assert len(dataset_seq._metadata.shard_indices) == 1  # type: ignore[attr-defined]

    strict_rows_config = replace(base_config, max_total_rows=3)
    with pytest.raises(RuntimeError):
        build_streaming_dataloader(parquet_path, metadata, strict_rows_config)


@pytest.mark.skipif(not (HAS_PANDAS and HAS_TORCH), reason="pandas and torch required")
def test_apply_streaming_limits_and_count_sequences(tmp_path: Path) -> None:
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
    parquet_path = tmp_path / "limits.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
        shard_row_budget=3,
    )
    total_rows = sum(metadata.instrument_row_counts.values())

    config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=0,
        num_workers=0,
        max_shards=2,
    )

    assert config.max_encoder_length == 2
    assert all(shard.row_count >= 3 for shard in metadata.shard_indices)

    limited, summary = apply_streaming_limits(metadata, config)
    assert isinstance(summary, StreamingLimitSummary)
    assert len(limited.shard_indices) == 2
    assert summary.skipped_shards == max(0, len(metadata.shard_indices) - 2)
    selected_rows = sum(limited.instrument_row_counts.values())
    assert selected_rows <= total_rows
    assert summary.skipped_rows == max(0, total_rows - selected_rows)

    selected_sequences = count_sequences(limited, config)
    assert selected_sequences >= 0
    assert summary.total_instrument_rows == dict(metadata.instrument_row_counts)
    assert summary.selected_instrument_rows == dict(limited.instrument_row_counts)
    expected_skipped_rows = {
        instrument: count - limited.instrument_row_counts.get(instrument, 0)
        for instrument, count in metadata.instrument_row_counts.items()
        if count != limited.instrument_row_counts.get(instrument, 0)
    }
    assert summary.skipped_instrument_rows == expected_skipped_rows
    assert sum(summary.selected_instrument_rows.values()) + sum(summary.skipped_instrument_rows.values()) == sum(
        summary.total_instrument_rows.values(),
    )

    loader = build_streaming_dataloader(
        parquet_path,
        limited,
        config,
        metadata_is_limited=True,
        limit_summary=summary,
    )
    batch_inputs, (targets, _) = next(iter(loader))
    assert "encoder_cont" in batch_inputs
    assert targets.shape[0] > 0


@pytest.mark.skipif(not (HAS_PANDAS and HAS_TORCH), reason="pandas and torch required")
def test_streaming_limits_round_robin_instrument_mix(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    frame = pd.DataFrame(
        {
            "time_index": np.arange(18, dtype=np.int64),
            "instrument_id": ["AAPL"] * 6 + ["MSFT"] * 6 + ["GOOG"] * 6,
            "y": np.linspace(0.0, 17.0, num=18, dtype=np.float32),
            "feature": np.linspace(5.0, 23.0, num=18, dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "round_robin.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
        shard_row_budget=3,
    )

    config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=0,
        num_workers=0,
        max_shards=4,
        max_total_rows=None,
        max_total_sequences=None,
    )

    limited, summary = apply_streaming_limits(metadata, config)
    assert len(limited.shard_indices) == 4
    assert set(limited.instrument_row_counts) == {"AAPL", "MSFT", "GOOG"}
    assert summary.total_instrument_rows == dict(metadata.instrument_row_counts)
    assert summary.selected_instrument_rows == dict(limited.instrument_row_counts)
    assert summary.skipped_instrument_rows
    assert sum(summary.selected_instrument_rows.values()) + sum(summary.skipped_instrument_rows.values()) == sum(
        summary.total_instrument_rows.values(),
    )


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
        target_col="y",
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


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_materialize_streaming_frame_matches_time_series_formatter(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")
    if not streaming_loader_module.HAS_PYARROW:
        check_ml_dependencies(["pyarrow"])
        pytest.skip("pyarrow import guard triggered")

    frame = pd.DataFrame(
        {
            "time_index": np.arange(6, dtype=np.int64),
            "ts_event": np.arange(6, dtype=np.int64) + 1,
            "instrument_id": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "y": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            "feature": np.array([1.0, 1.1, 1.2, 2.0, 2.1, 2.2], dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "dataset.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
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
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=0,
        num_workers=0,
    )

    streaming_frame = materialize_streaming_frame(
        parquet_path,
        metadata=metadata,
        config=config,
        max_rows=len(frame),
    )

    formatter = TimeSeriesFormatter()
    offline_frame = formatter.format_for_tft(
        frame,
        lookback_periods=config.max_encoder_length,
        use_polars=False,
        timestamp_col="ts_event",
        time_index_col="time_index",
        group_id_col="instrument_id",
    )

    streaming_pd = _to_pandas(streaming_frame).sort_values(
        ["instrument_id", "time_index"],
        kind="mergesort",
    )
    offline_pd = _to_pandas(offline_frame).sort_values(
        ["instrument_id", "time_index"],
        kind="mergesort",
    )

    streaming_pd = streaming_pd.reset_index(drop=True)
    offline_pd = offline_pd.reset_index(drop=True)

    assert "sequence_id" in streaming_pd.columns
    assert "sequence_id" in offline_pd.columns
    assert np.issubdtype(streaming_pd["time_index"].dtype, np.integer)
    assert np.issubdtype(offline_pd["time_index"].dtype, np.integer)

    common_cols = ["instrument_id", "time_index", "y", "feature"]
    pd.testing.assert_frame_equal(
        streaming_pd[common_cols],
        offline_pd[common_cols],
        check_dtype=False,
    )


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_resolve_shard_order_deterministic(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")
    if not streaming_loader_module.HAS_PYARROW:
        check_ml_dependencies(["pyarrow"])
        pytest.skip("pyarrow import guard triggered")

    frame = pd.DataFrame(
        {
            "time_index": np.arange(8, dtype=np.int64),
            "instrument_id": ["AAPL"] * 4 + ["MSFT"] * 4,
            "y": np.linspace(0.0, 0.7, num=8, dtype=np.float32),
            "feature": np.linspace(1.0, 1.7, num=8, dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "order.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
        shard_row_budget=2,
    )

    order_a = resolve_shard_order(metadata, shuffle=True, seed=11).tolist()
    order_b = resolve_shard_order(metadata, shuffle=True, seed=11).tolist()
    assert order_a == order_b
    if len(order_a) > 1:
        order_c = resolve_shard_order(metadata, shuffle=True, seed=12).tolist()
        assert order_a != order_c

    sequential = resolve_shard_order(metadata, shuffle=False, seed=99).tolist()
    assert sequential == list(range(len(metadata.shard_indices)))
try:  # optional dependency for regression parity test
    from pytorch_forecasting import TimeSeriesDataSet

    HAS_PYTORCH_FORECASTING = True
except Exception:  # pragma: no cover - optional dependency guard
    TimeSeriesDataSet = None  # type: ignore[assignment]
    HAS_PYTORCH_FORECASTING = False
