"""
Tests for Chronos -> ONNX feature alignment enforcement.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_POLARS
from ml.config.autogluon import AutoGluonDataConfig
from ml.config.autogluon import ChronosDistillationConfig
from ml.config.autogluon import ChronosOnnxDistillationConfig
from ml.config.autogluon import ChronosTrainingConfig
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash
from ml.training.autogluon.chronos_distillation import prepare_chronos_onnx_distillation_artifacts

if TYPE_CHECKING:
    import polars as pl


class DummyTimeSeriesFrame:
    """
    Lightweight stand-in for AutoGluon TimeSeriesDataFrame.
    """

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        *,
        id_column: str,
        timestamp_column: str,
    ) -> pd.DataFrame:
        ordered = [id_column, timestamp_column, "target"]
        ordered += [col for col in df.columns if col not in {id_column, timestamp_column, "target"}]
        return df[ordered]


class DummyPredictor:
    """
    Predictor stub emitting deterministic forecasts.
    """

    def __init__(self, prediction_length: int) -> None:
        self._prediction_length = prediction_length

    def make_future_data_frame(self, train_data: pd.DataFrame) -> pd.DataFrame:
        history = train_data.reset_index() if hasattr(train_data, "reset_index") else train_data
        last_ts = history["timestamp"].iloc[-1]
        if len(history) > 1:
            freq = history["timestamp"].iloc[-1] - history["timestamp"].iloc[-2]
        else:
            freq = pd.Timedelta(minutes=1)
        future_ts = [last_ts + freq * (step + 1) for step in range(self._prediction_length)]
        return pd.DataFrame(
            {
                "item_id": [str(history["item_id"].iloc[-1])] * self._prediction_length,
                "timestamp": future_ts,
            },
        )

    def predict(
        self,
        train_data: pd.DataFrame,
        *,
        known_covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        future_df = self.make_future_data_frame(train_data)
        output = future_df.copy()
        output["mean"] = np.zeros(len(output), dtype=float)
        return output


def _build_test_frame() -> pl.DataFrame:
    import polars as pl

    base_ts = 1704067200_000_000_000
    timestamps = [base_ts + i * 60_000_000_000 for i in range(6)]
    return pl.DataFrame(
        {
            "instrument_id": ["SPY"] * 6,
            "ts_event": timestamps,
            "time_index": list(range(6)),
            "forward_return": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "y": [0, 1, 0, 1, 0, 1],
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
    )


def _register_features(tmp_path: Path, feature_names: list[str]) -> tuple[str, str, str]:
    registry_dir = tmp_path / "feature_registry"
    registry = FeatureRegistry(registry_dir)
    dtypes = ["float32"] * len(feature_names)
    schema_hash = compute_schema_hash(feature_names, dtypes, "pipeline_sig")
    manifest = FeatureManifest(
        feature_set_id="feature_set_1",
        name="distill_features",
        version="1.0.0",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=feature_names,
        feature_dtypes=dtypes,
        schema_hash=schema_hash,
        pipeline_signature="pipeline_sig",
        pipeline_version="1.0",
    )
    feature_set_id = registry.register_feature_set(manifest)
    return str(registry_dir), feature_set_id, schema_hash


def _build_config(
    tmp_path: Path,
    *,
    feature_registry_dir: str,
    feature_set_id: str,
) -> ChronosOnnxDistillationConfig:
    teacher_config = ChronosTrainingConfig(
        prediction_length=2,
        data_config=AutoGluonDataConfig(),
    )
    distill_config = ChronosDistillationConfig(
        teacher_config=teacher_config,
        student_config=ChronosTrainingConfig(preset="bolt_small"),
        min_history=1,
        stride=1,
        forecast_step=1,
        label_strategy="teacher_only",
        max_windows_per_series=None,
        sample_fraction=None,
        export_soft_labels=False,
    )
    return ChronosOnnxDistillationConfig(
        distillation_config=distill_config,
        output_dir=str(tmp_path / "out"),
        feature_registry_dir=feature_registry_dir,
        feature_set_id=feature_set_id,
        registry_dir=str(tmp_path / "model_registry"),
        model_id="student_v1",
        parent_id="teacher_v1",
        train_fraction=0.5,
        output_transform="identity",
    )


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
def test_feature_alignment_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml.training.autogluon import soft_label_generator as slg

    monkeypatch.setattr(slg, "HAS_AUTOGLUON", True)
    monkeypatch.setattr(slg, "TimeSeriesDataFrame", DummyTimeSeriesFrame)

    df = _build_test_frame()
    feature_names = ["feature_a", "feature_missing"]
    registry_dir, feature_set_id, schema_hash = _register_features(tmp_path, feature_names)
    expected_hash = compute_schema_hash(
        feature_names,
        ["float32"] * len(feature_names),
        "pipeline_sig",
    )
    assert schema_hash == expected_hash
    config = _build_config(
        tmp_path,
        feature_registry_dir=registry_dir,
        feature_set_id=feature_set_id,
    )

    assert schema_hash
    with pytest.raises(ValueError, match="missing feature columns"):
        prepare_chronos_onnx_distillation_artifacts(
            df,
            DummyPredictor(2),
            config=config,
        )
