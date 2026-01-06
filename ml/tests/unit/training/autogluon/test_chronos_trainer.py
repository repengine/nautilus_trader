"""
Unit tests for Chronos trainer.

Tests cover configuration validation, data conversion, and trainer interface
without requiring actual AutoGluon installation (uses mocks where needed).

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ml._imports import HAS_AUTOGLUON
from ml._imports import HAS_POLARS
from ml.config.autogluon import AutoGluonDataConfig
from ml.config.autogluon import ChronosDistillationConfig
from ml.config.autogluon import ChronosTrainingConfig
from ml.config.autogluon import ChronosTuningConfig


if TYPE_CHECKING:
    import polars as pl


class TestChronosTrainingConfig:
    """Tests for ChronosTrainingConfig validation."""

    def test_default_config_valid(self) -> None:
        """Test that default configuration is valid."""
        config = ChronosTrainingConfig()

        assert config.prediction_length == 15
        assert config.preset == "chronos2"
        assert config.time_limit == 3600
        assert config.enable_ensemble is True
        assert config.num_val_windows == 1
        assert config.refit_every_n_windows == 1
        assert config.refit_full is False
        assert config.skip_model_selection is False
        assert config.freq == "min"
        assert config.eval_metric == "RMSE"
        assert config.enable_gpu is True

    def test_custom_config_valid(self) -> None:
        """Test custom configuration values."""
        config = ChronosTrainingConfig(
            prediction_length=30,
            preset="bolt_small",
            time_limit=1800,
            enable_ensemble=False,
            num_val_windows=2,
            refit_every_n_windows=2,
            refit_full=True,
            skip_model_selection=True,
            freq="h",
            eval_metric="MAE",
            enable_gpu=False,
        )

        assert config.prediction_length == 30
        assert config.preset == "bolt_small"
        assert config.time_limit == 1800
        assert config.enable_ensemble is False
        assert config.num_val_windows == 2
        assert config.refit_every_n_windows == 2
        assert config.refit_full is True
        assert config.skip_model_selection is True
        assert config.freq == "h"
        assert config.eval_metric == "MAE"
        assert config.enable_gpu is False

    def test_invalid_preset_raises(self) -> None:
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Invalid preset"):
            ChronosTrainingConfig(preset="invalid_preset")

    def test_invalid_eval_metric_raises(self) -> None:
        """Test that invalid eval_metric raises ValueError."""
        with pytest.raises(ValueError, match="Invalid eval_metric"):
            ChronosTrainingConfig(eval_metric="INVALID")

    def test_all_valid_presets(self) -> None:
        """Test that all documented presets are valid."""
        valid_presets = [
            "chronos_tiny",
            "chronos_mini",
            "chronos_small",
            "chronos_base",
            "chronos_large",
            "chronos2",
            "bolt_tiny",
            "bolt_mini",
            "bolt_small",
            "bolt_base",
        ]

        for preset in valid_presets:
            config = ChronosTrainingConfig(preset=preset)
            assert config.preset == preset

    def test_all_valid_eval_metrics(self) -> None:
        """Test that all documented eval metrics are valid."""
        # Standard uppercase metrics
        valid_metrics = ["RMSE", "MAE", "MAPE", "MASE", "SMAPE", "WAPE", "MSE"]

        for metric in valid_metrics:
            config = ChronosTrainingConfig(eval_metric=metric)
            assert config.eval_metric == metric

        # Special case: sMAPE with lowercase 's'
        config = ChronosTrainingConfig(eval_metric="sMAPE")
        assert config.eval_metric == "sMAPE"

    def test_get_data_config_returns_default(self) -> None:
        """Test get_data_config returns default when not specified."""
        config = ChronosTrainingConfig()
        data_config = config.get_data_config()

        assert isinstance(data_config, AutoGluonDataConfig)
        assert data_config.item_id_column == "instrument_id"
        assert data_config.timestamp_column == "ts_event"
        assert data_config.target_column == config.target_column

    def test_get_data_config_returns_custom(self) -> None:
        """Test get_data_config returns custom config when specified."""
        custom_data = AutoGluonDataConfig(
            item_id_column="symbol",
            timestamp_column="ts_event",
            target_column="y",
        )
        config = ChronosTrainingConfig(target_column="y", data_config=custom_data)

        data_config = config.get_data_config()
        assert data_config.item_id_column == "symbol"
        assert data_config.timestamp_column == "ts_event"
        assert data_config.target_column == "y"

    def test_data_config_target_column_mismatch_raises(self) -> None:
        """Test data_config target column mismatch raises ValueError."""
        with pytest.raises(ValueError, match=r"data_config\.target_column"):
            ChronosTrainingConfig(
                target_column="y",
                data_config=AutoGluonDataConfig(target_column="forward_return"),
            )

    def test_data_config_timestamp_mismatch_raises(self) -> None:
        """Test non-ts_event timestamp column raises ValueError."""
        with pytest.raises(ValueError, match="ts_event"):
            ChronosTrainingConfig(
                data_config=AutoGluonDataConfig(timestamp_column="timestamp"),
            )

    def test_tuning_requires_model_selection(self) -> None:
        """Test tuning config requires model selection enabled."""
        with pytest.raises(ValueError, match="tuning_config"):
            ChronosTrainingConfig(
                skip_model_selection=True,
                tuning_config=ChronosTuningConfig(num_trials=4),
            )


class TestAutoGluonDataConfig:
    """Tests for AutoGluonDataConfig."""

    def test_default_config(self) -> None:
        """Test default data configuration."""
        config = AutoGluonDataConfig()

        assert config.item_id_column == "instrument_id"
        assert config.timestamp_column == "ts_event"
        assert config.target_column == "forward_return"
        assert config.known_covariates == ()
        assert config.past_covariates == ()
        assert config.static_features == ()

    def test_custom_covariates(self) -> None:
        """Test configuration with custom covariates."""
        config = AutoGluonDataConfig(
            known_covariates=("hour", "dow", "holiday"),
            past_covariates=("return_1", "volume_ratio"),
            static_features=("asset_class", "exchange"),
        )

        assert len(config.known_covariates) == 3
        assert "hour" in config.known_covariates
        assert len(config.past_covariates) == 2
        assert len(config.static_features) == 2


class TestChronosDistillationConfig:
    """Tests for ChronosDistillationConfig."""

    def test_default_distillation_config(self) -> None:
        """Test default distillation configuration."""
        teacher = ChronosTrainingConfig(preset="chronos2")
        student = ChronosTrainingConfig(preset="bolt_small")

        config = ChronosDistillationConfig(
            teacher_config=teacher,
            student_config=student,
        )

        assert config.enable_distillation is True
        assert config.soft_label_temperature == 1.0
        assert config.distillation_alpha == 0.5
        assert config.export_soft_labels is True

    def test_invalid_distillation_alpha_raises(self) -> None:
        """Test that distillation_alpha > 1.0 raises ValueError."""
        teacher = ChronosTrainingConfig(preset="chronos2")
        student = ChronosTrainingConfig(preset="bolt_small")

        with pytest.raises(ValueError, match="distillation_alpha must be between 0 and 1"):
            ChronosDistillationConfig(
                teacher_config=teacher,
                student_config=student,
                distillation_alpha=1.5,
            )


class TestChronosDistillationPipeline:
    """Tests for the distillation pipeline wiring."""

    def test_train_teacher_student_builds_student_config(self) -> None:
        """Ensure distilled target column is used for the student."""
        from ml.training.autogluon.chronos_trainer import train_teacher_student

        teacher = ChronosTrainingConfig(preset="chronos2")
        student = ChronosTrainingConfig(preset="bolt_small")
        config = ChronosDistillationConfig(
            teacher_config=teacher,
            student_config=student,
        )

        mock_teacher = MagicMock()
        mock_teacher.train.return_value = {"metrics": {}}
        mock_teacher.predictor = MagicMock()

        mock_student = MagicMock()
        mock_student.train.return_value = {"metrics": {}}

        distillation_result = MagicMock()
        distillation_result.data = MagicMock()
        distillation_result.labels = MagicMock()
        distillation_result.stats.coverage = 1.0
        distillation_result.stats.generated = 1
        distillation_result.stats.total_candidates = 1
        distillation_result.stats.eligible_candidates = 1
        distillation_result.stats.used_series = 1
        distillation_result.stats.total_series = 1

        with patch("ml.training.autogluon.chronos_trainer.ChronosTrainer") as trainer_cls:
            trainer_cls.side_effect = [mock_teacher, mock_student]
            with patch(
                "ml.training.autogluon.chronos_trainer.build_distillation_dataset",
                return_value=distillation_result,
            ):
                result = train_teacher_student(MagicMock(), config)

        student_call = trainer_cls.call_args_list[1]
        student_config = student_call.args[0]
        assert student_config.target_column == config.distilled_target_column
        assert result["student"] is mock_student


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestDataConversion:
    """Tests for data conversion utilities."""

    def test_validate_nautilus_dataset_valid(self) -> None:
        """Test validation passes for valid dataset."""
        import polars as pl

        from ml.data.autogluon_adapter import validate_nautilus_dataset

        config = AutoGluonDataConfig()
        df = pl.DataFrame({
            "instrument_id": ["SPY", "SPY", "AAPL", "AAPL"],
            "ts_event": [1000000000, 2000000000, 1000000000, 2000000000],
            "forward_return": [0.01, -0.005, 0.02, 0.015],
        })

        errors = validate_nautilus_dataset(df, config)
        assert len(errors) == 0

    def test_validate_nautilus_dataset_missing_columns(self) -> None:
        """Test validation fails for missing columns."""
        import polars as pl

        from ml.data.autogluon_adapter import validate_nautilus_dataset

        config = AutoGluonDataConfig()
        df = pl.DataFrame({
            "symbol": ["SPY", "AAPL"],  # Wrong column name
            "ts_event": [1000000000, 2000000000],
        })

        errors = validate_nautilus_dataset(df, config)
        assert len(errors) > 0
        assert any("instrument_id" in e for e in errors)

    def test_validate_nautilus_dataset_accepts_timestamp_alias(self) -> None:
        """Test validation accepts timestamp alias for ts_event."""
        import polars as pl

        from ml.data.autogluon_adapter import validate_nautilus_dataset

        config = AutoGluonDataConfig()
        df = pl.DataFrame({
            "instrument_id": ["SPY", "SPY", "AAPL", "AAPL"],
            "timestamp": [1000000000, 2000000000, 1000000000, 2000000000],
            "forward_return": [0.01, -0.005, 0.02, 0.015],
        })

        errors = validate_nautilus_dataset(df, config)
        assert len(errors) == 0

    def test_compute_forward_return(self) -> None:
        """Test forward return computation."""
        import polars as pl

        from ml.data.autogluon_adapter import compute_forward_return

        df = pl.DataFrame({
            "instrument_id": ["SPY"] * 5,
            "ts_event": [1, 2, 3, 4, 5],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        })

        result = compute_forward_return(df, horizon=2, price_col="close")

        assert "forward_return" in result.columns
        # forward_return[0] = (102 - 100) / 100 = 0.02
        np.testing.assert_almost_equal(result["forward_return"][0], 0.02)

    def test_compute_forward_return_per_instrument(self) -> None:
        """Test forward return computation is per instrument."""
        import polars as pl

        from ml.data.autogluon_adapter import compute_forward_return

        df = pl.DataFrame({
            "instrument_id": ["SPY", "SPY", "AAPL", "AAPL"],
            "ts_event": [1, 2, 1, 2],
            "close": [100.0, 110.0, 200.0, 220.0],
        })

        result = compute_forward_return(df, horizon=1, price_col="close")

        # SPY return: (110 - 100) / 100 = 0.10
        spy_return = result.filter(pl.col("instrument_id") == "SPY")["forward_return"][0]
        np.testing.assert_almost_equal(spy_return, 0.10)

        # AAPL return: (220 - 200) / 200 = 0.10
        aapl_return = result.filter(pl.col("instrument_id") == "AAPL")["forward_return"][0]
        np.testing.assert_almost_equal(aapl_return, 0.10)

        # Last row per instrument should be null after shift
        assert result.filter(pl.col("instrument_id") == "SPY")["forward_return"][1] is None
        assert result.filter(pl.col("instrument_id") == "AAPL")["forward_return"][1] is None

    def test_compute_forward_return_accepts_timestamp_alias(self) -> None:
        """Test forward return computation accepts timestamp alias for ts_event."""
        import polars as pl

        from ml.data.autogluon_adapter import compute_forward_return

        df = pl.DataFrame({
            "instrument_id": ["SPY"] * 4,
            "timestamp": [1, 2, 3, 4],
            "close": [100.0, 101.0, 102.0, 103.0],
        })

        result = compute_forward_return(df, horizon=2, price_col="close")

        assert "ts_event" in result.columns
        assert "timestamp" not in result.columns
        assert "forward_return" in result.columns

    def test_extract_covariates(self) -> None:
        """Test covariate extraction."""
        import polars as pl

        from ml.data.autogluon_adapter import extract_covariates

        config = AutoGluonDataConfig(
            known_covariates=("hour", "dow", "missing_col"),
            past_covariates=("return_1",),
            static_features=("asset_class",),
        )
        df = pl.DataFrame({
            "hour": [9, 10, 11],
            "dow": [1, 2, 3],
            "return_1": [0.01, 0.02, 0.03],
            "asset_class": ["equity", "equity", "equity"],
        })

        covariates = extract_covariates(df, config)

        assert "hour" in covariates["known"]
        assert "dow" in covariates["known"]
        assert "missing_col" not in covariates["known"]  # Not in DataFrame
        assert "return_1" in covariates["past"]
        assert "asset_class" in covariates["static"]


@pytest.mark.skipif(not HAS_AUTOGLUON, reason="AutoGluon not available")
class TestChronosTrainerIntegration:
    """Integration tests requiring AutoGluon installation."""

    @pytest.fixture
    def sample_tsdf(self) -> MagicMock:
        """Create a mock TimeSeriesDataFrame for testing."""
        mock_tsdf = MagicMock()
        mock_tsdf.item_ids = ["SPY", "AAPL"]
        mock_tsdf.__len__ = lambda self: 100
        return mock_tsdf

    def test_trainer_initialization(self) -> None:
        """Test trainer initializes correctly."""
        from ml.training.autogluon.chronos_trainer import ChronosTrainer

        config = ChronosTrainingConfig(preset="bolt_small", time_limit=60)
        trainer = ChronosTrainer(config)

        assert trainer.config == config
        assert trainer.predictor is None
        assert trainer.is_fitted is False

    def test_trainer_not_fitted_raises_on_predict(self) -> None:
        """Test that predict raises when not fitted."""
        from ml.training.autogluon.chronos_trainer import ChronosTrainer

        config = ChronosTrainingConfig()
        trainer = ChronosTrainer(config)

        with pytest.raises(ValueError, match="must be trained"):
            trainer.predict(MagicMock())

    def test_trainer_not_fitted_raises_on_save(self) -> None:
        """Test that save raises when not fitted."""
        from ml.training.autogluon.chronos_trainer import ChronosTrainer

        config = ChronosTrainingConfig()
        trainer = ChronosTrainer(config)

        with pytest.raises(ValueError, match="must be trained"):
            trainer.save("/tmp/model")

    def test_get_model_info_not_fitted(self) -> None:
        """Test get_model_info when not fitted."""
        from ml.training.autogluon.chronos_trainer import ChronosTrainer

        config = ChronosTrainingConfig()
        trainer = ChronosTrainer(config)

        info = trainer.get_model_info()
        assert info["status"] == "not_fitted"


class TestChronosTeacherConfig:
    """Tests for ChronosTeacherConfig."""

    def test_default_teacher_config(self) -> None:
        """Test default teacher configuration."""
        from ml.training.teacher.chronos_teacher import ChronosTeacherConfig

        config = ChronosTeacherConfig()

        assert config.architecture == "Chronos-2"
        assert config.preset == "chronos2"
        assert config.prediction_length == 15
        assert config.time_limit == 3600

    def test_custom_teacher_config(self) -> None:
        """Test custom teacher configuration."""
        from ml.training.teacher.chronos_teacher import ChronosTeacherConfig

        config = ChronosTeacherConfig(
            preset="chronos_large",
            prediction_length=30,
            time_limit=7200,
        )

        assert config.preset == "chronos_large"
        assert config.prediction_length == 30
        assert config.time_limit == 7200


@pytest.mark.skipif(not HAS_AUTOGLUON, reason="AutoGluon not available")
class TestChronosTeacher:
    """Tests for ChronosTeacher."""

    def test_teacher_initialization(self) -> None:
        """Test teacher initializes correctly."""
        from ml.training.teacher.chronos_teacher import ChronosTeacher
        from ml.training.teacher.chronos_teacher import ChronosTeacherConfig

        config = ChronosTeacherConfig()
        teacher = ChronosTeacher(config)

        assert teacher.config == config
        assert teacher._is_fitted is False

    def test_teacher_not_fitted_raises_on_predict(self) -> None:
        """Test that predict raises when not fitted."""
        from ml.training.teacher.chronos_teacher import ChronosTeacher
        from ml.training.teacher.chronos_teacher import ChronosTeacherConfig

        config = ChronosTeacherConfig()
        teacher = ChronosTeacher(config)

        with pytest.raises(ValueError, match="must be fitted"):
            teacher.predict_logits(np.array([[1, 2, 3]]))

    def test_teacher_feature_schema_empty_when_not_fitted(self) -> None:
        """Test feature_schema returns empty dict when not fitted."""
        from ml.training.teacher.chronos_teacher import ChronosTeacher
        from ml.training.teacher.chronos_teacher import ChronosTeacherConfig

        config = ChronosTeacherConfig()
        teacher = ChronosTeacher(config)

        schema = teacher.feature_schema()
        assert schema == {}
