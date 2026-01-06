"""
Integration tests for Chronos training pipeline.

These tests verify end-to-end training workflows with AutoGluon TimeSeries,
including data conversion, model training, and soft label generation.

Note: These tests require AutoGluon to be installed and may use GPU resources.
They are marked appropriately for CI/CD filtering.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ml._imports import HAS_AUTOGLUON
from ml._imports import HAS_POLARS


if TYPE_CHECKING:
    import polars as pl


@pytest.fixture
def sample_training_data() -> MagicMock | pl.DataFrame:
    """Create sample training data for integration tests."""
    if not HAS_POLARS:
        pytest.skip("Polars not available")

    import polars as pl

    # Create synthetic time series data for multiple instruments
    n_per_instrument = 500
    instruments = ["SPY", "AAPL", "MSFT"]

    dfs = []
    for i, inst in enumerate(instruments):
        # Generate timestamps (1 minute apart, in nanoseconds)
        base_ts = 1704067200_000_000_000  # 2024-01-01 00:00:00
        timestamps = [base_ts + j * 60_000_000_000 for j in range(n_per_instrument)]

        # Generate synthetic price data with trend and noise
        np.random.seed(42 + i)
        base_price = 100.0 + i * 50
        prices = base_price + np.cumsum(np.random.randn(n_per_instrument) * 0.1)

        # Generate forward returns (will be computed fresh)
        close_prices = prices.tolist()

        df = pl.DataFrame({
            "instrument_id": [inst] * n_per_instrument,
            "ts_event": timestamps,
            "close": close_prices,
            "volume": np.random.randint(1000, 100000, n_per_instrument).tolist(),
            "hour": [(t // 3600_000_000_000) % 24 for t in timestamps],
            "dow": [(t // 86400_000_000_000) % 7 for t in timestamps],
        })
        dfs.append(df)

    return pl.concat(dfs)


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestDataConversionIntegration:
    """Integration tests for data conversion pipeline."""

    def test_full_data_conversion_pipeline(self, sample_training_data: pl.DataFrame) -> None:
        """Test complete data conversion from Polars to TimeSeriesDataFrame."""
        from ml.config.autogluon import AutoGluonDataConfig
        from ml.config.autogluon import ChronosTrainingConfig
        from ml.data.autogluon_adapter import compute_forward_return
        from ml.data.autogluon_adapter import validate_nautilus_dataset

        # Add forward returns
        df = compute_forward_return(
            sample_training_data,
            horizon=15,
            price_col="close",
            output_col="forward_return",
        )

        # Validate dataset
        config = AutoGluonDataConfig(
            known_covariates=("hour", "dow"),
        )
        errors = validate_nautilus_dataset(df, config)
        assert len(errors) == 0, f"Validation errors: {errors}"

        # Verify forward return computation
        assert "forward_return" in df.columns
        assert df["forward_return"].null_count() == 15 * 3  # 15 nulls per instrument

    @pytest.mark.skipif(not HAS_AUTOGLUON, reason="AutoGluon not available")
    def test_convert_to_timeseries_dataframe(self, sample_training_data: pl.DataFrame) -> None:
        """Test conversion to AutoGluon TimeSeriesDataFrame."""
        from ml.config.autogluon import ChronosTrainingConfig
        from ml.data.autogluon_adapter import compute_forward_return
        from ml.data.autogluon_adapter import convert_to_timeseries_dataframe

        # Add forward returns
        df = compute_forward_return(sample_training_data, horizon=15)

        config = ChronosTrainingConfig(
            prediction_length=15,
            preset="bolt_small",
        )

        tsdf = convert_to_timeseries_dataframe(df, config)

        # Verify TimeSeriesDataFrame structure
        assert tsdf is not None
        assert hasattr(tsdf, "item_ids")
        assert len(tsdf.item_ids) == 3  # SPY, AAPL, MSFT


@pytest.mark.skipif(not HAS_AUTOGLUON, reason="AutoGluon not available")
@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
@pytest.mark.integration
class TestChronosTrainerIntegration:
    """Integration tests for ChronosTrainer end-to-end workflows."""

    @pytest.mark.slow
    def test_trainer_fit_predict_cycle(
        self,
        sample_training_data: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test complete fit-predict cycle with small time limit.

        Note: This test actually trains a model, so it may take a few minutes.
        Use --slow flag to include this test.
        """
        from ml.config.autogluon import AutoGluonDataConfig
        from ml.config.autogluon import ChronosTrainingConfig
        from ml.data.autogluon_adapter import compute_forward_return
        from ml.training.autogluon.chronos_trainer import ChronosTrainer

        # Prepare data
        df = compute_forward_return(sample_training_data, horizon=15)

        # Use minimal config for fast testing
        config = ChronosTrainingConfig(
            prediction_length=5,  # Short horizon
            preset="bolt_tiny",  # Smallest/fastest model
            time_limit=60,  # 1 minute max
            enable_gpu=False,  # CPU for CI
            save_path=str(tmp_path / "test_model"),
            verbosity=0,  # Quiet
            data_config=AutoGluonDataConfig(
                known_covariates=("hour",),
            ),
        )

        # Train
        trainer = ChronosTrainer(config)
        result = trainer.train(df)

        # Verify training completed
        assert trainer.is_fitted
        assert "metrics" in result
        assert result["training_time"] > 0

        # Verify prediction works
        predictions = trainer.predict(df)
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)

        # Verify soft label generation
        soft_labels = trainer.generate_soft_labels(df)
        assert soft_labels.dtype == np.float64

    def test_trainer_model_info(self, sample_training_data: pl.DataFrame) -> None:
        """Test model info reporting for unfitted trainer."""
        from ml.config.autogluon import ChronosTrainingConfig
        from ml.training.autogluon.chronos_trainer import ChronosTrainer

        config = ChronosTrainingConfig()
        trainer = ChronosTrainer(config)

        info = trainer.get_model_info()

        assert info["status"] == "not_fitted"


@pytest.mark.skipif(not HAS_AUTOGLUON, reason="AutoGluon not available")
@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
@pytest.mark.integration
class TestChronosTeacherIntegration:
    """Integration tests for ChronosTeacher distillation."""

    def test_teacher_fit_generates_soft_labels(
        self,
        sample_training_data: pl.DataFrame,
    ) -> None:
        """Test teacher training and soft label generation.

        Uses mocking to avoid actual training in CI.
        """
        from ml.data.autogluon_adapter import compute_forward_return
        from ml.training.teacher.chronos_teacher import ChronosTeacher
        from ml.training.teacher.chronos_teacher import ChronosTeacherConfig

        # Prepare data
        df = compute_forward_return(sample_training_data, horizon=15)

        config = ChronosTeacherConfig(
            preset="bolt_tiny",
            prediction_length=5,
            time_limit=60,
        )

        # Mock the trainer to avoid actual training
        with patch("ml.training.teacher.chronos_teacher.ChronosTrainer") as mock_trainer_cls:
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {
                "metrics": {"RMSE": 0.01},
                "feature_names": ["hour"],
            }
            mock_trainer.generate_soft_labels.return_value = np.random.randn(len(df) - 45)
            mock_trainer_cls.return_value = mock_trainer

            teacher = ChronosTeacher(config)
            teacher.fit(df)

            assert teacher._is_fitted

            # Generate soft labels
            soft_labels = teacher.get_soft_labels(df)
            assert soft_labels is not None


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestCLIIntegration:
    """Integration tests for CLI entry point."""

    def test_cli_parse_args(self) -> None:
        """Test CLI argument parsing."""
        from ml.cli.train_chronos import parse_args

        args = parse_args([
            "--symbols", "SPY,AAPL",
            "--preset", "bolt_small",
            "--time-limit", "300",
        ])

        assert args.symbols == "SPY,AAPL"
        assert args.preset == "bolt_small"
        assert args.time_limit == 300

    def test_cli_parse_distillation_args(self) -> None:
        """Test CLI distillation argument parsing."""
        from ml.cli.train_chronos import parse_args

        args = parse_args([
            "--symbols", "SPY",
            "--distill",
            "--teacher-preset", "chronos2",
            "--student-preset", "bolt_small",
        ])

        assert args.distill is True
        assert args.teacher_preset == "chronos2"
        assert args.student_preset == "bolt_small"

    def test_cli_symbol_parsing(self) -> None:
        """Test symbol list parsing."""
        from ml.cli.train_chronos import _parse_symbols

        symbols = _parse_symbols("SPY, AAPL, MSFT")
        assert symbols == ["SPY", "AAPL", "MSFT"]

        symbols = _parse_symbols("spy,aapl")
        assert symbols == ["SPY", "AAPL"]

    def test_cli_symbol_parsing_empty_raises(self) -> None:
        """Test that empty symbol list raises error."""
        from ml.cli.train_chronos import _parse_symbols

        with pytest.raises(ValueError, match="At least one symbol"):
            _parse_symbols("")
