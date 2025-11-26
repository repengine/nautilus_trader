"""
Unit tests for DataPreparationComponent.

This module tests the data preparation component extracted from BaseMLTrainer
(lines 284-357 and 1496-1517). Tests verify:
- FeatureStore integration for training data preparation
- Label generation from features
- Train/validation data splitting
- Error handling for missing store or empty features
- Property tests for split invariants

Following the test design in reports/tests/phase_3_8_test_design_report.md.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings, strategies as st

from ml.training.common.data_preparation import (
    DataPreparationComponent,
    DataPreparationTrainerProtocol,
)


# ============================================================================
# Mock Fixtures
# ============================================================================


@dataclass
class MockConfig:
    """Mock training configuration for testing."""

    data_source: str = "memory"
    target_column: str = "target"
    train_test_split: float = 0.8
    db_connection: str | None = None


class MockFeatureStore:
    """Mock FeatureStore for testing."""

    def __init__(
        self,
        features: npt.NDArray[np.float64] | None = None,
        timestamps: npt.NDArray[np.int64] | None = None,
        feature_names: list[str] | None = None,
        compute_return_value: int = 0,
    ) -> None:
        """Initialize mock feature store with configurable data."""
        self._features = features
        self._timestamps = timestamps
        self._feature_names = feature_names or []
        self._compute_return_value = compute_return_value
        # Track method calls for assertions
        self.call_log: list[str] = []
        self.compute_calls: list[dict[str, Any]] = []
        self.get_training_data_calls: list[dict[str, Any]] = []

    def compute_and_store_historical(
        self,
        instrument_id: str,
        start: Any,
        end: Any,
        force_recompute: bool = False,
    ) -> int:
        """Mock implementation of compute_and_store_historical."""
        self.call_log.append("compute_and_store_historical")
        self.compute_calls.append({
            "instrument_id": instrument_id,
            "start": start,
            "end": end,
            "force_recompute": force_recompute,
        })
        return self._compute_return_value

    def get_training_data(
        self,
        instrument_id: str,
        start: Any,
        end: Any,
        include_bars: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
        """Mock implementation of get_training_data."""
        self.call_log.append("get_training_data")
        self.get_training_data_calls.append({
            "instrument_id": instrument_id,
            "start": start,
            "end": end,
            "include_bars": include_bars,
        })
        features = self._features if self._features is not None else np.array([])
        timestamps = self._timestamps if self._timestamps is not None else np.array([])
        return features, timestamps, self._feature_names


class TestableTrainer:
    """
    Concrete trainer implementation for testing DataPreparationComponent.

    Implements the DataPreparationTrainerProtocol interface with mock implementations.
    """

    def __init__(
        self,
        config: MockConfig | None = None,
        feature_store: MockFeatureStore | None = None,
    ) -> None:
        """Initialize testable trainer."""
        self._config = config or MockConfig()
        self._feature_store: MockFeatureStore | None = feature_store
        self._feature_names: list[str] = []
        # Track method calls for assertions
        self._call_log: list[str] = []
        self._info_messages: list[str] = []
        self._warning_messages: list[str] = []

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """Mock implementation of _log_info."""
        self._call_log.append("_log_info")
        formatted_message = message % args if args else message
        self._info_messages.append(formatted_message)

    def _log_warning(self, message: str, *args: object, **kwargs: Any) -> None:
        """Mock implementation of _log_warning."""
        self._call_log.append("_log_warning")
        formatted_message = message % args if args else message
        self._warning_messages.append(formatted_message)


# ============================================================================
# Mock DataFrame for Testing
# ============================================================================


class MockDataFrame:
    """Mock DataFrame for testing without polars dependency."""

    def __init__(self, data: dict[str, list[Any]]) -> None:
        self._data = data
        self._columns = list(data.keys())

    @property
    def columns(self) -> list[str]:
        return self._columns

    def __len__(self) -> int:
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def __getitem__(self, key: str | slice) -> Any:
        if isinstance(key, str):
            return MockColumn(self._data[key])
        elif isinstance(key, slice):
            new_data = {}
            for col_name, col_values in self._data.items():
                new_data[col_name] = col_values[key]
            return MockDataFrame(new_data)
        return self


class MockColumn:
    """Mock column for testing."""

    def __init__(self, data: list[Any]) -> None:
        self._data = data

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return np.array(self._data, dtype=np.float64)

    def __len__(self) -> int:
        return len(self._data)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_feature_array() -> npt.NDArray[np.float64]:
    """Create sample feature array for testing."""
    np.random.seed(42)
    return np.random.randn(100, 5).astype(np.float64)


@pytest.fixture
def sample_timestamps() -> npt.NDArray[np.int64]:
    """Create sample timestamps array for testing."""
    # Nanosecond timestamps starting from a fixed point
    start_ns = 1704067200000000000  # 2024-01-01 00:00:00 UTC
    interval_ns = 60_000_000_000  # 1 minute intervals
    return np.arange(start_ns, start_ns + 100 * interval_ns, interval_ns, dtype=np.int64)


@pytest.fixture
def mock_feature_store(
    sample_feature_array: npt.NDArray[np.float64],
    sample_timestamps: npt.NDArray[np.int64],
) -> MockFeatureStore:
    """Create mock FeatureStore with sample data."""
    return MockFeatureStore(
        features=sample_feature_array,
        timestamps=sample_timestamps,
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        compute_return_value=0,
    )


@pytest.fixture
def mock_feature_store_empty() -> MockFeatureStore:
    """Create mock FeatureStore that returns empty data."""
    return MockFeatureStore(
        features=np.array([], dtype=np.float64).reshape(0, 5),
        timestamps=np.array([], dtype=np.int64),
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        compute_return_value=0,
    )


@pytest.fixture
def mock_feature_store_with_computation(
    sample_feature_array: npt.NDArray[np.float64],
    sample_timestamps: npt.NDArray[np.int64],
) -> MockFeatureStore:
    """Create mock FeatureStore that indicates feature computation occurred."""
    return MockFeatureStore(
        features=sample_feature_array,
        timestamps=sample_timestamps,
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        compute_return_value=100,  # 100 rows computed
    )


@pytest.fixture
def trainer_fixture() -> TestableTrainer:
    """Create basic TestableTrainer instance without feature store."""
    return TestableTrainer(MockConfig())


@pytest.fixture
def trainer_with_feature_store_fixture(mock_feature_store: MockFeatureStore) -> TestableTrainer:
    """Create TestableTrainer with FeatureStore."""
    config = MockConfig(db_connection="postgresql://test")
    return TestableTrainer(config, feature_store=mock_feature_store)


@pytest.fixture
def sample_training_dataframe() -> MockDataFrame:
    """Create sample DataFrame for training tests."""
    np.random.seed(42)
    n_samples = 100
    return MockDataFrame({
        "feature_1": list(np.random.randn(n_samples)),
        "feature_2": list(np.random.randn(n_samples)),
        "feature_3": list(np.random.randn(n_samples)),
        "target": list(np.random.randint(0, 2, n_samples)),
    })


# ============================================================================
# Happy Path Tests
# ============================================================================


class TestPrepareDataWithFeatureStoreBasic:
    """Tests for basic FeatureStore integration."""

    def test_prepare_data_with_feature_store_basic(
        self,
        trainer_with_feature_store_fixture: TestableTrainer,
    ) -> None:
        """Verify feature store integration works and returns correct tuple."""
        data_prep = DataPreparationComponent(trainer_with_feature_store_fixture)
        feature_store = trainer_with_feature_store_fixture._feature_store
        assert feature_store is not None

        X, y, feature_names = data_prep.prepare_data_with_feature_store(
            instrument_id="BTC-USD",
            start="2024-01-01",
            end="2024-01-31",
        )

        # Verify return types and shapes
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)

        # Verify shapes match
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(feature_names)

        # Verify get_training_data was called
        assert "get_training_data" in feature_store.call_log

    def test_prepare_data_with_feature_store_updates_trainer_feature_names(
        self,
        trainer_with_feature_store_fixture: TestableTrainer,
    ) -> None:
        """Verify trainer's feature names are updated after data preparation."""
        data_prep = DataPreparationComponent(trainer_with_feature_store_fixture)

        # Feature names should be empty initially
        assert trainer_with_feature_store_fixture._feature_names == []

        _, _, feature_names = data_prep.prepare_data_with_feature_store(
            instrument_id="BTC-USD",
            start="2024-01-01",
            end="2024-01-31",
        )

        # Feature names should be updated
        assert trainer_with_feature_store_fixture._feature_names == feature_names
        assert len(trainer_with_feature_store_fixture._feature_names) == 5


class TestPrepareDataWithFeatureStoreComputesMissing:
    """Tests for feature computation when missing."""

    def test_prepare_data_with_feature_store_computes_missing(
        self,
        mock_feature_store_with_computation: MockFeatureStore,
    ) -> None:
        """Verify features computed when missing."""
        config = MockConfig(db_connection="postgresql://test")
        trainer = TestableTrainer(config, feature_store=mock_feature_store_with_computation)
        data_prep = DataPreparationComponent(trainer)

        data_prep.prepare_data_with_feature_store(
            instrument_id="BTC-USD",
            start="2024-01-01",
            end="2024-01-31",
            compute_if_missing=True,
        )

        # Verify compute_and_store_historical was called
        assert "compute_and_store_historical" in mock_feature_store_with_computation.call_log

        # Verify log message about computed rows
        assert any("Computed 100 feature rows" in msg for msg in trainer._info_messages)

    def test_prepare_data_with_feature_store_logs_loaded_samples(
        self,
        trainer_with_feature_store_fixture: TestableTrainer,
    ) -> None:
        """Verify log message about loaded samples."""
        data_prep = DataPreparationComponent(trainer_with_feature_store_fixture)

        data_prep.prepare_data_with_feature_store(
            instrument_id="BTC-USD",
            start="2024-01-01",
            end="2024-01-31",
        )

        # Verify log message about loaded samples
        assert any("Loaded 100 samples with 5 features" in msg
                   for msg in trainer_with_feature_store_fixture._info_messages)


class TestPrepareDataWithFeatureStoreSkipsCompute:
    """Tests for skipping feature computation."""

    def test_prepare_data_with_feature_store_skips_compute(
        self,
        trainer_with_feature_store_fixture: TestableTrainer,
    ) -> None:
        """Verify computation skipped when compute_if_missing=False."""
        data_prep = DataPreparationComponent(trainer_with_feature_store_fixture)
        feature_store = trainer_with_feature_store_fixture._feature_store
        assert feature_store is not None

        data_prep.prepare_data_with_feature_store(
            instrument_id="BTC-USD",
            start="2024-01-01",
            end="2024-01-31",
            compute_if_missing=False,
        )

        # Verify compute_and_store_historical was NOT called
        assert "compute_and_store_historical" not in feature_store.call_log

        # Verify get_training_data WAS called
        assert "get_training_data" in feature_store.call_log


class TestGenerateLabelsBasic:
    """Tests for label generation."""

    def test_generate_labels_basic(
        self,
        trainer_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_timestamps: npt.NDArray[np.int64],
    ) -> None:
        """Verify simple label generation."""
        data_prep = DataPreparationComponent(trainer_fixture)

        labels = data_prep._generate_labels(sample_feature_array, sample_timestamps)

        # Verify shape matches features
        assert labels.shape[0] == sample_feature_array.shape[0]

        # Verify labels are 0 or 1
        assert set(np.unique(labels)).issubset({0.0, 1.0})

    def test_generate_labels_returns_float64(
        self,
        trainer_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_timestamps: npt.NDArray[np.int64],
    ) -> None:
        """Verify labels are float64 type."""
        data_prep = DataPreparationComponent(trainer_fixture)

        labels = data_prep._generate_labels(sample_feature_array, sample_timestamps)

        assert labels.dtype == np.float64

    def test_generate_labels_empty_features(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify handling of empty features."""
        data_prep = DataPreparationComponent(trainer_fixture)

        empty_features = np.array([], dtype=np.float64).reshape(0, 0)
        empty_timestamps = np.array([], dtype=np.int64)

        labels = data_prep._generate_labels(empty_features, empty_timestamps)

        # Should return empty array
        assert len(labels) == 0


class TestSplitDataRespectsRatio:
    """Tests for data splitting."""

    def test_split_data_respects_ratio(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify data split follows config ratio."""
        data_prep = DataPreparationComponent(trainer_fixture)

        train_data, val_data = data_prep._split_data(sample_training_dataframe)

        # Default ratio is 0.8
        expected_train_size = int(100 * 0.8)
        expected_val_size = 100 - expected_train_size

        assert len(train_data) == expected_train_size
        assert len(val_data) == expected_val_size

    def test_split_data_with_custom_ratio(
        self,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify data split with custom ratio."""
        config = MockConfig(train_test_split=0.7)
        trainer = TestableTrainer(config)
        data_prep = DataPreparationComponent(trainer)

        train_data, val_data = data_prep._split_data(sample_training_dataframe)

        expected_train_size = int(100 * 0.7)
        expected_val_size = 100 - expected_train_size

        assert len(train_data) == expected_train_size
        assert len(val_data) == expected_val_size


# ============================================================================
# Error Condition Tests
# ============================================================================


class TestPrepareDataWithFeatureStoreRaisesWithoutStore:
    """Tests for error when FeatureStore not configured."""

    def test_prepare_data_with_feature_store_raises_without_store(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify error when feature store not configured."""
        data_prep = DataPreparationComponent(trainer_fixture)

        with pytest.raises(ValueError, match="FeatureStore not configured"):
            data_prep.prepare_data_with_feature_store(
                instrument_id="BTC-USD",
                start="2024-01-01",
                end="2024-01-31",
            )


class TestPrepareDataWithFeatureStoreRaisesOnNoFeatures:
    """Tests for error when no features found."""

    def test_prepare_data_with_feature_store_raises_on_no_features(
        self,
        mock_feature_store_empty: MockFeatureStore,
    ) -> None:
        """Verify error when no features found."""
        config = MockConfig(db_connection="postgresql://test")
        trainer = TestableTrainer(config, feature_store=mock_feature_store_empty)
        data_prep = DataPreparationComponent(trainer)

        with pytest.raises(ValueError, match="No features found for BTC-USD"):
            data_prep.prepare_data_with_feature_store(
                instrument_id="BTC-USD",
                start="2024-01-01",
                end="2024-01-31",
            )


# ============================================================================
# Property Tests (Hypothesis)
# ============================================================================


class TestSplitDataPropertyTests:
    """Property tests for data splitting invariants."""

    @given(st.integers(min_value=10, max_value=1000))
    @settings(max_examples=50)
    def test_split_data_preserves_total_count(self, n_samples: int) -> None:
        """Train + val count equals original count."""
        config = MockConfig(train_test_split=0.8)
        trainer = TestableTrainer(config)
        data_prep = DataPreparationComponent(trainer)

        # Create data with n_samples rows
        data = MockDataFrame({
            "feature_1": list(np.random.randn(n_samples)),
            "target": list(np.random.randint(0, 2, n_samples)),
        })

        train_data, val_data = data_prep._split_data(data)

        assert len(train_data) + len(val_data) == n_samples

    @given(st.floats(min_value=0.1, max_value=0.9))
    @settings(max_examples=50)
    def test_split_data_ratio_within_bounds(self, ratio: float) -> None:
        """Split ratio always produces valid subsets."""
        config = MockConfig(train_test_split=ratio)
        trainer = TestableTrainer(config)
        data_prep = DataPreparationComponent(trainer)

        n_samples = 100
        data = MockDataFrame({
            "feature_1": list(np.random.randn(n_samples)),
            "target": list(np.random.randint(0, 2, n_samples)),
        })

        train_data, val_data = data_prep._split_data(data)

        # Both subsets should be non-empty
        assert 0 < len(train_data) < n_samples
        assert 0 < len(val_data) < n_samples

        # Total should equal original
        assert len(train_data) + len(val_data) == n_samples

    @given(
        st.integers(min_value=10, max_value=500),
        st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=50)
    def test_split_data_train_size_correct(self, n_samples: int, ratio: float) -> None:
        """Train size matches expected based on ratio."""
        config = MockConfig(train_test_split=ratio)
        trainer = TestableTrainer(config)
        data_prep = DataPreparationComponent(trainer)

        data = MockDataFrame({
            "feature_1": list(np.random.randn(n_samples)),
            "target": list(np.random.randint(0, 2, n_samples)),
        })

        train_data, _ = data_prep._split_data(data)

        expected_train_size = int(n_samples * ratio)
        assert len(train_data) == expected_train_size


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_testable_trainer_implements_protocol(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify TestableTrainer implements DataPreparationTrainerProtocol."""
        # Check required attributes
        assert hasattr(trainer_fixture, "_config")
        assert hasattr(trainer_fixture, "_feature_store")
        assert hasattr(trainer_fixture, "_feature_names")

        # Check required methods
        assert callable(getattr(trainer_fixture, "_log_info", None))
        assert callable(getattr(trainer_fixture, "_log_warning", None))

    def test_data_preparation_component_accepts_protocol(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify DataPreparationComponent accepts protocol-compliant trainer."""
        # Should not raise
        component = DataPreparationComponent(trainer_fixture)
        assert component._trainer is trainer_fixture


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_split_data_with_numpy_array(
        self,
        trainer_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
    ) -> None:
        """Verify split works with numpy array."""
        data_prep = DataPreparationComponent(trainer_fixture)

        train_data, val_data = data_prep._split_data(sample_feature_array)

        assert len(train_data) + len(val_data) == len(sample_feature_array)

    def test_split_data_with_list(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify split works with list."""
        data_prep = DataPreparationComponent(trainer_fixture)

        data = list(range(100))
        train_data, val_data = data_prep._split_data(data)

        assert len(train_data) + len(val_data) == 100

    def test_generate_labels_single_column_features(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify label generation with single column features."""
        data_prep = DataPreparationComponent(trainer_fixture)

        features = np.random.randn(50, 1).astype(np.float64)
        timestamps = np.arange(50, dtype=np.int64)

        labels = data_prep._generate_labels(features, timestamps)

        assert labels.shape[0] == features.shape[0]

    def test_prepare_data_passes_correct_args_to_feature_store(
        self,
        trainer_with_feature_store_fixture: TestableTrainer,
    ) -> None:
        """Verify correct arguments passed to feature store methods."""
        data_prep = DataPreparationComponent(trainer_with_feature_store_fixture)
        feature_store = trainer_with_feature_store_fixture._feature_store
        assert feature_store is not None

        data_prep.prepare_data_with_feature_store(
            instrument_id="ETH-USD",
            start="2024-02-01",
            end="2024-02-28",
            compute_if_missing=True,
        )

        # Verify correct args to compute_and_store_historical
        assert len(feature_store.compute_calls) == 1
        assert feature_store.compute_calls[0]["instrument_id"] == "ETH-USD"
        assert feature_store.compute_calls[0]["start"] == "2024-02-01"
        assert feature_store.compute_calls[0]["end"] == "2024-02-28"
        assert feature_store.compute_calls[0]["force_recompute"] is False

        # Verify correct args to get_training_data
        assert len(feature_store.get_training_data_calls) == 1
        assert feature_store.get_training_data_calls[0]["instrument_id"] == "ETH-USD"
        assert feature_store.get_training_data_calls[0]["include_bars"] is True


# ============================================================================
# Logging Tests
# ============================================================================


class TestLogging:
    """Tests for logging behavior."""

    def test_prepare_data_logs_info_messages(
        self,
        trainer_with_feature_store_fixture: TestableTrainer,
    ) -> None:
        """Verify info messages are logged during data preparation."""
        data_prep = DataPreparationComponent(trainer_with_feature_store_fixture)

        data_prep.prepare_data_with_feature_store(
            instrument_id="BTC-USD",
            start="2024-01-01",
            end="2024-01-31",
        )

        # Verify _log_info was called
        assert "_log_info" in trainer_with_feature_store_fixture._call_log

        # Verify loaded samples message
        assert len(trainer_with_feature_store_fixture._info_messages) > 0


__all__ = [
    "MockConfig",
    "MockFeatureStore",
    "TestableTrainer",
]
