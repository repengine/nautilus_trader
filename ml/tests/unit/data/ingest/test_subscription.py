"""
Tests for unified subscription policy and checker.

Validates SubscriptionPolicy, SubscriptionChecker, and helper functions.
"""

from __future__ import annotations

import os
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.data.ingest.subscription import SubscriptionChecker
from ml.data.ingest.subscription import SubscriptionPolicy
from ml.data.ingest.subscription import get_effective_policy
from ml.data.ingest.subscription import get_max_lookback_days


class TestSubscriptionPolicy:
    """Test SubscriptionPolicy dataclass and methods."""

    def test_default_policy_initialization(self) -> None:
        """Test policy with default values."""
        policy = SubscriptionPolicy()

        assert policy.allowed_datasets is None
        assert policy.allowed_schemas is None
        assert policy.allowed_symbols is None
        assert policy.max_days is None
        assert policy.earliest is None
        assert policy.latest is None
        assert policy.max_symbols is None
        assert policy.strict is False
        assert policy.l0_max_lookback_days == 365 * 7
        assert policy.l1_max_lookback_days == 365
        assert policy.l2_max_lookback_days == 30
        assert policy.l3_max_lookback_days == 30

    def test_policy_with_custom_values(self) -> None:
        """Test policy with custom values."""
        policy = SubscriptionPolicy(
            allowed_datasets={"XNAS.ITCH"},
            allowed_schemas={"ohlcv-1m", "trades"},
            max_days=100,
            strict=True,
            l0_max_lookback_days=1000,
        )

        assert policy.allowed_datasets == {"XNAS.ITCH"}
        assert policy.allowed_schemas == {"ohlcv-1m", "trades"}
        assert policy.max_days == 100
        assert policy.strict is True
        assert policy.l0_max_lookback_days == 1000

    def test_allow_dataset(self) -> None:
        """Test dynamically allowing datasets."""
        policy = SubscriptionPolicy()
        assert policy.allowed_datasets is None

        policy.allow_dataset("XNAS.ITCH")
        assert policy.allowed_datasets == {"XNAS.ITCH"}

        policy.allow_dataset("GLBX.MDP3")
        assert policy.allowed_datasets == {"XNAS.ITCH", "GLBX.MDP3"}

        # Empty string should be ignored
        policy.allow_dataset("")
        assert policy.allowed_datasets == {"XNAS.ITCH", "GLBX.MDP3"}

    def test_validate_dataset_schema_success(self) -> None:
        """Test successful validation of dataset and schema."""
        policy = SubscriptionPolicy(
            allowed_datasets={"XNAS.ITCH"},
            allowed_schemas={"ohlcv-1m"},
        )

        # Should not raise
        policy.validate_dataset_schema(dataset="XNAS.ITCH", schema="ohlcv-1m")

    def test_validate_dataset_schema_dataset_not_allowed(self) -> None:
        """Test validation fails when dataset not in allowlist."""
        policy = SubscriptionPolicy(allowed_datasets={"XNAS.ITCH"})

        with pytest.raises(
            PermissionError,
            match=r"Dataset 'GLBX.MDP3' is not in allowed set",
        ):
            policy.validate_dataset_schema(dataset="GLBX.MDP3", schema="trades")

    def test_validate_dataset_schema_schema_not_allowed(self) -> None:
        """Test validation fails when schema not in allowlist."""
        policy = SubscriptionPolicy(
            allowed_datasets={"XNAS.ITCH"},
            allowed_schemas={"ohlcv-1m"},
        )

        with pytest.raises(PermissionError, match="Schema 'trades' is not in allowed set"):
            policy.validate_dataset_schema(dataset="XNAS.ITCH", schema="trades")

    def test_clamp_range_no_constraints(self) -> None:
        """Test range clamping with no policy constraints."""
        policy = SubscriptionPolicy()
        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2021, 1, 1, tzinfo=UTC)

        clamped_start, clamped_end = policy.clamp_range(start, end)

        assert clamped_start == start
        assert clamped_end == end

    def test_clamp_range_earliest_constraint(self) -> None:
        """Test range clamping with earliest date constraint."""
        earliest = datetime(2020, 6, 1, tzinfo=UTC)
        policy = SubscriptionPolicy(earliest=earliest)

        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2021, 1, 1, tzinfo=UTC)

        clamped_start, clamped_end = policy.clamp_range(start, end)

        assert clamped_start == earliest
        assert clamped_end == end

    def test_clamp_range_latest_constraint(self) -> None:
        """Test range clamping with latest date constraint."""
        latest = datetime(2020, 12, 1, tzinfo=UTC)
        policy = SubscriptionPolicy(latest=latest)

        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2021, 6, 1, tzinfo=UTC)

        clamped_start, clamped_end = policy.clamp_range(start, end)

        assert clamped_start == start
        assert clamped_end == latest

    def test_clamp_range_max_days_constraint(self) -> None:
        """Test range clamping with max_days constraint."""
        policy = SubscriptionPolicy(max_days=30)

        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2020, 12, 1, tzinfo=UTC)  # 335 days

        clamped_start, clamped_end = policy.clamp_range(start, end)

        # Should clamp start to end - 30 days
        expected_start = end - timedelta(days=30)
        assert clamped_start == expected_start
        assert clamped_end == end

    def test_clamp_range_schema_specific_max_days(self) -> None:
        """Test range clamping with schema-specific max_days."""
        policy = SubscriptionPolicy(
            max_days=100,
            max_days_by_schema={"ohlcv-1m": 365, "mbp-1": 30},
        )

        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2021, 1, 1, tzinfo=UTC)  # 366 days

        # Schema-specific override (365 days for ohlcv-1m)
        clamped_start, clamped_end = policy.clamp_range(start, end, schema="ohlcv-1m")
        expected_start = end - timedelta(days=365)
        assert clamped_start == expected_start
        assert clamped_end == end

        # Schema-specific override (30 days for mbp-1)
        clamped_start, clamped_end = policy.clamp_range(start, end, schema="mbp-1")
        expected_start = end - timedelta(days=30)
        assert clamped_start == expected_start
        assert clamped_end == end

        # No schema specified, use global max_days (100)
        clamped_start, clamped_end = policy.clamp_range(start, end)
        expected_start = end - timedelta(days=100)
        assert clamped_start == expected_start
        assert clamped_end == end

    def test_clamp_range_empty_window_non_strict(self) -> None:
        """Test clamping that results in empty window (non-strict mode)."""
        earliest = datetime(2021, 1, 1, tzinfo=UTC)
        policy = SubscriptionPolicy(earliest=earliest, strict=False)

        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2020, 6, 1, tzinfo=UTC)

        # Should clamp but not raise
        clamped_start, clamped_end = policy.clamp_range(start, end)
        assert clamped_start == earliest
        assert clamped_end == end

    def test_clamp_range_empty_window_strict(self) -> None:
        """Test clamping that results in empty window (strict mode)."""
        earliest = datetime(2021, 1, 1, tzinfo=UTC)
        policy = SubscriptionPolicy(earliest=earliest, strict=True)

        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2020, 6, 1, tzinfo=UTC)

        # Should raise in strict mode
        with pytest.raises(PermissionError, match="Clamped window is empty"):
            policy.clamp_range(start, end)

    def test_filter_symbols_no_constraints(self) -> None:
        """Test symbol filtering with no constraints."""
        policy = SubscriptionPolicy()
        symbols = ["AAPL", "MSFT", "GOOGL"]

        filtered = policy.filter_symbols(symbols)

        assert filtered == symbols

    def test_filter_symbols_allowlist(self) -> None:
        """Test symbol filtering with allowlist."""
        policy = SubscriptionPolicy(allowed_symbols={"AAPL", "MSFT"})
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]

        filtered = policy.filter_symbols(symbols)

        assert filtered == ["AAPL", "MSFT"]

    def test_filter_symbols_max_symbols(self) -> None:
        """Test symbol filtering with max_symbols constraint."""
        policy = SubscriptionPolicy(max_symbols=2)
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]

        filtered = policy.filter_symbols(symbols)

        assert len(filtered) == 2
        assert filtered == ["AAPL", "MSFT"]

    def test_filter_symbols_empty_non_strict(self) -> None:
        """Test symbol filtering that results in empty list (non-strict)."""
        policy = SubscriptionPolicy(allowed_symbols={"XYZ"}, strict=False)
        symbols = ["AAPL", "MSFT"]

        filtered = policy.filter_symbols(symbols)

        assert filtered == []

    def test_filter_symbols_empty_strict(self) -> None:
        """Test symbol filtering that results in empty list (strict)."""
        policy = SubscriptionPolicy(allowed_symbols={"XYZ"}, strict=True)
        symbols = ["AAPL", "MSFT"]

        with pytest.raises(PermissionError, match="No symbols permitted by policy"):
            policy.filter_symbols(symbols)

    def test_get_lookback_days_for_level_l0(self) -> None:
        """Test getting lookback days for L0 data levels."""
        policy = SubscriptionPolicy(l0_max_lookback_days=2000)

        assert policy.get_lookback_days_for_level("L0") == 2000
        assert policy.get_lookback_days_for_level("bars") == 2000
        assert policy.get_lookback_days_for_level("ohlcv") == 2000
        assert policy.get_lookback_days_for_level("ohlcv-1m") == 2000
        assert policy.get_lookback_days_for_level("OHLCV-1D") == 2000  # Case insensitive

    def test_get_lookback_days_for_level_l1(self) -> None:
        """Test getting lookback days for L1 data levels."""
        policy = SubscriptionPolicy(l1_max_lookback_days=500)

        assert policy.get_lookback_days_for_level("L1") == 500
        assert policy.get_lookback_days_for_level("quotes") == 500
        assert policy.get_lookback_days_for_level("trades") == 500
        assert policy.get_lookback_days_for_level("tbbo") == 500
        assert policy.get_lookback_days_for_level("BBO") == 500

    def test_get_lookback_days_for_level_l2(self) -> None:
        """Test getting lookback days for L2 data levels."""
        policy = SubscriptionPolicy(l2_max_lookback_days=60)

        assert policy.get_lookback_days_for_level("L2") == 60
        assert policy.get_lookback_days_for_level("mbp") == 60
        assert policy.get_lookback_days_for_level("mbp-1") == 60
        assert policy.get_lookback_days_for_level("mbp-10") == 60
        assert policy.get_lookback_days_for_level("orderbook") == 60

    def test_get_lookback_days_for_level_l3(self) -> None:
        """Test getting lookback days for L3 data levels."""
        policy = SubscriptionPolicy(l3_max_lookback_days=45)

        assert policy.get_lookback_days_for_level("L3") == 45
        assert policy.get_lookback_days_for_level("mbo") == 45
        assert policy.get_lookback_days_for_level("depth") == 45

    def test_get_lookback_days_for_level_unknown_defaults_to_l2(self) -> None:
        """Test that unknown levels default to L2 lookback."""
        policy = SubscriptionPolicy(
            l0_max_lookback_days=2000,
            l1_max_lookback_days=500,
            l2_max_lookback_days=60,
            l3_max_lookback_days=45,
        )

        # Unknown level should default to L2
        assert policy.get_lookback_days_for_level("unknown") == 60
        assert policy.get_lookback_days_for_level("custom_schema") == 60

    def test_from_env_empty_environment(self) -> None:
        """Test policy construction from empty environment."""
        with patch.dict(os.environ, {}, clear=True):
            policy = SubscriptionPolicy.from_env()

            assert policy.allowed_datasets is None
            assert policy.allowed_schemas is None
            assert policy.max_days is None
            assert policy.strict is False
            assert policy.l0_max_lookback_days == 365 * 7
            assert policy.l1_max_lookback_days == 365
            assert policy.l2_max_lookback_days == 30
            assert policy.l3_max_lookback_days == 30

    def test_from_env_with_coverage_variables(self) -> None:
        """Test policy construction with coverage environment variables."""
        env = {
            "DATABENTO_ALLOWED_DATASETS": "XNAS.ITCH,GLBX.MDP3",
            "DATABENTO_ALLOWED_SCHEMAS": "ohlcv-1m,trades",
            "DATABENTO_MAX_DAYS": "100",
            "DATABENTO_EARLIEST_DATE": "2020-01-01",
            "DATABENTO_LATEST_DATE": "2023-12-31",
            "DATABENTO_MAX_SYMBOLS": "50",
            "DATABENTO_POLICY_STRICT": "1",
            "DATABENTO_MAX_DAYS_BY_SCHEMA": "ohlcv-1m:365,mbp-1:30",
        }

        with patch.dict(os.environ, env, clear=True):
            policy = SubscriptionPolicy.from_env()

            assert policy.allowed_datasets == {"XNAS.ITCH", "GLBX.MDP3"}
            assert policy.allowed_schemas == {"ohlcv-1m", "trades"}
            assert policy.max_days == 100
            assert policy.earliest == datetime(2020, 1, 1, tzinfo=UTC)
            assert policy.latest == datetime(2023, 12, 31, tzinfo=UTC)
            assert policy.max_symbols == 50
            assert policy.strict is True
            assert policy.max_days_by_schema == {"ohlcv-1m": 365, "mbp-1": 30}

    def test_from_env_with_lookback_variables(self) -> None:
        """Test policy construction with lookback environment variables."""
        env = {
            "ML_L0_LOOKBACK_DAYS": "3000",
            "ML_L1_LOOKBACK_DAYS": "730",
            "ML_L2_LOOKBACK_DAYS": "90",
            "ML_L3_LOOKBACK_DAYS": "60",
        }

        with patch.dict(os.environ, env, clear=True):
            policy = SubscriptionPolicy.from_env()

            assert policy.l0_max_lookback_days == 3000
            assert policy.l1_max_lookback_days == 730
            assert policy.l2_max_lookback_days == 90
            assert policy.l3_max_lookback_days == 60

    def test_from_env_strict_parsing(self) -> None:
        """Test various strict flag parsing."""
        test_cases = [
            ("1", True),
            ("0", False),
            ("true", True),
            ("TRUE", True),
            ("false", False),
            ("", False),
        ]

        for strict_value, expected in test_cases:
            env = {"DATABENTO_POLICY_STRICT": strict_value}
            with patch.dict(os.environ, env, clear=True):
                policy = SubscriptionPolicy.from_env()
                assert policy.strict is expected, f"Failed for strict_value={strict_value}"


class TestSubscriptionChecker:
    """Test SubscriptionChecker class."""

    def test_initialization_without_api_key(self) -> None:
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("ml._imports.HAS_DATABENTO", True):
                mock_db = MagicMock()
                with patch.dict("sys.modules", {"databento": mock_db}):
                    with pytest.raises(ValueError, match="DATABENTO_API_KEY must be provided"):
                        SubscriptionChecker()

    def test_initialization_with_explicit_api_key(self) -> None:
        """Test initialization with explicit API key."""
        with patch("ml._imports.HAS_DATABENTO", True):
            # Create a mock databento module
            mock_db = MagicMock()
            mock_historical = MagicMock()
            mock_db.Historical.return_value = mock_historical

            with patch.dict("sys.modules", {"databento": mock_db}):
                checker = SubscriptionChecker(api_key="test_key")

                mock_db.Historical.assert_called_once_with("test_key")
                assert checker.client == mock_historical

    def test_initialization_with_env_api_key(self) -> None:
        """Test initialization with API key from environment."""
        with patch.dict(os.environ, {"DATABENTO_API_KEY": "env_test_key"}):
            with patch("ml._imports.HAS_DATABENTO", True):
                # Create a mock databento module
                mock_db = MagicMock()
                mock_historical = MagicMock()
                mock_db.Historical.return_value = mock_historical

                with patch.dict("sys.modules", {"databento": mock_db}):
                    checker = SubscriptionChecker()

                    mock_db.Historical.assert_called_once_with("env_test_key")
                    assert checker.client == mock_historical

    def test_check_available_datasets_success(self) -> None:
        """Test successful dataset availability check."""
        with patch("ml._imports.HAS_DATABENTO", True):
            mock_db = MagicMock()
            mock_client = MagicMock()
            mock_client.metadata.list_datasets.return_value = [
                "XNAS.ITCH",
                "GLBX.MDP3",
            ]
            mock_db.Historical.return_value = mock_client

            with patch.dict("sys.modules", {"databento": mock_db}):
                checker = SubscriptionChecker(api_key="test_key")
                datasets = checker.check_available_datasets()

                assert datasets == ["XNAS.ITCH", "GLBX.MDP3"]
                assert checker.results["available_datasets"] == ["XNAS.ITCH", "GLBX.MDP3"]

    def test_check_available_datasets_error(self) -> None:
        """Test dataset check handles errors gracefully."""
        with patch("ml._imports.HAS_DATABENTO", True):
            mock_db = MagicMock()
            mock_client = MagicMock()
            mock_client.metadata.list_datasets.side_effect = Exception("API Error")
            mock_db.Historical.return_value = mock_client

            with patch.dict("sys.modules", {"databento": mock_db}):
                checker = SubscriptionChecker(api_key="test_key")
                datasets = checker.check_available_datasets()

                assert datasets == []
                assert "API Error" in checker.results["warnings"][0]

    def test_check_dataset_range_success(self) -> None:
        """Test successful dataset range check."""
        with patch("ml._imports.HAS_DATABENTO", True):
            mock_db = MagicMock()
            mock_client = MagicMock()
            mock_client.metadata.get_dataset_range.return_value = {
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
            }
            mock_db.Historical.return_value = mock_client

            with patch.dict("sys.modules", {"databento": mock_db}):
                checker = SubscriptionChecker(api_key="test_key")
                range_info = checker.check_dataset_range("XNAS.ITCH")

                assert range_info["start_date"] == "2020-01-01"
                assert range_info["end_date"] == "2023-12-31"
                assert "XNAS.ITCH" in checker.results["datasets"]
                assert checker.results["datasets"]["XNAS.ITCH"]["days"] == 1460  # Approx

    def test_check_available_schemas_success(self) -> None:
        """Test successful schema availability check."""
        with patch("ml._imports.HAS_DATABENTO", True):
            mock_db = MagicMock()
            mock_client = MagicMock()
            mock_client.metadata.list_schemas.return_value = [
                "ohlcv-1m",
                "trades",
                "mbp-1",
            ]
            mock_db.Historical.return_value = mock_client

            with patch.dict("sys.modules", {"databento": mock_db}):
                checker = SubscriptionChecker(api_key="test_key")
                schemas = checker.check_available_schemas("XNAS.ITCH")

                assert schemas == ["ohlcv-1m", "trades", "mbp-1"]
                assert (
                    checker.results["datasets"]["XNAS.ITCH"]["schemas"]
                    == ["ohlcv-1m", "trades", "mbp-1"]
                )

    def test_get_results(self) -> None:
        """Test retrieving accumulated results."""
        with patch("ml._imports.HAS_DATABENTO", True):
            mock_db = MagicMock()
            mock_db.Historical.return_value = MagicMock()

            with patch.dict("sys.modules", {"databento": mock_db}):
                checker = SubscriptionChecker(api_key="test_key")
                checker.results["warnings"].append("Test warning")

                results = checker.get_results()

                assert "warnings" in results
                assert results["warnings"] == ["Test warning"]


class TestModuleHelpers:
    """Test module-level helper functions."""

    def test_get_effective_policy_with_explicit_policy(self) -> None:
        """Test get_effective_policy with explicit policy."""
        custom_policy = SubscriptionPolicy(max_days=999)

        result = get_effective_policy(custom_policy)

        assert result is custom_policy
        assert result.max_days == 999

    def test_get_effective_policy_from_env(self) -> None:
        """Test get_effective_policy constructs from environment."""
        env = {"DATABENTO_MAX_DAYS": "500"}

        with patch.dict(os.environ, env, clear=True):
            result = get_effective_policy()

            assert result.max_days == 500

    def test_get_max_lookback_days_default_policy(self) -> None:
        """Test get_max_lookback_days with default policy."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_max_lookback_days("bars") == 365 * 7
            assert get_max_lookback_days("trades") == 365
            assert get_max_lookback_days("mbp-1") == 30

    def test_get_max_lookback_days_custom_policy(self) -> None:
        """Test get_max_lookback_days with custom policy."""
        policy = SubscriptionPolicy(
            l0_max_lookback_days=3000,
            l1_max_lookback_days=800,
        )

        assert get_max_lookback_days("bars", policy) == 3000
        assert get_max_lookback_days("trades", policy) == 800
        assert get_max_lookback_days("mbp-1", policy) == 30  # Uses default L2
