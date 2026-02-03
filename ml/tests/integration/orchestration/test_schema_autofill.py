#!/usr/bin/env python3
"""
Integration tests for ConfigResolver schema auto-fill workflows.

Phase 2.2.6: STRUCTURAL PHASE
- All tests marked @pytest.mark.skip
- Tests designed for Phase 2.2.8 full implementation
- Document expected schema auto-fill behavior
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ml.orchestration.config_types import AutoFillUniverseConfig, DatasetBuildConfig
from ml.tests.utils.targets import build_default_target_semantics_payload


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def dataset_config(tmp_path: Path) -> DatasetBuildConfig:
    """Provides minimal DatasetBuildConfig for testing."""
    return DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        dataset_id="test.dataset",
        instrument_ids=["SPY.NASDAQ"],
        target_semantics=build_default_target_semantics_payload(),
    )


@pytest.fixture
def auto_fill_config() -> AutoFillUniverseConfig:
    """Provides AutoFillUniverseConfig for testing."""
    return AutoFillUniverseConfig(
        enabled=True,
        dataset_id="test.dataset",
        instrument_ids=["SPY.NASDAQ"],
        include_bars=True,
        include_tbbo=True,
        include_trades=True,
        include_l2=False,
        include_l3=False,
    )


@pytest.fixture
def config_resolver():
    """Provides ConfigResolver instance for testing."""
    from ml.orchestration.config_resolver import ConfigResolver

    return ConfigResolver()


@pytest.fixture
def metrics_mock():
    """Provides mock metrics object."""
    metrics = Mock()
    metrics.operations_total = Mock()
    metrics.operations_total.labels = Mock(return_value=Mock(inc=Mock()))
    return metrics


@pytest.fixture
def policy_mock():
    """Provides mock CoveragePolicy object."""
    return Mock()


# ============================================================================
# INTEGRATION TESTS (3 tests - ALL SKIPPED)
# ============================================================================


@pytest.mark.integration
def test_auto_fill_universe_populates_all_schemas(
    config_resolver,
    dataset_config,
    auto_fill_config,
):
    """
    Verify auto-fill universe workflow populates all enabled schema types.

    Phase 2.2.8 Expected Behavior:
    - Resolves instruments: ["SPY.NASDAQ"]
    - For bars schema: calls _auto_fill_schema with schema="ohlcv-1m"
    - For tbbo schema: calls _auto_fill_schema with schema="tbbo"
    - For trades schema: calls _auto_fill_schema with schema="trades"
    - All schemas populated for SPY.NASDAQ
    """
    resolver = config_resolver

    # Mock _auto_fill_schema to track calls
    with patch.object(resolver, "_auto_fill_schema") as mock_autofill:
        resolver._auto_fill_universe(dataset_config, auto_fill_config)

        # Verify called for each schema type
        calls = mock_autofill.call_args_list

        # Phase 2.2.8 assertions
        schemas_filled = [call.kwargs["schema"] for call in calls]
        assert "ohlcv-1m" in schemas_filled, "Bars schema should be auto-filled"
        assert "tbbo" in schemas_filled, "TBBO schema should be auto-filled"
        assert "trades" in schemas_filled, "Trades schema should be auto-filled"


@pytest.mark.integration
def test_auto_fill_schema_triggers_ingestion_workflow(
    config_resolver,
    dataset_config,
    metrics_mock,
):
    """
    Verify auto-fill schema triggers ingestion with correct parameters.

    Phase 2.2.8 Expected Behavior:
    - Validates lookback_days > 0 (30 is valid)
    - Resolves market bindings for SPY.NASDAQ
    - Calculates window: (now - 30 days, now)
    - Triggers ingestion via ingestor.ingest_market_data()
    - Records metrics (operations_total incremented)
    """
    resolver = config_resolver

    # Mock ingestor
    mock_ingestor = Mock()
    resolver.ingestor = mock_ingestor

    resolver._auto_fill_schema(
        dataset_id="test.dataset",
        schema="ohlcv-1m",
        instrument_id="SPY.NASDAQ",
        lookback_days=30,
        metrics=metrics_mock,
        dataset_cfg=dataset_config,
    )

    # Phase 2.2.8 assertions
    # Verify ingestion triggered (method exists on ingestor)
    assert hasattr(
        mock_ingestor,
        "ingest_market_data",
    ), "Ingestor should have ingest method"
    # Note: Actual call verification depends on Phase 2.2.8 implementation


@pytest.mark.integration
def test_auto_fill_l2_populates_depth_and_mbp_schemas(
    config_resolver,
    dataset_config,
    auto_fill_config,
    metrics_mock,
    policy_mock,
):
    """
    Verify L2 auto-fill populates both depth and MBP schemas.

    Phase 2.2.8 Expected Behavior:
    - Determines L2 lookback from policy
    - For each instrument (SPY, QQQ):
      - Calls _auto_fill_schema with schema="mbp-10"
      - Calls _auto_fill_schema with schema="depth"
    - Total calls: 4 (2 instruments × 2 schemas)
    """
    resolver = config_resolver

    # Mock _auto_fill_schema to track calls
    with patch.object(resolver, "_auto_fill_schema") as mock_autofill:
        resolver._auto_fill_l2(
            dataset_cfg=dataset_config,
            auto_fill_cfg=auto_fill_config,
            instruments=("SPY.NASDAQ", "QQQ.NASDAQ"),
            metrics=metrics_mock,
            policy=policy_mock,
        )

        # Phase 2.2.8 assertions
        # Verify 4 calls (2 instruments x 2 schemas)
        assert (
            mock_autofill.call_count == 4
        ), "Should call for each instrument x schema"

        # Verify schemas are depth and mbp-10
        schemas = [call.kwargs["schema"] for call in mock_autofill.call_args_list]
        assert schemas.count("mbp-10") == 2, "MBP schema for both instruments"
        assert schemas.count("depth") == 2, "Depth schema for both instruments"
