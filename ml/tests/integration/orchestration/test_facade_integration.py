"""
Integration tests for MLPipelineOrchestratorFacade with stores and registries.

Phase 2.2.8: Verify facade integrates correctly with all 4 stores and 4 registries.
Tests require PostgreSQL database via cloned_test_database fixture.

Test Design: reports/tests/phase_2_2_8_test_design_report.md

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest


if TYPE_CHECKING:
    from ml.stores.feature_store import FeatureStore
    from ml.stores.model_store import ModelStore
    from ml.stores.strategy_store import StrategyStore


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_coverage_provider() -> Mock:
    """
    Mock CoverageProviderProtocol for testing.
    """
    provider = Mock()
    provider.read_bucket_coverage.return_value = set()
    return provider


@pytest.fixture
def mock_writer() -> Mock:
    """
    Mock MarketDataWriterProtocol for testing.
    """
    writer = Mock()
    writer.write.return_value = 0
    return writer


@pytest.fixture
def mock_build_main() -> Mock:
    """
    Mock CLI main function for dataset building.
    """
    return Mock(return_value=0)


@pytest.fixture
def mock_teacher_main() -> Mock:
    """
    Mock CLI main function for teacher training.
    """
    return Mock(return_value=0)


# ============================================================================
# REGISTRY INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("cloned_test_database")
class TestDataRegistryIntegration:
    """
    Integration tests for DataRegistry operations through facade.
    """

    def test_data_registry_dataset_registration(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify DataRegistry operations work through facade.

        Given:
        - Facade with real DataRegistry
        - Valid dataset to register

        When:
        - Running pipeline that registers dataset

        Then:
        - Dataset manifest registered correctly
        - Manifest contains required fields

        """
        # Test will verify data_registry.register_dataset called correctly

    def test_data_registry_manifest_retrieval(
        self,
    ) -> None:
        """
        Verify registered datasets can be retrieved.

        Given:
        - Previously registered dataset

        When:
        - Querying data_registry

        Then:
        - Manifest returned with correct fields

        """


@pytest.mark.integration
@pytest.mark.usefixtures("cloned_test_database")
class TestFeatureRegistryIntegration:
    """
    Integration tests for FeatureRegistry operations through facade.
    """

    def test_feature_registry_schema_registration(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify FeatureRegistry operations work through facade.

        Given:
        - Facade with real FeatureRegistry
        - Dataset build that creates features

        When:
        - Running dataset build with register_features=True

        Then:
        - Feature schemas registered
        - Schema hash computed correctly

        """

    def test_feature_registry_version_tracking(
        self,
    ) -> None:
        """
        Verify feature version tracking works.

        Given:
        - Existing feature registration

        When:
        - Updating feature schema

        Then:
        - New version created
        - Old version preserved

        """


@pytest.mark.integration
@pytest.mark.usefixtures("cloned_test_database")
class TestModelRegistryIntegration:
    """
    Integration tests for ModelRegistry operations through facade.
    """

    def test_model_registry_training_registration(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify ModelRegistry operations work through facade.

        Given:
        - Facade with real ModelRegistry
        - Successful model training

        When:
        - Training completes

        Then:
        - Model registered with version
        - Metadata captured correctly

        """

    def test_model_registry_deployment_status(
        self,
    ) -> None:
        """
        Verify model deployment status tracking.

        Given:
        - Registered model

        When:
        - Updating deployment status

        Then:
        - Status correctly updated
        - A/B routing configured

        """


@pytest.mark.integration
@pytest.mark.usefixtures("cloned_test_database")
class TestStrategyRegistryIntegration:
    """
    Integration tests for StrategyRegistry operations through facade.
    """

    def test_strategy_registry_manifest_registration(
        self,
    ) -> None:
        """
        Verify StrategyRegistry operations work through facade.

        Given:
        - Facade with real StrategyRegistry
        - Strategy manifest to register

        When:
        - Registering strategy

        Then:
        - Manifest stored correctly
        - Dependencies tracked

        """


# ============================================================================
# STORE INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("cloned_test_database")
class TestDataStoreIntegration:
    """
    Integration tests for DataStore operations through facade.
    """

    def test_data_store_write_read_cycle(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify DataStore read/write through facade.

        Given:
        - Facade with real DataStore
        - Data to persist

        When:
        - Writing data via facade

        Then:
        - Data persisted
        - Read returns same data

        """

    def test_data_store_time_range_query(
        self,
    ) -> None:
        """
        Verify time range queries work.

        Given:
        - Persisted time series data

        When:
        - Querying by time range

        Then:
        - Correct data returned
        - Filtering applied

        """


@pytest.mark.integration
@pytest.mark.usefixtures("cloned_test_database")
class TestFeatureStoreIntegration:
    """
    Integration tests for FeatureStore operations through facade.
    """

    def test_feature_store_write_read_cycle(
        self,
        feature_store: FeatureStore,
    ) -> None:
        """
        Verify FeatureStore operations work.

        Given:
        - Facade with real FeatureStore
        - Computed features

        When:
        - Writing features

        Then:
        - Features persisted
        - Retrievable by instrument

        """

    def test_feature_store_instrument_filtering(
        self,
        feature_store: FeatureStore,
    ) -> None:
        """
        Verify instrument-based filtering.

        Given:
        - Features for multiple instruments

        When:
        - Querying specific instrument

        Then:
        - Only matching features returned

        """


@pytest.mark.integration
@pytest.mark.usefixtures("cloned_test_database")
class TestModelStoreIntegration:
    """
    Integration tests for ModelStore operations through facade.
    """

    def test_model_store_artifact_persistence(
        self,
        model_store: ModelStore,
    ) -> None:
        """
        Verify ModelStore artifact operations.

        Given:
        - Facade with real ModelStore
        - Trained model artifact

        When:
        - Saving model

        Then:
        - Artifact persisted
        - Loadable

        """

    def test_model_store_prediction_logging(
        self,
        model_store: ModelStore,
    ) -> None:
        """
        Verify prediction logging works.

        Given:
        - Deployed model

        When:
        - Logging predictions

        Then:
        - Predictions stored
        - Retrievable by timestamp

        """


@pytest.mark.integration
@pytest.mark.usefixtures("cloned_test_database")
class TestStrategyStoreIntegration:
    """
    Integration tests for StrategyStore operations through facade.
    """

    def test_strategy_store_state_persistence(
        self,
        strategy_store: StrategyStore,
    ) -> None:
        """
        Verify StrategyStore state operations.

        Given:
        - Facade with real StrategyStore
        - Strategy state to persist

        When:
        - Saving state

        Then:
        - State persisted
        - Retrievable

        """

    def test_strategy_store_state_history(
        self,
        strategy_store: StrategyStore,
    ) -> None:
        """
        Verify state history tracking.

        Given:
        - Multiple state updates

        When:
        - Querying history

        Then:
        - All states returned
        - Chronologically ordered

        """


# ============================================================================
# CROSS-COMPONENT INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("cloned_test_database")
class TestCrossComponentIntegration:
    """
    Tests for integration across multiple components.
    """

    def test_pipeline_uses_all_stores(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify full pipeline touches all stores.

        Given:
        - Facade with all stores configured

        When:
        - Running full pipeline

        Then:
        - DataStore used for ingestion
        - FeatureStore used for features
        - ModelStore used for model
        - StrategyStore used for strategy

        """

    def test_pipeline_uses_all_registries(
        self,
    ) -> None:
        """
        Verify full pipeline touches all registries.

        Given:
        - Facade with all registries configured

        When:
        - Running full pipeline

        Then:
        - DataRegistry has dataset manifest
        - FeatureRegistry has feature schema
        - ModelRegistry has model entry
        - StrategyRegistry has strategy manifest

        """

    def test_store_registry_synchronization(
        self,
    ) -> None:
        """
        Verify stores and registries stay in sync.

        Given:
        - Data written to stores

        When:
        - Checking corresponding registries

        Then:
        - Registry entries match store contents
        - Metadata consistent

        """


# ============================================================================
# MESSAGE BUS INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
class TestMessageBusIntegration:
    """
    Tests for message bus event emission.
    """

    def test_pipeline_emits_stage_events(self) -> None:
        """
        Verify pipeline emits stage transition events.

        Given:
        - Message bus configured

        When:
        - Running pipeline

        Then:
        - PipelineStarted event emitted
        - StageCompleted events for each stage
        - PipelineCompleted event at end

        """

    def test_error_events_emitted_on_failure(self) -> None:
        """
        Verify error events emitted on failure.

        Given:
        - Pipeline configured to fail

        When:
        - Running pipeline

        Then:
        - PipelineFailed event emitted
        - Error details included

        """
