"""
Integration test for FeatureRegistry, FeatureStore, and L2/L3 microstructure features.

Validates that the new L2/L3 features integrate seamlessly with the registry and store.

"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ml._imports import HAS_PANDAS
from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.microstructure import L2MicrostructureFeatures
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.feature_store import FeatureStore


pytestmark = pytest.mark.skipif(
    not HAS_POLARS or not HAS_PANDAS,
    reason="Requires polars and pandas",
)


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.usefixtures("clean_postgres_db_class")
class TestL2L3RegistryStoreIntegration:
    """
    Test that L2/L3 features integrate properly with registry and store.
    """

    def test_feature_config_includes_microstructure(self) -> None:
        """
        Test that FeatureConfig properly includes microstructure features.
        """
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=True,
        )

        feature_names = config.get_feature_names()

        # Check microstructure features are included
        assert "spread_mean" in feature_names
        assert "spread_std" in feature_names
        assert "spread_relative" in feature_names
        assert "size_imbalance_mean" in feature_names
        assert "size_imbalance_std" in feature_names
        assert "mid_return_std" in feature_names
        assert "mid_return_autocorr" in feature_names

        # Check trade flow features are included
        assert "trade_flow_imbalance" in feature_names
        assert "vwap" in feature_names
        assert "trade_intensity" in feature_names
        assert "avg_price_impact" in feature_names

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_engineer_delegates_to_l2_calculator(self) -> None:
        """
        Test that FeatureEngineer properly delegates to L2MicrostructureFeatures.
        """
        config = FeatureConfig(include_microstructure=True)
        engineer = FeatureEngineer(config)

        # Create mock data with L2 depth
        df = pl.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000, 2000],
                "ts_event": [1000, 2000],
                "ts_init": [1000, 2000],
                # L2 depth data
                "bid_price_0": [100.5, 101.5],
                "ask_price_0": [101.5, 102.5],
                "bid_size_0": [100, 200],
                "ask_size_0": [150, 250],
            },
        )

        # Patch L2MicrostructureFeatures to verify delegation
        with patch("ml.features.microstructure.L2MicrostructureFeatures") as mock_l2:
            mock_instance = MagicMock()
            mock_instance.compute_all_features.return_value = {
                "spread": [1.0, 1.0],
                "spread_bps": [100.0, 100.0],
            }
            mock_l2.return_value = mock_instance

            features = engineer._calculate_microstructure_features_batch(df, 1)

            # Verify L2 calculator was used
            mock_l2.assert_called_once()
            mock_instance.compute_all_features.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_registry_manifest_with_l2_features(self, test_database) -> None:
        """
        Test creating feature manifest with L2/L3 capabilities.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string=test_database.connection_string,
            )
            registry = FeatureRegistry(
                registry_path=Path(tmpdir),
                persistence_config=persistence_config
            )

            # Create manifest with L2/L3 features
            config = FeatureConfig(
                include_microstructure=True,
                include_trade_flow=True,
            )
            feature_names = config.get_feature_names()
            feature_dtypes = ["float32"] * len(feature_names)

            # Create pipeline signature including L2/L3 transforms
            pipeline_spec = PipelineSpec(
                transforms=[
                    TransformSpec(name="returns"),
                    TransformSpec(name="microstructure"),
                    TransformSpec(name="trade_flow"),
                ],
            )
            pipeline_runner = PipelineRunner(pipeline_spec, DataRequirements.L1_L2)
            pipeline_signature = pipeline_runner.compute_signature()

            schema_hash = compute_schema_hash(
                feature_names,
                feature_dtypes,
                pipeline_signature,
            )

            manifest = FeatureManifest(
                feature_set_id="l2_l3_features_v1",
                name="L2/L3 Microstructure Features",
                version="1.0.0",
                role=FeatureRole.TEACHER,
                data_requirements=DataRequirements.L1_L2,
                feature_names=feature_names,
                feature_dtypes=feature_dtypes,
                schema_hash=schema_hash,
                pipeline_signature=pipeline_signature,
                pipeline_version="1.0.0",
                capability_flags={
                    "has_microstructure": True,
                    "has_trade_flow": True,
                    "has_l2_depth": True,
                },
                constraints={
                    "min_lookback": 20,
                    "max_latency_ms": 5,
                },
                stage=FeatureStage.CANDIDATE,
            )

            # Register the manifest
            feature_id = registry.register_feature_set(manifest)

            # Verify registration
            retrieved = registry.get_feature_manifest(feature_id)
            assert retrieved is not None
            assert retrieved.feature_set_id == "l2_l3_features_v1"
            assert retrieved.capability_flags["has_microstructure"] is True
            assert retrieved.capability_flags["has_l2_depth"] is True
            assert "spread_mean" in retrieved.feature_names
            assert retrieved.data_requirements == DataRequirements.L1_L2

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_store_computes_l2_features(
        self,
        test_database,
    ) -> None:
        """
        Test that FeatureStore properly computes L2/L3 features.
        """
        # Create store with L2/L3 features enabled using real PostgreSQL
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=True,
        )

        store = FeatureStore(
            connection_string=test_database.connection_string,
            feature_config=config,
        )

        # Verify the feature engineer has the right config
        assert store.feature_engineer.config.include_microstructure is True
        assert store.feature_engineer.config.include_trade_flow is True

        # Check feature names include L2/L3
        feature_names = store._get_feature_names()
        assert "spread_mean" in feature_names
        assert "trade_flow_imbalance" in feature_names

        # Test that the feature count is correct
        n_features = len(feature_names)

        # Features should include base + microstructure + trade flow
        assert n_features > 30  # At least 30 features with all enabled

    @pytest.mark.database
    @pytest.mark.serial
    def test_pipeline_integration_with_l2_transforms(self) -> None:
        """
        Test that pipeline properly includes L2/L3 transforms.
        """
        # Create pipeline with L2/L3 transforms
        pipeline_spec = PipelineSpec(
            transforms=[
                TransformSpec(name="returns", params={"periods": [1, 5, 10]}),
                TransformSpec(name="volatility"),
                TransformSpec(name="microstructure"),  # L2 features
                TransformSpec(name="trade_flow"),  # L3 features
                TransformSpec(name="calendar"),  # Known-future
                TransformSpec(name="macro_indicators"),  # Known-future
            ],
        )

        # Create runner with L1_L2 data requirements
        runner = PipelineRunner(pipeline_spec, DataRequirements.L1_L2)

        # Get feature names from pipeline
        feature_names = runner.compute_feature_names()

        # Verify L2/L3 features are included
        assert "spread_mean" in feature_names
        assert "spread_std" in feature_names
        assert "trade_flow_imbalance" in feature_names
        assert "vwap" in feature_names

        # Verify known-future features are included
        assert "hour_sin" in feature_names
        assert "vix" in feature_names

        # Compute signature for versioning
        signature = runner.compute_signature()
        assert len(signature) == 64  # SHA256 hex digest

    @pytest.mark.database
    @pytest.mark.serial
    def test_l2_feature_computation_with_real_data(self) -> None:
        """
        Test L2 feature computation with realistic order book data.
        """
        calculator = L2MicrostructureFeatures(n_levels=5, lookback_window=10)

        # Create realistic L2 order book data
        n_samples = 20
        n_levels = 5

        # Generate realistic bid/ask ladder
        mid_price = 100.0
        spread = 0.01

        bid_prices = np.zeros((n_samples, n_levels))
        ask_prices = np.zeros((n_samples, n_levels))
        bid_sizes = np.zeros((n_samples, n_levels))
        ask_sizes = np.zeros((n_samples, n_levels))

        for i in range(n_samples):
            # Add some price movement
            from numpy.random import default_rng

            _rng = default_rng(0)
            mid_price += float(_rng.standard_normal()) * 0.005

            for level in range(n_levels):
                # Bid prices decrease with level
                bid_prices[i, level] = mid_price - spread / 2 - level * 0.001
                # Ask prices increase with level
                ask_prices[i, level] = mid_price + spread / 2 + level * 0.001

                # Sizes typically decrease with level (more volume at best prices)
                from numpy.random import default_rng

                _rng = default_rng(0)
                bid_sizes[i, level] = (
                    1000 * (1.0 - level * 0.15) + float(_rng.standard_normal()) * 50
                )
                ask_sizes[i, level] = (
                    1000 * (1.0 - level * 0.15) + float(_rng.standard_normal()) * 50
                )

        # Compute features
        spread_features = calculator.compute_spread_features(
            bid_prices,
            ask_prices,
            bid_sizes,
            ask_sizes,
        )

        imbalance_features = calculator.compute_imbalance_features(
            bid_sizes,
            ask_sizes,
        )

        # Validate computed features
        assert "spread" in spread_features
        assert "spread_bps" in spread_features
        assert spread_features["spread"] > 0  # Spread should be positive
        assert spread_features["spread_bps"] > 0  # Basis points should be positive

        assert "imbalance_l1" in imbalance_features
        assert -1 <= imbalance_features["imbalance_l1"] <= 1  # Normalized imbalance

        # Test shape features
        shape_features = calculator.compute_shape_features(
            bid_prices,
            ask_prices,
            bid_sizes,
            ask_sizes,
        )

        # Check that we get shape features (actual names may vary)
        assert len(shape_features) > 0
        assert any("bid" in k for k in shape_features.keys())
        assert any("ask" in k for k in shape_features.keys())

    @pytest.mark.database
    @pytest.mark.serial
    def test_end_to_end_l2_feature_persistence(self, test_database) -> None:
        """
        Test end-to-end flow from L2 data to persisted features.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup registry with PostgreSQL backend
            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string=test_database.connection_string,
            )
            registry = FeatureRegistry(
                registry_path=Path(tmpdir),
                persistence_config=persistence_config
            )

            # Create config with L2 features
            config = FeatureConfig(
                include_microstructure=True,
                include_trade_flow=False,  # Only test L2 for simplicity
            )

            # Create feature manifest
            feature_names = config.get_feature_names()
            manifest = FeatureManifest(
                feature_set_id="l2_test_v1",
                name="L2 Test Features",
                version="1.0.0",
                role=FeatureRole.STUDENT,  # Test student role
                data_requirements=DataRequirements.L1_L2,
                feature_names=feature_names,
                feature_dtypes=["float32"] * len(feature_names),
                schema_hash=hashlib.sha256(
                    json.dumps(feature_names).encode(),
                ).hexdigest(),
                pipeline_signature="test_sig",
                pipeline_version="1.0.0",
                capability_flags={"has_l2": True},
                stage=FeatureStage.STAGING,
            )

            # Register manifest
            feature_id = registry.register_feature_set(manifest)

            # Verify registration
            retrieved = registry.get_feature_manifest(feature_id)
            assert retrieved is not None
            assert retrieved.feature_set_id == "l2_test_v1"
            assert retrieved.capability_flags["has_l2"] is True

            # Verify we can list by role
            student_features = registry.list_by_role(FeatureRole.STUDENT)

            assert len(student_features) >= 1
            assert any(f.manifest.feature_set_id == "l2_test_v1" for f in student_features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
