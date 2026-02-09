"""
Tests for ML configuration classes.
"""

import pytest
from msgspec import ValidationError

from ml.config.base import CanaryDeploymentConfig
from ml.config.base import CorrelationDataConfig
from ml.config.base import ExposurePriceConfig
from ml.config.base import ExposurePriceSource
from ml.config.base import MLActorConfig
from ml.config.base import MLTrainingConfig
from ml.config.base import ModelDeploymentConfig
from ml.config.base import ModelRegistryConfig
from ml.config.base import MultiModelStrategyConfig
from ml.config.base import PositionsConfig
from ml.config.base import PositionsSource
from ml.config.policy import RegistryCompatibilityPolicyConfig
from ml.config.registry import RegistryPolicyConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


@pytest.mark.parallel_safe
@pytest.mark.unit
class TestModelRegistryConfig:
    """
    Tests for ModelRegistryConfig.
    """

    def test_default_values(self) -> None:
        """
        Test ModelRegistryConfig default values.
        """
        config = ModelRegistryConfig()

        assert config.registry_path == "ml/registry"
        assert config.enable_mlflow is False
        assert config.mlflow_tracking_uri is None
        assert config.auto_versioning is True
        assert config.max_versions_per_model == 10

    def test_custom_values(self) -> None:
        """
        Test ModelRegistryConfig with custom values.
        """
        config = ModelRegistryConfig(
            registry_path="/custom/path",
            enable_mlflow=True,
            mlflow_tracking_uri="http://localhost:5000",
            auto_versioning=False,
            max_versions_per_model=5,
        )

        assert config.registry_path == "/custom/path"
        assert config.enable_mlflow is True
        assert config.mlflow_tracking_uri == "http://localhost:5000"
        assert config.auto_versioning is False
        assert config.max_versions_per_model == 5


class TestMLActorConfig:
    """
    Tests for MLActorConfig with model_id field.
    """

    def test_model_id_required(self) -> None:
        """
        Test that model_id is required in MLActorConfig.
        """
        config = MLActorConfig(
            model_path="/path/to/model.onnx",
            model_id="test_model_v1",  # NEW required field
            bar_type=BarType.from_str("EURUSD.SIM-1-MINUTE-BID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.SIM"),
        )

        assert config.model_id == "test_model_v1"
        assert config.model_path == "/path/to/model.onnx"


@pytest.mark.parallel_safe
@pytest.mark.unit
class TestRegistryPolicyConfig:
    """
    Tests for RegistryPolicyConfig scaffolding defaults and env loading.
    """

    def test_default_compatibility_policy_values(self) -> None:
        config = RegistryPolicyConfig()
        assert config.compatibility_policy == RegistryCompatibilityPolicyConfig()
        assert config.compatibility_policy.strict_model_compatibility is False
        assert config.compatibility_policy.allow_compatibility_migration_override is True
        assert config.compatibility_policy.allow_unsigned_artifacts is False
        assert config.compatibility_policy.require_output_semantics is False

    def test_from_env_parses_compatibility_policy(self) -> None:
        config = RegistryPolicyConfig.from_env(
            env={
                "ML_STRICT_MODEL_COMPATIBILITY": "true",
                "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE": "false",
                "ML_ALLOW_UNSIGNED_ARTIFACTS": "true",
                "ML_REQUIRE_OUTPUT_SEMANTICS": "true",
            },
        )
        assert config.compatibility_policy.strict_model_compatibility is True
        assert config.compatibility_policy.allow_compatibility_migration_override is False
        assert config.compatibility_policy.allow_unsigned_artifacts is True
        assert config.compatibility_policy.require_output_semantics is True

    def test_from_env_defaults_to_strict_compatibility_policy(self) -> None:
        config = RegistryPolicyConfig.from_env(env={})
        assert config.compatibility_policy.strict_model_compatibility is True
        assert config.compatibility_policy.allow_compatibility_migration_override is False
        assert config.compatibility_policy.allow_unsigned_artifacts is False
        assert config.compatibility_policy.require_output_semantics is True


class TestMLTrainingConfig:
    """
    Tests for MLTrainingConfig.
    """

    def test_target_semantics_required(self) -> None:
        """
        Test that target_semantics is required for training configs.
        """
        with pytest.raises(ValidationError, match="target_semantics must be provided"):
            MLTrainingConfig(
                data_source="unit-test",
                target_semantics=None,  # type: ignore[arg-type]
            )


class TestMultiModelStrategyConfig:
    """
    Tests for MultiModelStrategyConfig.
    """

    def test_multi_model_config(self) -> None:
        """
        Test MultiModelStrategyConfig creation.
        """
        config = MultiModelStrategyConfig(
            instrument_id=InstrumentId.from_str("EURUSD.SIM"),
            ml_signal_source="ML_ACTOR",
            position_size_pct=0.1,
            target_model_ids=["model1", "model2", "model3"],
            aggregation_mode="weighted_average",
            model_weights={"model1": 0.5, "model2": 0.3, "model3": 0.2},
            required_models=2,
        )

        assert config.target_model_ids == ["model1", "model2", "model3"]
        assert config.aggregation_mode == "weighted_average"
        assert config.model_weights == {"model1": 0.5, "model2": 0.3, "model3": 0.2}
        assert config.required_models == 2

    def test_voting_aggregation(self) -> None:
        """
        Test voting aggregation mode.
        """
        config = MultiModelStrategyConfig(
            instrument_id=InstrumentId.from_str("EURUSD.SIM"),
            ml_signal_source="ML_ACTOR",
            position_size_pct=0.1,
            target_model_ids=["model1", "model2"],
            aggregation_mode="voting",
            required_models=1,
        )

        assert config.aggregation_mode == "voting"
        assert config.model_weights is None  # Not needed for voting


class TestModelDeploymentConfig:
    """
    Tests for ModelDeploymentConfig.
    """

    def test_immediate_deployment(self) -> None:
        """
        Test immediate deployment configuration.
        """
        config = ModelDeploymentConfig(
            deployment_target="actor",
            rollout_strategy="immediate",
            rollout_percentage=100.0,
        )

        assert config.deployment_target == "actor"
        assert config.rollout_strategy == "immediate"
        assert config.rollout_percentage == 100.0
        assert config.auto_rollback_on_error is True

    def test_gradual_deployment(self) -> None:
        """
        Test gradual deployment configuration.
        """
        config = ModelDeploymentConfig(
            deployment_target="strategy",
            rollout_strategy="gradual",
            rollout_percentage=25.0,
            health_check_interval=30,
        )

        assert config.rollout_strategy == "gradual"
        assert config.rollout_percentage == 25.0
        assert config.health_check_interval == 30

    def test_invalid_rollout_percentage(self) -> None:
        """
        Test that rollout_percentage must be <= 100.
        """
        with pytest.raises(ValidationError, match="rollout_percentage must be between"):
            ModelDeploymentConfig(
                deployment_target="both",
                rollout_strategy="canary",
                rollout_percentage=150.0,  # Invalid
            )


class TestCanaryDeploymentConfig:
    """
    Tests for CanaryDeploymentConfig.
    """

    def test_default_canary_config(self) -> None:
        """
        Test CanaryDeploymentConfig default values.
        """
        config = CanaryDeploymentConfig()

        assert config.initial_traffic_percentage == 10.0
        assert config.increment_percentage == 10.0
        assert config.promotion_interval_seconds == 300
        assert config.error_threshold_percentage == 5.0
        assert config.latency_threshold_ms == 100.0
        assert config.auto_promote is True
        assert config.auto_rollback is True

    def test_custom_canary_config(self) -> None:
        """
        Test CanaryDeploymentConfig with custom values.
        """
        config = CanaryDeploymentConfig(
            initial_traffic_percentage=5.0,
            increment_percentage=5.0,
            promotion_interval_seconds=600,
            error_threshold_percentage=2.0,
            latency_threshold_ms=50.0,
            auto_promote=False,
            auto_rollback=False,
        )

        assert config.initial_traffic_percentage == 5.0
        assert config.increment_percentage == 5.0
        assert config.promotion_interval_seconds == 600
        assert config.error_threshold_percentage == 2.0
        assert config.latency_threshold_ms == 50.0
        assert config.auto_promote is False
        assert config.auto_rollback is False


@pytest.mark.parallel_safe
@pytest.mark.unit
class TestPositionsConfig:
    """
    Tests for PositionsConfig defaults and validation.
    """

    def test_default_values(self) -> None:
        """
        Test PositionsConfig default values.
        """
        config = PositionsConfig()

        assert config.positions_required_for_live is True
        assert config.allow_degraded is True
        assert [source.value for source in config.source_priority] == [
            "cache_positions_open",
            "cache_positions",
            "portfolio_net_position",
            "portfolio_positions",
            "portfolio_positions_open",
        ]

    def test_rejects_empty_priority(self) -> None:
        """
        Test that source_priority cannot be empty.
        """
        with pytest.raises(ValidationError, match="source_priority must contain at least one"):
            PositionsConfig(source_priority=[])

    def test_rejects_duplicate_sources(self) -> None:
        """
        Test that source_priority must be unique.
        """
        with pytest.raises(ValidationError, match="source_priority entries must be unique"):
            PositionsConfig(
                source_priority=[
                    PositionsSource.CACHE_OPEN,
                    PositionsSource.CACHE_OPEN,
                ],
            )

    def test_invalid_percentage_values(self) -> None:
        """
        Test validation of percentage values.
        """
        with pytest.raises(ValidationError):
            CanaryDeploymentConfig(
                initial_traffic_percentage=150.0,  # Invalid > 100
            )


@pytest.mark.parallel_safe
@pytest.mark.unit
class TestExposurePriceConfig:
    """
    Tests for ExposurePriceConfig defaults and validation.
    """

    def test_default_values(self) -> None:
        """
        Test ExposurePriceConfig default values.
        """
        config = ExposurePriceConfig()

        assert [source.value for source in config.source_priority] == [
            "quote_mid",
            "position_avg",
            "cache_last",
        ]

    def test_rejects_empty_priority(self) -> None:
        """
        Test that source_priority cannot be empty.
        """
        with pytest.raises(ValidationError, match="source_priority must contain at least one"):
            ExposurePriceConfig(source_priority=[])

    def test_rejects_duplicate_sources(self) -> None:
        """
        Test that source_priority must be unique.
        """
        with pytest.raises(ValidationError, match="source_priority entries must be unique"):
            ExposurePriceConfig(
                source_priority=[
                    ExposurePriceSource.QUOTE_MID,
                    ExposurePriceSource.QUOTE_MID,
                ],
            )


@pytest.mark.parallel_safe
@pytest.mark.unit
class TestCorrelationDataConfig:
    """
    Tests for CorrelationDataConfig defaults and validation.
    """

    def test_default_values(self) -> None:
        """
        Test CorrelationDataConfig default values.
        """
        config = CorrelationDataConfig()

        assert config.max_age_seconds == 300
        assert config.fallback_value == 0.0

    def test_rejects_out_of_range_fallback(self) -> None:
        """
        Test CorrelationDataConfig rejects invalid fallback values.
        """
        with pytest.raises(ValidationError, match="fallback_value must be between"):
            CorrelationDataConfig(fallback_value=1.5)


class TestConfigIntegration:
    """
    Test integration between different config classes.
    """

    def test_registry_with_deployment(self) -> None:
        """
        Test using registry config with deployment config.
        """
        registry_config = ModelRegistryConfig(
            registry_path="/models/registry",
            enable_mlflow=True,
        )

        deployment_config = ModelDeploymentConfig(
            deployment_target="actor",
            rollout_strategy="canary",
            rollout_percentage=10.0,
        )

        # These should work together
        assert registry_config.registry_path == "/models/registry"
        assert deployment_config.rollout_strategy == "canary"

    def test_multi_model_with_actor_configs(self) -> None:
        """
        Test multi-model strategy with multiple actor configs.
        """
        # Create configs for multiple actors
        actor1_config = MLActorConfig(
            model_path="/models/xgb_v1.json",
            model_id="xgb_v1",
            bar_type=BarType.from_str("EURUSD.SIM-1-MINUTE-BID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.SIM"),
        )

        actor2_config = MLActorConfig(
            model_path="/models/lgb_v1.txt",
            model_id="lgb_v1",
            bar_type=BarType.from_str("EURUSD.SIM-1-MINUTE-BID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.SIM"),
        )

        # Strategy that consumes signals from both actors
        strategy_config = MultiModelStrategyConfig(
            instrument_id=InstrumentId.from_str("EURUSD.SIM"),
            ml_signal_source="ML_ACTORS",
            position_size_pct=0.1,
            target_model_ids=["xgb_v1", "lgb_v1"],
            aggregation_mode="weighted_average",
            model_weights={"xgb_v1": 0.6, "lgb_v1": 0.4},
            required_models=2,
        )

        # Verify model_ids match
        assert actor1_config.model_id in strategy_config.target_model_ids
        assert actor2_config.model_id in strategy_config.target_model_ids
