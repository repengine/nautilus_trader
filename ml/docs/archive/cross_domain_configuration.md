# Cross-Domain Configuration Guide

## Overview

This guide provides a comprehensive strategy for unified configuration management across all ML domains in Nautilus Trader. It addresses configuration consistency, environment variable consolidation, validation patterns, and deployment best practices to ensure seamless cross-domain integration.

## Configuration Architecture

### Unified Configuration Hierarchy

The ML configuration system follows a hierarchical structure that supports cross-domain coordination:

```
System Configuration
├── Environment Configuration (dev/staging/prod)
├── Domain-Specific Configuration
│   ├── Data Domain Config
│   ├── Feature Domain Config
│   ├── Model Domain Config
│   └── Strategy Domain Config
├── Component Configuration
│   ├── Store Configuration
│   ├── Registry Configuration
│   └── Actor Configuration
└── Runtime Configuration
    ├── Performance Tuning
    ├── Resource Limits
    └── Fallback Settings
```

### Core Configuration Classes

#### Base Configuration

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass(frozen=True)
class BaseMLConfiguration:
    """Base configuration for all ML components."""

    # Environment settings
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = False

    # Database configuration
    db_connection: str = "postgresql://postgres:postgres@localhost:5432/nautilus"

    # System behavior
    auto_start_postgres: bool = True
    auto_migrate: bool = True
    strict_protocol_validation: bool = True

    # Performance settings
    enable_metrics: bool = True
    enable_health_checks: bool = True
    hot_path_optimization: bool = True

    # Fallback configuration
    fallback_enabled: bool = True
    fallback_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate configuration consistency."""
        if self.environment == Environment.PRODUCTION:
            if self.debug_mode:
                raise ValueError("Debug mode not allowed in production")
            if self.auto_start_postgres:
                raise ValueError("Auto-start postgres not allowed in production")

        if not self.db_connection:
            raise ValueError("Database connection string required")
```

#### Domain Configuration Classes

```python
@dataclass(frozen=True)
class DataDomainConfig:
    """Configuration for the Data Domain."""

    # Data ingestion settings
    ingestion_enabled: bool = True
    max_ingestion_rate_per_second: int = 10000
    ingestion_buffer_size: int = 50000

    # Data quality settings
    quality_checks_enabled: bool = True
    outlier_detection_enabled: bool = True
    staleness_threshold_seconds: int = 300

    # Storage settings
    parquet_compression: str = "snappy"
    partition_by_date: bool = True
    retention_days: int = 365

    # Backfill settings
    backfill_enabled: bool = True
    backfill_batch_size: int = 10000
    max_concurrent_backfills: int = 3

@dataclass(frozen=True)
class FeatureDomainConfig:
    """Configuration for the Feature Domain."""

    # Feature computation settings
    enable_technical_indicators: bool = True
    enable_microstructure_features: bool = True
    enable_cross_sectional_features: bool = False

    # Performance settings
    max_feature_computation_time_ms: float = 100.0
    feature_cache_size: int = 10000
    batch_computation_enabled: bool = True

    # Validation settings
    parity_validation_enabled: bool = True
    parity_tolerance: float = 1e-10
    drift_detection_enabled: bool = True

    # Pipeline settings
    pipeline_parallel_workers: int = 4
    pipeline_chunk_size: int = 1000

@dataclass(frozen=True)
class ModelDomainConfig:
    """Configuration for the Model Domain."""

    # Model loading settings
    model_cache_size: int = 32
    lazy_loading_enabled: bool = True
    model_warming_enabled: bool = True

    # Inference settings
    max_inference_latency_ms: float = 5.0
    batch_inference_enabled: bool = True
    inference_timeout_ms: float = 10.0

    # Model lifecycle settings
    auto_retraining_enabled: bool = False
    performance_monitoring_enabled: bool = True
    drift_detection_enabled: bool = True

    # A/B testing settings
    ab_testing_enabled: bool = False
    champion_challenger_ratio: float = 0.9

@dataclass(frozen=True)
class StrategyDomainConfig:
    """Configuration for the Strategy Domain."""

    # Signal generation settings
    signal_generation_enabled: bool = True
    max_signal_latency_ms: float = 10.0
    signal_aggregation_enabled: bool = True

    # Risk management settings
    risk_checks_enabled: bool = True
    position_size_limits_enabled: bool = True
    max_position_size_ratio: float = 0.1

    # Strategy execution settings
    strategy_parallelism_enabled: bool = False
    execution_delay_ms: int = 0
    dry_run_mode: bool = False
```

#### Unified System Configuration

```python
@dataclass(frozen=True)
class MLSystemConfiguration(BaseMLConfiguration):
    """Complete ML system configuration integrating all domains."""

    # Domain configurations
    data_domain: DataDomainConfig = field(default_factory=DataDomainConfig)
    feature_domain: FeatureDomainConfig = field(default_factory=FeatureDomainConfig)
    model_domain: ModelDomainConfig = field(default_factory=ModelDomainConfig)
    strategy_domain: StrategyDomainConfig = field(default_factory=StrategyDomainConfig)

    # Cross-domain settings
    event_correlation_enabled: bool = True
    cross_domain_validation_enabled: bool = True
    unified_monitoring_enabled: bool = True

    # Integration settings
    integration_health_check_interval_seconds: int = 60
    cross_domain_timeout_seconds: int = 30
    event_propagation_enabled: bool = True

    def get_domain_config(self, domain: str) -> Any:
        """Get configuration for specific domain."""
        domain_configs = {
            "data": self.data_domain,
            "feature": self.feature_domain,
            "model": self.model_domain,
            "strategy": self.strategy_domain,
        }
        return domain_configs.get(domain)

    @classmethod
    def for_environment(cls, env: Environment) -> "MLSystemConfiguration":
        """Create configuration optimized for specific environment."""
        if env == Environment.DEVELOPMENT:
            return cls(
                environment=env,
                debug_mode=True,
                auto_start_postgres=True,
                auto_migrate=True,
                strict_protocol_validation=True,
            )

        elif env == Environment.STAGING:
            return cls(
                environment=env,
                debug_mode=False,
                auto_start_postgres=False,
                auto_migrate=True,
                strict_protocol_validation=True,
            )

        elif env == Environment.PRODUCTION:
            return cls(
                environment=env,
                debug_mode=False,
                auto_start_postgres=False,
                auto_migrate=False,
                strict_protocol_validation=False,
                hot_path_optimization=True,
                feature_domain=FeatureDomainConfig(
                    max_feature_computation_time_ms=50.0,  # Stricter in prod
                    feature_cache_size=50000,              # Larger cache
                ),
                model_domain=ModelDomainConfig(
                    max_inference_latency_ms=3.0,          # Stricter in prod
                    model_cache_size=64,                   # Larger cache
                ),
            )

        return cls()
```

## Environment Variable Consolidation

### Environment Variable Strategy

All configuration should be overrideable via environment variables with clear naming conventions:

```python
import os
from typing import TypeVar, Type, Union

T = TypeVar('T')

class EnvironmentConfigLoader:
    """Load configuration from environment variables."""

    @staticmethod
    def load_from_env(config_class: Type[T]) -> T:
        """Load configuration from environment variables."""

        # Get environment configuration first
        env_name = os.getenv("ML_ENVIRONMENT", "development").lower()
        environment = Environment(env_name)

        # Base configuration overrides
        base_overrides = {
            "environment": environment,
            "debug_mode": EnvironmentConfigLoader._get_bool("ML_DEBUG_MODE"),
            "db_connection": os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL"),
            "auto_start_postgres": EnvironmentConfigLoader._get_bool("ML_AUTO_START_DB"),
            "auto_migrate": EnvironmentConfigLoader._get_bool("ML_AUTO_MIGRATE"),
            "strict_protocol_validation": EnvironmentConfigLoader._get_bool("ML_STRICT_VALIDATION"),
        }

        # Domain-specific overrides
        domain_overrides = {
            "data_domain": EnvironmentConfigLoader._load_data_domain_from_env(),
            "feature_domain": EnvironmentConfigLoader._load_feature_domain_from_env(),
            "model_domain": EnvironmentConfigLoader._load_model_domain_from_env(),
            "strategy_domain": EnvironmentConfigLoader._load_strategy_domain_from_env(),
        }

        # Filter None values
        overrides = {k: v for k, v in {**base_overrides, **domain_overrides}.items() if v is not None}

        # Create configuration with overrides
        if hasattr(config_class, 'for_environment'):
            config = config_class.for_environment(environment)
            # Apply overrides using dataclass replace
            return replace(config, **overrides)
        else:
            return config_class(**overrides)

    @staticmethod
    def _get_bool(env_var: str, default: bool = None) -> bool | None:
        """Get boolean from environment variable."""
        value = os.getenv(env_var)
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _get_int(env_var: str, default: int = None) -> int | None:
        """Get integer from environment variable."""
        value = os.getenv(env_var)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _get_float(env_var: str, default: float = None) -> float | None:
        """Get float from environment variable."""
        value = os.getenv(env_var)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _load_data_domain_from_env() -> DataDomainConfig | None:
        """Load data domain config from environment."""
        overrides = {}

        if (max_rate := EnvironmentConfigLoader._get_int("ML_DATA_MAX_INGESTION_RATE")) is not None:
            overrides["max_ingestion_rate_per_second"] = max_rate

        if (buffer_size := EnvironmentConfigLoader._get_int("ML_DATA_BUFFER_SIZE")) is not None:
            overrides["ingestion_buffer_size"] = buffer_size

        if (retention := EnvironmentConfigLoader._get_int("ML_DATA_RETENTION_DAYS")) is not None:
            overrides["retention_days"] = retention

        return DataDomainConfig(**overrides) if overrides else None

    @staticmethod
    def _load_feature_domain_from_env() -> FeatureDomainConfig | None:
        """Load feature domain config from environment."""
        overrides = {}

        if (max_time := EnvironmentConfigLoader._get_float("ML_FEATURE_MAX_COMPUTATION_TIME_MS")) is not None:
            overrides["max_feature_computation_time_ms"] = max_time

        if (cache_size := EnvironmentConfigLoader._get_int("ML_FEATURE_CACHE_SIZE")) is not None:
            overrides["feature_cache_size"] = cache_size

        if (workers := EnvironmentConfigLoader._get_int("ML_FEATURE_PARALLEL_WORKERS")) is not None:
            overrides["pipeline_parallel_workers"] = workers

        return FeatureDomainConfig(**overrides) if overrides else None

    @staticmethod
    def _load_model_domain_from_env() -> ModelDomainConfig | None:
        """Load model domain config from environment."""
        overrides = {}

        if (max_latency := EnvironmentConfigLoader._get_float("ML_MODEL_MAX_INFERENCE_LATENCY_MS")) is not None:
            overrides["max_inference_latency_ms"] = max_latency

        if (cache_size := EnvironmentConfigLoader._get_int("ML_MODEL_CACHE_SIZE")) is not None:
            overrides["model_cache_size"] = cache_size

        return ModelDomainConfig(**overrides) if overrides else None

    @staticmethod
    def _load_strategy_domain_from_env() -> StrategyDomainConfig | None:
        """Load strategy domain config from environment."""
        overrides = {}

        if (max_latency := EnvironmentConfigLoader._get_float("ML_STRATEGY_MAX_SIGNAL_LATENCY_MS")) is not None:
            overrides["max_signal_latency_ms"] = max_latency

        if (dry_run := EnvironmentConfigLoader._get_bool("ML_STRATEGY_DRY_RUN")) is not None:
            overrides["dry_run_mode"] = dry_run

        return StrategyDomainConfig(**overrides) if overrides else None
```

### Environment Variable Reference

#### Core System Variables

```bash
# Environment and debugging
ML_ENVIRONMENT=development|staging|production
ML_DEBUG_MODE=true|false

# Database configuration
DB_CONNECTION=postgresql://user:pass@host:port/db
DATABASE_URL=postgresql://user:pass@host:port/db  # Alternative

# System behavior
ML_AUTO_START_DB=true|false
ML_AUTO_MIGRATE=true|false
ML_STRICT_VALIDATION=true|false

# Performance and monitoring
ML_ENABLE_METRICS=true|false
ML_ENABLE_HEALTH_CHECKS=true|false
ML_HOT_PATH_OPTIMIZATION=true|false

# Fallback and resilience
ML_FALLBACK_ENABLED=true|false
ML_FALLBACK_TIMEOUT_SECONDS=30
```

#### Data Domain Variables

```bash
# Data ingestion
ML_DATA_INGESTION_ENABLED=true|false
ML_DATA_MAX_INGESTION_RATE=10000
ML_DATA_BUFFER_SIZE=50000

# Data quality
ML_DATA_QUALITY_CHECKS_ENABLED=true|false
ML_DATA_OUTLIER_DETECTION_ENABLED=true|false
ML_DATA_STALENESS_THRESHOLD_SECONDS=300

# Data storage
ML_DATA_RETENTION_DAYS=365
ML_DATA_PARTITION_BY_DATE=true|false
ML_DATA_PARQUET_COMPRESSION=snappy

# Backfill
ML_DATA_BACKFILL_ENABLED=true|false
ML_DATA_BACKFILL_BATCH_SIZE=10000
ML_DATA_MAX_CONCURRENT_BACKFILLS=3
```

#### Feature Domain Variables

```bash
# Feature computation
ML_FEATURE_TECHNICAL_INDICATORS=true|false
ML_FEATURE_MICROSTRUCTURE=true|false
ML_FEATURE_CROSS_SECTIONAL=true|false

# Performance
ML_FEATURE_MAX_COMPUTATION_TIME_MS=100.0
ML_FEATURE_CACHE_SIZE=10000
ML_FEATURE_BATCH_COMPUTATION=true|false

# Pipeline
ML_FEATURE_PARALLEL_WORKERS=4
ML_FEATURE_CHUNK_SIZE=1000

# Validation
ML_FEATURE_PARITY_VALIDATION=true|false
ML_FEATURE_PARITY_TOLERANCE=1e-10
ML_FEATURE_DRIFT_DETECTION=true|false
```

#### Model Domain Variables

```bash
# Model loading and caching
ML_MODEL_CACHE_SIZE=32
ML_MODEL_LAZY_LOADING=true|false
ML_MODEL_WARMING_ENABLED=true|false

# Inference performance
ML_MODEL_MAX_INFERENCE_LATENCY_MS=5.0
ML_MODEL_BATCH_INFERENCE=true|false
ML_MODEL_INFERENCE_TIMEOUT_MS=10.0

# Model lifecycle
ML_MODEL_AUTO_RETRAINING=true|false
ML_MODEL_PERFORMANCE_MONITORING=true|false
ML_MODEL_DRIFT_DETECTION=true|false

# A/B testing
ML_MODEL_AB_TESTING_ENABLED=true|false
ML_MODEL_CHAMPION_CHALLENGER_RATIO=0.9
```

#### Strategy Domain Variables

```bash
# Signal generation
ML_STRATEGY_SIGNAL_GENERATION=true|false
ML_STRATEGY_MAX_SIGNAL_LATENCY_MS=10.0
ML_STRATEGY_SIGNAL_AGGREGATION=true|false

# Risk management
ML_STRATEGY_RISK_CHECKS=true|false
ML_STRATEGY_POSITION_SIZE_LIMITS=true|false
ML_STRATEGY_MAX_POSITION_SIZE_RATIO=0.1

# Execution
ML_STRATEGY_PARALLELISM=true|false
ML_STRATEGY_EXECUTION_DELAY_MS=0
ML_STRATEGY_DRY_RUN=true|false
```

## Configuration Validation Patterns

### Multi-Level Validation

```python
from typing import Protocol, Dict, List, Any
from abc import abstractmethod

class ConfigurationValidator(Protocol):
    """Protocol for configuration validators."""

    @abstractmethod
    def validate(self, config: Any) -> List[str]:
        """Validate configuration and return list of issues."""
        pass

class BaseConfigurationValidator:
    """Base validator with common validation patterns."""

    def validate_required_fields(self, config: Any, required_fields: List[str]) -> List[str]:
        """Validate that required fields are present and not None."""
        issues = []
        for field in required_fields:
            if not hasattr(config, field):
                issues.append(f"Required field missing: {field}")
            elif getattr(config, field) is None:
                issues.append(f"Required field is None: {field}")
        return issues

    def validate_ranges(self, config: Any, range_specs: Dict[str, tuple]) -> List[str]:
        """Validate that numeric fields are within specified ranges."""
        issues = []
        for field, (min_val, max_val) in range_specs.items():
            if hasattr(config, field):
                value = getattr(config, field)
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        issues.append(f"Field {field}={value} outside range [{min_val}, {max_val}]")
        return issues

    def validate_dependencies(self, config: Any, dependencies: Dict[str, List[str]]) -> List[str]:
        """Validate field dependencies."""
        issues = []
        for field, required_fields in dependencies.items():
            if hasattr(config, field) and getattr(config, field):
                for required_field in required_fields:
                    if not hasattr(config, required_field) or not getattr(config, required_field):
                        issues.append(f"Field {field} requires {required_field} to be enabled")
        return issues

class SystemConfigurationValidator(BaseConfigurationValidator):
    """Validator for complete system configuration."""

    def validate(self, config: MLSystemConfiguration) -> List[str]:
        """Validate complete system configuration."""
        issues = []

        # Base configuration validation
        issues.extend(self._validate_base_config(config))

        # Domain configuration validation
        issues.extend(self._validate_domain_configs(config))

        # Cross-domain validation
        issues.extend(self._validate_cross_domain_consistency(config))

        # Environment-specific validation
        issues.extend(self._validate_environment_specific(config))

        return issues

    def _validate_base_config(self, config: MLSystemConfiguration) -> List[str]:
        """Validate base configuration."""
        issues = []

        # Required fields
        issues.extend(self.validate_required_fields(config, ["environment", "db_connection"]))

        # Connection string validation
        if config.db_connection:
            if not config.db_connection.startswith(("postgresql://", "sqlite://")):
                issues.append("db_connection must be postgresql:// or sqlite:// URL")

        # Environment-specific validation
        if config.environment == Environment.PRODUCTION:
            if config.debug_mode:
                issues.append("debug_mode must be False in production")
            if config.auto_start_postgres:
                issues.append("auto_start_postgres must be False in production")

        return issues

    def _validate_domain_configs(self, config: MLSystemConfiguration) -> List[str]:
        """Validate individual domain configurations."""
        issues = []

        # Data domain validation
        data_issues = self._validate_data_domain(config.data_domain)
        issues.extend([f"data_domain.{issue}" for issue in data_issues])

        # Feature domain validation
        feature_issues = self._validate_feature_domain(config.feature_domain)
        issues.extend([f"feature_domain.{issue}" for issue in feature_issues])

        # Model domain validation
        model_issues = self._validate_model_domain(config.model_domain)
        issues.extend([f"model_domain.{issue}" for issue in model_issues])

        # Strategy domain validation
        strategy_issues = self._validate_strategy_domain(config.strategy_domain)
        issues.extend([f"strategy_domain.{issue}" for issue in strategy_issues])

        return issues

    def _validate_data_domain(self, config: DataDomainConfig) -> List[str]:
        """Validate data domain configuration."""
        issues = []

        # Range validation
        issues.extend(self.validate_ranges(config, {
            "max_ingestion_rate_per_second": (1, 100000),
            "ingestion_buffer_size": (1000, 1000000),
            "staleness_threshold_seconds": (1, 3600),
            "retention_days": (1, 3650),
            "backfill_batch_size": (100, 100000),
            "max_concurrent_backfills": (1, 10),
        }))

        # Dependency validation
        issues.extend(self.validate_dependencies(config, {
            "backfill_enabled": ["ingestion_enabled"],
            "outlier_detection_enabled": ["quality_checks_enabled"],
        }))

        return issues

    def _validate_feature_domain(self, config: FeatureDomainConfig) -> List[str]:
        """Validate feature domain configuration."""
        issues = []

        # Range validation
        issues.extend(self.validate_ranges(config, {
            "max_feature_computation_time_ms": (0.1, 1000.0),
            "feature_cache_size": (100, 100000),
            "parity_tolerance": (1e-15, 1e-3),
            "pipeline_parallel_workers": (1, 32),
            "pipeline_chunk_size": (10, 10000),
        }))

        # Dependency validation
        issues.extend(self.validate_dependencies(config, {
            "parity_validation_enabled": ["batch_computation_enabled"],
            "drift_detection_enabled": ["parity_validation_enabled"],
        }))

        return issues

    def _validate_model_domain(self, config: ModelDomainConfig) -> List[str]:
        """Validate model domain configuration."""
        issues = []

        # Range validation
        issues.extend(self.validate_ranges(config, {
            "model_cache_size": (1, 1000),
            "max_inference_latency_ms": (0.1, 100.0),
            "inference_timeout_ms": (1.0, 1000.0),
            "champion_challenger_ratio": (0.0, 1.0),
        }))

        # Consistency validation
        if config.max_inference_latency_ms >= config.inference_timeout_ms:
            issues.append("max_inference_latency_ms must be less than inference_timeout_ms")

        # Dependency validation
        issues.extend(self.validate_dependencies(config, {
            "ab_testing_enabled": ["performance_monitoring_enabled"],
            "auto_retraining_enabled": ["drift_detection_enabled"],
        }))

        return issues

    def _validate_strategy_domain(self, config: StrategyDomainConfig) -> List[str]:
        """Validate strategy domain configuration."""
        issues = []

        # Range validation
        issues.extend(self.validate_ranges(config, {
            "max_signal_latency_ms": (0.1, 100.0),
            "max_position_size_ratio": (0.001, 1.0),
            "execution_delay_ms": (0, 10000),
        }))

        # Dependency validation
        issues.extend(self.validate_dependencies(config, {
            "position_size_limits_enabled": ["risk_checks_enabled"],
            "signal_aggregation_enabled": ["signal_generation_enabled"],
        }))

        return issues

    def _validate_cross_domain_consistency(self, config: MLSystemConfiguration) -> List[str]:
        """Validate consistency across domains."""
        issues = []

        # Performance consistency
        total_latency = (
            config.feature_domain.max_feature_computation_time_ms +
            config.model_domain.max_inference_latency_ms +
            config.strategy_domain.max_signal_latency_ms
        )

        if total_latency > 50.0:  # Total pipeline should be under 50ms
            issues.append(f"Total pipeline latency {total_latency}ms exceeds recommended 50ms")

        # Cache consistency
        if config.model_domain.model_cache_size > config.feature_domain.feature_cache_size:
            issues.append("model_cache_size should not exceed feature_cache_size")

        # Capability consistency
        if (config.feature_domain.enable_microstructure_features and
            not config.data_domain.quality_checks_enabled):
            issues.append("Microstructure features require data quality checks")

        return issues

    def _validate_environment_specific(self, config: MLSystemConfiguration) -> List[str]:
        """Validate environment-specific constraints."""
        issues = []

        if config.environment == Environment.PRODUCTION:
            # Production-specific validations
            if config.feature_domain.max_feature_computation_time_ms > 50.0:
                issues.append("Feature computation time too high for production")

            if config.model_domain.max_inference_latency_ms > 5.0:
                issues.append("Model inference latency too high for production")

            if not config.data_domain.quality_checks_enabled:
                issues.append("Data quality checks required in production")

            if not config.model_domain.performance_monitoring_enabled:
                issues.append("Model performance monitoring required in production")

        elif config.environment == Environment.DEVELOPMENT:
            # Development-specific validations
            if not config.debug_mode:
                issues.append("Debug mode recommended in development")

        return issues

# Usage
validator = SystemConfigurationValidator()
config = MLSystemConfiguration.for_environment(Environment.PRODUCTION)
validation_issues = validator.validate(config)

if validation_issues:
    for issue in validation_issues:
        print(f"❌ {issue}")
else:
    print("✅ Configuration valid")
```

## Deployment Configuration Best Practices

### Configuration Management Strategy

#### 1. Configuration Sources Hierarchy

```python
from enum import Enum
from typing import Dict, Any, Optional

class ConfigSource(Enum):
    DEFAULT = "default"           # Built-in defaults
    FILE = "file"                # Configuration files
    ENVIRONMENT = "environment"   # Environment variables
    RUNTIME = "runtime"          # Runtime overrides

class ConfigurationManager:
    """Manages configuration from multiple sources with precedence."""

    def __init__(self):
        self.sources: Dict[ConfigSource, Dict[str, Any]] = {}
        self.precedence_order = [
            ConfigSource.DEFAULT,
            ConfigSource.FILE,
            ConfigSource.ENVIRONMENT,
            ConfigSource.RUNTIME,
        ]

    def load_configuration(self, config_class: Type[MLSystemConfiguration]) -> MLSystemConfiguration:
        """Load configuration from all sources with precedence."""

        # 1. Load defaults
        default_config = config_class()
        self.sources[ConfigSource.DEFAULT] = asdict(default_config)

        # 2. Load from configuration files
        file_config = self._load_from_files()
        if file_config:
            self.sources[ConfigSource.FILE] = file_config

        # 3. Load from environment variables
        env_config = EnvironmentConfigLoader.load_from_env(config_class)
        self.sources[ConfigSource.ENVIRONMENT] = asdict(env_config)

        # 4. Runtime overrides (if any)
        runtime_config = self._get_runtime_overrides()
        if runtime_config:
            self.sources[ConfigSource.RUNTIME] = runtime_config

        # Merge configurations with precedence
        merged_config = self._merge_configurations()

        # Create final configuration object
        return config_class(**merged_config)

    def _load_from_files(self) -> Optional[Dict[str, Any]]:
        """Load configuration from files."""
        config_files = [
            "ml_config.yaml",
            "ml_config.yml",
            "config/ml_config.yaml",
            "/etc/nautilus/ml_config.yaml",
        ]

        for config_file in config_files:
            try:
                import yaml
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")

        return None

    def _get_runtime_overrides(self) -> Optional[Dict[str, Any]]:
        """Get runtime configuration overrides."""
        # This could come from command line arguments,
        # remote configuration service, etc.
        return None

    def _merge_configurations(self) -> Dict[str, Any]:
        """Merge configurations according to precedence order."""
        merged = {}

        for source in self.precedence_order:
            if source in self.sources:
                merged = self._deep_merge(merged, self.sources[source])

        return merged

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result
```

#### 2. Configuration File Templates

##### Development Configuration (ml_config.dev.yaml)

```yaml
# Development Environment Configuration
environment: "development"
debug_mode: true

# Database settings
db_connection: "postgresql://dev_user:dev_pass@localhost:5432/nautilus_dev"
auto_start_postgres: true
auto_migrate: true

# Development-optimized settings
data_domain:
  ingestion_enabled: true
  max_ingestion_rate_per_second: 1000  # Lower rate for development
  quality_checks_enabled: true
  retention_days: 30  # Shorter retention

feature_domain:
  enable_technical_indicators: true
  enable_microstructure_features: false  # Simplified for dev
  max_feature_computation_time_ms: 200.0  # More relaxed
  feature_cache_size: 1000  # Smaller cache

model_domain:
  model_cache_size: 8  # Smaller cache
  max_inference_latency_ms: 10.0  # More relaxed
  auto_retraining_enabled: false  # Disabled in dev

strategy_domain:
  dry_run_mode: true  # Safety in development
  risk_checks_enabled: true
  max_signal_latency_ms: 20.0  # More relaxed

# Monitoring and debugging
enable_metrics: true
enable_health_checks: true
strict_protocol_validation: true
```

##### Staging Configuration (ml_config.staging.yaml)

```yaml
# Staging Environment Configuration
environment: "staging"
debug_mode: false

# Database settings
db_connection: "postgresql://staging_user:staging_pass@staging-db:5432/nautilus_staging"
auto_start_postgres: false  # Managed externally
auto_migrate: true

# Production-like settings with some relaxation
data_domain:
  ingestion_enabled: true
  max_ingestion_rate_per_second: 5000
  quality_checks_enabled: true
  retention_days: 90

feature_domain:
  enable_technical_indicators: true
  enable_microstructure_features: true
  max_feature_computation_time_ms: 100.0
  feature_cache_size: 5000
  parity_validation_enabled: true

model_domain:
  model_cache_size: 16
  max_inference_latency_ms: 7.0
  performance_monitoring_enabled: true
  ab_testing_enabled: true  # Test A/B in staging

strategy_domain:
  dry_run_mode: false  # Real execution in staging
  risk_checks_enabled: true
  max_signal_latency_ms: 15.0

# Full monitoring enabled
enable_metrics: true
enable_health_checks: true
strict_protocol_validation: true
```

##### Production Configuration (ml_config.prod.yaml)

```yaml
# Production Environment Configuration
environment: "production"
debug_mode: false

# Database settings - use connection pooling
db_connection: "postgresql://prod_user:${PROD_DB_PASSWORD}@prod-db-cluster:5432/nautilus_prod"
auto_start_postgres: false  # Externally managed
auto_migrate: false  # Manual migration process

# Production-optimized settings
data_domain:
  ingestion_enabled: true
  max_ingestion_rate_per_second: 10000
  ingestion_buffer_size: 100000  # Larger buffer
  quality_checks_enabled: true
  outlier_detection_enabled: true
  retention_days: 365

feature_domain:
  enable_technical_indicators: true
  enable_microstructure_features: true
  max_feature_computation_time_ms: 50.0  # Strict SLA
  feature_cache_size: 20000  # Large cache
  parity_validation_enabled: true
  drift_detection_enabled: true

model_domain:
  model_cache_size: 64  # Large cache
  max_inference_latency_ms: 3.0  # Strict SLA
  lazy_loading_enabled: true
  performance_monitoring_enabled: true
  drift_detection_enabled: true
  ab_testing_enabled: true

strategy_domain:
  signal_generation_enabled: true
  max_signal_latency_ms: 10.0  # Strict SLA
  risk_checks_enabled: true
  position_size_limits_enabled: true
  dry_run_mode: false

# Production monitoring
enable_metrics: true
enable_health_checks: true
strict_protocol_validation: false  # Performance optimized
hot_path_optimization: true

# Cross-domain settings
event_correlation_enabled: true
unified_monitoring_enabled: true
```

#### 3. Container and Kubernetes Configuration

##### Docker Compose for Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: nautilus_dev
      POSTGRES_USER: dev_user
      POSTGRES_PASSWORD: dev_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data

  ml-system:
    build:
      context: .
      dockerfile: Dockerfile.dev
    environment:
      ML_ENVIRONMENT: development
      ML_DEBUG_MODE: "true"
      DB_CONNECTION: "postgresql://dev_user:dev_pass@postgres:5432/nautilus_dev"
      ML_AUTO_START_DB: "false"  # Use service postgres
      ML_AUTO_MIGRATE: "true"
    depends_on:
      - postgres
    ports:
      - "8080:8080"  # Health check endpoint
      - "9090:9090"  # Metrics endpoint
    volumes:
      - ./config:/app/config
      - ./ml_models:/app/ml_models

volumes:
  postgres_dev_data:
```

##### Kubernetes Configuration for Production

```yaml
# k8s/ml-system-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-system-config
  namespace: nautilus-ml
data:
  ML_ENVIRONMENT: "production"
  ML_DEBUG_MODE: "false"
  ML_AUTO_START_DB: "false"
  ML_AUTO_MIGRATE: "false"
  ML_ENABLE_METRICS: "true"
  ML_ENABLE_HEALTH_CHECKS: "true"
  ML_HOT_PATH_OPTIMIZATION: "true"

  # Performance tuning
  ML_FEATURE_MAX_COMPUTATION_TIME_MS: "50.0"
  ML_MODEL_MAX_INFERENCE_LATENCY_MS: "3.0"
  ML_STRATEGY_MAX_SIGNAL_LATENCY_MS: "10.0"

  # Cache sizes
  ML_FEATURE_CACHE_SIZE: "20000"
  ML_MODEL_CACHE_SIZE: "64"

---
apiVersion: v1
kind: Secret
metadata:
  name: ml-system-secrets
  namespace: nautilus-ml
type: Opaque
stringData:
  DB_CONNECTION: "postgresql://prod_user:secure_password@postgres-cluster:5432/nautilus_prod"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-system
  namespace: nautilus-ml
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-system
  template:
    metadata:
      labels:
        app: ml-system
    spec:
      containers:
      - name: ml-system
        image: nautilus-ml:latest
        ports:
        - containerPort: 8080
          name: health
        - containerPort: 9090
          name: metrics
        envFrom:
        - configMapRef:
            name: ml-system-config
        - secretRef:
            name: ml-system-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: health
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: health
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
```

#### 4. Configuration Validation and Testing

```python
class ConfigurationTestSuite:
    """Test suite for configuration validation."""

    def test_development_configuration(self):
        """Test development configuration."""
        config = MLSystemConfiguration.for_environment(Environment.DEVELOPMENT)
        validator = SystemConfigurationValidator()
        issues = validator.validate(config)

        assert len(issues) == 0, f"Development config issues: {issues}"
        assert config.debug_mode is True
        assert config.auto_start_postgres is True

    def test_production_configuration(self):
        """Test production configuration."""
        config = MLSystemConfiguration.for_environment(Environment.PRODUCTION)
        validator = SystemConfigurationValidator()
        issues = validator.validate(config)

        assert len(issues) == 0, f"Production config issues: {issues}"
        assert config.debug_mode is False
        assert config.auto_start_postgres is False

    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        import os

        # Set test environment variables
        os.environ["ML_DEBUG_MODE"] = "false"
        os.environ["ML_FEATURE_CACHE_SIZE"] = "15000"

        try:
            config = EnvironmentConfigLoader.load_from_env(MLSystemConfiguration)
            assert config.debug_mode is False
            assert config.feature_domain.feature_cache_size == 15000
        finally:
            # Clean up
            del os.environ["ML_DEBUG_MODE"]
            del os.environ["ML_FEATURE_CACHE_SIZE"]

    def test_cross_domain_consistency(self):
        """Test cross-domain configuration consistency."""
        config = MLSystemConfiguration(
            feature_domain=FeatureDomainConfig(max_feature_computation_time_ms=30.0),
            model_domain=ModelDomainConfig(max_inference_latency_ms=15.0),
            strategy_domain=StrategyDomainConfig(max_signal_latency_ms=5.0),
        )

        validator = SystemConfigurationValidator()
        issues = validator.validate(config)

        # Should pass total latency check (30 + 15 + 5 = 50ms)
        latency_issues = [i for i in issues if "pipeline latency" in i]
        assert len(latency_issues) == 0
```

This comprehensive cross-domain configuration guide ensures consistent, validated, and environment-appropriate configuration management across all ML domains in Nautilus Trader. It provides the foundation for reliable deployments and effective system operation across development, staging, and production environments.
