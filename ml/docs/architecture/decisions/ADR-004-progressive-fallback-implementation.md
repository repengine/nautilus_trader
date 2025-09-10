# ADR-004: Progressive Fallback Implementation

## Status
**ACCEPTED** - 2024-01-15

## Context

The Nautilus Trader ML system operates in environments with varying degrees of reliability and resource availability:

**Production Trading Environment**:

- PostgreSQL database may be temporarily unavailable due to maintenance/network issues
- External APIs (market data, model serving) may experience outages
- Container/Kubernetes environments may have resource constraints
- Network partitions between services can occur
- System must continue trading operations even with degraded capabilities

**Edge Deployment Scenarios**:

- Limited local storage and memory
- Intermittent network connectivity
- No guarantee of external service availability
- Must operate autonomously for extended periods

**Development/Testing Environments**:

- Services may not be fully configured
- Database might not be running
- Mock services need to work seamlessly

Previously, the system had binary failure modes - components either worked perfectly or failed completely. This caused:

- Complete system shutdowns when PostgreSQL was unavailable
- Loss of trading opportunities during temporary outages
- Complex manual recovery procedures
- Difficulty in testing with partial infrastructure

## Decision

**Implement progressive fallback chains for ALL external dependencies, providing graceful degradation rather than hard failures.**

### Fallback Philosophy

- **Primary**: Full functionality with all services available
- **Degraded**: Reduced functionality using local caches/files
- **Minimal**: Basic operations using in-memory stores
- **Safe**: System shutdown only as last resort with proper cleanup

### Mandatory Fallback Levels

1. **Level 1 (Primary)**: Full external service functionality
2. **Level 2 (Cached)**: Local cache with periodic sync attempts
3. **Level 3 (File-based)**: Local file storage
4. **Level 4 (In-memory)**: Dummy implementations with warnings

### Implementation Requirements

- **Automatic Detection**: System detects service availability and selects appropriate level
- **Transparent Operation**: Business logic remains unchanged across fallback levels
- **Recovery Detection**: Automatic promotion back to higher levels when services recover
- **Monitoring Integration**: All fallback activations tracked in metrics and logs

## Consequences

### Positive

- **Continuous Operation**: System continues running even when dependencies fail
- **Graceful Degradation**: Performance degrades smoothly rather than failing catastrophically
- **Automated Recovery**: No manual intervention required for common failure scenarios
- **Development Flexibility**: Easy testing with partial infrastructure
- **Deployment Resilience**: Robust operation in various environment configurations

### Negative

- **Implementation Complexity**: Each component needs multiple implementations
- **Resource Overhead**: Maintaining caches and fallback state requires memory
- **Data Consistency**: Risk of stale data when operating in fallback modes
- **Testing Burden**: Must test all fallback levels and transitions

### Risks

- **Silent Degradation**: Operations may continue with stale/incorrect data
- **Resource Leaks**: Fallback implementations may not be as optimized
- **Recovery Thrashing**: Rapid switching between levels during unstable conditions

## Implementation Details

### Fallback Level Enumeration

```python
from enum import Enum, auto
from typing import Protocol, TypeVar, Generic

class FallbackLevel(Enum):
    PRIMARY = "primary"           # Full external service functionality
    CACHED = "cached"            # Local cache with sync attempts
    FILE_BASED = "file_based"    # Local file storage only
    IN_MEMORY = "in_memory"      # Dummy implementation with warnings

T = TypeVar('T')

class FallbackChain(Generic[T]):
    """Generic fallback chain implementation."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.current_level = FallbackLevel.IN_MEMORY
        self.implementations: dict[FallbackLevel, T] = {}
        self.last_failure_time: dict[FallbackLevel, float] = {}
        self.recovery_timeout_seconds = 60

    def register_implementation(self, level: FallbackLevel, implementation: T) -> None:
        """Register implementation for specific fallback level."""
        self.implementations[level] = implementation

    def get_current_implementation(self) -> T:
        """Get current active implementation."""
        # Try to promote to higher level if recovery time has passed
        self._attempt_recovery()

        return self.implementations[self.current_level]

    def handle_failure(self, failed_level: FallbackLevel, error: Exception) -> None:
        """Handle failure and fallback to next level."""
        logger.warning(f"{self.component_name}: {failed_level.value} failed: {error}")

        self.last_failure_time[failed_level] = time.time()

        # Find next available fallback level
        fallback_order = [FallbackLevel.PRIMARY, FallbackLevel.CACHED,
                         FallbackLevel.FILE_BASED, FallbackLevel.IN_MEMORY]

        current_index = fallback_order.index(failed_level)
        for next_level in fallback_order[current_index + 1:]:
            if next_level in self.implementations:
                self._activate_level(next_level)
                break

    def _attempt_recovery(self) -> None:
        """Attempt to recover to higher fallback level."""
        if self.current_level == FallbackLevel.PRIMARY:
            return  # Already at highest level

        # Try to promote to next higher level
        fallback_order = [FallbackLevel.IN_MEMORY, FallbackLevel.FILE_BASED,
                         FallbackLevel.CACHED, FallbackLevel.PRIMARY]

        current_index = fallback_order.index(self.current_level)
        for higher_level in reversed(fallback_order[:current_index]):
            if self._can_attempt_recovery(higher_level):
                try:
                    # Test if higher level is now available
                    implementation = self.implementations[higher_level]
                    self._test_implementation(implementation)

                    # Success - promote to higher level
                    self._activate_level(higher_level)
                    logger.info(f"{self.component_name}: Recovered to {higher_level.value}")
                    break

                except Exception as e:
                    logger.debug(f"{self.component_name}: Recovery attempt to {higher_level.value} failed: {e}")
                    self.last_failure_time[higher_level] = time.time()

    def _can_attempt_recovery(self, level: FallbackLevel) -> bool:
        """Check if enough time has passed to attempt recovery."""
        last_failure = self.last_failure_time.get(level, 0)
        return time.time() - last_failure > self.recovery_timeout_seconds

    def _activate_level(self, level: FallbackLevel) -> None:
        """Activate specific fallback level."""
        old_level = self.current_level
        self.current_level = level

        # Emit metrics for monitoring
        from ml.common.metrics_bootstrap import get_counter
        fallback_counter = get_counter("ml_fallback_transitions_total", "Fallback level transitions")
        fallback_counter.inc(labels={
            "component": self.component_name,
            "from_level": old_level.value,
            "to_level": level.value
        })

        logger.info(f"{self.component_name}: Activated fallback level {level.value}")
```

### Store Implementation Example

```python
class ResilientFeatureStore:
    """Feature store with progressive fallback chain."""

    def __init__(self, connection_string: str, cache_dir: Path):
        self.fallback_chain = FallbackChain[FeatureStoreProtocol]("FeatureStore")
        self._initialize_fallback_implementations(connection_string, cache_dir)

    def _initialize_fallback_implementations(self, connection_string: str, cache_dir: Path) -> None:
        """Initialize all fallback level implementations."""

        # Level 1: Primary PostgreSQL store
        try:
            primary_store = PostgreSQLFeatureStore(connection_string)
            self.fallback_chain.register_implementation(FallbackLevel.PRIMARY, primary_store)
        except Exception as e:
            logger.warning(f"Primary feature store initialization failed: {e}")

        # Level 2: Cached store with periodic sync
        try:
            cached_store = CachedFeatureStore(
                primary_connection=connection_string,
                cache_dir=cache_dir,
                sync_interval_seconds=300,  # 5 minute sync attempts
                max_cache_age_seconds=3600  # 1 hour max stale data
            )
            self.fallback_chain.register_implementation(FallbackLevel.CACHED, cached_store)
        except Exception as e:
            logger.warning(f"Cached feature store initialization failed: {e}")

        # Level 3: File-based store
        try:
            file_store = FileFeatureStore(cache_dir / "features")
            self.fallback_chain.register_implementation(FallbackLevel.FILE_BASED, file_store)
        except Exception as e:
            logger.warning(f"File-based feature store initialization failed: {e}")

        # Level 4: In-memory dummy store (always succeeds)
        dummy_store = DummyFeatureStore()
        self.fallback_chain.register_implementation(FallbackLevel.IN_MEMORY, dummy_store)

        # Start at highest available level
        self._find_initial_level()

    def _find_initial_level(self) -> None:
        """Find highest available fallback level at startup."""
        for level in [FallbackLevel.PRIMARY, FallbackLevel.CACHED,
                     FallbackLevel.FILE_BASED, FallbackLevel.IN_MEMORY]:
            try:
                implementation = self.fallback_chain.implementations[level]
                self.fallback_chain._test_implementation(implementation)
                self.fallback_chain._activate_level(level)
                break
            except Exception:
                continue

    def write_features(self, instrument_id: str, features: dict[str, float],
                      ts_event: int, ts_init: int) -> None:
        """Write features with automatic fallback on failure."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                current_store = self.fallback_chain.get_current_implementation()
                current_store.write_features(instrument_id, features, ts_event, ts_init)
                return  # Success

            except Exception as e:
                logger.warning(f"Feature store write failed (attempt {attempt + 1}): {e}")

                if attempt < max_retries - 1:  # Not last attempt
                    # Try fallback to next level
                    self.fallback_chain.handle_failure(self.fallback_chain.current_level, e)
                else:
                    # Final attempt failed
                    logger.error(f"All feature store fallback levels failed: {e}")
                    raise FeatureStoreError(f"Write failed after {max_retries} attempts") from e

    def get_latest_features(self, instrument_id: str, ts_event: int) -> dict[str, float] | None:
        """Get features with fallback and staleness warnings."""
        try:
            current_store = self.fallback_chain.get_current_implementation()
            features = current_store.get_latest_features(instrument_id, ts_event)

            # Warn if using degraded data
            if self.fallback_chain.current_level != FallbackLevel.PRIMARY:
                logger.warning(f"Using features from {self.fallback_chain.current_level.value} store")

                # Add metadata about data freshness
                if features and self.fallback_chain.current_level == FallbackLevel.CACHED:
                    features['_cache_age_seconds'] = self._get_cache_age(instrument_id, ts_event)

            return features

        except Exception as e:
            # Try fallback
            self.fallback_chain.handle_failure(self.fallback_chain.current_level, e)

            # Recursive call with fallback store
            return self.get_latest_features(instrument_id, ts_event)
```

### Model Registry Fallback

```python
class ResilientModelRegistry:
    """Model registry with file system fallback."""

    def __init__(self, registry_path: Path, fallback_paths: list[Path]):
        self.fallback_chain = FallbackChain[ModelRegistryProtocol]("ModelRegistry")
        self._initialize_fallbacks(registry_path, fallback_paths)

    def load_model(self, model_id: str) -> Any:
        """Load model with progressive fallback."""

        for attempt_level in [FallbackLevel.PRIMARY, FallbackLevel.FILE_BASED, FallbackLevel.IN_MEMORY]:
            try:
                registry = self.fallback_chain.implementations.get(attempt_level)
                if registry:
                    model = registry.load_model(model_id)

                    # Cache successful loads in higher levels
                    self._cache_model_in_higher_levels(model_id, model)

                    return model

            except ModelNotFoundError:
                continue  # Try next level
            except Exception as e:
                logger.warning(f"Model loading failed at level {attempt_level}: {e}")
                continue

        # All fallback levels failed
        raise ModelNotFoundError(f"Model {model_id} not found in any fallback level")

    def _cache_model_in_higher_levels(self, model_id: str, model: Any) -> None:
        """Cache successfully loaded model in higher-level stores."""
        # This enables faster subsequent loads and provides redundancy
        for level in [FallbackLevel.PRIMARY, FallbackLevel.FILE_BASED]:
            try:
                registry = self.fallback_chain.implementations.get(level)
                if registry and hasattr(registry, 'cache_model'):
                    registry.cache_model(model_id, model)
            except Exception:
                pass  # Non-critical operation
```

### Circuit Breaker Integration

```python
from datetime import datetime, timedelta

class CircuitBreakerFallback:
    """Circuit breaker with automatic fallback activation."""

    def __init__(self, fallback_chain: FallbackChain,
                 failure_threshold: int = 5,
                 recovery_timeout: timedelta = timedelta(minutes=2)):
        self.fallback_chain = fallback_chain
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failure_counts: dict[FallbackLevel, int] = {}
        self.circuit_open_time: dict[FallbackLevel, datetime] = {}

    def execute_with_circuit_breaker(self, operation: callable, *args, **kwargs):
        """Execute operation with circuit breaker protection."""
        current_level = self.fallback_chain.current_level

        # Check if circuit is open for current level
        if self._is_circuit_open(current_level):
            if self._should_try_recovery(current_level):
                # Try to close circuit
                self._attempt_circuit_recovery(current_level)
            else:
                # Circuit still open, use fallback immediately
                self.fallback_chain.handle_failure(current_level,
                    CircuitBreakerOpenError(f"Circuit open for {current_level}"))
                return self.execute_with_circuit_breaker(operation, *args, **kwargs)

        try:
            result = operation(*args, **kwargs)
            self._record_success(current_level)
            return result

        except Exception as e:
            self._record_failure(current_level)

            if self.failure_counts[current_level] >= self.failure_threshold:
                self._open_circuit(current_level)
                self.fallback_chain.handle_failure(current_level, e)
                return self.execute_with_circuit_breaker(operation, *args, **kwargs)
            else:
                raise  # Failure count not yet at threshold

    def _is_circuit_open(self, level: FallbackLevel) -> bool:
        """Check if circuit is open for given level."""
        return level in self.circuit_open_time

    def _should_try_recovery(self, level: FallbackLevel) -> bool:
        """Check if enough time has passed to try circuit recovery."""
        open_time = self.circuit_open_time.get(level)
        if open_time:
            return datetime.now() - open_time > self.recovery_timeout
        return False

    def _open_circuit(self, level: FallbackLevel) -> None:
        """Open circuit for given level."""
        self.circuit_open_time[level] = datetime.now()
        logger.warning(f"Circuit breaker opened for {level.value}")

        # Emit metrics
        from ml.common.metrics_bootstrap import get_counter
        circuit_counter = get_counter("ml_circuit_breaker_opens_total", "Circuit breaker opens")
        circuit_counter.inc(labels={"level": level.value})
```

### Network Partition Handling

```python
class NetworkPartitionResilience:
    """Handle network partitions with local operations."""

    def __init__(self, fallback_chain: FallbackChain):
        self.fallback_chain = fallback_chain
        self.partition_detected = False
        self.queued_operations: list[dict] = []
        self.max_queue_size = 10000

    def execute_with_partition_handling(self, operation: callable,
                                       operation_data: dict, *args, **kwargs):
        """Execute operation with network partition handling."""

        if self.partition_detected:
            # We're in partition mode - queue operation and use local fallback
            self._queue_operation(operation, operation_data, args, kwargs)
            return self._execute_local_fallback(operation, *args, **kwargs)

        try:
            # Try normal execution
            result = operation(*args, **kwargs)

            # If we have queued operations, try to sync them
            if self.queued_operations:
                self._sync_queued_operations()

            return result

        except NetworkError as e:
            # Network partition detected
            self._activate_partition_mode(e)
            return self.execute_with_partition_handling(operation, operation_data, *args, **kwargs)

    def _activate_partition_mode(self, error: Exception) -> None:
        """Activate partition handling mode."""
        self.partition_detected = True
        logger.warning(f"Network partition detected: {error}")

        # Force fallback to local-only operations
        if self.fallback_chain.current_level in [FallbackLevel.PRIMARY, FallbackLevel.CACHED]:
            self.fallback_chain.handle_failure(self.fallback_chain.current_level, error)

        # Start partition recovery monitoring
        self._start_partition_recovery_monitoring()

    def _queue_operation(self, operation: callable, operation_data: dict,
                        args: tuple, kwargs: dict) -> None:
        """Queue operation for later synchronization."""
        if len(self.queued_operations) >= self.max_queue_size:
            # Remove oldest operation to make room
            self.queued_operations.pop(0)
            logger.warning("Operation queue full, dropping oldest operation")

        self.queued_operations.append({
            'operation': operation.__name__,
            'data': operation_data,
            'args': args,
            'kwargs': kwargs,
            'timestamp': time.time(),
        })

    def _sync_queued_operations(self) -> None:
        """Synchronize queued operations after partition recovery."""
        logger.info(f"Syncing {len(self.queued_operations)} queued operations")

        synced_count = 0
        failed_count = 0

        for queued_op in self.queued_operations.copy():
            try:
                # Attempt to replay operation
                self._replay_operation(queued_op)
                synced_count += 1
                self.queued_operations.remove(queued_op)

            except Exception as e:
                logger.warning(f"Failed to sync operation: {e}")
                failed_count += 1

                # If operation is too old, discard it
                if time.time() - queued_op['timestamp'] > 3600:  # 1 hour
                    self.queued_operations.remove(queued_op)

        logger.info(f"Sync complete: {synced_count} synced, {failed_count} failed")
```

## Monitoring and Observability

### Fallback Metrics

```python
from ml.common.metrics_bootstrap import get_counter, get_gauge, get_histogram

class FallbackMetrics:
    """Metrics for fallback chain monitoring."""

    def __init__(self):
        self.fallback_activations = get_counter(
            "ml_fallback_activations_total",
            "Total fallback activations",
            labels=["component", "from_level", "to_level", "reason"]
        )

        self.current_fallback_level = get_gauge(
            "ml_current_fallback_level",
            "Current fallback level (0=primary, 1=cached, 2=file, 3=memory)",
            labels=["component"]
        )

        self.recovery_attempts = get_counter(
            "ml_fallback_recovery_attempts_total",
            "Fallback recovery attempts",
            labels=["component", "to_level", "success"]
        )

        self.operation_latency_by_level = get_histogram(
            "ml_operation_latency_by_fallback_level_seconds",
            "Operation latency by fallback level",
            buckets=[0.001, 0.01, 0.1, 1.0, 10.0],
            labels=["component", "operation", "fallback_level"]
        )

    def record_fallback_activation(self, component: str, from_level: str,
                                 to_level: str, reason: str) -> None:
        """Record fallback activation."""
        self.fallback_activations.inc(labels={
            "component": component,
            "from_level": from_level,
            "to_level": to_level,
            "reason": reason
        })

        # Update current level gauge
        level_mapping = {"primary": 0, "cached": 1, "file_based": 2, "in_memory": 3}
        self.current_fallback_level.set(
            level_mapping.get(to_level, 3),
            labels={"component": component}
        )
```

### Health Checks with Fallback Awareness

```python
class FallbackAwareHealthCheck:
    """Health checks that understand fallback states."""

    def check_component_health(self, component_with_fallback) -> dict[str, Any]:
        """Check health including fallback status."""
        current_level = component_with_fallback.fallback_chain.current_level

        health_status = {
            "status": "healthy" if current_level != FallbackLevel.IN_MEMORY else "degraded",
            "fallback_level": current_level.value,
            "available_levels": list(component_with_fallback.fallback_chain.implementations.keys()),
            "last_failures": component_with_fallback.fallback_chain.last_failure_time,
        }

        # Test all levels to see what's currently working
        level_health = {}
        for level, implementation in component_with_fallback.fallback_chain.implementations.items():
            try:
                # Quick health test
                implementation.get_health_status()
                level_health[level.value] = "available"
            except Exception:
                level_health[level.value] = "unavailable"

        health_status["level_availability"] = level_health
        return health_status
```

## Testing Strategy

### Fallback Transition Testing

```python
import pytest
from unittest.mock import Mock, patch

class TestFallbackTransitions:
    """Test fallback level transitions."""

    def test_automatic_fallback_on_primary_failure(self):
        """Test automatic fallback when primary level fails."""
        # Setup resilient store with mocked implementations
        primary_store = Mock()
        cached_store = Mock()

        primary_store.write_features.side_effect = ConnectionError("Database unavailable")
        cached_store.write_features.return_value = None

        resilient_store = ResilientFeatureStore("postgresql://fake", Path("/tmp"))
        resilient_store.fallback_chain.register_implementation(FallbackLevel.PRIMARY, primary_store)
        resilient_store.fallback_chain.register_implementation(FallbackLevel.CACHED, cached_store)
        resilient_store.fallback_chain._activate_level(FallbackLevel.PRIMARY)

        # Execute operation that should trigger fallback
        resilient_store.write_features("EUR/USD", {"close": 1.1000}, time.time_ns(), time.time_ns())

        # Verify fallback occurred
        assert resilient_store.fallback_chain.current_level == FallbackLevel.CACHED
        cached_store.write_features.assert_called_once()

    def test_recovery_promotion(self):
        """Test automatic promotion back to higher level on recovery."""
        resilient_store = self._create_test_store()

        # Start in fallback mode
        resilient_store.fallback_chain._activate_level(FallbackLevel.CACHED)

        # Simulate recovery conditions
        resilient_store.fallback_chain.last_failure_time[FallbackLevel.PRIMARY] = time.time() - 120  # 2 minutes ago

        # Mock primary store as now working
        primary_store = resilient_store.fallback_chain.implementations[FallbackLevel.PRIMARY]
        primary_store.get_health_status.return_value = {"status": "ok"}

        # Trigger recovery check
        resilient_store.get_latest_features("EUR/USD", time.time_ns())

        # Should have promoted back to primary
        assert resilient_store.fallback_chain.current_level == FallbackLevel.PRIMARY

    @pytest.mark.integration
    def test_end_to_end_fallback_scenario(self):
        """Test complete fallback scenario with real services."""

        # This test requires actual PostgreSQL and file system
        with patch('sqlalchemy.create_engine') as mock_engine:
            # Simulate database connection failure
            mock_engine.side_effect = ConnectionError("Connection refused")

            resilient_store = ResilientFeatureStore(
                "postgresql://fake:fake@fake:5432/fake",
                Path("/tmp/test_cache")
            )

            # Should have fallen back to file-based storage
            assert resilient_store.fallback_chain.current_level == FallbackLevel.FILE_BASED

            # Should still be able to write/read features
            test_features = {"close": 1.1000, "volume": 10000}
            resilient_store.write_features("EUR/USD", test_features, time.time_ns(), time.time_ns())

            retrieved_features = resilient_store.get_latest_features("EUR/USD", time.time_ns())
            assert retrieved_features is not None
            assert retrieved_features["close"] == 1.1000
```

## Migration Strategy

### Phase 1: Core Fallback Infrastructure

- Implement `FallbackChain` generic class
- Create fallback level enumeration and management
- Add metrics and monitoring for fallback states

### Phase 2: Store Implementation

- Implement progressive fallback for each store type
- Add file-based and in-memory fallback implementations
- Integrate circuit breakers and recovery logic

### Phase 3: Registry and Service Fallback

- Implement model registry fallback to local files
- Add network partition handling for external services
- Create cached implementations for external APIs

### Phase 4: Testing and Validation

- Comprehensive fallback transition testing
- Failure injection testing in CI/CD
- Load testing with various failure scenarios

### Phase 5: Monitoring and Operations

- Grafana dashboards for fallback status monitoring
- Alerting on fallback activations and recovery failures
- Documentation and runbooks for operations teams

## Related ADRs

- ADR-001: 4-Store + 4-Registry Mandatory Pattern
- ADR-002: Protocol-First Interface Design
- ADR-003: Hot/Cold Path Separation Strategy
- ADR-005: Centralized Metrics Bootstrap Pattern

## References

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Fallback Implementation Examples](../../stores/)
- [Resilience Testing Strategy](../integration_testing_strategy.md)
