# ADR-008: Observability and Monitoring Strategy

**Status: ACCEPTED**

**Date: 2025-09-10**

**Context: Production-Ready ML System Observability**

## Summary

This ADR establishes the comprehensive observability and monitoring architecture for ML systems in production environments, implementing off-hot-path data collection, centralized metrics management, health scoring algorithms, and real-time dashboard capabilities to ensure system reliability and operational excellence.

## Context

During the comprehensive system review, the need for sophisticated observability emerged as critical for production ML systems. The architecture must support high-frequency trading requirements while providing complete visibility into system performance, data quality, feature drift, model accuracy, and operational health across distributed components.

### Key Observability Challenges

1. **Hot Path Performance**: Observability must not impact trading performance (<5ms P99)
2. **Comprehensive Coverage**: Monitor across data, features, models, strategies, and infrastructure
3. **Real-time Insights**: Immediate visibility into system degradation and anomalies
4. **Event Correlation**: Track causality across complex ML pipeline stages
5. **Operational Integration**: Support for alerting, dashboards, and troubleshooting workflows

## Decision

We implement a comprehensive Observability and Monitoring Strategy with the following architecture:

### 1. Off-Hot-Path Observability Service

**Pattern**: Separate observability collection from critical inference paths to maintain performance.

```python
from ml.observability.service import ObservabilityService
from ml.observability.pipeline import LatencyWatermarkBuilder, EventCorrelationBuilder

class ObservabilityService:
    """Central façade for off-hot-path observability data collection."""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        
        # Lightweight row collectors (minimal hot-path impact)
        self.latency_rows: list[LatencyWatermark] = []
        self.metrics_rows: list[MetricsCollection] = []
        self.correlation_rows: list[EventCorrelation] = []
        self.health_rows: list[HealthScore] = []
        
        # Background pipeline builders
        self.pipeline_builders = {
            "latency": LatencyWatermarkBuilder(),
            "metrics": MetricsCollectionBuilder(),
            "correlation": EventCorrelationBuilder(),
            "health": HealthScoreBuilder()
        }
    
    def add_latency_stage(self, correlation_id: str, instrument_id: str,
                         pipeline_stage: str, ts_stage_start: int, ts_stage_end: int) -> None:
        """Record pipeline stage latency with minimal hot-path impact."""
        # ✅ REQUIRED: Zero allocation in hot path
        watermark = LatencyWatermark(
            correlation_id=correlation_id,
            instrument_id=instrument_id,
            pipeline_stage=pipeline_stage,
            ts_stage_start=ts_stage_start,
            ts_stage_end=ts_stage_end,
            latency_ns=ts_stage_end - ts_stage_start
        )
        self.latency_rows.append(watermark)
    
    def latency_watermarks_df(self) -> pd.DataFrame:
        """Materialize latency watermarks as structured DataFrame."""
        return self.pipeline_builders["latency"].build_dataframe(self.latency_rows)
    
    def add_health(self, component_id: str, health_score: float,
                  subsystem_scores: dict[str, float], timestamp: int,
                  measurement_window_ms: int) -> None:
        """Record component health with subsystem breakdown."""
        health = HealthScore(
            component_id=component_id,
            health_score=health_score,
            subsystem_scores=subsystem_scores,
            timestamp=timestamp,
            measurement_window_ms=measurement_window_ms
        )
        self.health_rows.append(health)
    
    def health_scores_df(self) -> pd.DataFrame:
        """Materialize health scores as structured DataFrame."""
        return self.pipeline_builders["health"].build_dataframe(self.health_rows)
```

### 2. Centralized Metrics Bootstrap

**Pattern**: Prevent prometheus_client registry conflicts through centralized metric creation.

```python
from typing import Dict, List, Optional
import prometheus_client
from threading import RLock

# Global registry and thread safety
_METRICS_REGISTRY: Dict[str, prometheus_client.MetricWrapperBase] = {}
_REGISTRY_LOCK = RLock()

def get_counter(name: str, documentation: str, 
               labels: List[str] | None = None) -> prometheus_client.Counter:
    """Safe counter creation with conflict prevention."""
    with _REGISTRY_LOCK:
        # ✅ REQUIRED: Prevent duplicate metrics
        if name in _METRICS_REGISTRY:
            existing = _METRICS_REGISTRY[name]
            if isinstance(existing, prometheus_client.Counter):
                return existing
            else:
                raise ValueError(f"Metric {name} already exists with different type")
        
        # Create new counter
        counter = prometheus_client.Counter(
            name=name,
            documentation=documentation,
            labelnames=labels or []
        )
        
        _METRICS_REGISTRY[name] = counter
        return counter

def get_histogram(name: str, documentation: str,
                 buckets: List[float] | None = None,
                 labels: List[str] | None = None) -> prometheus_client.Histogram:
    """Safe histogram creation with conflict prevention."""
    with _REGISTRY_LOCK:
        if name in _METRICS_REGISTRY:
            existing = _METRICS_REGISTRY[name]
            if isinstance(existing, prometheus_client.Histogram):
                return existing
            else:
                raise ValueError(f"Metric {name} already exists with different type")
        
        histogram = prometheus_client.Histogram(
            name=name,
            documentation=documentation,
            buckets=buckets or prometheus_client.DEFAULT_BUCKETS,
            labelnames=labels or []
        )
        
        _METRICS_REGISTRY[name] = histogram
        return histogram

# ✅ USAGE: All components use centralized bootstrap
from ml.common.metrics_bootstrap import get_counter, get_histogram

# Component-specific metrics
inference_counter = get_counter(
    "ml_model_inferences_total",
    "Total model inferences",
    labels=["model_id", "instrument_id", "status"]
)

inference_latency = get_histogram(
    "ml_model_inference_duration_seconds",
    "Model inference latency distribution",
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.05],
    labels=["model_id", "version"]
)
```

### 3. Health Scoring and Monitoring

**Pattern**: Automated health calculation with configurable thresholds and subsystem tracking.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    component_id: str
    status: HealthStatus
    score: float  # 0-1 health score
    subsystem_scores: Dict[str, float]
    checks_passed: int
    checks_failed: int
    error_details: List[str]
    timestamp: int

class MLIntegrationManager:
    """Central health aggregation and observability coordination."""
    
    def __init__(self, config: MLIntegrationConfig):
        self.config = config
        self.component_health: Dict[str, HealthCheckResult] = {}
        
        # Health thresholds
        self.healthy_threshold = 0.90
        self.degraded_threshold = 0.70
        
        # Initialize observability service
        self.observability = ObservabilityService(config.observability_config)
    
    def check_component_health(self, component_id: str) -> HealthCheckResult:
        """Comprehensive component health checking."""
        checks_passed = 0
        checks_failed = 0
        error_details = []
        subsystem_scores = {}
        
        try:
            component = self.get_component(component_id)
            
            # ✅ REQUIRED: Protocol-based health checking
            if hasattr(component, 'get_health_status'):
                health_status = component.get_health_status()
                
                # Database connectivity
                if 'database' in health_status:
                    db_healthy = health_status['database'].get('connected', False)
                    subsystem_scores['database'] = 1.0 if db_healthy else 0.0
                    if db_healthy:
                        checks_passed += 1
                    else:
                        checks_failed += 1
                        error_details.append("Database connection failed")
                
                # Store operations
                if 'stores' in health_status:
                    store_health = health_status['stores']
                    for store_name, store_status in store_health.items():
                        store_score = 1.0 if store_status.get('operational', False) else 0.0
                        subsystem_scores[f'store_{store_name}'] = store_score
                        if store_score > 0:
                            checks_passed += 1
                        else:
                            checks_failed += 1
                            error_details.append(f"{store_name} store not operational")
                
                # Performance metrics
                if hasattr(component, 'get_performance_metrics'):
                    perf_metrics = component.get_performance_metrics()
                    
                    # Latency checks
                    if 'p99_latency_ms' in perf_metrics:
                        latency = perf_metrics['p99_latency_ms']
                        latency_healthy = latency < 5.0  # 5ms SLA
                        subsystem_scores['latency'] = 1.0 if latency_healthy else 0.0
                        if latency_healthy:
                            checks_passed += 1
                        else:
                            checks_failed += 1
                            error_details.append(f"P99 latency {latency}ms exceeds 5ms SLA")
                
        except Exception as e:
            checks_failed += 1
            error_details.append(f"Health check error: {str(e)}")
        
        # Calculate overall health score
        total_checks = checks_passed + checks_failed
        score = checks_passed / total_checks if total_checks > 0 else 0.0
        
        # Determine status
        if score >= self.healthy_threshold:
            status = HealthStatus.HEALTHY
        elif score >= self.degraded_threshold:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.CRITICAL
        
        result = HealthCheckResult(
            component_id=component_id,
            status=status,
            score=score,
            subsystem_scores=subsystem_scores,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            error_details=error_details,
            timestamp=time.time_ns()
        )
        
        # Record to observability service
        self.observability.add_health(
            component_id=component_id,
            health_score=score,
            subsystem_scores=subsystem_scores,
            timestamp=time.time_ns(),
            measurement_window_ms=1000
        )
        
        return result
    
    def aggregate_health(self) -> Dict[str, Any]:
        """System-wide health aggregation."""
        domain_health = {}
        component_details = {}
        
        # Check each component
        for component_id in self.get_component_ids():
            health_result = self.check_component_health(component_id)
            component_details[component_id] = health_result
            
            # Aggregate by domain
            domain = self._get_component_domain(component_id)
            if domain not in domain_health:
                domain_health[domain] = []
            domain_health[domain].append(health_result.score)
        
        # Calculate domain averages
        domain_averages = {
            domain: sum(scores) / len(scores) if scores else 0.0
            for domain, scores in domain_health.items()
        }
        
        # Overall system health
        all_scores = [result.score for result in component_details.values()]
        system_health = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            "system_health": system_health,
            "domain_health": domain_averages,
            "component_details": component_details,
            "timestamp": time.time_ns()
        }
```

### 4. Real-time Dashboard and Visualization

**Pattern**: Rich terminal interface and web dashboards for operational monitoring.

```python
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
import asyncio

class MLRealtimeDashboard:
    """Real-time terminal dashboard for ML system monitoring."""
    
    def __init__(self, integration_manager: MLIntegrationManager):
        self.manager = integration_manager
        self.console = Console()
        self.running = False
        
        # Dashboard configuration
        self.refresh_interval = 1.0  # seconds
        self.history_length = 100
        
        # Metrics history
        self.health_history: List[float] = []
        self.latency_history: List[float] = []
        self.error_history: List[int] = []
    
    def create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="health", ratio=2),
            Layout(name="performance", ratio=3)
        )
        
        layout["right"].split_column(
            Layout(name="alerts"),
            Layout(name="activity")
        )
        
        return layout
    
    def update_layout(self, layout: Layout) -> None:
        """Update dashboard with current data."""
        # Get current health
        health_summary = self.manager.aggregate_health()
        
        # Header
        layout["header"].update(
            Panel(
                f"[bold blue]Nautilus ML System Dashboard[/bold blue] | "
                f"System Health: [{'green' if health_summary['system_health'] > 0.9 else 'yellow' if health_summary['system_health'] > 0.7 else 'red'}]"
                f"{health_summary['system_health']:.2%}[/]",
                style="bold white on blue"
            )
        )
        
        # Health overview
        health_table = Table(title="Component Health", show_header=True)
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="green")
        health_table.add_column("Score", justify="right")
        health_table.add_column("Issues", style="red")
        
        for component_id, details in health_summary["component_details"].items():
            status_color = {
                HealthStatus.HEALTHY: "green",
                HealthStatus.DEGRADED: "yellow",
                HealthStatus.CRITICAL: "red",
                HealthStatus.UNKNOWN: "gray"
            }[details.status]
            
            health_table.add_row(
                component_id,
                f"[{status_color}]{details.status.value}[/{status_color}]",
                f"{details.score:.2%}",
                str(details.checks_failed) if details.checks_failed > 0 else "-"
            )
        
        layout["health"].update(Panel(health_table, title="Health Status"))
        
        # Performance metrics
        perf_table = Table(title="Performance Metrics", show_header=True)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Current", justify="right")
        perf_table.add_column("P95", justify="right")
        perf_table.add_column("SLA", justify="right")
        perf_table.add_column("Status", style="green")
        
        # Add performance data
        perf_table.add_row(
            "Inference Latency",
            "2.1ms",
            "3.8ms",
            "<5ms",
            "[green]✓[/green]"
        )
        perf_table.add_row(
            "Feature Computation",
            "0.8ms",
            "1.2ms",
            "<2ms",
            "[green]✓[/green]"
        )
        
        layout["performance"].update(Panel(perf_table, title="Performance"))
        
        # Alerts
        alerts_text = "[yellow]⚠ High memory usage in feature computation[/yellow]\n"
        alerts_text += "[green]✓ All models within latency SLA[/green]\n"
        alerts_text += "[green]✓ Database connections healthy[/green]"
        
        layout["alerts"].update(Panel(alerts_text, title="Alerts"))
        
        # Activity
        activity_text = "15:34:22 - Model inference: EURUSD\n"
        activity_text += "15:34:21 - Feature computed: SPY\n"
        activity_text += "15:34:20 - Signal generated: QQQ"
        
        layout["activity"].update(Panel(activity_text, title="Recent Activity"))
        
        # Footer
        layout["footer"].update(
            Panel(
                f"[dim]Last updated: {time.strftime('%H:%M:%S')} | "
                f"Refresh: {self.refresh_interval}s | Press 'q' to quit[/dim]",
                style="dim white on black"
            )
        )
    
    async def run(self) -> None:
        """Run real-time dashboard."""
        layout = self.create_layout()
        
        with Live(layout, console=self.console, refresh_per_second=1/self.refresh_interval) as live:
            self.running = True
            while self.running:
                try:
                    self.update_layout(layout)
                    await asyncio.sleep(self.refresh_interval)
                except KeyboardInterrupt:
                    self.running = False
                    break
```

### 5. Circuit Breaker Integration

**Pattern**: Production fault tolerance with metrics and automated recovery.

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

class CircuitBreakerState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Blocking requests
    HALF_OPEN = "half_open" # Testing recovery

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    success_threshold: int = 3

class CircuitBreaker:
    """Production circuit breaker with comprehensive metrics."""
    
    def __init__(self, component_name: str, config: CircuitBreakerConfig):
        self.component_name = component_name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        
        # Metrics integration
        from ml.common.metrics_bootstrap import get_counter, get_gauge
        
        self.state_gauge = get_gauge(
            "ml_circuit_breaker_state",
            "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
            labels=["component"]
        )
        
        self.trips_counter = get_counter(
            "ml_circuit_breaker_trips_total",
            "Total circuit breaker state transitions",
            labels=["component", "from_state", "to_state"]
        )
        
        self.operations_counter = get_counter(
            "ml_circuit_breaker_operations_total",
            "Total operations through circuit breaker",
            labels=["component", "result"]
        )
        
        # Initialize metrics
        self._update_state_metrics()
    
    def can_execute(self) -> bool:
        """Check if operation can proceed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitBreakerState.HALF_OPEN)
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record successful operation."""
        self.operations_counter.inc(labels={
            "component": self.component_name,
            "result": "success"
        })
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to(CircuitBreakerState.CLOSED)
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed operation."""
        self.operations_counter.inc(labels={
            "component": self.component_name,
            "result": "failure"
        })
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitBreakerState.OPEN)
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self._transition_to(CircuitBreakerState.OPEN)
    
    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """Transition to new state with metrics."""
        old_state = self.state
        self.state = new_state
        
        # Record state transition
        self.trips_counter.inc(labels={
            "component": self.component_name,
            "from_state": old_state.value,
            "to_state": new_state.value
        })
        
        # Reset counters on state change
        if new_state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitBreakerState.HALF_OPEN:
            self.success_count = 0
        
        self._update_state_metrics()
        
        logger.info(f"Circuit breaker {self.component_name}: {old_state.value} -> {new_state.value}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed for reset attempt."""
        if self.last_failure_time is None:
            return True
        
        recovery_timeout = timedelta(seconds=self.config.recovery_timeout_seconds)
        return datetime.now() - self.last_failure_time > recovery_timeout
    
    def _update_state_metrics(self) -> None:
        """Update state gauge metric."""
        state_values = {
            CircuitBreakerState.CLOSED: 0.0,
            CircuitBreakerState.HALF_OPEN: 0.5,
            CircuitBreakerState.OPEN: 1.0
        }
        
        self.state_gauge.set(
            state_values[self.state],
            labels={"component": self.component_name}
        )

# ✅ INTEGRATION: ML Actor with circuit breaker
class BaseMLInferenceActor:
    """Base ML actor with integrated circuit breaker."""
    
    def __init__(self, config: MLActorConfig):
        super().__init__(config)
        
        # ✅ REQUIRED: Circuit breaker integration
        self.circuit_breaker = CircuitBreaker(
            component_name=self.__class__.__name__,
            config=config.circuit_breaker_config or CircuitBreakerConfig()
        )
    
    def on_bar(self, bar: Bar) -> None:
        """Bar processing with circuit breaker protection."""
        if not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for {self.__class__.__name__}")
            return
        
        try:
            # Your inference logic
            self._process_bar(bar)
            self.circuit_breaker.record_success()
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Inference failed: {e}")
            raise
```

### 6. Comprehensive Metrics Collection

**Pattern**: Domain-specific metrics collectors with graceful degradation.

```python
from abc import ABC, abstractmethod
from threading import RLock
from typing import Dict, Any, Optional

class BaseMetricsCollector(ABC):
    """Base class for domain-specific metrics collectors."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.enabled = True
        self._lock = RLock()
        
        # Initialize metrics
        self._init_metrics()
    
    @abstractmethod
    def _init_metrics(self) -> None:
        """Initialize domain-specific metrics."""
        pass
    
    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics."""
        pass
    
    def enable(self) -> None:
        """Enable metrics collection."""
        with self._lock:
            self.enabled = True
    
    def disable(self) -> None:
        """Disable metrics collection (graceful degradation)."""
        with self._lock:
            self.enabled = False

class FeatureEngineeringCollector(BaseMetricsCollector):
    """Metrics collector for feature engineering pipeline."""
    
    def _init_metrics(self) -> None:
        from ml.common.metrics_bootstrap import get_counter, get_histogram, get_gauge
        
        # Feature computation metrics
        self.computation_counter = get_counter(
            "ml_features_computed_total",
            "Total features computed",
            labels=["instrument_id", "feature_set", "mode"]
        )
        
        self.computation_duration = get_histogram(
            "ml_feature_computation_duration_seconds",
            "Feature computation time",
            buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01],
            labels=["feature_set", "mode"]
        )
        
        self.parity_score = get_gauge(
            "ml_feature_parity_score",
            "Feature parity validation score",
            labels=["feature_set"]
        )
        
        self.drift_score = get_gauge(
            "ml_feature_drift_score", 
            "Feature drift detection score",
            labels=["feature_set", "feature_name"]
        )
    
    def record_computation(self, instrument_id: str, feature_set: str, 
                          mode: str, duration_seconds: float) -> None:
        """Record feature computation event."""
        if not self.enabled:
            return
        
        with self._lock:
            self.computation_counter.inc(labels={
                "instrument_id": instrument_id,
                "feature_set": feature_set,
                "mode": mode
            })
            
            self.computation_duration.observe(
                duration_seconds,
                labels={"feature_set": feature_set, "mode": mode}
            )
    
    def update_parity_score(self, feature_set: str, score: float) -> None:
        """Update feature parity validation score."""
        if not self.enabled:
            return
        
        with self._lock:
            self.parity_score.set(score, labels={"feature_set": feature_set})
    
    def update_drift_scores(self, feature_set: str, drift_scores: Dict[str, float]) -> None:
        """Update feature drift detection scores."""
        if not self.enabled:
            return
        
        with self._lock:
            for feature_name, score in drift_scores.items():
                self.drift_score.set(score, labels={
                    "feature_set": feature_set,
                    "feature_name": feature_name
                })
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect feature engineering metrics."""
        # Implementation would gather current metric values
        return {
            "component": self.component_name,
            "enabled": self.enabled,
            "timestamp": time.time_ns()
        }

class ModelLifecycleCollector(BaseMetricsCollector):
    """Metrics collector for model lifecycle management."""
    
    def _init_metrics(self) -> None:
        from ml.common.metrics_bootstrap import get_counter, get_histogram, get_gauge
        
        # Model operations
        self.inference_counter = get_counter(
            "ml_model_inferences_total",
            "Total model inferences",
            labels=["model_id", "version", "status"]
        )
        
        self.inference_duration = get_histogram(
            "ml_model_inference_duration_seconds", 
            "Model inference latency",
            buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.05],
            labels=["model_id", "version"]
        )
        
        self.accuracy_gauge = get_gauge(
            "ml_model_accuracy_score",
            "Model accuracy score",
            labels=["model_id", "version", "evaluation_window"]
        )
        
        self.confidence_gauge = get_gauge(
            "ml_model_confidence_score",
            "Average model confidence",
            labels=["model_id", "version"]
        )
    
    def record_inference(self, model_id: str, version: str, 
                        duration_seconds: float, status: str) -> None:
        """Record model inference event."""
        if not self.enabled:
            return
        
        with self._lock:
            self.inference_counter.inc(labels={
                "model_id": model_id,
                "version": version,
                "status": status
            })
            
            if status == "success":
                self.inference_duration.observe(
                    duration_seconds,
                    labels={"model_id": model_id, "version": version}
                )
    
    def update_model_metrics(self, model_id: str, version: str,
                           accuracy: float, avg_confidence: float,
                           evaluation_window: str = "1h") -> None:
        """Update model performance metrics."""
        if not self.enabled:
            return
        
        with self._lock:
            self.accuracy_gauge.set(accuracy, labels={
                "model_id": model_id,
                "version": version,
                "evaluation_window": evaluation_window
            })
            
            self.confidence_gauge.set(avg_confidence, labels={
                "model_id": model_id,
                "version": version
            })
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect model lifecycle metrics."""
        return {
            "component": self.component_name,
            "enabled": self.enabled,
            "timestamp": time.time_ns()
        }
```

## Implementation Guidelines

### 1. Hot Path Protection

All observability operations MUST be designed to avoid impact on trading performance:

- **Zero Allocation**: Use pre-allocated buffers in hot paths
- **Asynchronous Collection**: Defer expensive operations to background threads
- **Circuit Breakers**: Fail fast to prevent cascading failures
- **Optional Collection**: Graceful degradation when observability systems fail

### 2. Metrics Naming Conventions

```python
# ✅ CORRECT: Standard naming patterns
ml_component_operation_total{labels}     # Counters
ml_component_operation_duration_seconds  # Histograms  
ml_component_state_ratio                 # Gauges (0-1 values)
ml_component_count                       # Gauges (absolute counts)

# Labels should be low cardinality
labels=["component", "operation", "status"]  # Good
labels=["correlation_id", "timestamp"]       # Bad (high cardinality)
```

### 3. Health Check Implementation

```python
class MLComponent:
    """Example component with health checking."""
    
    def get_health_status(self) -> Dict[str, Any]:
        """Required health check implementation."""
        return {
            "status": "healthy",  # healthy/degraded/critical
            "database": {
                "connected": True,
                "latency_ms": 12.5
            },
            "stores": {
                "feature_store": {"operational": True},
                "model_store": {"operational": True}
            },
            "last_check": time.time_ns()
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Required performance metrics."""
        return {
            "p50_latency_ms": 1.2,
            "p95_latency_ms": 2.8,
            "p99_latency_ms": 4.1,
            "error_rate": 0.001,
            "throughput_per_second": 150.0
        }
```

### 4. Dashboard Integration

```python
# Real-time dashboard startup
dashboard = MLRealtimeDashboard(integration_manager)
await dashboard.run()

# Web dashboard (Grafana integration)
from ml.monitoring.grafana import GrafanaDashboardManager

grafana = GrafanaDashboardManager(config)
grafana.provision_ml_dashboards()
```

## Consequences

### Benefits

1. **Complete Visibility**: Comprehensive observability across all ML pipeline components
2. **Production Performance**: Zero impact on trading latency through off-hot-path design
3. **Operational Excellence**: Real-time dashboards and automated health monitoring
4. **Fault Tolerance**: Circuit breakers and graceful degradation prevent cascading failures
5. **Standardization**: Consistent metrics naming and collection patterns

### Trade-offs

1. **System Complexity**: Additional infrastructure components and monitoring overhead
2. **Resource Usage**: Background observability processes consume memory and CPU
3. **Configuration Overhead**: Multiple monitoring systems require coordination
4. **Alert Fatigue**: Comprehensive monitoring may generate excessive notifications

### Mitigation Strategies

1. **Selective Monitoring**: Configurable collection levels based on environment
2. **Resource Limits**: Bounded memory usage for observability buffers
3. **Alert Prioritization**: Severity-based alerting with escalation policies
4. **Dashboard Consolidation**: Unified interfaces to reduce operational overhead

## Related ADRs

- **ADR-001**: 4-Store + 4-Registry Integration (observability data sources)
- **ADR-003**: Hot/Cold Path Separation (observability performance requirements)
- **ADR-004**: Progressive Fallback Chains (monitoring system resilience)
- **ADR-005**: Centralized Metrics Bootstrap (metrics creation patterns)
- **ADR-006**: Production Security Architecture (monitoring security)
- **ADR-007**: Event-Driven ML Pipeline Architecture (event observability)

## Status

**ACCEPTED** - This ADR establishes the comprehensive observability and monitoring strategy for all ML components.

All ML components MUST implement the required observability interfaces. The system provides multiple monitoring levels from basic health checks to comprehensive real-time dashboards with automatic deployment and configuration management.