# ADR-006: Production Security Architecture

**Status: ACCEPTED**

**Date: 2025-09-10**

**Context: Production ML System Security Hardening**

## Summary

This ADR establishes mandatory security patterns for ML components in production environments, implementing defense-in-depth strategies that protect against common ML security vulnerabilities while maintaining system performance and operability.

## Context

During the comprehensive system review, critical security patterns emerged from production deployment requirements and threat modeling. The ML system handles financial market data, trading signals, and model artifacts that require strict security controls to prevent data exfiltration, model tampering, and unauthorized access.

### Key Security Challenges Identified

1. **Model Artifact Security**: Prevention of arbitrary code execution through malicious model files
2. **Data Protection**: Safeguarding sensitive financial data in memory and at rest
3. **Access Control**: Ensuring only authorized components can access critical resources
4. **Supply Chain Security**: Protecting against compromised dependencies and model artifacts
5. **Runtime Security**: Preventing privilege escalation and resource exhaustion attacks

## Decision

We implement a comprehensive Production Security Architecture with the following mandatory patterns:

### 1. ONNX-Only Model Loading in Production

**Pattern**: Production environments MUST use ONNX models exclusively unless explicitly enabled via environment variable.

```python
# ✅ REQUIRED: Production security check
class ONNXMLInferenceActor(BaseMLInferenceActor):
    def _load_model(self) -> None:
        if not os.getenv("ML_TEST_ALLOW_NON_ONNX", "false").lower() == "true":
            if not self.config.model_path.suffix == ".onnx":
                raise SecurityError(
                    "Production security: Only ONNX models allowed. "
                    "Set ML_TEST_ALLOW_NON_ONNX=true for testing only."
                )

        # Safe ONNX loading with restricted providers
        self.model = self._load_onnx_with_restrictions()

# ❌ DEPRECATED: Security vulnerability
class PickleMLInferenceActor(BaseMLInferenceActor):
    def _load_model(self) -> None:
        raise SecurityError(
            "Pickle model loading deprecated for security. "
            "Use ONNX models in production."
        )
```

**Rationale**: ONNX models provide:

- No arbitrary code execution (unlike pickle)
- Standard format with defined security boundaries
- Performance optimization through compiled inference
- Cross-platform compatibility with minimal attack surface

### 2. Secure Model Registry with Cryptographic Verification

**Pattern**: All model artifacts MUST be cryptographically signed and verified before loading.

```python
class SecureModelRegistry(ModelRegistry):
    """Production model registry with cryptographic verification."""

    def __init__(self, config: ModelRegistryConfig):
        super().__init__(config)
        self.signature_verifier = ModelSignatureVerifier(
            public_key_path=config.verification_key_path
        )

    def load_model(self, model_id: str, version: str) -> ModelManifest:
        """Load model with mandatory signature verification."""
        manifest = super().load_model(model_id, version)

        # ✅ REQUIRED: Cryptographic verification
        if not self.signature_verifier.verify_model(manifest):
            raise SecurityError(
                f"Model {model_id} v{version} failed signature verification"
            )

        # ✅ REQUIRED: Hash verification
        if not self._verify_model_hash(manifest):
            raise SecurityError(
                f"Model {model_id} v{version} hash mismatch - potential tampering"
            )

        return manifest

    def _verify_model_hash(self, manifest: ModelManifest) -> bool:
        """Verify model file integrity."""
        computed_hash = hashlib.sha256(
            Path(manifest.model_path).read_bytes()
        ).hexdigest()
        return computed_hash == manifest.expected_hash
```

### 3. Memory Protection and Data Sanitization

**Pattern**: Implement secure memory handling to prevent data leakage and ensure proper cleanup.

```python
import mlock
from contextlib import contextmanager

class SecureFeatureStore(FeatureStore):
    """Feature store with memory protection."""

    @contextmanager
    def secure_feature_computation(self, size_bytes: int):
        """Secure memory allocation for sensitive computations."""
        # ✅ REQUIRED: Lock memory to prevent swapping
        buffer = mlock.mlockall()
        try:
            secured_array = np.zeros(size_bytes, dtype=np.uint8)
            mlock.mlock(secured_array)
            yield secured_array
        finally:
            # ✅ REQUIRED: Explicit memory clearing
            if 'secured_array' in locals():
                secured_array.fill(0)  # Zero memory
                mlock.munlock(secured_array)
            mlock.munlockall()

    def write_features(self, features: dict[str, float], **kwargs) -> None:
        """Write features with data sanitization."""
        # ✅ REQUIRED: Sanitize before persistence
        sanitized_features = self._sanitize_features(features)
        super().write_features(sanitized_features, **kwargs)

    def _sanitize_features(self, features: dict[str, float]) -> dict[str, float]:
        """Remove potentially sensitive data patterns."""
        sanitized = {}
        for key, value in features.items():
            # Remove NaN/inf values that could indicate data corruption
            if not np.isfinite(value):
                continue
            # Clamp extreme values that could indicate attacks
            if abs(value) > 1e10:
                continue
            sanitized[key] = value
        return sanitized
```

### 4. Access Control and Resource Limits

**Pattern**: Implement fine-grained access control and resource limits to prevent abuse.

```python
class SecureMLActor(BaseMLInferenceActor):
    """ML actor with security controls."""

    def __init__(self, config: MLActorConfig):
        super().__init__(config)
        self._init_security_controls()

    def _init_security_controls(self) -> None:
        """Initialize security controls."""
        # ✅ REQUIRED: Resource limits
        self.max_memory_mb = int(os.getenv("ML_MAX_MEMORY_MB", "1024"))
        self.max_inference_rate = int(os.getenv("ML_MAX_INFERENCE_RATE", "1000"))

        # ✅ REQUIRED: Rate limiting
        self.rate_limiter = TokenBucket(
            capacity=self.max_inference_rate,
            refill_rate=self.max_inference_rate / 60  # Per minute
        )

        # ✅ REQUIRED: Memory monitoring
        self.memory_monitor = MemoryMonitor(limit_mb=self.max_memory_mb)

    def on_bar(self, bar: Bar) -> None:
        """Bar handler with security checks."""
        # ✅ REQUIRED: Rate limiting
        if not self.rate_limiter.consume():
            self._record_security_event("rate_limit_exceeded", bar.instrument_id)
            return

        # ✅ REQUIRED: Memory limit check
        if not self.memory_monitor.check_memory_limit():
            self._record_security_event("memory_limit_exceeded", bar.instrument_id)
            return

        # ✅ REQUIRED: Input validation
        if not self._validate_bar_data(bar):
            self._record_security_event("invalid_input_data", bar.instrument_id)
            return

        super().on_bar(bar)

    def _validate_bar_data(self, bar: Bar) -> bool:
        """Validate bar data for security."""
        # Check for obviously invalid values that could indicate attacks
        if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
            return False
        if bar.volume < 0:
            return False
        if bar.high < bar.low:
            return False
        if not (bar.low <= bar.open <= bar.high):
            return False
        if not (bar.low <= bar.close <= bar.high):
            return False
        return True
```

### 5. Audit Logging and Security Monitoring

**Pattern**: Comprehensive security event logging with structured audit trails.

```python
from ml.common.metrics_bootstrap import get_counter

class SecurityEventLogger:
    """Security event logging with structured audit trails."""

    def __init__(self):
        self.security_events_counter = get_counter(
            "ml_security_events_total",
            "Total security events recorded",
            labels=["event_type", "severity", "component"]
        )

        self.audit_logger = logging.getLogger("ml.security.audit")
        self.audit_logger.setLevel(logging.INFO)

    def record_security_event(self,
                            event_type: str,
                            component: str,
                            severity: str = "medium",
                            details: dict | None = None) -> None:
        """Record security event with metrics and audit log."""

        # ✅ REQUIRED: Metrics recording
        self.security_events_counter.inc(labels={
            "event_type": event_type,
            "severity": severity,
            "component": component
        })

        # ✅ REQUIRED: Structured audit log
        audit_event = {
            "timestamp": time.time_ns(),
            "event_type": event_type,
            "component": component,
            "severity": severity,
            "details": details or {},
            "correlation_id": self._generate_correlation_id()
        }

        self.audit_logger.info(
            f"SECURITY_EVENT: {event_type}",
            extra={"audit_event": audit_event}
        )

        # ✅ REQUIRED: High severity alerts
        if severity == "high":
            self._trigger_security_alert(audit_event)

    def _trigger_security_alert(self, event: dict) -> None:
        """Trigger immediate security alert for high severity events."""
        # Implementation depends on alerting infrastructure
        # Could integrate with PagerDuty, Slack, or other systems
        pass
```

### 6. Secure Configuration Management

**Pattern**: Environment-driven security configuration with validation.

```python
@frozen
class SecurityConfig(NautilusConfig):
    """Security configuration with validation."""

    # Model loading security
    enforce_onnx_only: bool = True
    model_signature_verification: bool = True
    model_hash_verification: bool = True

    # Access control
    max_memory_mb: int = 1024
    max_inference_rate_per_minute: int = 1000
    max_concurrent_actors: int = 10

    # Data protection
    enable_memory_locking: bool = True
    enable_data_sanitization: bool = True
    log_security_events: bool = True

    # Network security
    allowed_registry_hosts: list[str] = field(default_factory=lambda: ["localhost"])
    require_tls: bool = True
    certificate_path: str | None = None

    def __post_init__(self) -> None:
        """Validate security configuration."""
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")

        if self.max_inference_rate_per_minute <= 0:
            raise ValueError("max_inference_rate_per_minute must be positive")

        if self.require_tls and not self.certificate_path:
            raise ValueError("certificate_path required when require_tls=True")
```

## Implementation Guidelines

### 1. Mandatory Security Checks

All ML components MUST implement these security checks:

- **Model Loading**: ONNX-only in production with signature verification
- **Input Validation**: Sanitize all external data before processing
- **Resource Limits**: Enforce memory and rate limits
- **Access Control**: Validate permissions before sensitive operations
- **Audit Logging**: Record all security-relevant events

### 2. Security Testing Requirements

```python
class TestProductionSecurity:
    """Security testing for production ML components."""

    def test_model_loading_security(self):
        """Test model loading security controls."""
        # Test ONNX-only enforcement
        with pytest.raises(SecurityError):
            actor = ONNXMLInferenceActor(config_with_pickle_model)

        # Test signature verification
        with pytest.raises(SecurityError):
            registry.load_model("unsigned_model", "v1.0")

    def test_resource_limits(self):
        """Test resource limit enforcement."""
        actor = SecureMLActor(config)

        # Test rate limiting
        for _ in range(1001):  # Exceed limit
            actor.on_bar(test_bar)

        # Should have triggered rate limiting
        assert actor.rate_limiter.is_empty()

    def test_input_validation(self):
        """Test input validation."""
        actor = SecureMLActor(config)

        # Test invalid bar data
        invalid_bar = Bar(open=-1, high=100, low=50, close=75, volume=1000)
        result = actor._validate_bar_data(invalid_bar)
        assert not result
```

### 3. Security Metrics and Monitoring

Required security metrics:

```python
# Security event counters
ml_security_events_total{event_type, severity, component}

# Resource usage gauges
ml_memory_usage_ratio{component}
ml_inference_rate_current{component}

# Authentication/authorization
ml_auth_failures_total{component, reason}
ml_unauthorized_access_attempts_total{component}
```

## Consequences

### Benefits

1. **Defense in Depth**: Multiple security layers prevent single points of failure
2. **Compliance Ready**: Structured audit trails support regulatory requirements
3. **Attack Surface Reduction**: ONNX-only models eliminate arbitrary code execution
4. **Resource Protection**: Limits prevent denial-of-service attacks
5. **Comprehensive Monitoring**: Security events are tracked and alerted

### Trade-offs

1. **Performance Overhead**: Security checks add latency (typically <1ms)
2. **Operational Complexity**: Additional configuration and monitoring required
3. **Development Workflow**: Stricter requirements for model deployment
4. **Resource Constraints**: Memory and rate limits may require tuning

### Mitigation Strategies

1. **Hot Path Optimization**: Security checks optimized for minimal latency impact
2. **Automated Testing**: Comprehensive security test suite prevents regressions
3. **Clear Documentation**: Security patterns documented with examples
4. **Gradual Rollout**: Environment-based feature flags enable controlled deployment

## Related ADRs

- **ADR-001**: 4-Store + 4-Registry Integration (security boundaries)
- **ADR-003**: Hot/Cold Path Separation (security in hot paths)
- **ADR-004**: Progressive Fallback Chains (secure fallback strategies)
- **ADR-005**: Centralized Metrics Bootstrap (security metrics)

## Status

**ACCEPTED** - This ADR establishes the mandatory security architecture for all ML components in production environments.

All new ML components MUST implement these security patterns. Existing components have a grace period for migration but MUST NOT be deployed to production without security compliance.
