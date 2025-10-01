# Actor Management Implementation Plan

## Overview

This document outlines the implementation strategy for the "🤖 Actors & Models" tab in the ML dashboard, focusing on actor lifecycle management, model deployment, and real-time status monitoring.

## UI Elements Analysis & Implementation Mapping

### 1. Model Performance & P&L Table

**UI Elements:**
- Model ID column
- Type (Teacher/Student/Signal)
- Daily P&L
- Sharpe Ratio
- Win Rate
- Status indicator (Active/Paused/Training/Failed)

**Implementation Mapping:**

```python
# Dashboard Service Extension
class ActorManagementService:
    def list_active_actors(self) -> list[dict[str, Any]]:
        """
        Retrieve running actors and their performance metrics.
        Maps to BaseMLInferenceActor.get_health_status()
        """
        actors = []
        # Query ModelRegistry for deployed models
        registry = self._get_model_registry()
        active_models = registry.get_active_models()

        for model_info in active_models:
            # Get performance from ModelStore
            perf_history = registry.get_performance_history(model_info.manifest.model_id)

            # Calculate metrics
            daily_pnl = self._calculate_daily_pnl(model_info.manifest.model_id)
            sharpe_ratio = self._calculate_sharpe_ratio(perf_history)
            win_rate = self._calculate_win_rate(perf_history)

            actors.append({
                "model_id": model_info.manifest.model_id,
                "type": model_info.manifest.role.value,  # TEACHER/STUDENT/SIGNAL
                "architecture": model_info.manifest.architecture,
                "daily_pnl": daily_pnl,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "status": self._get_actor_status(model_info),
                "deployed_to": model_info.deployed_to,
                "last_update": model_info.last_updated,
                "health": self._query_actor_health(model_info.deployed_to)
            })

        return actors

    def _calculate_daily_pnl(self, model_id: str) -> float:
        """Calculate daily P&L from StrategyStore"""
        # Query strategy_store for signals and their outcomes
        # Aggregate by day and return latest
        pass

    def _calculate_sharpe_ratio(self, perf_history: list[dict]) -> float:
        """Calculate Sharpe ratio from performance history"""
        if not perf_history:
            return 0.0

        returns = [p.get("daily_return", 0) for p in perf_history[-30:]]  # Last 30 days
        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5

        return mean_return / std_return if std_return > 0 else 0.0

    def _get_actor_status(self, model_info: ModelInfo) -> str:
        """Determine actor status from health probes"""
        if not model_info.deployed_to:
            return "inactive"

        # Probe actor health endpoints
        for target in model_info.deployed_to:
            health_url = self._get_actor_health_url(target)
            try:
                response = requests.get(health_url, timeout=2.0)
                if response.ok:
                    health_data = response.json()
                    return health_data.get("status", "unknown")
            except Exception:
                return "unreachable"

        return "inactive"
```

### 2. Action Buttons Per Model

**UI Elements:**
- Pause/Resume toggle
- Config button (opens modal)
- Promote button (Teacher → Student promotion)
- Stop button (graceful shutdown)

**Implementation Mapping:**

```python
class ActorControlService:
    def pause_actor(self, model_id: str, target: str) -> dict[str, Any]:
        """
        Pause actor without stopping the process.
        Uses circuit breaker to block new predictions.
        """
        actor_url = self._get_actor_control_url(target)
        try:
            response = requests.post(
                f"{actor_url}/control/pause",
                json={"model_id": model_id},
                timeout=5.0
            )
            return {
                "ok": response.ok,
                "model_id": model_id,
                "target": target,
                "action": "paused"
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def resume_actor(self, model_id: str, target: str) -> dict[str, Any]:
        """Resume paused actor by clearing circuit breaker."""
        actor_url = self._get_actor_control_url(target)
        try:
            response = requests.post(
                f"{actor_url}/control/resume",
                json={"model_id": model_id},
                timeout=5.0
            )
            return {
                "ok": response.ok,
                "model_id": model_id,
                "target": target,
                "action": "resumed"
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def update_actor_config(
        self,
        model_id: str,
        target: str,
        config_updates: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Update actor configuration at runtime.
        Maps to MLSignalActorConfig hot-reload capability.
        """
        actor_url = self._get_actor_control_url(target)
        try:
            response = requests.post(
                f"{actor_url}/control/config",
                json={
                    "model_id": model_id,
                    "updates": config_updates
                },
                timeout=5.0
            )
            return {
                "ok": response.ok,
                "model_id": model_id,
                "target": target,
                "action": "config_updated"
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def promote_model(self, teacher_id: str, target_role: str) -> dict[str, Any]:
        """
        Promote teacher model to student for deployment.
        Uses ModelRegistry.deploy_model() with role transition.
        """
        registry = self._get_model_registry()

        # Get teacher model
        teacher_info = registry.get_model(teacher_id)
        if not teacher_info or teacher_info.manifest.role != ModelRole.TEACHER:
            return {"ok": False, "error": "Invalid teacher model"}

        # Create student manifest
        student_manifest = ModelManifest(
            model_id=f"{teacher_id}_student_{int(time.time())}",
            role=ModelRole.STUDENT,
            data_requirements=teacher_info.manifest.data_requirements,
            architecture=teacher_info.manifest.architecture,
            feature_schema=teacher_info.manifest.feature_schema,
            feature_schema_hash=teacher_info.manifest.feature_schema_hash,
            parent_id=teacher_id,
            performance_metrics=teacher_info.manifest.performance_metrics,
            deployment_constraints={
                "max_latency_ms": 5,  # Student must be faster
                "max_memory_mb": 100
            }
        )

        # Register and deploy student
        registry.register_model(student_manifest, teacher_info.model_path)
        success = registry.deploy_model(student_manifest.model_id, target_role)

        return {
            "ok": success,
            "teacher_id": teacher_id,
            "student_id": student_manifest.model_id,
            "action": "promoted"
        }

    def stop_actor(self, model_id: str, target: str, graceful: bool = True) -> dict[str, Any]:
        """
        Stop actor process gracefully or forcefully.
        Triggers BaseMLInferenceActor.on_stop() for cleanup.
        """
        if graceful:
            # Send graceful shutdown signal
            actor_url = self._get_actor_control_url(target)
            try:
                response = requests.post(
                    f"{actor_url}/control/shutdown",
                    json={"model_id": model_id, "graceful": True},
                    timeout=10.0
                )
                success = response.ok
            except Exception:
                success = False
        else:
            # Force stop via service controller
            success = self.controller.stop(target)

        # Update deployment status in registry
        if success:
            registry = self._get_model_registry()
            registry.undeploy_model(model_id, target)

        return {
            "ok": success,
            "model_id": model_id,
            "target": target,
            "action": "stopped",
            "graceful": graceful
        }
```

### 3. Deploy New Actor Section

**UI Elements:**
- Model selection dropdown
- Target environment dropdown (dev/staging/prod)
- Configuration form (prediction threshold, signal strategy, etc.)
- Deploy button

**Implementation Mapping:**

```python
class ActorDeploymentService:
    def get_deployable_models(self) -> list[dict[str, Any]]:
        """Get models available for deployment"""
        registry = self._get_model_registry()
        models = registry.get_all_models()

        deployable = []
        for model_info in models:
            if model_info.deployment_status == DeploymentStatus.READY:
                deployable.append({
                    "model_id": model_info.manifest.model_id,
                    "role": model_info.manifest.role.value,
                    "architecture": model_info.manifest.architecture,
                    "version": model_info.manifest.version,
                    "feature_count": len(model_info.manifest.feature_schema),
                    "constraints": model_info.manifest.deployment_constraints
                })

        return deployable

    def deploy_new_actor(
        self,
        model_id: str,
        target_env: str,
        actor_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Deploy new actor instance with specified configuration.
        Creates MLSignalActorConfig and starts actor process.
        """
        registry = self._get_model_registry()
        model_info = registry.get_model(model_id)

        if not model_info:
            return {"ok": False, "error": "Model not found"}

        # Build actor configuration
        config = self._build_actor_config(model_info, actor_config)

        # Validate configuration against model constraints
        validation_result = self._validate_deployment(model_info, config)
        if not validation_result.is_valid:
            return {
                "ok": False,
                "error": f"Validation failed: {validation_result.message}"
            }

        # Deploy via service controller
        try:
            # Start actor service with configuration
            success = self._start_actor_service(target_env, config)

            if success:
                # Update registry deployment status
                registry.deploy_model(model_id, target_env)

                return {
                    "ok": True,
                    "model_id": model_id,
                    "target": target_env,
                    "actor_id": config.get("component_id"),
                    "action": "deployed"
                }
            else:
                return {"ok": False, "error": "Service start failed"}

        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _build_actor_config(
        self,
        model_info: ModelInfo,
        user_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Build MLSignalActorConfig from model manifest and user inputs"""
        from ml.config.actors import MLSignalActorConfig

        # Extract model-specific settings
        manifest = model_info.manifest

        config = {
            "component_id": f"{manifest.model_id}_{int(time.time())}",
            "model_id": manifest.model_id,
            "model_path": model_info.model_path,

            # User-configurable parameters
            "prediction_threshold": user_config.get("prediction_threshold", 0.7),
            "signal_strategy": user_config.get("signal_strategy", "threshold"),
            "bar_type": user_config.get("bar_type", "SPY.NASDAQ-1-MINUTE-BID-EXTERNAL"),
            "warm_up_period": user_config.get("warm_up_period", 100),
            "enable_hot_reload": user_config.get("enable_hot_reload", True),
            "enable_health_monitoring": True,
            "publish_signals": True,
            "log_predictions": user_config.get("log_predictions", False),

            # Model-driven configuration
            "use_manifest_features": True,
            "feature_set_id": manifest.feature_schema_hash,

            # Performance constraints from manifest
            "max_inference_latency_ms": manifest.deployment_constraints.get("max_latency_ms", 5),
            "max_feature_latency_ms": 1.0,

            # Store connections
            "db_connection": self.config.db_connection,
            "persist_features": True,
        }

        return config
```

### 4. Hot Reload Functionality

**UI Elements:**
- Hot reload status indicator
- "Reload Now" button
- Model version display
- Reload history

**Implementation Mapping:**

```python
class HotReloadService:
    def trigger_hot_reload(self, model_id: str, target: str) -> dict[str, Any]:
        """
        Trigger immediate hot reload of model in running actor.
        Maps to BaseMLInferenceActor._execute_hot_reload()
        """
        registry = self._get_model_registry()

        # Check for newer model version
        model_info = registry.get_model(model_id)
        if not model_info:
            return {"ok": False, "error": "Model not found"}

        # Trigger hot reload via actor API
        actor_url = self._get_actor_control_url(target)
        try:
            response = requests.post(
                f"{actor_url}/control/hot_reload",
                json={
                    "model_id": model_id,
                    "force": True
                },
                timeout=10.0
            )

            if response.ok:
                # Log reload event
                self._log_reload_event(model_id, target, success=True)
                return {
                    "ok": True,
                    "model_id": model_id,
                    "target": target,
                    "new_version": model_info.manifest.version,
                    "action": "hot_reloaded"
                }
            else:
                error = response.json().get("error", "Unknown error")
                self._log_reload_event(model_id, target, success=False, error=error)
                return {"ok": False, "error": error}

        except Exception as e:
            self._log_reload_event(model_id, target, success=False, error=str(e))
            return {"ok": False, "error": str(e)}

    def get_reload_status(self, model_id: str, target: str) -> dict[str, Any]:
        """Get hot reload status and history"""
        actor_url = self._get_actor_control_url(target)
        try:
            response = requests.get(f"{actor_url}/status/reload", timeout=2.0)
            if response.ok:
                return response.json()
            else:
                return {"available": False, "error": "Status unavailable"}
        except Exception:
            return {"available": False, "error": "Connection failed"}

    def _log_reload_event(
        self,
        model_id: str,
        target: str,
        success: bool,
        error: str | None = None
    ) -> None:
        """Log reload event for audit trail"""
        event = {
            "timestamp": time.time(),
            "model_id": model_id,
            "target": target,
            "success": success,
            "error": error
        }

        # Store in events table or send to message bus
        try:
            cfg = MessageBusConfig.from_env()
            pub = publisher_from_config(cfg)
            topic = build_topic_for_stage(
                Stage.MODEL_DEPLOYED,
                instrument_id="ALL",
                scheme=cfg.scheme,
                prefix=cfg.topic_prefix
            )
            pub.publish(topic, event)
        except Exception:
            logger.debug("Failed to publish reload event", exc_info=True)
```

### 5. A/B Testing Status Indicators

**UI Elements:**
- A/B test status badges
- Traffic split percentages
- Performance comparison
- "End Test" button

**Implementation Mapping:**

```python
class ABTestingService:
    def list_active_ab_tests(self) -> list[dict[str, Any]]:
        """Get currently running A/B tests"""
        registry = self._get_model_registry()
        ab_tests = registry.list_ab_tests(active_only=True)

        results = []
        for test in ab_tests:
            # Get performance metrics for both variants
            control_metrics = self._get_variant_metrics(test.control_model_id)
            treatment_metrics = self._get_variant_metrics(test.treatment_model_id)

            results.append({
                "test_id": test.test_id,
                "name": test.name,
                "control_model": test.control_model_id,
                "treatment_model": test.treatment_model_id,
                "traffic_split": test.traffic_split,
                "start_time": test.start_time,
                "duration_hours": test.duration_hours,
                "control_metrics": control_metrics,
                "treatment_metrics": treatment_metrics,
                "statistical_significance": self._calculate_significance(
                    control_metrics, treatment_metrics
                ),
                "status": test.status
            })

        return results

    def end_ab_test(
        self,
        test_id: str,
        winner: str,
        reason: str = "manual_end"
    ) -> dict[str, Any]:
        """End A/B test and promote winner"""
        registry = self._get_model_registry()

        try:
            # End the test
            test_result = registry.end_ab_test(test_id, winner, reason)

            # Deploy winner to all traffic
            if winner == "treatment":
                # Promote treatment model
                success = registry.promote_ab_winner(test_id)
            else:
                # Keep control model, undeploy treatment
                success = registry.rollback_ab_test(test_id)

            return {
                "ok": success,
                "test_id": test_id,
                "winner": winner,
                "action": "test_ended"
            }

        except Exception as e:
            return {"ok": False, "error": str(e)}
```

## State Management Strategy

### 1. Real-time Status Updates

```python
class ActorStatusManager:
    def __init__(self):
        self._status_cache = TTLCache(ttl_seconds=5.0, max_entries=100)
        self._health_endpoints = {}

    def get_real_time_status(self, model_id: str, target: str) -> dict[str, Any]:
        """Get cached or fresh status from actor"""
        cache_key = f"{model_id}:{target}"

        # Try cache first
        cached = self._status_cache.get(cache_key)
        if cached:
            return cached

        # Fetch fresh status
        status = self._fetch_actor_status(model_id, target)
        self._status_cache.put(cache_key, status)

        return status

    def _fetch_actor_status(self, model_id: str, target: str) -> dict[str, Any]:
        """Fetch status from actor health endpoint"""
        health_url = self._get_actor_health_url(target)

        try:
            response = requests.get(
                f"{health_url}/health/detailed",
                params={"model_id": model_id},
                timeout=2.0
            )

            if response.ok:
                health_data = response.json()
                return {
                    "status": health_data.get("status", "unknown"),
                    "uptime": health_data.get("uptime_seconds", 0),
                    "predictions_made": health_data.get("predictions_made", 0),
                    "avg_inference_time_ms": health_data.get("avg_inference_time_ms", 0),
                    "success_rate": health_data.get("success_rate", 0),
                    "last_prediction_time": health_data.get("last_prediction_time", 0),
                    "circuit_breaker": health_data.get("circuit_breaker", {}),
                    "model_version": health_data.get("model_version", "unknown"),
                    "is_warmed_up": health_data.get("is_warmed_up", False)
                }
            else:
                return {"status": "unreachable", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"status": "error", "error": str(e)}
```

### 2. Performance Metrics Collection

```python
class PerformanceCollector:
    def collect_model_performance(self, model_id: str) -> dict[str, Any]:
        """Collect comprehensive performance metrics"""
        # Get from ModelStore
        model_store = self._get_model_store()
        predictions = model_store.get_recent_predictions(
            model_id=model_id,
            hours=24
        )

        # Get from StrategyStore
        strategy_store = self._get_strategy_store()
        signals = strategy_store.get_recent_signals(
            model_id=model_id,
            hours=24
        )

        # Calculate metrics
        metrics = {
            "total_predictions": len(predictions),
            "total_signals": len(signals),
            "signal_rate": len(signals) / max(len(predictions), 1),
            "avg_confidence": np.mean([p["confidence"] for p in predictions]) if predictions else 0,
            "avg_inference_time": np.mean([p["inference_time_ms"] for p in predictions]) if predictions else 0,
        }

        # Calculate P&L if signals have outcomes
        if signals:
            pnl_data = self._calculate_signal_pnl(signals)
            metrics.update(pnl_data)

        return metrics
```

## API Endpoints Summary

```python
# Add to ml/dashboard/app.py

@app.get("/api/actors")
def list_actors() -> tuple[Any, int]:
    """List all active actors with performance metrics"""
    data = svc.list_active_actors()
    return jsonify(data), 200

@app.post("/api/actors/<model_id>/pause")
def pause_actor(model_id: str) -> tuple[Any, int]:
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401
    payload = cast(dict[str, Any], request.get_json(silent=True) or {})
    target = payload.get("target")
    result = svc.pause_actor(model_id, target)
    return jsonify(result), 202 if result.get("ok") else 400

@app.post("/api/actors/<model_id>/resume")
def resume_actor(model_id: str) -> tuple[Any, int]:
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401
    payload = cast(dict[str, Any], request.get_json(silent=True) or {})
    target = payload.get("target")
    result = svc.resume_actor(model_id, target)
    return jsonify(result), 202 if result.get("ok") else 400

@app.post("/api/actors/<model_id>/promote")
def promote_actor(model_id: str) -> tuple[Any, int]:
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401
    payload = cast(dict[str, Any], request.get_json(silent=True) or {})
    target_role = payload.get("target_role", "ml_signal_actor")
    result = svc.promote_model(model_id, target_role)
    return jsonify(result), 202 if result.get("ok") else 400

@app.post("/api/actors/deploy")
def deploy_new_actor() -> tuple[Any, int]:
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401
    payload = cast(dict[str, Any], request.get_json(silent=True) or {})
    result = svc.deploy_new_actor(
        model_id=payload.get("model_id"),
        target_env=payload.get("target_env"),
        actor_config=payload.get("config", {})
    )
    return jsonify(result), 202 if result.get("ok") else 400

@app.post("/api/actors/<model_id>/hot_reload")
def hot_reload_actor(model_id: str) -> tuple[Any, int]:
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401
    payload = cast(dict[str, Any], request.get_json(silent=True) or {})
    target = payload.get("target")
    result = svc.trigger_hot_reload(model_id, target)
    return jsonify(result), 202 if result.get("ok") else 400

@app.get("/api/actors/<model_id>/status")
def get_actor_status(model_id: str) -> tuple[Any, int]:
    target = request.args.get("target")
    status = svc.get_real_time_status(model_id, target)
    return jsonify(status), 200

@app.get("/api/ab_tests")
def list_ab_tests() -> tuple[Any, int]:
    data = svc.list_active_ab_tests()
    return jsonify(data), 200

@app.post("/api/ab_tests/<test_id>/end")
def end_ab_test(test_id: str) -> tuple[Any, int]:
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401
    payload = cast(dict[str, Any], request.get_json(silent=True) or {})
    winner = payload.get("winner")
    result = svc.end_ab_test(test_id, winner)
    return jsonify(result), 202 if result.get("ok") else 200
```

## Integration Points

### 1. ModelRegistry Integration

- **Model Lifecycle**: Register → Deploy → Monitor → Hot Reload → Undeploy
- **Version Management**: Semantic versioning with automatic rollback capability
- **Performance Tracking**: Store metrics in `performance_history` field
- **Deployment Status**: Track which models are deployed where

### 2. MLSignalActor Integration

- **Health Endpoints**: Expose `/health/detailed` for status monitoring
- **Control Endpoints**: Expose `/control/{pause,resume,config,shutdown}` for management
- **Hot Reload**: Use existing `_execute_hot_reload()` mechanism
- **Circuit Breaker**: Leverage existing protection for pause/resume

### 3. Store Integration

- **ModelStore**: Query prediction performance and inference metrics
- **StrategyStore**: Query signal generation and P&L outcomes
- **FeatureStore**: Monitor feature quality and drift

### 4. Real-time Updates

- **WebSocket Connection**: Push status updates to dashboard
- **Event Streaming**: Use Redis streams for actor lifecycle events
- **Metric Collection**: Poll actor health endpoints every 5 seconds
- **Caching Strategy**: Cache status for 5 seconds, metrics for 30 seconds

## Code Examples

### Actor Health Monitoring

```python
class ActorHealthMonitor:
    def __init__(self, dashboard_service: DashboardService):
        self.dashboard_service = dashboard_service
        self.health_cache = TTLCache(ttl_seconds=5.0, max_entries=100)

    async def monitor_actor_health(self, model_id: str, target: str) -> dict[str, Any]:
        """Monitor single actor health with caching"""
        cache_key = f"{model_id}:{target}"

        # Check cache first
        cached_health = self.health_cache.get(cache_key)
        if cached_health:
            return cached_health

        # Fetch fresh health data
        health_url = self._get_actor_health_url(target)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{health_url}/health/detailed") as response:
                    if response.status == 200:
                        health_data = await response.json()

                        # Enhance with registry data
                        registry = self.dashboard_service._get_model_registry()
                        model_info = registry.get_model(model_id)

                        enhanced_health = {
                            **health_data,
                            "model_version": model_info.manifest.version if model_info else "unknown",
                            "deployment_status": model_info.deployment_status.value if model_info else "unknown",
                            "last_checked": time.time()
                        }

                        # Cache the result
                        self.health_cache.put(cache_key, enhanced_health)
                        return enhanced_health
                    else:
                        return {"status": "unreachable", "http_status": response.status}

        except Exception as e:
            return {"status": "error", "error": str(e)}
```

This implementation plan provides a comprehensive approach to managing ML actors through the dashboard, with proper integration into the existing Nautilus Trader ML infrastructure while maintaining the hot/cold path separation and performance requirements.