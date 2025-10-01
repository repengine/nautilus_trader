# Terminal & Settings Implementation Plan

## Executive Summary

This document provides a comprehensive implementation plan for the Nautilus Trader ML Dashboard's "💻 Terminal" and "⚙️ Settings" tabs. The plan covers command execution architecture, secure shell access, configuration management systems, and real-time settings synchronization across running ML components.

## 1. Terminal Tab Implementation

### UI Elements Analysis
- **Command Input Field**: Multi-line code editor with syntax highlighting
- **Output Display Area**: Real-time streaming output with color support
- **Command History**: Persistent history with search and recall
- **Auto-completion**: Context-aware suggestions for ML commands
- **Session Management**: Multiple terminal sessions with tabs

### Command Execution Architecture

#### Security-First Command Parser
```python
# Secure command execution with whitelist approach
class SecureCommandExecutor:
    """
    Secure command executor with strict whitelisting and sandboxing.

    Only ML-specific commands are allowed, with parameter validation
    and output sanitization.
    """

    ALLOWED_COMMANDS = {
        'ml': {
            'data': ['ingest', 'backfill', 'validate', 'coverage'],
            'model': ['train', 'deploy', 'rollback', 'evaluate'],
            'feature': ['compute', 'promote', 'deprecate', 'validate'],
            'pipeline': ['run', 'status', 'stop', 'schedule'],
            'registry': ['list', 'show', 'search', 'health']
        },
        'system': {
            'status': ['services', 'health', 'metrics'],
            'logs': ['tail', 'search', 'export'],
            'config': ['show', 'validate', 'reload']
        }
    }

    def __init__(self, auth_token: str, db_connection: str):
        self.auth_token = auth_token
        self.db_connection = db_connection
        self.command_validators = self._build_validators()

    def execute_command(self, command: str, session_id: str) -> AsyncGenerator[str, None]:
        """Execute command with streaming output and security validation."""
        parsed = self.parse_command(command)
        if not self.validate_command(parsed):
            yield f"Error: Command '{command}' not allowed or invalid syntax"
            return

        async for output in self._execute_secure(parsed, session_id):
            yield output
```

#### Terminal Session Management
```python
# Multi-session terminal with persistence
class TerminalSessionManager:
    """
    Manages multiple terminal sessions with isolation and persistence.
    """

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.active_sessions: dict[str, TerminalSession] = {}

    async def create_session(self, user_token: str) -> str:
        """Create new isolated terminal session."""
        session_id = f"term_{uuid.uuid4().hex[:8]}"
        session = TerminalSession(
            session_id=session_id,
            user_token=user_token,
            working_dir="/ml",  # Restricted to ML workspace
            environment=self._build_ml_environment()
        )
        self.active_sessions[session_id] = session
        await self._persist_session(session)
        return session_id

    def _build_ml_environment(self) -> dict[str, str]:
        """Build secure environment with ML-specific paths and settings."""
        return {
            'ML_REGISTRY_PATH': './ml_registry',
            'ML_DATA_PATH': './ml_data',
            'PYTHONPATH': '/nautilus_trader/ml',
            'PATH': '/usr/local/bin:/usr/bin:/bin',  # Restricted PATH
        }
```

#### Command Categories and Handlers

**Data Management Commands:**
```python
# ML data management command handlers
class MLDataCommands:
    """Handlers for ML data operations."""

    @command('ml data ingest')
    async def ingest_data(self, symbols: list[str], start_date: str,
                         end_date: str, data_types: list[str]) -> AsyncGenerator:
        """Ingest historical data with streaming progress."""
        # Validate inputs
        validated_symbols = self._validate_symbols(symbols)
        date_range = self._validate_date_range(start_date, end_date)

        # Stream ingestion progress
        ingestion_service = DatabentoIngestionService(self.config)
        async for progress in ingestion_service.ingest_historical(
            symbols=validated_symbols,
            start=date_range[0],
            end=date_range[1],
            data_types=data_types
        ):
            yield f"Progress: {progress['completed']}/{progress['total']} - {progress['symbol']}"

    @command('ml data coverage')
    async def show_coverage(self, dataset: str, instrument: str = None) -> AsyncGenerator:
        """Show data coverage report."""
        reporter = CoverageReporter(self.db_connection)
        coverage_data = await reporter.generate_report(
            dataset=dataset,
            instrument=instrument,
            days_back=30
        )

        # Format as table
        yield self._format_coverage_table(coverage_data)
```

**Model Management Commands:**
```python
# ML model management command handlers
class MLModelCommands:
    """Handlers for ML model operations."""

    @command('ml model train')
    async def train_model(self, config_path: str, model_id: str) -> AsyncGenerator:
        """Train model with streaming logs."""
        # Load and validate training config
        config = MLTrainingConfig.from_yaml(config_path)

        # Stream training progress
        trainer = self._build_trainer(config)
        async for log_entry in trainer.train_with_streaming(model_id):
            yield f"[{log_entry.timestamp}] {log_entry.level}: {log_entry.message}"

    @command('ml model deploy')
    async def deploy_model(self, model_id: str, target: str = "ml_signal_actor") -> AsyncGenerator:
        """Deploy model to target actor."""
        registry = self._get_model_registry()

        try:
            success = registry.deploy_model(model_id, target)
            if success:
                yield f"✓ Model {model_id} deployed to {target}"
                yield f"Monitoring deployment health..."

                # Stream deployment health
                async for health in self._monitor_deployment(model_id, target):
                    yield f"Health: {health['status']} - Latency: {health['latency_ms']}ms"
            else:
                yield f"✗ Failed to deploy model {model_id}"
        except Exception as e:
            yield f"✗ Deployment error: {str(e)}"
```

### Auto-completion Engine
```python
# Context-aware command completion
class TerminalAutoCompletion:
    """
    Provides intelligent auto-completion for ML commands.
    """

    def __init__(self, registry_clients: dict):
        self.registries = registry_clients
        self.completion_cache = TTLCache(maxsize=1000, ttl=300)

    async def get_completions(self, partial_command: str,
                            session_context: dict) -> list[Completion]:
        """Get context-aware completions."""
        tokens = partial_command.split()

        if len(tokens) <= 1:
            return self._get_command_completions(tokens[0] if tokens else "")

        base_command = tokens[0:2]  # e.g., ['ml', 'model']

        if base_command == ['ml', 'model']:
            return await self._get_model_completions(tokens[2:])
        elif base_command == ['ml', 'feature']:
            return await self._get_feature_completions(tokens[2:])
        elif base_command == ['ml', 'data']:
            return await self._get_data_completions(tokens[2:])

        return []

    async def _get_model_completions(self, remaining_tokens: list[str]) -> list[Completion]:
        """Get model-specific completions."""
        if not remaining_tokens:
            return [
                Completion('train', 'Train a new model'),
                Completion('deploy', 'Deploy model to production'),
                Completion('list', 'List available models'),
                Completion('show', 'Show model details')
            ]

        if remaining_tokens[0] == 'deploy' and len(remaining_tokens) == 1:
            # Complete with available model IDs
            models = await self.registries['model'].get_all_models()
            return [Completion(m.manifest.model_id, f"Deploy {m.manifest.model_id}")
                   for m in models]

        return []
```

## 2. Settings Tab Implementation

### Configuration Management Architecture

#### Hierarchical Configuration System
```python
# Multi-level configuration management
class DashboardConfigManager:
    """
    Manages configuration across system, user, and session levels.

    Configuration precedence (highest to lowest):
    1. Session overrides (temporary)
    2. User preferences (persistent)
    3. Environment variables
    4. System defaults
    """

    def __init__(self, db_connection: str, redis_client: Redis):
        self.db = db_connection
        self.redis = redis_client
        self.config_schema = self._load_schema()
        self.watchers: list[ConfigWatcher] = []

    async def get_effective_config(self, user_token: str,
                                 section: str = None) -> dict[str, Any]:
        """Get merged configuration with precedence rules."""
        # Load all layers
        system_config = self._load_system_defaults()
        env_config = self._load_environment_overrides()
        user_config = await self._load_user_preferences(user_token)
        session_config = await self._load_session_overrides(user_token)

        # Merge with precedence
        effective = deep_merge(
            system_config,
            env_config,
            user_config,
            session_config
        )

        # Validate against schema
        self._validate_config(effective)

        return effective if section is None else effective.get(section, {})

    async def update_config(self, user_token: str, section: str,
                          updates: dict[str, Any], persist_level: str = 'user') -> bool:
        """Update configuration at specified persistence level."""
        # Validate updates
        if not self._validate_section_updates(section, updates):
            raise ConfigValidationError(f"Invalid configuration for section {section}")

        # Apply updates
        if persist_level == 'session':
            await self._update_session_config(user_token, section, updates)
        elif persist_level == 'user':
            await self._update_user_config(user_token, section, updates)
        else:
            raise ValueError(f"Invalid persist_level: {persist_level}")

        # Notify watchers
        await self._notify_config_change(section, updates)
        return True
```

#### Configuration Schema Definition
```python
# Configuration schema with validation
CONFIG_SCHEMA = {
    "system": {
        "dashboard": {
            "type": "object",
            "properties": {
                "theme": {"type": "string", "enum": ["light", "dark", "auto"]},
                "refresh_interval": {"type": "integer", "minimum": 1, "maximum": 300},
                "max_concurrent_sessions": {"type": "integer", "minimum": 1, "maximum": 10},
                "enable_websockets": {"type": "boolean"},
                "terminal_history_size": {"type": "integer", "minimum": 100, "maximum": 10000}
            }
        },
        "security": {
            "type": "object",
            "properties": {
                "session_timeout_minutes": {"type": "integer", "minimum": 5, "maximum": 1440},
                "require_token_for_terminal": {"type": "boolean"},
                "allowed_terminal_commands": {"type": "array", "items": {"type": "string"}},
                "max_command_execution_seconds": {"type": "integer", "minimum": 1, "maximum": 3600}
            }
        }
    },
    "trading": {
        "portfolio": {
            "type": "object",
            "properties": {
                "default_position_size": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                "max_positions_per_instrument": {"type": "integer", "minimum": 1, "maximum": 100},
                "risk_limit_percent": {"type": "number", "minimum": 0.01, "maximum": 0.5},
                "stop_loss_default": {"type": "number", "minimum": 0.001, "maximum": 0.1}
            }
        },
        "execution": {
            "type": "object",
            "properties": {
                "order_timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                "slippage_tolerance": {"type": "number", "minimum": 0.0001, "maximum": 0.01},
                "enable_paper_trading": {"type": "boolean"},
                "confirm_live_orders": {"type": "boolean"}
            }
        }
    },
    "ml": {
        "models": {
            "type": "object",
            "properties": {
                "max_inference_latency_ms": {"type": "number", "minimum": 1.0, "maximum": 100.0},
                "prediction_confidence_threshold": {"type": "number", "minimum": 0.1, "maximum": 0.99},
                "enable_model_hot_reload": {"type": "boolean"},
                "model_check_interval_seconds": {"type": "integer", "minimum": 30, "maximum": 3600}
            }
        },
        "features": {
            "type": "object",
            "properties": {
                "feature_computation_timeout": {"type": "integer", "minimum": 1, "maximum": 60},
                "enable_feature_caching": {"type": "boolean"},
                "cache_ttl_seconds": {"type": "integer", "minimum": 60, "maximum": 86400}
            }
        },
        "data": {
            "type": "object",
            "properties": {
                "ingestion_batch_size": {"type": "integer", "minimum": 100, "maximum": 10000},
                "max_backfill_days": {"type": "integer", "minimum": 1, "maximum": 365},
                "enable_data_validation": {"type": "boolean"},
                "data_retention_days": {"type": "integer", "minimum": 7, "maximum": 3650}
            }
        }
    }
}
```

### Settings UI Components

#### System Settings Panel
```python
# System-level configuration UI
class SystemSettingsPanel:
    """Configuration panel for system-wide settings."""

    def render_dashboard_settings(self, current_config: dict) -> dict:
        """Render dashboard appearance and behavior settings."""
        return {
            "title": "Dashboard Settings",
            "fields": [
                {
                    "key": "theme",
                    "label": "Theme",
                    "type": "select",
                    "options": ["light", "dark", "auto"],
                    "value": current_config.get("theme", "light"),
                    "description": "Dashboard visual theme"
                },
                {
                    "key": "refresh_interval",
                    "label": "Refresh Interval (seconds)",
                    "type": "number",
                    "min": 1,
                    "max": 300,
                    "value": current_config.get("refresh_interval", 30),
                    "description": "How often to refresh dashboard data"
                },
                {
                    "key": "enable_websockets",
                    "label": "Enable WebSocket Updates",
                    "type": "boolean",
                    "value": current_config.get("enable_websockets", True),
                    "description": "Use WebSockets for real-time updates"
                }
            ]
        }

    def render_security_settings(self, current_config: dict) -> dict:
        """Render security and access control settings."""
        return {
            "title": "Security Settings",
            "fields": [
                {
                    "key": "session_timeout_minutes",
                    "label": "Session Timeout (minutes)",
                    "type": "number",
                    "min": 5,
                    "max": 1440,
                    "value": current_config.get("session_timeout_minutes", 60),
                    "description": "Auto-logout timeout for idle sessions"
                },
                {
                    "key": "require_token_for_terminal",
                    "label": "Require Authentication for Terminal",
                    "type": "boolean",
                    "value": current_config.get("require_token_for_terminal", True),
                    "description": "Require valid token to access terminal"
                }
            ]
        }
```

#### ML Configuration Panel
```python
# ML-specific configuration management
class MLSettingsPanel:
    """Configuration panel for ML components."""

    def render_model_settings(self, current_config: dict) -> dict:
        """Render model inference and deployment settings."""
        return {
            "title": "Model Settings",
            "fields": [
                {
                    "key": "max_inference_latency_ms",
                    "label": "Max Inference Latency (ms)",
                    "type": "number",
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "value": current_config.get("max_inference_latency_ms", 5.0),
                    "description": "Maximum allowed model inference time"
                },
                {
                    "key": "prediction_confidence_threshold",
                    "label": "Prediction Confidence Threshold",
                    "type": "number",
                    "min": 0.1,
                    "max": 0.99,
                    "step": 0.01,
                    "value": current_config.get("prediction_confidence_threshold", 0.7),
                    "description": "Minimum confidence to act on predictions"
                },
                {
                    "key": "enable_model_hot_reload",
                    "label": "Enable Model Hot Reload",
                    "type": "boolean",
                    "value": current_config.get("enable_model_hot_reload", False),
                    "description": "Allow models to be reloaded without restart"
                }
            ]
        }

    def render_feature_settings(self, current_config: dict) -> dict:
        """Render feature engineering configuration."""
        return {
            "title": "Feature Engineering",
            "fields": [
                {
                    "key": "enable_feature_caching",
                    "label": "Enable Feature Caching",
                    "type": "boolean",
                    "value": current_config.get("enable_feature_caching", True),
                    "description": "Cache computed features to improve performance"
                },
                {
                    "key": "cache_ttl_seconds",
                    "label": "Cache TTL (seconds)",
                    "type": "number",
                    "min": 60,
                    "max": 86400,
                    "value": current_config.get("cache_ttl_seconds", 3600),
                    "description": "How long to cache feature values",
                    "depends_on": "enable_feature_caching"
                }
            ]
        }
```

### Real-time Configuration Synchronization

#### Configuration Change Propagation
```python
# Real-time config sync across components
class ConfigurationSynchronizer:
    """
    Synchronizes configuration changes across all ML components in real-time.
    """

    def __init__(self, message_bus: MessageBusConfig):
        self.bus = publisher_from_config(message_bus)
        self.component_registry: dict[str, ComponentInfo] = {}

    async def register_component(self, component_id: str,
                               config_sections: list[str],
                               reload_callback: str) -> None:
        """Register a component for configuration updates."""
        self.component_registry[component_id] = ComponentInfo(
            component_id=component_id,
            config_sections=config_sections,
            reload_callback=reload_callback,
            last_updated=time.time()
        )

    async def propagate_config_change(self, section: str,
                                    changes: dict[str, Any],
                                    user_token: str) -> dict[str, bool]:
        """Propagate configuration changes to affected components."""
        affected_components = [
            comp for comp in self.component_registry.values()
            if section in comp.config_sections
        ]

        results = {}
        for component in affected_components:
            try:
                # Publish configuration change event
                topic = f"ml.config.change.{component.component_id}"
                message = {
                    "section": section,
                    "changes": changes,
                    "timestamp": time.time(),
                    "user_token": user_token,
                    "reload_callback": component.reload_callback
                }

                success = self.bus.publish(topic, message)
                results[component.component_id] = success

                if success:
                    component.last_updated = time.time()

            except Exception as e:
                logger.error(f"Failed to notify {component.component_id}: {e}")
                results[component.component_id] = False

        return results

    async def validate_component_sync(self, timeout_seconds: int = 30) -> dict[str, bool]:
        """Validate that all components have synced configuration."""
        sync_status = {}
        cutoff_time = time.time() - timeout_seconds

        for component_id, component in self.component_registry.items():
            # Check if component acknowledged recent config changes
            is_synced = component.last_updated > cutoff_time
            sync_status[component_id] = is_synced

            if not is_synced:
                logger.warning(f"Component {component_id} may be out of sync")

        return sync_status
```

#### Configuration Persistence Strategy
```python
# Multi-tier configuration persistence
class ConfigPersistenceManager:
    """
    Manages configuration persistence across multiple storage tiers.

    Persistence Levels:
    1. Session (Redis) - Temporary overrides, cleared on logout
    2. User (PostgreSQL) - Persistent user preferences
    3. System (Files + DB) - System-wide defaults and policies
    """

    def __init__(self, db_connection: str, redis_client: Redis):
        self.db = EngineManager.get_engine(db_connection)
        self.redis = redis_client

    async def persist_user_config(self, user_token: str, section: str,
                                config: dict[str, Any]) -> bool:
        """Persist user configuration to PostgreSQL."""
        try:
            async with self.db.begin() as conn:
                # Upsert user configuration
                query = text("""
                    INSERT INTO ml_user_configs (user_token_hash, section, config_json, updated_at)
                    VALUES (:token_hash, :section, :config, NOW())
                    ON CONFLICT (user_token_hash, section)
                    DO UPDATE SET config_json = :config, updated_at = NOW()
                """)

                token_hash = hashlib.sha256(user_token.encode()).hexdigest()
                await conn.execute(query, {
                    "token_hash": token_hash,
                    "section": section,
                    "config": json.dumps(config)
                })

            return True

        except Exception as e:
            logger.error(f"Failed to persist user config: {e}")
            return False

    async def persist_session_config(self, user_token: str, section: str,
                                   config: dict[str, Any], ttl: int = 3600) -> bool:
        """Persist session configuration to Redis with TTL."""
        try:
            key = f"ml:config:session:{hashlib.sha256(user_token.encode()).hexdigest()}:{section}"
            await self.redis.setex(key, ttl, json.dumps(config))
            return True

        except Exception as e:
            logger.error(f"Failed to persist session config: {e}")
            return False
```

### Settings Validation and Security

#### Configuration Validation Engine
```python
# Comprehensive configuration validation
class ConfigurationValidator:
    """
    Validates configuration changes against schema and security policies.
    """

    def __init__(self, schema: dict, security_policies: dict):
        self.schema = schema
        self.security_policies = security_policies

    def validate_configuration(self, section: str, config: dict[str, Any]) -> ValidationResult:
        """Comprehensive configuration validation."""
        errors = []
        warnings = []

        # Schema validation
        schema_errors = self._validate_against_schema(section, config)
        errors.extend(schema_errors)

        # Security policy validation
        security_errors = self._validate_security_policies(section, config)
        errors.extend(security_errors)

        # Business logic validation
        business_warnings = self._validate_business_logic(section, config)
        warnings.extend(business_warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _validate_security_policies(self, section: str, config: dict) -> list[str]:
        """Validate against security policies."""
        errors = []

        # Check for dangerous settings
        if section == "system.security":
            if config.get("session_timeout_minutes", 0) > 1440:
                errors.append("Session timeout cannot exceed 24 hours")

        if section == "ml.models":
            if config.get("max_inference_latency_ms", 0) > 100:
                errors.append("Model latency limit too high for production")

        return errors

    def _validate_business_logic(self, section: str, config: dict) -> list[str]:
        """Validate business logic constraints."""
        warnings = []

        if section == "trading.portfolio":
            position_size = config.get("default_position_size", 0)
            risk_limit = config.get("risk_limit_percent", 0)

            if position_size > risk_limit:
                warnings.append("Position size exceeds risk limit")

        return warnings
```

## 3. Architecture Integration

### API Endpoints for Terminal & Settings

#### Terminal API Endpoints
```python
# Terminal-specific API endpoints
@app.post("/api/terminal/session")
def create_terminal_session() -> tuple[Any, int]:
    """Create new terminal session."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    try:
        session_id = terminal_manager.create_session(
            user_token=request.headers.get("X-ML-DASHBOARD-TOKEN")
        )
        return jsonify({"session_id": session_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/api/terminal/<session_id>/execute")
def execute_terminal_command(session_id: str) -> tuple[Any, int]:
    """Execute command in terminal session."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json()
    command = payload.get("command", "").strip()

    if not command:
        return jsonify({"error": "empty_command"}), 400

    try:
        execution_id = terminal_manager.execute_command(session_id, command)
        return jsonify({"execution_id": execution_id}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/terminal/<session_id>/output/<execution_id>")
def get_command_output(session_id: str, execution_id: str) -> tuple[Any, int]:
    """Stream command output."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    try:
        output = terminal_manager.get_output(session_id, execution_id)
        return jsonify({"output": output, "complete": output.get("complete", False)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

#### Settings API Endpoints
```python
# Settings-specific API endpoints
@app.get("/api/settings/schema")
def get_settings_schema() -> tuple[Any, int]:
    """Get configuration schema for UI rendering."""
    schema = config_manager.get_schema()
    return jsonify(schema), 200

@app.get("/api/settings/<section>")
def get_settings_section(section: str) -> tuple[Any, int]:
    """Get current configuration for a section."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    try:
        user_token = request.headers.get("X-ML-DASHBOARD-TOKEN")
        config = config_manager.get_effective_config(user_token, section)
        return jsonify(config), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/api/settings/<section>")
def update_settings_section(section: str) -> tuple[Any, int]:
    """Update configuration for a section."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json()
    updates = payload.get("updates", {})
    persist_level = payload.get("persist_level", "user")

    try:
        user_token = request.headers.get("X-ML-DASHBOARD-TOKEN")

        # Validate configuration
        validation = config_validator.validate_configuration(section, updates)
        if not validation.valid:
            return jsonify({
                "error": "validation_failed",
                "details": validation.errors
            }), 400

        # Update configuration
        success = config_manager.update_config(
            user_token, section, updates, persist_level
        )

        if success:
            # Propagate changes to components
            sync_results = await config_synchronizer.propagate_config_change(
                section, updates, user_token
            )

            return jsonify({
                "success": True,
                "sync_results": sync_results,
                "warnings": validation.warnings
            }), 200
        else:
            return jsonify({"error": "update_failed"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### WebSocket Integration for Real-time Updates

#### Terminal WebSocket Handler
```python
# Real-time terminal communication
class TerminalWebSocketHandler:
    """Handle WebSocket connections for terminal sessions."""

    def __init__(self, terminal_manager: TerminalSessionManager):
        self.terminal_manager = terminal_manager
        self.connections: dict[str, WebSocket] = {}

    async def handle_connection(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connection for terminal session."""
        await websocket.accept()
        self.connections[session_id] = websocket

        try:
            while True:
                # Listen for commands from client
                data = await websocket.receive_json()

                if data.get("type") == "execute_command":
                    command = data.get("command", "")

                    # Execute command with streaming output
                    async for output_chunk in self.terminal_manager.execute_command_stream(
                        session_id, command
                    ):
                        await websocket.send_json({
                            "type": "command_output",
                            "data": output_chunk
                        })

                elif data.get("type") == "autocomplete":
                    partial_command = data.get("partial_command", "")
                    completions = await self.terminal_manager.get_completions(
                        session_id, partial_command
                    )

                    await websocket.send_json({
                        "type": "completions",
                        "data": completions
                    })

        except WebSocketDisconnect:
            logger.info(f"Terminal WebSocket disconnected: {session_id}")
        finally:
            self.connections.pop(session_id, None)
```

#### Settings WebSocket Handler
```python
# Real-time settings synchronization
class SettingsWebSocketHandler:
    """Handle WebSocket connections for settings updates."""

    async def handle_connection(self, websocket: WebSocket, user_token: str):
        """Handle WebSocket connection for settings updates."""
        await websocket.accept()

        # Register for configuration change notifications
        config_synchronizer.register_websocket(user_token, websocket)

        try:
            while True:
                # Listen for settings updates from client
                data = await websocket.receive_json()

                if data.get("type") == "update_config":
                    section = data.get("section")
                    updates = data.get("updates", {})

                    # Validate and update configuration
                    try:
                        success = await config_manager.update_config(
                            user_token, section, updates, "user"
                        )

                        # Send acknowledgment
                        await websocket.send_json({
                            "type": "config_updated",
                            "section": section,
                            "success": success
                        })

                    except Exception as e:
                        await websocket.send_json({
                            "type": "config_error",
                            "section": section,
                            "error": str(e)
                        })

        except WebSocketDisconnect:
            logger.info(f"Settings WebSocket disconnected: {user_token}")
        finally:
            config_synchronizer.unregister_websocket(user_token)
```

## 4. Security and Sandboxing

### Command Execution Security
- **Whitelist-only Commands**: Only pre-approved ML commands allowed
- **Parameter Validation**: Strict validation of all command parameters
- **Resource Limits**: CPU, memory, and execution time limits
- **Path Restrictions**: Commands restricted to ML workspace directories
- **User Context**: Commands execute with limited ML user privileges

### Configuration Security
- **Schema Validation**: All config changes validated against strict schemas
- **Security Policies**: Additional security rules beyond schema validation
- **Audit Logging**: All configuration changes logged with user attribution
- **Rollback Capability**: Ability to rollback dangerous configuration changes

## 5. Performance Optimization

### Terminal Performance
- **Command Caching**: Cache frequently used command results
- **Output Streaming**: Stream large command outputs to avoid memory issues
- **Session Persistence**: Maintain session state across connections
- **Compression**: Compress large terminal outputs

### Settings Performance
- **Configuration Caching**: Multi-level cache for configuration data
- **Batch Updates**: Batch multiple configuration changes
- **Lazy Loading**: Load configuration sections on demand
- **Change Detection**: Only propagate actual configuration changes

## 6. Implementation Timeline

### Phase 1: Core Infrastructure (Weeks 1-2)
1. Implement secure command execution framework
2. Build configuration management system
3. Create WebSocket handlers for real-time updates
4. Set up security validation and sandboxing

### Phase 2: Terminal Implementation (Weeks 3-4)
1. Build command parser and validators
2. Implement ML-specific command handlers
3. Add auto-completion engine
4. Create session management system

### Phase 3: Settings Implementation (Weeks 5-6)
1. Build configuration schema system
2. Implement settings UI components
3. Add real-time configuration synchronization
4. Create configuration persistence layer

### Phase 4: Integration & Testing (Weeks 7-8)
1. Integrate with existing dashboard
2. Add comprehensive error handling
3. Performance testing and optimization
4. Security testing and hardening

This implementation plan provides a comprehensive roadmap for building secure, performant Terminal and Settings capabilities that integrate seamlessly with the Nautilus Trader ML infrastructure while maintaining the highest security standards.