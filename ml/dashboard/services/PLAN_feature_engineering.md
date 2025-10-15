# Feature Engineering Tab Implementation Plan

## Overview

This document provides a comprehensive implementation plan for the dashboard's "🔬 Features" tab, analyzing UI elements and mapping them to the underlying FeatureEngineer, FeatureConfig, and FeatureRegistry components. The plan follows Nautilus Trader's mandatory patterns and hot/cold path separation.

## Current UI Elements Analysis

### 1. Feature Designer Section

**UI Components Identified**:
- Feature Set Name input field (`#feature-set-name`)
- Base Features checkboxes:
  - ✅ Price Features (returns, momentum)
  - ✅ Volume Features (volume moving averages)
  - ⬜ Microstructure (bid-ask spread, order imbalance)
  - ⬜ Order Flow (trade direction, flow toxicity)
- Technical Indicators multi-select (`#technical-indicators`):
  - RSI, MACD, Bollinger Bands, EMA, ATR, VWAP
- Lookback Periods input (`#lookback-periods`): "10,20,50,100,200"
- Generate Features button (`onclick="generateFeatures()"`)

**Backend Mapping**:
```python
# ml/dashboard/services/feature_designer_service.py
class FeatureDesignerService:
    """Manages feature set design and generation workflow."""

    def __init__(self):
        self.feature_registry = self._get_feature_registry()
        self.feature_store = self._get_feature_store()

    def create_feature_config_from_ui(self, ui_params: dict) -> FeatureConfig:
        """Convert UI parameters to FeatureConfig object."""
        return FeatureConfig(
            # Price Features
            return_periods=ui_params.get("lookback_periods", [10, 20, 50, 100, 200]),
            momentum_periods=ui_params.get("lookback_periods", [10, 20, 50]),

            # Technical Indicators
            rsi_period=14 if "rsi" in ui_params.get("technical_indicators", []) else None,
            bb_period=20 if "bb" in ui_params.get("technical_indicators", []) else None,
            bb_std=2.0,
            atr_period=20 if "atr" in ui_params.get("technical_indicators", []) else None,

            # Moving Averages
            ema_fast=12 if "ema" in ui_params.get("technical_indicators", []) else None,
            ema_slow=26 if "ema" in ui_params.get("technical_indicators", []) else None,
            macd_signal=9 if "macd" in ui_params.get("technical_indicators", []) else None,

            # Volume Features
            volume_ma_periods=[5, 10, 20] if ui_params.get("volume_features") else [],

            # Advanced Features
            include_microstructure=ui_params.get("microstructure", False),
            include_order_flow=ui_params.get("order_flow", False),
        )

    async def generate_features(self, feature_set_name: str, ui_params: dict) -> dict[str, Any]:
        """Generate features based on UI configuration."""
        try:
            # 1. Create FeatureConfig from UI parameters
            config = self.create_feature_config_from_ui(ui_params)

            # 2. Initialize FeatureEngineer (cold path)
            engineer = FeatureEngineer(config)

            # 3. Generate pipeline specification
            pipeline_spec = self._create_pipeline_spec(config, ui_params)

            # 4. Compute feature schema and names
            feature_names = self._compute_feature_names(config)

            # 5. Create FeatureManifest
            manifest = FeatureManifest(
                feature_set_id=feature_set_name,
                name=f"Generated Feature Set: {feature_set_name}",
                version="1.0.0",
                role=FeatureRole.INFERENCE_SUPPORT,
                data_requirements=self._determine_data_requirements(ui_params),
                feature_names=feature_names,
                feature_dtypes=["float32"] * len(feature_names),
                schema_hash=compute_schema_hash(feature_names),
                pipeline_signature=pipeline_spec.compute_signature(),
                pipeline_version="1.0",
                capability_flags={
                    "microstructure": ui_params.get("microstructure", False),
                    "order_flow": ui_params.get("order_flow", False),
                    "high_frequency": True,
                },
                constraints={
                    "max_latency_ms": 5.0,  # Hot path requirement
                    "warmup_bars": max(ui_params.get("lookback_periods", [20])),
                    "memory_mb": 64,
                },
                parity_tolerance=1e-10,
                parity_digest={},
                perf_digest={},
                parent_feature_set_id=None,
                metadata={
                    "created_via": "dashboard_ui",
                    "ui_config": ui_params,
                },
                created_at=time.time(),
                last_modified=time.time(),
                stage=FeatureStage.CANDIDATE,
            )

            # 6. Register with FeatureRegistry
            if self.feature_registry:
                self.feature_registry.register_feature_set(manifest)

            return {
                "success": True,
                "feature_set_id": feature_set_name,
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "config": config,
                "manifest": manifest
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "feature_set_id": feature_set_name
            }
```

### 2. Technical Indicators Multi-Select

**UI Component**: `<select multiple id="technical-indicators">`

**FeatureConfig Mapping**:
```python
INDICATOR_CONFIG_MAPPING = {
    "rsi": {"rsi_period": 14},
    "macd": {"ema_fast": 12, "ema_slow": 26, "macd_signal": 9},
    "bb": {"bb_period": 20, "bb_std": 2.0},
    "ema": {"ema_fast": 12, "ema_slow": 26},
    "atr": {"atr_period": 20},
    "vwap": {"vwap_enabled": True},
}

def map_ui_indicators_to_config(selected_indicators: list[str]) -> dict:
    """Map UI indicator selections to FeatureConfig parameters."""
    config_params = {}
    for indicator in selected_indicators:
        if indicator in INDICATOR_CONFIG_MAPPING:
            config_params.update(INDICATOR_CONFIG_MAPPING[indicator])
    return config_params
```

### 3. Lookback Periods Input

**UI Component**: `<input type="text" id="lookback-periods" placeholder="10,20,50,100,200">`

**Processing Logic**:
```python
def parse_lookback_periods(periods_str: str) -> list[int]:
    """Parse comma-separated lookback periods from UI."""
    try:
        periods = [int(p.strip()) for p in periods_str.split(",") if p.strip()]
        # Validation: periods must be between 1 and 500
        periods = [p for p in periods if 1 <= p <= 500]
        return sorted(periods) if periods else [10, 20, 50]
    except (ValueError, AttributeError):
        return [10, 20, 50]  # Fallback defaults
```

### 4. Custom Feature Code Editor (Monaco)

**UI Component**: Monaco Editor (`#feature-editor`)

**Implementation Strategy**:
```python
# ml/dashboard/services/code_execution_service.py
class FeatureCodeExecutionService:
    """Secure execution environment for custom feature code."""

    def __init__(self):
        self.sandbox = self._create_sandbox()
        self.allowed_imports = {
            'pandas', 'numpy', 'polars',
            'nautilus_trader.model.data',
            'ml.features.engineering',
            'ml.features.microstructure',
        }

    def validate_custom_code(self, code: str) -> dict[str, Any]:
        """Validate custom feature code before execution."""
        try:
            # 1. AST parsing and security validation
            tree = ast.parse(code)
            validator = CodeValidator(allowed_imports=self.allowed_imports)
            security_issues = validator.validate(tree)

            if security_issues:
                return {
                    "valid": False,
                    "errors": security_issues,
                    "security_risk": True
                }

            # 2. Function signature validation
            required_signature = "def compute_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:"
            if not self._check_function_signature(tree, required_signature):
                return {
                    "valid": False,
                    "errors": ["Function must match required signature"],
                    "signature_error": True
                }

            # 3. Syntax compilation check
            compile(code, '<feature_code>', 'exec')

            return {"valid": True, "errors": []}

        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"Syntax error: {e.msg}"],
                "syntax_error": True
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }

    def execute_custom_code(self, code: str, test_data: pd.DataFrame) -> dict[str, Any]:
        """Execute custom feature code in sandboxed environment."""
        # Implementation with restricted globals, timeout protection
        # and comprehensive error handling
        pass

class CodeValidator(ast.NodeVisitor):
    """AST visitor for security validation of custom feature code."""

    def __init__(self, allowed_imports: set[str]):
        self.allowed_imports = allowed_imports
        self.issues = []
        self.dangerous_calls = {
            'eval', 'exec', 'open', '__import__',
            'getattr', 'setattr', 'delattr',
            'globals', 'locals', 'vars'
        }

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in self.allowed_imports:
                self.issues.append(f"Unauthorized import: {alias.name}")

    def visit_Call(self, node):
        if hasattr(node.func, 'id') and node.func.id in self.dangerous_calls:
            self.issues.append(f"Dangerous function call: {node.func.id}")
        self.generic_visit(node)

    def validate(self, tree):
        self.visit(tree)
        return self.issues
```

### 5. Feature Analysis Section

**UI Components**:
- Total Features count display
- Feature Importance method selector (SHAP)
- Correlation metric display

**Backend Service**:
```python
# ml/dashboard/services/feature_analysis_service.py
class FeatureAnalysisService:
    """Provides feature analysis metrics for dashboard."""

    def __init__(self, feature_store):
        self.feature_store = feature_store

    def get_feature_statistics(self, feature_set_id: str) -> dict[str, Any]:
        """Compute feature analysis statistics."""
        try:
            # Get feature manifest
            registry = self._get_feature_registry()
            manifest = registry.get_manifest(feature_set_id) if registry else None

            if not manifest:
                return {"error": "Feature set not found"}

            # Get recent feature data
            recent_features = self.feature_store.get_recent_features(
                feature_set_id,
                limit=1000
            )

            if recent_features is None or len(recent_features) == 0:
                return {
                    "total_features": len(manifest.feature_names),
                    "data_available": False
                }

            # Compute statistics
            df = recent_features.to_pandas() if hasattr(recent_features, 'to_pandas') else recent_features

            # Feature correlation matrix
            correlation_matrix = df[manifest.feature_names].corr()
            avg_correlation = correlation_matrix.abs().mean().mean()

            # Feature importance (if target available)
            feature_importance = self._compute_feature_importance(df, manifest.feature_names)

            return {
                "total_features": len(manifest.feature_names),
                "data_available": True,
                "avg_correlation": float(avg_correlation),
                "max_correlation": float(correlation_matrix.abs().max().max()),
                "feature_importance_method": "SHAP" if feature_importance else "N/A",
                "top_features": feature_importance[:5] if feature_importance else [],
                "data_quality": {
                    "completeness": float((~df[manifest.feature_names].isnull()).mean().mean()),
                    "recent_observations": len(df),
                }
            }

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
```

## Feature Computation Pipeline Architecture

### 1. Pipeline Specification

```python
# ml/features/pipeline.py (extensions)
@dataclass
class DashboardPipelineSpec(PipelineSpec):
    """Extended pipeline spec for dashboard-generated features."""

    ui_config: dict[str, Any] = field(default_factory=dict)
    custom_code: str | None = None
    validation_results: dict[str, Any] = field(default_factory=dict)

    def create_feature_engineer(self) -> FeatureEngineer:
        """Create FeatureEngineer from pipeline specification."""
        config = self._build_feature_config()
        engineer = FeatureEngineer(config)

        # Inject custom code if provided
        if self.custom_code and self.validation_results.get("valid"):
            engineer.add_custom_transform(self.custom_code)

        return engineer

    def _build_feature_config(self) -> FeatureConfig:
        """Build FeatureConfig from UI parameters."""
        ui = self.ui_config

        return FeatureConfig(
            # Map UI selections to config parameters
            return_periods=self._parse_periods(ui.get("lookback_periods", "10,20,50")),
            momentum_periods=self._parse_periods(ui.get("lookback_periods", "10,20")),
            rsi_period=14 if "rsi" in ui.get("technical_indicators", []) else 0,
            bb_period=20 if "bb" in ui.get("technical_indicators", []) else 0,
            # ... (complete mapping)
            include_microstructure=ui.get("microstructure", False),
            include_order_flow=ui.get("order_flow", False),
        )
```

### 2. Hot/Cold Path Implementation

```python
# ml/dashboard/services/feature_pipeline_service.py
class FeaturePipelineService:
    """Manages feature generation pipeline with hot/cold path separation."""

    def __init__(self):
        self.cold_path_executor = ThreadPoolExecutor(max_workers=4)
        self.hot_path_cache = {}  # Pre-computed feature configs

    async def generate_features_cold_path(
        self,
        pipeline_spec: DashboardPipelineSpec,
        data_slice: pl.DataFrame | pd.DataFrame
    ) -> dict[str, Any]:
        """Execute feature generation in cold path (batch processing)."""

        def _compute_batch_features():
            # Create FeatureEngineer
            engineer = pipeline_spec.create_feature_engineer()

            # Configure for batch processing
            engineer.configure_for_batch(
                use_polars=True,
                memory_efficient=True,
                parallel_processing=True
            )

            # Compute features
            start_time = time.time()
            features_df = engineer.compute_features_batch(data_slice)
            computation_time = time.time() - start_time

            # Run parity validation
            parity_results = self._validate_feature_parity(engineer, data_slice.tail(100))

            return {
                "features": features_df,
                "computation_time": computation_time,
                "feature_count": len(features_df.columns) if hasattr(features_df, 'columns') else 0,
                "parity_validation": parity_results,
                "pipeline_signature": pipeline_spec.compute_signature()
            }

        # Execute in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.cold_path_executor, _compute_batch_features)

    def prepare_hot_path_config(self, pipeline_spec: DashboardPipelineSpec) -> str:
        """Prepare feature configuration for hot path deployment."""
        # Create optimized config for real-time inference
        engineer = pipeline_spec.create_feature_engineer()

        # Pre-allocate arrays and optimize for speed
        engineer.configure_for_realtime(
            preallocate_arrays=True,
            max_latency_ms=5.0,
            memory_limit_mb=64
        )

        # Cache config for hot path actors
        config_id = f"dashboard_{pipeline_spec.compute_signature()[:8]}"
        self.hot_path_cache[config_id] = engineer.config

        return config_id
```

## Code Validation and Execution Strategy

### 1. Security Sandbox

```python
# ml/dashboard/security/code_sandbox.py
class FeatureCodeSandbox:
    """Secure execution environment for user-provided feature code."""

    ALLOWED_MODULES = {
        'pandas', 'numpy', 'polars', 'datetime', 'math', 're',
        'nautilus_trader.model.data', 'ml.features.microstructure'
    }

    FORBIDDEN_PATTERNS = [
        r'__.*__',  # Dunder methods
        r'eval\s*\(',  # eval calls
        r'exec\s*\(',  # exec calls
        r'open\s*\(',  # file operations
        r'import\s+os',  # os module
        r'subprocess',  # subprocess module
    ]

    def __init__(self):
        self.timeout_seconds = 30
        self.memory_limit_mb = 256

    def validate_and_execute(self, code: str, test_data: pd.DataFrame) -> dict[str, Any]:
        """Validate and execute feature code with comprehensive security."""

        # 1. Security validation
        security_check = self._check_security(code)
        if not security_check["safe"]:
            return {
                "success": False,
                "error": "Security violation",
                "details": security_check["violations"]
            }

        # 2. Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "success": False,
                "error": "Syntax error",
                "details": str(e)
            }

        # 3. Execute in restricted environment
        return self._execute_restricted(code, test_data)

    def _execute_restricted(self, code: str, test_data: pd.DataFrame) -> dict[str, Any]:
        """Execute code with resource limits and restricted globals."""

        # Create restricted globals
        restricted_globals = {
            '__builtins__': {
                'len', 'range', 'enumerate', 'zip', 'abs', 'max', 'min',
                'sum', 'round', 'float', 'int', 'str', 'bool', 'list', 'dict'
            },
            'pd': pd,
            'np': np,
            'datetime': __import__('datetime'),
            'math': __import__('math'),
        }

        # Add allowed nautilus imports
        try:
            from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
            restricted_globals.update({
                'Bar': Bar,
                'QuoteTick': QuoteTick,
                'TradeTick': TradeTick
            })
        except ImportError:
            pass

        # Execute with timeout and memory monitoring
        try:
            with resource_limits(
                timeout_seconds=self.timeout_seconds,
                memory_mb=self.memory_limit_mb
            ):
                # Compile and execute
                compiled_code = compile(code, '<feature_code>', 'exec')
                exec_locals = {'data': test_data}
                exec(compiled_code, restricted_globals, exec_locals)

                # Extract the custom feature function
                if 'compute_custom_features' in exec_locals:
                    feature_func = exec_locals['compute_custom_features']

                    # Test execution
                    result = feature_func(None, test_data.head(50))  # Test with small dataset

                    return {
                        "success": True,
                        "result_type": type(result).__name__,
                        "result_shape": result.shape if hasattr(result, 'shape') else None,
                        "columns": list(result.columns) if hasattr(result, 'columns') else None
                    }
                else:
                    return {
                        "success": False,
                        "error": "Function 'compute_custom_features' not found"
                    }

        except TimeoutError:
            return {"success": False, "error": "Execution timeout"}
        except MemoryError:
            return {"success": False, "error": "Memory limit exceeded"}
        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}
```

### 2. Feature Registry Integration

```python
# ml/dashboard/services/registry_integration_service.py
class FeatureRegistryIntegrationService:
    """Integrates dashboard feature generation with FeatureRegistry."""

    def __init__(self):
        self.registry = self._get_feature_registry()
        self.store = self._get_feature_store()

    async def register_dashboard_features(
        self,
        feature_set_id: str,
        pipeline_spec: DashboardPipelineSpec,
        computed_features: pl.DataFrame
    ) -> dict[str, Any]:
        """Register dashboard-generated features with registry."""

        try:
            # 1. Create feature manifest
            manifest = self._create_feature_manifest(
                feature_set_id, pipeline_spec, computed_features
            )

            # 2. Run quality gates
            quality_results = await self._run_quality_gates(manifest, computed_features)

            # 3. Register with appropriate stage
            initial_stage = FeatureStage.CANDIDATE
            if quality_results["passed_all"]:
                initial_stage = FeatureStage.STAGING

            manifest.stage = initial_stage

            # 4. Store in registry
            if self.registry:
                registration_result = self.registry.register_feature_set(manifest)

                # 5. Store feature data
                if self.store and registration_result:
                    await self.store.store_features(
                        feature_set_id,
                        computed_features,
                        metadata={"source": "dashboard_ui"}
                    )

                return {
                    "success": True,
                    "feature_set_id": feature_set_id,
                    "stage": initial_stage.value,
                    "manifest_id": manifest.feature_set_id,
                    "quality_gates": quality_results
                }
            else:
                return {
                    "success": False,
                    "error": "Feature registry not available",
                    "fallback_used": True
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Registration failed: {str(e)}"
            }

    async def promote_dashboard_features(
        self,
        feature_set_id: str,
        target_stage: str
    ) -> dict[str, Any]:
        """Promote dashboard features through registry stages."""

        if not self.registry:
            return {"success": False, "error": "Registry unavailable"}

        try:
            # Run promotion quality gates based on target stage
            quality_gates = self._get_promotion_gates(target_stage)

            # Execute promotion
            result = self.registry.promote_feature_set(
                feature_set_id,
                target_stage,
                quality_gates
            )

            return {
                "success": result,
                "feature_set_id": feature_set_id,
                "new_stage": target_stage,
                "promotion_time": time.time()
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Promotion failed: {str(e)}"
            }
```

## Performance Considerations

### 1. Hot Path Optimization

**Requirements**:
- Feature computation: <5ms P99 latency
- Memory usage: <64MB per feature set
- Zero allocations in inference path

**Implementation Strategy**:
```python
# ml/features/hot_path_optimizer.py
class HotPathFeatureOptimizer:
    """Optimizes feature computation for real-time inference."""

    def __init__(self):
        self.preallocated_arrays = {}
        self.cached_indicators = {}

    def optimize_for_production(self, config: FeatureConfig) -> dict[str, Any]:
        """Optimize feature config for production deployment."""

        # 1. Pre-allocate numpy arrays
        max_lookback = max(config.return_periods + config.momentum_periods)
        buffer_size = max_lookback + 100  # Safety buffer

        self.preallocated_arrays[config.id] = {
            'price_buffer': np.zeros(buffer_size, dtype=np.float32),
            'volume_buffer': np.zeros(buffer_size, dtype=np.float32),
            'feature_output': np.zeros(len(config.feature_names), dtype=np.float32),
        }

        # 2. Pre-initialize indicators
        self._preinitialize_indicators(config)

        # 3. Generate optimized computation graph
        computation_graph = self._build_computation_graph(config)

        return {
            "optimized": True,
            "memory_allocated_mb": self._calculate_memory_usage(config),
            "expected_latency_ms": self._estimate_latency(config),
            "computation_graph": computation_graph
        }

    def _estimate_latency(self, config: FeatureConfig) -> float:
        """Estimate P99 latency based on feature complexity."""
        base_latency = 0.5  # Base processing time

        # Add latency for each feature type
        latency_factors = {
            'returns': 0.1 * len(config.return_periods),
            'momentum': 0.15 * len(config.momentum_periods),
            'rsi': 0.2 if config.rsi_period > 0 else 0,
            'macd': 0.3 if config.ema_fast > 0 else 0,
            'bollinger': 0.25 if config.bb_period > 0 else 0,
            'atr': 0.2 if config.atr_period > 0 else 0,
            'microstructure': 1.0 if config.include_microstructure else 0,
            'order_flow': 1.5 if config.include_order_flow else 0,
        }

        total_latency = base_latency + sum(latency_factors.values())
        return min(total_latency, 4.5)  # Cap at 4.5ms for safety
```

### 2. Cold Path Scalability

**Batch Processing Strategy**:
```python
# ml/features/batch_processor.py
class BatchFeatureProcessor:
    """Optimized batch processing for large-scale feature generation."""

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.polars_available = HAS_POLARS

    async def process_large_dataset(
        self,
        data: pl.DataFrame | pd.DataFrame,
        pipeline_spec: DashboardPipelineSpec,
        chunk_size: int = 10000
    ) -> pl.DataFrame:
        """Process large dataset in chunks with parallel processing."""

        if len(data) <= chunk_size:
            # Small dataset - process directly
            return await self._process_single_chunk(data, pipeline_spec)

        # Large dataset - chunk processing
        chunks = self._create_chunks(data, chunk_size)

        # Process chunks in parallel
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                self.executor,
                self._process_chunk_sync,
                chunk,
                pipeline_spec
            )
            for chunk in chunks
        ]

        processed_chunks = await asyncio.gather(*tasks)

        # Combine results
        if self.polars_available and isinstance(processed_chunks[0], pl.DataFrame):
            return pl.concat(processed_chunks)
        else:
            return pd.concat(processed_chunks, ignore_index=True)
```

## API Endpoints Implementation

```python
# ml/dashboard/app.py (additions)

@app.post("/api/features/generate")
def features_generate() -> tuple[Any, int]:
    """Generate features based on UI configuration."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    payload = cast(dict[str, Any], request.get_json(silent=True) or {})

    feature_set_name = payload.get("feature_set_name", "").strip()
    if not feature_set_name:
        return jsonify({"error": "feature_set_name required"}), 400

    try:
        from ml.dashboard.services.feature_designer_service import FeatureDesignerService

        service = FeatureDesignerService()
        result = asyncio.run(service.generate_features(feature_set_name, payload))

        return jsonify(result), 200 if result.get("success") else 400

    except Exception as e:
        return jsonify({"error": f"Feature generation failed: {str(e)}"}), 500

@app.post("/api/features/validate_code")
def features_validate_code() -> tuple[Any, int]:
    """Validate custom feature code."""
    payload = cast(dict[str, Any], request.get_json(silent=True) or {})
    code = payload.get("code", "").strip()

    if not code:
        return jsonify({"valid": False, "error": "No code provided"}), 400

    try:
        from ml.dashboard.services.code_execution_service import FeatureCodeExecutionService

        service = FeatureCodeExecutionService()
        result = service.validate_custom_code(code)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "valid": False,
            "error": f"Validation failed: {str(e)}"
        }), 500

@app.get("/api/features/analysis/<feature_set_id>")
def features_analysis(feature_set_id: str) -> tuple[Any, int]:
    """Get feature analysis statistics."""
    try:
        from ml.dashboard.services.feature_analysis_service import FeatureAnalysisService

        # Get feature store (with fallback)
        feature_store = svc._get_feature_store()  # From dashboard service
        service = FeatureAnalysisService(feature_store)

        result = service.get_feature_statistics(feature_set_id)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
```

## Frontend JavaScript Implementation

```javascript
// Enhanced feature generation workflow
async function generateFeatures() {
    try {
        // Collect UI parameters
        const uiParams = {
            feature_set_name: document.getElementById('feature-set-name').value.trim(),
            price_features: document.querySelector('input[type="checkbox"]:nth-of-type(1)').checked,
            volume_features: document.querySelector('input[type="checkbox"]:nth-of-type(2)').checked,
            microstructure: document.querySelector('input[type="checkbox"]:nth-of-type(3)').checked,
            order_flow: document.querySelector('input[type="checkbox"]:nth-of-type(4)').checked,
            technical_indicators: Array.from(document.getElementById('technical-indicators').selectedOptions)
                .map(option => option.value),
            lookback_periods: document.getElementById('lookback-periods').value,
            custom_code: featureEditor ? featureEditor.getValue() : null
        };

        // Validation
        if (!uiParams.feature_set_name) {
            showErrorMessage('Please provide a feature set name');
            return;
        }

        // Validate custom code if provided
        if (uiParams.custom_code && uiParams.custom_code.trim()) {
            const codeValidation = await validateCustomCode(uiParams.custom_code);
            if (!codeValidation.valid) {
                showErrorMessage(`Code validation failed: ${codeValidation.errors.join(', ')}`);
                return;
            }
        }

        // Show loading state
        showLoadingMessage('🧮 Generating features...');

        // API call
        const response = await fetch('/api/features/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-ML-DASHBOARD-TOKEN': getAuthToken()
            },
            body: JSON.stringify(uiParams)
        });

        const result = await response.json();

        if (result.success) {
            // Update UI with results
            updateFeatureAnalysisSection({
                total_features: result.feature_count,
                feature_names: result.feature_names,
                feature_set_id: result.feature_set_id
            });

            showSuccessMessage(`✅ Generated ${result.feature_count} features successfully`);

            // Automatically load analysis
            setTimeout(() => loadFeatureAnalysis(result.feature_set_id), 1000);

        } else {
            showErrorMessage(`❌ Feature generation failed: ${result.error}`);
        }

    } catch (error) {
        showErrorMessage(`❌ Error: ${error.message}`);
    } finally {
        hideLoadingMessage();
    }
}

async function validateCustomCode(code) {
    try {
        const response = await fetch('/api/features/validate_code', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ code: code })
        });

        return await response.json();
    } catch (error) {
        return {
            valid: false,
            errors: [`Validation request failed: ${error.message}`]
        };
    }
}

async function loadFeatureAnalysis(featureSetId) {
    try {
        const response = await fetch(`/api/features/analysis/${featureSetId}`);
        const analysis = await response.json();

        if (analysis.error) {
            console.warn('Feature analysis unavailable:', analysis.error);
            return;
        }

        // Update analysis section
        updateFeatureAnalysisSection(analysis);

    } catch (error) {
        console.error('Failed to load feature analysis:', error);
    }
}

function updateFeatureAnalysisSection(analysis) {
    // Update total features
    const totalFeaturesElement = document.querySelector('[class*="metric-value"]:first-of-type');
    if (totalFeaturesElement && analysis.total_features) {
        totalFeaturesElement.textContent = analysis.total_features;
    }

    // Update correlation
    const correlationElement = document.querySelector('[class*="metric-value"]:last-of-type');
    if (correlationElement && analysis.avg_correlation !== undefined) {
        correlationElement.textContent = analysis.avg_correlation.toFixed(2);
    }

    // Update feature importance method
    const importanceElement = document.querySelector('[class*="metric-value"]:nth-of-type(2)');
    if (importanceElement && analysis.feature_importance_method) {
        importanceElement.textContent = analysis.feature_importance_method;
    }
}

// Initialize feature editor with proper configuration
function initializeFeatureEditor() {
    if (typeof monaco !== 'undefined') {
        featureEditor = monaco.editor.create(document.getElementById('feature-editor'), {
            value: `def compute_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute custom features from market data.

    Parameters:
    -----------
    data : pd.DataFrame
        Market data with OHLCV columns and timestamps

    Returns:
    --------
    pd.DataFrame
        DataFrame with custom feature columns
    """
    features = pd.DataFrame(index=data.index)

    # Example: Price velocity (rate of change)
    features['price_velocity'] = data['close'].pct_change().rolling(5).mean()

    # Example: Volume-weighted returns
    features['vw_returns'] = (
        data['close'].pct_change() * data['volume'] / data['volume'].rolling(20).mean()
    )

    # Example: High-low ratio
    features['hl_ratio'] = data['high'] / data['low']

    return features`,
            language: 'python',
            theme: 'vs-dark',
            automaticLayout: true,
            fontSize: 12,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            wordWrap: 'on'
        });

        // Add code validation on change
        featureEditor.getModel().onDidChangeContent(() => {
            clearTimeout(featureEditor._validationTimeout);
            featureEditor._validationTimeout = setTimeout(async () => {
                const code = featureEditor.getValue();
                if (code.trim()) {
                    const validation = await validateCustomCode(code);
                    // Update editor decorations based on validation
                    updateEditorValidation(validation);
                }
            }, 2000);
        });
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('feature-editor')) {
        initializeFeatureEditor();
    }
});
```

## Implementation Priority & Timeline

### Phase 1 (Week 1): Core Infrastructure
1. FeatureDesignerService with basic UI mapping
2. Pipeline specification extensions
3. Security validation framework
4. Basic API endpoints

### Phase 2 (Week 2): Feature Generation Pipeline
1. Hot/cold path feature computation
2. FeatureRegistry integration
3. Quality gates and validation
4. Batch processing optimization

### Phase 3 (Week 3): Code Execution & Analysis
1. Secure code sandbox implementation
2. Feature analysis service
3. Performance optimization
4. Enhanced error handling

### Phase 4 (Week 4): UI Integration & Testing
1. Complete JavaScript client implementation
2. Real-time validation feedback
3. Comprehensive testing
4. Performance benchmarking

## Risk Mitigation & Safety

1. **Code Security**: Multi-layer validation with AST parsing and restricted execution
2. **Performance**: Latency budgets enforced with circuit breakers
3. **Data Quality**: Comprehensive validation gates before registry promotion
4. **Error Handling**: Graceful degradation with meaningful error messages
5. **Resource Management**: Memory and CPU limits with proper cleanup

This implementation plan provides a robust, secure, and performant foundation for the dashboard's feature engineering capabilities while maintaining full compatibility with Nautilus Trader's architecture patterns.
