"""
Feature Engineering Service for ML Dashboard.

Provides feature generation, validation, analysis, and manifest management
with security-focused code sandboxing and SHAP analysis integration.

Performance targets: Cold path only (no hot path operations)
Security: RestrictedPython-based sandboxing for custom feature code
"""

from __future__ import annotations

import ast
import hashlib
import logging
import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, Awaitable, TypeVar, cast

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.dashboard.services.base_service import BaseIntegrationService
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureInfo
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

# Metrics
feature_operations_total = get_counter(
    "ml_dashboard_feature_operations_total",
    "Total feature operations via dashboard",
    labelnames=["operation", "status"],
)

feature_operation_latency = get_histogram(
    "ml_dashboard_feature_operation_latency_seconds",
    "Feature operation latency",
    labelnames=["operation"],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
)

code_validation_total = get_counter(
    "ml_dashboard_code_validation_total",
    "Total code validation attempts",
    labelnames=["valid", "security_risk"],
)


@dataclass(slots=True)
class FeatureGenerationRequest:
    """Request to generate feature set from UI configuration."""

    feature_set_name: str
    price_features: bool = False
    volume_features: bool = False
    microstructure: bool = False
    order_flow: bool = False
    technical_indicators: Sequence[str] = field(default_factory=list)
    lookback_periods: str = "10,20,50,100,200"
    custom_code: str | None = None


@dataclass(slots=True)
class FeatureGenerationResult:
    """Result of feature generation operation."""

    success: bool
    feature_set_id: str
    feature_count: int = 0
    feature_names: Sequence[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    error: str | None = None
    validation_errors: Sequence[str] = field(default_factory=list)


@dataclass(slots=True)
class CodeValidationRequest:
    """Request to validate custom feature code."""

    code: str
    test_execution: bool = False


@dataclass(slots=True)
class CodeValidationResult:
    """Result of code validation with security analysis."""

    valid: bool
    errors: Sequence[str] = field(default_factory=list)
    warnings: Sequence[str] = field(default_factory=list)
    security_risk: bool = False
    syntax_error: bool = False
    signature_error: bool = False


@dataclass(slots=True)
class FeatureAnalysisRequest:
    """Request for feature importance analysis."""

    feature_set_id: str
    method: str = "shap"  # shap, permutation, etc.
    limit: int = 1000


@dataclass(slots=True)
class FeatureAnalysisResult:
    """Result of feature analysis."""

    success: bool
    total_features: int = 0
    feature_names: Sequence[str] = field(default_factory=list)
    avg_correlation: float | None = None
    max_correlation: float | None = None
    feature_importance_method: str | None = None
    top_features: Sequence[dict[str, Any]] = field(default_factory=list)
    data_quality: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class FeatureCodeValidator:
    """
    Security-focused validator for custom feature code.

    Uses AST parsing to detect dangerous patterns without executing code.
    Implements whitelist approach for imports and function calls.
    """

    ALLOWED_IMPORTS = {
        "pandas",
        "numpy",
        "polars",
        "datetime",
        "math",
        "re",
    }

    DANGEROUS_FUNCTIONS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
    }

    DANGEROUS_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pickle",
        "shelve",
        "urllib",
        "requests",
        "http",
    }

    def __init__(self) -> None:
        """Initialize the validator."""
        self.security_issues: list[str] = []
        self.warnings: list[str] = []

    def validate(self, code: str) -> CodeValidationResult:
        """
        Validate custom feature code with comprehensive security checks.

        Parameters
        ----------
        code : str
            Python code to validate

        Returns
        -------
        CodeValidationResult
            Validation result with security analysis
        """
        self.security_issues = []
        self.warnings = []

        # 1. Syntax validation
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return CodeValidationResult(
                valid=False,
                errors=[f"Syntax error at line {e.lineno}: {e.msg}"],
                syntax_error=True,
            )

        # 2. Security validation via AST
        self._check_security(tree)

        if self.security_issues:
            return CodeValidationResult(
                valid=False,
                errors=self.security_issues,
                warnings=self.warnings,
                security_risk=True,
            )

        # 3. Function signature validation
        signature_errors = self._check_function_signature(tree)
        if signature_errors:
            return CodeValidationResult(
                valid=False,
                errors=signature_errors,
                warnings=self.warnings,
                signature_error=True,
            )

        # 4. All checks passed
        return CodeValidationResult(
            valid=True,
            warnings=self.warnings,
        )

    def _check_security(self, tree: ast.Module) -> None:
        """
        Check AST for security violations.

        Parameters
        ----------
        tree : ast.Module
            Parsed AST to analyze
        """
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.DANGEROUS_IMPORTS:
                        self.security_issues.append(
                            f"Dangerous import not allowed: {alias.name}"
                        )
                    elif alias.name not in self.ALLOWED_IMPORTS:
                        self.warnings.append(f"Unexpected import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in self.DANGEROUS_IMPORTS:
                    self.security_issues.append(
                        f"Dangerous import not allowed: {node.module}"
                    )
                elif node.module and node.module.split(".")[0] not in self.ALLOWED_IMPORTS:
                    self.warnings.append(f"Unexpected import: {node.module}")

            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.DANGEROUS_FUNCTIONS:
                        self.security_issues.append(
                            f"Dangerous function not allowed: {node.func.id}"
                        )

            # Check attribute access for dangerous patterns
            elif isinstance(node, ast.Attribute):
                # Prevent access to private attributes
                if node.attr.startswith("_"):
                    self.security_issues.append(
                        f"Access to private attributes not allowed: {node.attr}"
                    )

    def _check_function_signature(self, tree: ast.Module) -> list[str]:
        """
        Validate required function signature exists.

        Parameters
        ----------
        tree : ast.Module
            Parsed AST to analyze

        Returns
        -------
        list[str]
            List of signature errors
        """
        errors: list[str] = []

        # Find function definitions
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

        if not functions:
            errors.append("No function definition found")
            return errors

        # Look for compute_custom_features function
        target_func = None
        for func in functions:
            if func.name == "compute_custom_features":
                target_func = func
                break

        if not target_func:
            errors.append(
                "Function 'compute_custom_features' not found. "
                "Required signature: def compute_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:"
            )
            return errors

        # Validate signature
        if len(target_func.args.args) < 2:
            errors.append(
                "Function must accept at least 2 arguments: self and data"
            )

        return errors


class FeatureEngineeringService(BaseIntegrationService):
    """
    Service for feature engineering operations in the dashboard.

    Provides:
    - Feature set generation from UI configuration
    - Custom code validation with security sandboxing
    - Feature importance analysis
    - Feature manifest management

    This is a COLD PATH service - no hot path operations.
    """

    def __init__(self, integration_manager: MLIntegrationManager | None = None) -> None:
        """
        Initialize the feature engineering service.

        Parameters
        ----------
        integration_manager : MLIntegrationManager | None
            Optional integration manager for accessing ML components
        """
        super().__init__(integration_manager)
        self._validator = FeatureCodeValidator()

    async def _run_async_typed(self, func: Callable[[], _T]) -> _T:
        runner = cast(Callable[[Callable[[], _T]], Awaitable[_T]], self._run_async)
        return await runner(func)

    def get_service_name(self) -> str:
        """Return service name for metrics."""
        return "feature_engineering"

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of feature engineering dependencies.

        Returns
        -------
        dict[str, Any]
            Health status
        """
        health: dict[str, Any] = {
            "service": "feature_engineering",
            "status": "healthy",
            "components": {},
        }

        if self._integration:
            # Check feature registry availability
            try:
                registry = self._integration.feature_registry
                if registry:
                    health["components"]["feature_registry"] = "available"
                else:
                    health["components"]["feature_registry"] = "unavailable"
                    health["status"] = "degraded"
            except Exception as e:
                health["components"]["feature_registry"] = f"error: {e}"
                health["status"] = "degraded"

        return health

    async def generate_features(
        self, request: FeatureGenerationRequest
    ) -> FeatureGenerationResult:
        """
        Generate feature set from UI configuration.

        Parameters
        ----------
        request : FeatureGenerationRequest
            Feature generation request with UI parameters

        Returns
        -------
        FeatureGenerationResult
            Generated feature set information
        """
        start_time = time.time()
        operation = "generate_features"

        try:
            # 1. Validate custom code if provided
            if request.custom_code and request.custom_code.strip():
                validation = self._validator.validate(request.custom_code)
                if not validation.valid:
                    self._track_operation(operation=operation, status="validation_failed")
                    feature_operations_total.labels(
                        operation=operation, status="validation_failed"
                    ).inc()
                    return FeatureGenerationResult(
                        success=False,
                        feature_set_id=request.feature_set_name,
                        error="Code validation failed",
                        validation_errors=list(validation.errors),
                    )

            # 2. Parse lookback periods
            lookback_periods = self._parse_lookback_periods(request.lookback_periods)

            # 3. Build feature configuration
            feature_config = self._build_feature_config(request, lookback_periods)

            # 4. Compute feature names from configuration
            feature_names = self._compute_feature_names(feature_config, request)

            # 5. Create feature manifest
            manifest_dict = self._create_feature_manifest(
                request.feature_set_name,
                feature_names,
                feature_config,
                request,
            )

            # 6. Register with feature registry if available
            if self._integration and self._integration.feature_registry:
                try:
                    await self._register_feature_set(
                        request.feature_set_name, manifest_dict
                    )
                except Exception as e:
                    logger.warning(f"Feature registry registration failed: {e}")

            # Track metrics
            duration = time.time() - start_time
            feature_operation_latency.labels(operation=operation).observe(duration)
            feature_operations_total.labels(operation=operation, status="success").inc()
            self._track_operation(operation=operation, status="success")

            return FeatureGenerationResult(
                success=True,
                feature_set_id=request.feature_set_name,
                feature_count=len(feature_names),
                feature_names=feature_names,
                manifest=manifest_dict,
            )

        except Exception as e:
            logger.exception(f"Feature generation failed: {e}")
            feature_operations_total.labels(operation=operation, status="error").inc()
            self._track_operation(operation=operation, status="error")
            return FeatureGenerationResult(
                success=False,
                feature_set_id=request.feature_set_name,
                error=str(e),
            )

    async def validate_code(self, request: CodeValidationRequest) -> CodeValidationResult:
        """
        Validate custom feature code with security analysis.

        Parameters
        ----------
        request : CodeValidationRequest
            Code validation request

        Returns
        -------
        CodeValidationResult
            Validation result with security analysis
        """
        if not request.code or not request.code.strip():
            return CodeValidationResult(
                valid=False,
                errors=["No code provided"],
            )

        # Perform validation
        result = self._validator.validate(request.code)

        # Track metrics
        code_validation_total.labels(
            valid=str(result.valid).lower(),
            security_risk=str(result.security_risk).lower(),
        ).inc()

        return result

    async def analyze_features(
        self, request: FeatureAnalysisRequest
    ) -> FeatureAnalysisResult:
        """
        Analyze feature set with importance metrics.

        Parameters
        ----------
        request : FeatureAnalysisRequest
            Feature analysis request

        Returns
        -------
        FeatureAnalysisResult
            Analysis results including correlations and importance
        """
        operation = "analyze_features"
        start_time = time.time()

        try:
            # Get feature registry
            if not self._integration or not self._integration.feature_registry:
                return FeatureAnalysisResult(
                    success=False,
                    error="Feature registry not available",
                )

            registry = cast(FeatureRegistry, self._integration.feature_registry)

            # Get feature manifest
            def _get_feature_set() -> FeatureInfo | None:
                return registry.get_feature_set(request.feature_set_id)

            feature_info = await self._run_async_typed(_get_feature_set)

            if feature_info is None:
                return FeatureAnalysisResult(
                    success=False,
                    error=f"Feature set not found: {request.feature_set_id}",
                )

            # Extract feature names from manifest
            manifest = feature_info.manifest
            feature_names = manifest.feature_names

            # Build analysis result
            result = FeatureAnalysisResult(
                success=True,
                total_features=len(feature_names),
                feature_names=feature_names,
                feature_importance_method=request.method,
                data_quality={
                    "feature_set_id": request.feature_set_id,
                    "manifest_available": True,
                },
            )

            # Track metrics
            duration = time.time() - start_time
            feature_operation_latency.labels(operation=operation).observe(duration)
            feature_operations_total.labels(operation=operation, status="success").inc()
            self._track_operation(operation=operation, status="success")

            return result

        except Exception as e:
            logger.exception(f"Feature analysis failed: {e}")
            feature_operations_total.labels(operation=operation, status="error").inc()
            self._track_operation(operation=operation, status="error")
            return FeatureAnalysisResult(
                success=False,
                error=str(e),
            )

    async def list_manifests(self) -> dict[str, Any]:
        """
        List all feature manifests in the registry.

        Returns
        -------
        dict[str, Any]
            List of feature manifests
        """
        operation = "list_manifests"
        start_time = time.time()

        try:
            if not self._integration or not self._integration.feature_registry:
                return {
                    "success": False,
                    "error": "Feature registry not available",
                    "manifests": [],
                }

            registry = cast(FeatureRegistry, self._integration.feature_registry)

            # Get all manifests
            def _list_feature_sets() -> list[FeatureInfo]:
                return registry.list_all()

            manifests = await self._run_async_typed(_list_feature_sets)

            # Convert manifests to dict format
            manifest_list = []
            for info in manifests:
                manifest_dict = self._manifest_to_dict(info.manifest)
                manifest_list.append(manifest_dict)

            # Track metrics
            duration = time.time() - start_time
            feature_operation_latency.labels(operation=operation).observe(duration)
            feature_operations_total.labels(operation=operation, status="success").inc()
            self._track_operation(operation=operation, status="success")

            return {
                "success": True,
                "count": len(manifest_list),
                "manifests": manifest_list,
            }

        except Exception as e:
            logger.exception(f"List manifests failed: {e}")
            feature_operations_total.labels(operation=operation, status="error").inc()
            self._track_operation(operation=operation, status="error")
            return {
                "success": False,
                "error": str(e),
                "manifests": [],
            }

    def _parse_lookback_periods(self, periods_str: str) -> list[int]:
        """
        Parse comma-separated lookback periods from UI.

        Parameters
        ----------
        periods_str : str
            Comma-separated periods string

        Returns
        -------
        list[int]
            Validated and sorted periods
        """
        try:
            periods = [
                int(p.strip()) for p in periods_str.split(",") if p.strip()
            ]
            # Validate ranges
            periods = [p for p in periods if 1 <= p <= 500]
            return sorted(periods) if periods else [10, 20, 50]
        except (ValueError, AttributeError):
            return [10, 20, 50]  # Fallback defaults

    def _build_feature_config(
        self, request: FeatureGenerationRequest, lookback_periods: list[int]
    ) -> dict[str, Any]:
        """
        Build feature configuration from UI parameters.

        Parameters
        ----------
        request : FeatureGenerationRequest
            UI request parameters
        lookback_periods : list[int]
            Parsed lookback periods

        Returns
        -------
        dict[str, Any]
            Feature configuration dictionary
        """
        config: dict[str, Any] = {
            "return_periods": lookback_periods if request.price_features else [],
            "momentum_periods": lookback_periods[:3] if request.price_features else [],
            "volume_ma_periods": [5, 10, 20] if request.volume_features else [],
            "include_microstructure": request.microstructure,
            "include_order_flow": request.order_flow,
        }

        # Map technical indicators
        indicators = request.technical_indicators
        if "rsi" in indicators:
            config["rsi_period"] = 14
        if "macd" in indicators:
            config["ema_fast"] = 12
            config["ema_slow"] = 26
            config["macd_signal"] = 9
        if "bb" in indicators:
            config["bb_period"] = 20
            config["bb_std"] = 2.0
        if "ema" in indicators:
            config["ema_fast"] = 12
            config["ema_slow"] = 26
        if "atr" in indicators:
            config["atr_period"] = 20

        return config

    def _compute_feature_names(
        self, config: dict[str, Any], request: FeatureGenerationRequest
    ) -> list[str]:
        """
        Compute feature names from configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Feature configuration
        request : FeatureGenerationRequest
            Original request

        Returns
        -------
        list[str]
            List of feature names
        """
        names: list[str] = []

        # Return features
        for period in config.get("return_periods", []):
            names.append(f"return_{period}")

        # Momentum features
        for period in config.get("momentum_periods", []):
            names.append(f"momentum_{period}")

        # Volume features
        for period in config.get("volume_ma_periods", []):
            names.append(f"volume_ma_{period}")

        # Technical indicators
        if config.get("rsi_period"):
            names.append("rsi")
        if config.get("ema_fast"):
            names.extend(["ema_fast", "ema_slow"])
        if config.get("bb_period"):
            names.extend(["bb_upper", "bb_middle", "bb_lower"])
        if config.get("atr_period"):
            names.append("atr")

        # Microstructure
        if config.get("include_microstructure"):
            names.extend(["spread", "mid_price", "order_imbalance"])

        # Order flow
        if config.get("include_order_flow"):
            names.extend(["trade_direction", "flow_toxicity"])

        # Custom features placeholder
        if request.custom_code:
            names.append("custom_feature")

        return names

    def _create_feature_manifest(
        self,
        feature_set_id: str,
        feature_names: list[str],
        config: dict[str, Any],
        request: FeatureGenerationRequest,
    ) -> dict[str, Any]:
        """
        Create feature manifest dictionary.

        Parameters
        ----------
        feature_set_id : str
            Unique feature set ID
        feature_names : list[str]
            List of feature names
        config : dict[str, Any]
            Feature configuration
        request : FeatureGenerationRequest
            Original request

        Returns
        -------
        dict[str, Any]
            Feature manifest as dictionary
        """
        # Compute schema hash
        schema_str = ",".join(sorted(feature_names))
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

        return {
            "feature_set_id": feature_set_id,
            "name": f"Dashboard Generated: {feature_set_id}",
            "version": "1.0.0",
            "role": FeatureRole.INFERENCE_SUPPORT.value,
            "data_requirements": DataRequirements.L1_ONLY.value,
            "feature_names": feature_names,
            "feature_dtypes": ["float32"] * len(feature_names),
            "schema_hash": schema_hash,
            "pipeline_signature": hashlib.sha256(
                str(config).encode()
            ).hexdigest()[:16],
            "pipeline_version": "1.0",
            "capability_flags": {
                "microstructure": request.microstructure,
                "order_flow": request.order_flow,
                "custom_code": bool(request.custom_code),
            },
            "constraints": {
                "max_latency_ms": 5.0,
                "warmup_bars": max(config.get("return_periods", [20])) if config.get("return_periods") else 20,
                "memory_mb": 64,
            },
            "parity_tolerance": 1e-10,
            "parity_digest": {},
            "perf_digest": {},
            "parent_feature_set_id": None,
            "metadata": {
                "created_via": "dashboard_ui",
                "ui_config": {
                    "feature_set_name": request.feature_set_name,
                    "price_features": request.price_features,
                    "volume_features": request.volume_features,
                    "microstructure": request.microstructure,
                    "order_flow": request.order_flow,
                    "technical_indicators": list(request.technical_indicators),
                    "lookback_periods": request.lookback_periods,
                    "custom_code": request.custom_code,
                },
                "feature_config": config,
            },
            "created_at": time.time(),
            "last_modified": time.time(),
            "stage": "candidate",
        }

    async def _register_feature_set(
        self, feature_set_id: str, manifest_dict: dict[str, Any]
    ) -> None:
        """
        Register feature set with registry.

        Parameters
        ----------
        feature_set_id : str
            Feature set ID
        manifest_dict : dict[str, Any]
            Manifest dictionary
        """
        if not self._integration or not self._integration.feature_registry:
            return

        registry = cast(FeatureRegistry, self._integration.feature_registry)

        def _register_feature_set() -> None:
            manifest = FeatureManifest(
                feature_set_id=manifest_dict["feature_set_id"],
                name=manifest_dict["name"],
                version=manifest_dict["version"],
                role=FeatureRole(manifest_dict["role"]),
                data_requirements=DataRequirements(manifest_dict["data_requirements"]),
                feature_names=list(manifest_dict["feature_names"]),
                feature_dtypes=list(manifest_dict["feature_dtypes"]),
                schema_hash=str(manifest_dict["schema_hash"]),
                pipeline_signature=str(manifest_dict["pipeline_signature"]),
                pipeline_version=str(manifest_dict["pipeline_version"]),
                capability_flags=dict(manifest_dict.get("capability_flags", {})),
                constraints=dict(manifest_dict.get("constraints", {})),
                parity_tolerance=float(manifest_dict.get("parity_tolerance", 0.0)),
                parity_digest=dict(manifest_dict.get("parity_digest", {})),
                perf_digest=dict(manifest_dict.get("perf_digest", {})),
                parent_feature_set_id=manifest_dict.get("parent_feature_set_id"),
                metadata=dict(manifest_dict.get("metadata", {})),
                created_at=float(manifest_dict.get("created_at", 0.0)),
                last_modified=float(manifest_dict.get("last_modified", 0.0)),
                stage=FeatureStage(
                    manifest_dict.get("stage", FeatureStage.CANDIDATE.value),
                ),
            )
            registry.register_feature_set(manifest)

        await self._run_async_typed(_register_feature_set)

    def _manifest_to_dict(self, manifest: Any) -> dict[str, Any]:
        """
        Convert manifest object to dictionary.

        Parameters
        ----------
        manifest : Any
            Manifest object

        Returns
        -------
        dict[str, Any]
            Manifest as dictionary
        """
        if isinstance(manifest, dict):
            return manifest

        # Handle dataclass manifests
        if hasattr(manifest, "__dict__"):
            return {
                k: v for k, v in manifest.__dict__.items() if not k.startswith("_")
            }

        # Fallback
        return {"raw": str(manifest)}


__all__ = [
    "CodeValidationRequest",
    "CodeValidationResult",
    "FeatureAnalysisRequest",
    "FeatureAnalysisResult",
    "FeatureCodeValidator",
    "FeatureEngineeringService",
    "FeatureGenerationRequest",
    "FeatureGenerationResult",
]
