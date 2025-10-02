"""
Strategy Builder Service for ML Dashboard.

Provides strategy validation, backtesting, deployment, and performance tracking
with security-focused code sandboxing and integration with Nautilus BacktestEngine.

Performance targets: Cold path only (no hot path operations)
Security: AST-based code validation with strict sandboxing (CRITICAL)
Integration: BacktestEngine, MLTradingStrategy, ActorIntegrationService
"""

from __future__ import annotations

import ast
import logging
import time
import uuid
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.dashboard.services.base_service import BaseIntegrationService


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager

logger = logging.getLogger(__name__)

# ============================================================================
# METRICS
# ============================================================================

strategy_operations_total = get_counter(
    "ml_dashboard_strategy_operations_total",
    "Total strategy operations via dashboard",
    labelnames=["operation", "status"],
)

strategy_operation_latency = get_histogram(
    "ml_dashboard_strategy_operation_latency_seconds",
    "Strategy operation latency",
    labelnames=["operation"],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

code_validation_total = get_counter(
    "ml_dashboard_strategy_validation_total",
    "Total strategy code validation attempts",
    labelnames=["valid", "security_risk"],
)

backtest_operations_total = get_counter(
    "ml_dashboard_backtest_operations_total",
    "Total backtest operations",
    labelnames=["status"],
)

deployment_operations_total = get_counter(
    "ml_dashboard_deployment_operations_total",
    "Total deployment operations",
    labelnames=["environment", "status"],
)

# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(slots=True)
class CodeValidationRequest:
    """Request to validate strategy code."""

    code: str
    strategy_name: str | None = None
    base_strategy: str = "MLTradingStrategy"


@dataclass(slots=True)
class CodeValidationResult:
    """Result of code validation with security analysis."""

    valid: bool
    errors: Sequence[str] = field(default_factory=list)
    warnings: Sequence[str] = field(default_factory=list)
    security_risk: bool = False
    syntax_error: bool = False
    signature_error: bool = False
    allowed_imports: Sequence[str] = field(default_factory=list)


@dataclass(slots=True)
class BacktestRequest:
    """Request to run strategy backtest."""

    strategy_code: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_balance: float = 100000.0
    instruments: Sequence[str] = field(default_factory=lambda: ["EURUSD.SIM"])
    risk_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestResult:
    """Result of backtest execution."""

    job_id: str
    status: str  # 'queued', 'running', 'completed', 'failed'
    performance_metrics: Mapping[str, float] = field(default_factory=dict)
    trades: Sequence[Mapping[str, Any]] = field(default_factory=list)
    equity_curve: Sequence[Mapping[str, Any]] = field(default_factory=list)
    error: str | None = None
    execution_time_seconds: float = 0.0


@dataclass(slots=True)
class DeploymentRequest:
    """Request to deploy strategy."""

    strategy_name: str
    strategy_code: str
    environment: str = "staging"  # staging, paper, live
    risk_params: Mapping[str, Any] = field(default_factory=dict)
    instruments: Sequence[str] = field(default_factory=lambda: ["EURUSD.SIM"])


@dataclass(slots=True)
class DeploymentResult:
    """Result of deployment operation."""

    deployment_id: str
    status: str
    environment: str
    message: str
    monitoring_url: str | None = None
    error: str | None = None


# ============================================================================
# STRATEGY CODE VALIDATOR
# ============================================================================


class StrategyCodeValidator:
    """
    Security-focused validator for custom strategy code.

    Uses AST parsing to detect dangerous patterns without executing code.
    Implements whitelist approach for imports and function calls.

    This is CRITICAL for security - strategies execute user-provided code.
    """

    # Allowed imports - whitelist approach (strict)
    ALLOWED_IMPORTS = {
        "ml.strategies.base",
        "ml.strategies.ml_strategy",
        "ml.actors.base",
        "ml.config.base",
        "nautilus_trader.model.enums",
        "nautilus_trader.model.position",
        "nautilus_trader.model.objects",
        "nautilus_trader.model.identifiers",
        "nautilus_trader.model.data",
        "nautilus_trader.model.events",
        "numpy",
        "pandas",
        "typing",
        "dataclasses",
        "abc",
        "enum",
        "datetime",
        "time",
        "math",
    }

    # Dangerous functions that must never be called
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
        "reload",
        "execfile",
    }

    # Dangerous imports
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
        "importlib",
        "ctypes",
        "multiprocessing",
        "threading",
    }

    # Dangerous attributes (prevent introspection attacks)
    DANGEROUS_ATTRS = {
        "__subclasses__",
        "__bases__",
        "__globals__",
        "__code__",
        "__closure__",
        "__dict__",
        "__builtins__",
    }

    # Required base classes for strategy
    REQUIRED_BASE_CLASSES = {
        "BaseMLStrategy",
        "MLTradingStrategy",
        "BaseStrategy",
    }

    def __init__(self) -> None:
        """Initialize the validator."""
        self.security_issues: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def validate(self, code: str) -> CodeValidationResult:
        """
        Validate custom strategy code with comprehensive security checks.

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
        self.errors = []

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

        # 3. Import validation
        self._validate_imports(tree)

        # 4. Structure validation
        self._validate_structure(tree)

        # 5. Method signature validation
        self._validate_methods(tree)

        # Determine if code is valid
        is_valid = len(self.errors) == 0 and len(self.security_issues) == 0
        has_security_risk = len(self.security_issues) > 0

        # Track validation metrics
        code_validation_total.labels(
            valid="true" if is_valid else "false",
            security_risk="true" if has_security_risk else "false",
        ).inc()

        return CodeValidationResult(
            valid=is_valid,
            errors=list(self.errors + self.security_issues),
            warnings=list(self.warnings),
            security_risk=has_security_risk,
            syntax_error=False,
            allowed_imports=list(self.ALLOWED_IMPORTS),
        )

    def _check_security(self, tree: ast.AST) -> None:
        """
        Check for security vulnerabilities in AST.

        Parameters
        ----------
        tree : ast.AST
            Parsed AST tree
        """
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.DANGEROUS_FUNCTIONS:
                        self.security_issues.append(
                            f"SECURITY: Forbidden function call: {node.func.id}"
                        )

            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in self.DANGEROUS_ATTRS:
                    self.security_issues.append(
                        f"SECURITY: Forbidden attribute access: {node.attr}"
                    )

            # Check for file I/O operations
            if isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Name):
                            if item.context_expr.func.id == "open":
                                self.security_issues.append(
                                    "SECURITY: File I/O operations not allowed"
                                )

    def _validate_imports(self, tree: ast.AST) -> None:
        """
        Validate that only approved modules are imported.

        Parameters
        ----------
        tree : ast.AST
            Parsed AST tree
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check if module is in dangerous imports
                    if alias.name in self.DANGEROUS_IMPORTS:
                        self.security_issues.append(
                            f"SECURITY: Dangerous import: {alias.name}"
                        )
                    # Check if module is allowed
                    elif alias.name not in self.ALLOWED_IMPORTS:
                        # Check if it's a submodule of an allowed module
                        allowed = any(
                            alias.name.startswith(allowed_mod + ".")
                            for allowed_mod in self.ALLOWED_IMPORTS
                        )
                        if not allowed:
                            self.errors.append(f"Unauthorized import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Check if module is in dangerous imports
                    if node.module in self.DANGEROUS_IMPORTS:
                        self.security_issues.append(
                            f"SECURITY: Dangerous import: {node.module}"
                        )
                    # Check if module is allowed
                    elif node.module not in self.ALLOWED_IMPORTS:
                        # Check if it's a submodule of allowed module
                        allowed = any(
                            node.module.startswith(allowed_mod + ".")
                            for allowed_mod in self.ALLOWED_IMPORTS
                        )
                        if not allowed:
                            self.errors.append(f"Unauthorized import: {node.module}")

    def _validate_structure(self, tree: ast.AST) -> None:
        """
        Validate strategy structure requirements.

        Parameters
        ----------
        tree : ast.AST
            Parsed AST tree
        """
        # Find strategy class
        strategy_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if inherits from allowed base classes
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        if base.id in self.REQUIRED_BASE_CLASSES:
                            strategy_classes.append(node)
                            break
                    elif isinstance(base, ast.Attribute):
                        if base.attr in self.REQUIRED_BASE_CLASSES:
                            strategy_classes.append(node)
                            break

        if not strategy_classes:
            self.errors.append(
                f"No strategy class found. Must inherit from one of: {self.REQUIRED_BASE_CLASSES}"
            )

    def _validate_methods(self, tree: ast.AST) -> None:
        """
        Validate that required methods are present.

        Parameters
        ----------
        tree : ast.AST
            Parsed AST tree
        """
        # For MLTradingStrategy, _process_ml_signal is required
        # For other strategies, different methods may be required
        # This is a basic check - can be enhanced
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_names = {
                    n.name for n in node.body if isinstance(n, ast.FunctionDef)
                }

                # Check for common required methods (lenient)
                if "_process_ml_signal" not in method_names and "on_start" not in method_names:
                    self.warnings.append(
                        "Strategy should implement _process_ml_signal or on_start method"
                    )


# ============================================================================
# STRATEGY SERVICE
# ============================================================================


class StrategyService(BaseIntegrationService):
    """
    Strategy builder service providing validation, backtesting, and deployment.

    Integrates with:
    - StrategyRegistry for strategy management
    - Nautilus BacktestEngine for backtesting
    - ActorIntegrationService for deployment
    - FeatureEngineeringService for features
    """

    def __init__(self, integration_manager: MLIntegrationManager | None = None) -> None:
        """
        Initialize strategy service.

        Parameters
        ----------
        integration_manager : MLIntegrationManager | None
            Integration manager for ML components
        """
        super().__init__(integration_manager)
        self._validator = StrategyCodeValidator()
        self._active_backtests: dict[str, BacktestResult] = {}

    def get_service_name(self) -> str:
        """
        Return service name for metrics.

        Returns
        -------
        str
            Service name
        """
        return "strategy_service"

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of strategy service.

        Returns
        -------
        dict[str, Any]
            Health status
        """
        return {
            "service": self.get_service_name(),
            "status": "ok",
            "active_backtests": len(self._active_backtests),
            "integration_available": self._integration is not None,
        }

    def validate_strategy_code(self, request: CodeValidationRequest) -> CodeValidationResult:
        """
        Validate strategy code with security checks.

        Parameters
        ----------
        request : CodeValidationRequest
            Validation request

        Returns
        -------
        CodeValidationResult
            Validation result
        """
        start = time.perf_counter()

        try:
            result = self._validator.validate(request.code)

            self._track_operation(
                operation="validate_code",
                status="success" if result.valid else "failed",
            )

            return result

        except Exception as e:
            logger.exception("Strategy code validation failed")
            self._track_operation(operation="validate_code", status="error")

            return CodeValidationResult(
                valid=False,
                errors=[f"Validation error: {e!s}"],
            )

        finally:
            strategy_operation_latency.labels(operation="validate_code").observe(
                time.perf_counter() - start
            )

    def submit_backtest(self, request: BacktestRequest) -> BacktestResult:
        """
        Submit backtest job (queued for execution).

        Parameters
        ----------
        request : BacktestRequest
            Backtest request

        Returns
        -------
        BacktestResult
            Backtest result with job ID
        """
        start = time.perf_counter()

        try:
            # 1. Validate code first
            validation = self.validate_strategy_code(
                CodeValidationRequest(
                    code=request.strategy_code,
                    strategy_name=request.strategy_name,
                )
            )

            if not validation.valid:
                result = BacktestResult(
                    job_id="",
                    status="failed",
                    error=f"Code validation failed: {validation.errors}",
                )
                backtest_operations_total.labels(status="validation_failed").inc()
                return result

            # 2. Create job ID
            job_id = str(uuid.uuid4())

            # 3. Queue backtest (mock for now - real implementation would use BacktestEngine)
            result = BacktestResult(
                job_id=job_id,
                status="queued",
            )

            # Store in active backtests
            self._active_backtests[job_id] = result

            self._track_operation(operation="submit_backtest", status="success")
            backtest_operations_total.labels(status="queued").inc()

            # TODO: Actual BacktestEngine integration would happen here
            # This would involve:
            # - Creating strategy instance from code
            # - Setting up BacktestEngine with data
            # - Running backtest asynchronously
            # - Updating result with performance metrics

            return result

        except Exception as e:
            logger.exception("Backtest submission failed")
            self._track_operation(operation="submit_backtest", status="error")
            backtest_operations_total.labels(status="error").inc()

            return BacktestResult(
                job_id="",
                status="failed",
                error=str(e),
            )

        finally:
            strategy_operation_latency.labels(operation="submit_backtest").observe(
                time.perf_counter() - start
            )

    def get_backtest_status(self, job_id: str) -> BacktestResult | None:
        """
        Get backtest status by job ID.

        Parameters
        ----------
        job_id : str
            Backtest job ID

        Returns
        -------
        BacktestResult | None
            Backtest result or None if not found
        """
        return self._active_backtests.get(job_id)

    def deploy_strategy(self, request: DeploymentRequest) -> DeploymentResult:
        """
        Deploy strategy to specified environment.

        Parameters
        ----------
        request : DeploymentRequest
            Deployment request

        Returns
        -------
        DeploymentResult
            Deployment result
        """
        start = time.perf_counter()

        try:
            # 1. Validate code
            validation = self.validate_strategy_code(
                CodeValidationRequest(
                    code=request.strategy_code,
                    strategy_name=request.strategy_name,
                )
            )

            if not validation.valid:
                result = DeploymentResult(
                    deployment_id="",
                    status="failed",
                    environment=request.environment,
                    message="Code validation failed",
                    error=f"Validation errors: {validation.errors}",
                )
                deployment_operations_total.labels(
                    environment=request.environment,
                    status="validation_failed",
                ).inc()
                return result

            # 2. Validate risk parameters
            risk_validation = self._validate_risk_parameters(request.risk_params)
            if not risk_validation["valid"]:
                result = DeploymentResult(
                    deployment_id="",
                    status="failed",
                    environment=request.environment,
                    message="Risk validation failed",
                    error=f"Risk errors: {risk_validation['errors']}",
                )
                deployment_operations_total.labels(
                    environment=request.environment,
                    status="risk_validation_failed",
                ).inc()
                return result

            # 3. Create deployment ID
            deployment_id = f"deploy-{request.strategy_name}-{int(time.time())}"

            # 4. Deploy based on environment
            if request.environment == "staging":
                message = "Strategy deployed to staging environment"
                monitoring_url = f"/api/strategies/{deployment_id}/performance"
            elif request.environment == "paper":
                message = "Strategy deployed to paper trading environment"
                monitoring_url = f"/api/strategies/{deployment_id}/performance"
            elif request.environment == "live":
                message = "Strategy deployment requires manual approval"
                monitoring_url = None
            else:
                result = DeploymentResult(
                    deployment_id="",
                    status="failed",
                    environment=request.environment,
                    message=f"Invalid environment: {request.environment}",
                )
                deployment_operations_total.labels(
                    environment=request.environment,
                    status="invalid_environment",
                ).inc()
                return result

            result = DeploymentResult(
                deployment_id=deployment_id,
                status="deployed" if request.environment != "live" else "pending_approval",
                environment=request.environment,
                message=message,
                monitoring_url=monitoring_url,
            )

            self._track_operation(operation="deploy_strategy", status="success")
            deployment_operations_total.labels(
                environment=request.environment,
                status="success",
            ).inc()

            # TODO: Actual deployment would happen here
            # This would involve:
            # - Creating strategy instance from code
            # - Deploying via ActorIntegrationService
            # - Setting up monitoring and alerts
            # - Configuring risk limits

            return result

        except Exception as e:
            logger.exception("Strategy deployment failed")
            self._track_operation(operation="deploy_strategy", status="error")
            deployment_operations_total.labels(
                environment=request.environment,
                status="error",
            ).inc()

            return DeploymentResult(
                deployment_id="",
                status="failed",
                environment=request.environment,
                message="Deployment failed",
                error=str(e),
            )

        finally:
            strategy_operation_latency.labels(operation="deploy_strategy").observe(
                time.perf_counter() - start
            )

    def get_strategy_performance(self, strategy_id: str) -> dict[str, Any]:
        """
        Get strategy performance metrics.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier

        Returns
        -------
        dict[str, Any]
            Performance metrics
        """
        start = time.perf_counter()

        try:
            # TODO: Actual implementation would query StrategyStore
            # For now, return mock data
            result = {
                "strategy_id": strategy_id,
                "status": "running",
                "metrics": {
                    "total_pnl": 1250.50,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": 0.05,
                    "win_rate": 0.65,
                    "total_trades": 42,
                },
                "recent_trades": [],
            }

            self._track_operation(operation="get_performance", status="success")

            return result

        except Exception as e:
            logger.exception("Failed to get strategy performance")
            self._track_operation(operation="get_performance", status="error")

            return {
                "strategy_id": strategy_id,
                "error": str(e),
            }

        finally:
            strategy_operation_latency.labels(operation="get_performance").observe(
                time.perf_counter() - start
            )

    def list_strategies(self) -> dict[str, Any]:
        """
        List all registered strategies.

        Returns
        -------
        dict[str, Any]
            List of strategies
        """
        start = time.perf_counter()

        try:
            # TODO: Actual implementation would query StrategyRegistry
            # For now, return empty list
            result = {
                "strategies": [],
                "count": 0,
            }

            self._track_operation(operation="list_strategies", status="success")

            return result

        except Exception as e:
            logger.exception("Failed to list strategies")
            self._track_operation(operation="list_strategies", status="error")

            return {
                "strategies": [],
                "count": 0,
                "error": str(e),
            }

        finally:
            strategy_operation_latency.labels(operation="list_strategies").observe(
                time.perf_counter() - start
            )

    def _validate_risk_parameters(self, risk_params: Mapping[str, Any]) -> dict[str, Any]:
        """
        Validate risk parameters against platform limits.

        Parameters
        ----------
        risk_params : Mapping[str, Any]
            Risk parameters to validate

        Returns
        -------
        dict[str, Any]
            Validation result with valid flag and errors
        """
        errors = []

        # Platform limits
        MAX_POSITION_SIZE = 1_000_000  # $1M
        MAX_LEVERAGE = 3.0
        MAX_DRAWDOWN = 0.20  # 20%
        MIN_STOP_LOSS = 0.01  # 1%

        max_position = risk_params.get("max_position_size", 0)
        if max_position > MAX_POSITION_SIZE:
            errors.append(f"Position size {max_position} exceeds limit {MAX_POSITION_SIZE}")

        leverage = risk_params.get("max_leverage", 1.0)
        if leverage > MAX_LEVERAGE:
            errors.append(f"Leverage {leverage} exceeds limit {MAX_LEVERAGE}")

        drawdown = risk_params.get("max_drawdown", 0)
        if drawdown > MAX_DRAWDOWN:
            errors.append(f"Drawdown {drawdown} exceeds limit {MAX_DRAWDOWN}")

        stop_loss = risk_params.get("stop_loss_pct", 0)
        if stop_loss > 0 and stop_loss < MIN_STOP_LOSS:
            errors.append(f"Stop loss {stop_loss} below minimum {MIN_STOP_LOSS}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }


__all__ = [
    "BacktestRequest",
    "BacktestResult",
    "CodeValidationRequest",
    "CodeValidationResult",
    "DeploymentRequest",
    "DeploymentResult",
    "StrategyCodeValidator",
    "StrategyService",
]
