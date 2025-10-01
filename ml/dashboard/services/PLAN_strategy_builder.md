# Strategy Builder Implementation Plan

## Overview

The **📈 Strategies** tab in the Nautilus ML Dashboard provides a comprehensive strategy development and deployment platform. This plan outlines the implementation of all UI elements: Strategy Builder form, code validation, backtesting infrastructure, and live deployment pipeline.

## System Architecture

### 1. Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dashboard Strategy Tab                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Strategy Builder │  │ Code Validation │  │ Backtest Engine │ │
│  │     Form        │  │ & Sandbox       │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Performance     │  │ Risk Management │  │ Live Deployment │ │
│  │ Visualization   │  │  Enforcement    │  │    Pipeline     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Backend Services Layer                        │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐ │
│ │ StrategyService │ │ ValidationService│ │ BacktestService     │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────────┘ │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐ │
│ │ DeploymentSvc   │ │ RiskGuardSvc    │ │ PerformanceSvc      │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                             │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐ │
│ │ StrategyRegistry│ │ Nautilus Engine │ │ Security Sandbox    │ │
│ │ + ModelRegistry │ │ + BacktestEngine│ │ + AST Validator     │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## UI Elements Analysis & Implementation

### 1. Strategy Builder Form

**Current Elements:**
- Strategy Name input
- Base Strategy dropdown (MLTradingStrategy, SignalBasedStrategy, HybridStrategy, Custom)
- Risk Parameters: Max Position Size, Stop Loss %, Take Profit %, Max Drawdown %

**Implementation:**

#### Frontend (JavaScript/HTML):
```javascript
class StrategyBuilderForm {
    constructor() {
        this.formData = {
            strategyName: '',
            baseStrategy: 'MLTradingStrategy',
            riskParams: {
                maxPositionSize: 100000,
                stopLossPercent: 2.0,
                takeProfitPercent: 5.0,
                maxDrawdownPercent: 10.0
            },
            code: '',
            instruments: [],
            timeframes: []
        };
    }

    validateForm() {
        // Client-side validation
        const errors = [];
        if (!this.formData.strategyName.match(/^[a-zA-Z0-9_]+$/)) {
            errors.push('Strategy name must be alphanumeric with underscores');
        }
        if (this.formData.riskParams.stopLossPercent <= 0) {
            errors.push('Stop loss must be positive');
        }
        // ... additional validations
        return errors;
    }
}
```

#### Backend API Endpoint:
```python
@app.post("/api/strategies")
def create_strategy() -> tuple[Any, int]:
    """Create a new strategy from form data."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    payload = cast(dict[str, Any], request.get_json(silent=True) or {})

    # Validate strategy configuration
    strategy_service = StrategyService()
    validation_result = strategy_service.validate_strategy_config(payload)

    if not validation_result.is_valid:
        return jsonify({
            "error": "validation_failed",
            "details": validation_result.errors
        }), 400

    # Create strategy manifest and register
    strategy_id = strategy_service.create_strategy(payload)
    return jsonify({"strategy_id": strategy_id, "status": "created"}), 201
```

**Map to MLTradingStrategy:**
- Form data maps to `MLStrategyConfig` parameters:
  - `maxPositionSize` → `position_size_pct` (converted to percentage)
  - `stopLossPercent` → `stop_loss_pct`
  - `takeProfitPercent` → `take_profit_pct`
  - Risk parameters validate against `CircuitBreakerConfig` constraints

### 2. Risk Parameter Fields

**Enhanced Risk Management:**

#### Risk Parameter Validation:
```python
@dataclass
class RiskParameterLimits:
    """Global risk limits enforced by the platform."""

    max_position_size_usd: float = 1_000_000  # $1M max per position
    max_portfolio_leverage: float = 3.0       # 3x max leverage
    max_daily_drawdown: float = 0.05          # 5% daily drawdown limit
    max_strategy_correlation: float = 0.8     # Max correlation with existing strategies
    min_sharpe_ratio_threshold: float = 0.5   # Minimum acceptable Sharpe ratio

    def validate_strategy_params(self, params: dict[str, Any]) -> ValidationResult:
        """Validate strategy parameters against platform limits."""
        errors = []

        if params.get('max_position_size', 0) > self.max_position_size_usd:
            errors.append(f"Position size exceeds limit: {self.max_position_size_usd:,.0f} USD")

        if params.get('stop_loss_pct', 0) > self.max_daily_drawdown * 100:
            errors.append(f"Stop loss exceeds daily drawdown limit: {self.max_daily_drawdown:.1%}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

#### UI Enhancements:
- **Real-time validation**: Red borders and error messages for invalid values
- **Dynamic limits**: Display platform limits next to input fields
- **Risk calculator**: Show portfolio impact preview
- **Tooltips**: Explain each risk parameter with examples

### 3. Strategy Logic Python Code Editor (Monaco Editor)

**Implementation Plan:**

#### Monaco Editor Integration:
```javascript
class StrategyCodeEditor {
    constructor(containerId) {
        this.editor = null;
        this.containerId = containerId;
        this.template = this.getDefaultTemplate();
    }

    async initialize() {
        // Load Monaco Editor
        await this.loadMonaco();

        this.editor = monaco.editor.create(document.getElementById(this.containerId), {
            value: this.template,
            language: 'python',
            theme: 'vs-dark',
            automaticLayout: true,
            minimap: { enabled: true },
            scrollBeyondLastLine: false,
            fontSize: 14,
            tabSize: 4,
            wordWrap: 'on'
        });

        // Add Python type hints and autocomplete
        this.setupPythonIntelliSense();

        // Real-time syntax validation
        this.editor.onDidChangeModelContent(() => {
            this.validateSyntax();
        });
    }

    getDefaultTemplate() {
        return `"""
Custom ML Trading Strategy Implementation

This template provides the basic structure for a custom strategy.
Inherit from BaseMLStrategy or MLTradingStrategy as needed.
"""

from ml.strategies.base import BaseMLStrategy
from ml.actors.base import MLSignal
from ml.config.base import MLStrategyConfig
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.position import Position


class CustomStrategy(BaseMLStrategy):
    """Custom trading strategy implementation."""

    def __init__(self, config: MLStrategyConfig, stores=None):
        super().__init__(config, stores)

        # Initialize custom state variables
        self.custom_indicator = None
        self.signal_history = []

    def on_start(self) -> None:
        """Initialize strategy on start."""
        super().on_start()

        # Add custom initialization logic
        self.log.info(f"Starting {self.__class__.__name__}")

    def _process_ml_signal(self, signal: MLSignal) -> None:
        """
        Process ML signal and execute trading logic.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process
        """
        # Store signal in history
        self.signal_history.append(signal)

        # Custom signal processing logic
        if signal.confidence < self._config.min_confidence:
            self.log.debug("Signal below confidence threshold")
            return

        # Determine position action
        current_position = self._get_current_position()
        target_side = self.target_side_from_prediction(signal.prediction, 0.5)

        if current_position is None:
            # Enter new position
            self._enter_position(target_side, signal)
        elif self.should_reverse(current_position, target_side):
            # Reverse position
            self._reverse_position(current_position, target_side, signal)
        else:
            # Hold position
            self.log.debug("Holding current position")

    def _enter_position(self, side: OrderSide, signal: MLSignal) -> None:
        """Enter a new position based on signal."""
        quantity = self.size_and_validate(signal)
        if quantity is None:
            return

        # Place order with custom logic
        order_id = self._submit_smart_order(side, quantity, signal)
        if order_id:
            self._active_positions += 1
            self.log.info(f"Entered {side.name} position: {quantity} units")

    def _reverse_position(self, current_position: Position, target_side: OrderSide, signal: MLSignal) -> None:
        """Reverse existing position."""
        # Close current position
        close_side = OrderSide.SELL if current_position.side.name == "LONG" else OrderSide.BUY
        self._place_market_order(close_side, current_position.quantity, reduce_only=True)

        # Open new position
        quantity = self.size_and_validate(signal)
        if quantity:
            self._submit_smart_order(target_side, quantity, signal)


# Strategy configuration
def create_strategy_config(**kwargs) -> MLStrategyConfig:
    """Create strategy configuration with overrides."""
    defaults = {
        'position_size_pct': 0.1,
        'min_confidence': 0.7,
        'max_positions': 1,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'execute_trades': True
    }
    defaults.update(kwargs)

    return MLStrategyConfig(**defaults)
`;
    }

    setupPythonIntelliSense() {
        // Add Nautilus Trader and ML module type definitions
        monaco.languages.registerCompletionItemProvider('python', {
            provideCompletionItems: (model, position) => {
                const suggestions = [
                    {
                        label: 'MLSignal',
                        kind: monaco.languages.CompletionItemKind.Class,
                        documentation: 'ML signal data structure',
                        insertText: 'MLSignal'
                    },
                    {
                        label: '_process_ml_signal',
                        kind: monaco.languages.CompletionItemKind.Method,
                        documentation: 'Process ML signal method',
                        insertText: '_process_ml_signal(self, signal: MLSignal) -> None:'
                    }
                    // Add more Nautilus-specific completions
                ];
                return { suggestions };
            }
        });
    }

    async validateSyntax() {
        const code = this.editor.getValue();

        // Send to backend for validation
        try {
            const response = await fetch('/api/strategies/validate-syntax', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code })
            });

            const result = await response.json();

            if (!result.valid) {
                // Show syntax errors in editor
                this.showSyntaxErrors(result.errors);
            } else {
                this.clearSyntaxErrors();
            }
        } catch (error) {
            console.error('Syntax validation failed:', error);
        }
    }
}
```

### 4. Code Validation and Security Approach

**Multi-Layer Security Architecture:**

#### 1. Client-Side Validation (Monaco Editor):
- **Syntax highlighting** with Python language server
- **Real-time error detection** using Pyflakes/AST parsing
- **Import restriction warnings** for non-approved modules

#### 2. Server-Side AST Validation:
```python
import ast
import sys
from typing import Any, Set, List
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    security_issues: List[str]

class StrategyCodeValidator:
    """Secure Python code validator for trading strategies."""

    # Allowed imports - whitelist approach
    ALLOWED_MODULES = {
        'ml.strategies.base',
        'ml.actors.base',
        'ml.config.base',
        'nautilus_trader.model.enums',
        'nautilus_trader.model.position',
        'nautilus_trader.model.objects',
        'nautilus_trader.model.identifiers',
        'numpy',
        'pandas',
        'typing',
        'dataclasses',
        'abc',
        'enum',
        'datetime'
    }

    # Forbidden operations
    FORBIDDEN_CALLS = {
        'exec', 'eval', 'compile', 'open', '__import__',
        'getattr', 'setattr', 'delattr', 'globals', 'locals',
        'input', 'raw_input', 'file', 'reload'
    }

    # Forbidden attributes (prevent system access)
    FORBIDDEN_ATTRS = {
        '__subclasses__', '__bases__', '__globals__',
        '__code__', '__closure__', '__dict__'
    }

    def validate_code(self, code: str) -> ValidationResult:
        """Comprehensive code validation with security checks."""
        errors = []
        warnings = []
        security_issues = []

        try:
            # Parse into AST
            tree = ast.parse(code)

            # Security analysis
            security_issues.extend(self._check_security(tree))

            # Import validation
            import_errors = self._validate_imports(tree)
            errors.extend(import_errors)

            # Structural validation
            struct_errors = self._validate_structure(tree)
            errors.extend(struct_errors)

            # Strategy-specific validation
            strategy_warnings = self._validate_strategy_patterns(tree)
            warnings.extend(strategy_warnings)

        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0 and len(security_issues) == 0,
            errors=errors,
            warnings=warnings,
            security_issues=security_issues
        )

    def _check_security(self, tree: ast.AST) -> List[str]:
        """Check for security vulnerabilities."""
        issues = []

        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
                    issues.append(f"Forbidden function call: {node.func.id}")

            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in self.FORBIDDEN_ATTRS:
                    issues.append(f"Forbidden attribute access: {node.attr}")

            # Check for file I/O operations
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ('open', 'file'):
                    issues.append("File I/O operations not allowed")

        return issues

    def _validate_imports(self, tree: ast.AST) -> List[str]:
        """Validate that only approved modules are imported."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_MODULES:
                        errors.append(f"Unauthorized import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in self.ALLOWED_MODULES:
                    # Check if it's a submodule of allowed module
                    allowed = any(
                        node.module.startswith(allowed_mod + '.')
                        for allowed_mod in self.ALLOWED_MODULES
                    )
                    if not allowed:
                        errors.append(f"Unauthorized import: {node.module}")

        return errors

    def _validate_structure(self, tree: ast.AST) -> List[str]:
        """Validate strategy structure requirements."""
        errors = []

        # Find strategy class
        strategy_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if inherits from BaseMLStrategy or MLTradingStrategy
                for base in node.bases:
                    if isinstance(base, ast.Name) and 'Strategy' in base.id:
                        strategy_classes.append(node.name)

        if not strategy_classes:
            errors.append("No strategy class found (must inherit from BaseMLStrategy)")

        # Check for required methods
        required_methods = {'_process_ml_signal'}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in strategy_classes:
                method_names = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
                missing_methods = required_methods - method_names
                if missing_methods:
                    errors.append(f"Missing required methods: {missing_methods}")

        return errors

# API endpoint for validation
@app.post("/api/strategies/validate-syntax")
def validate_strategy_syntax() -> tuple[Any, int]:
    """Validate strategy code syntax and security."""
    payload = cast(dict[str, Any], request.get_json(silent=True) or {})
    code = payload.get('code', '')

    if not code.strip():
        return jsonify({"valid": False, "errors": ["Empty code"]}), 400

    validator = StrategyCodeValidator()
    result = validator.validate_code(code)

    return jsonify({
        "valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "security_issues": result.security_issues
    }), 200
```

#### 3. Sandboxed Execution Environment:
```python
import subprocess
import tempfile
import os
from pathlib import Path

class StrategyExecutionSandbox:
    """Secure execution environment for strategy validation."""

    def __init__(self):
        self.timeout_seconds = 10
        self.max_memory_mb = 100

    def validate_execution(self, code: str) -> dict[str, Any]:
        """Execute strategy code in sandboxed environment."""

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute in restricted environment
            result = subprocess.run([
                sys.executable, '-c', f"""
import sys
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, ({self.max_memory_mb * 1024 * 1024}, -1))

# Import and validate
with open('{temp_file}', 'r') as f:
    code = f.read()

# Try to compile
try:
    compile(code, '{temp_file}', 'exec')
    print("VALIDATION_SUCCESS")
except Exception as e:
    print(f"VALIDATION_ERROR: {{e}}")
"""
            ],
            timeout=self.timeout_seconds,
            capture_output=True,
            text=True
            )

            if result.returncode == 0 and "VALIDATION_SUCCESS" in result.stdout:
                return {"valid": True, "message": "Code validation successful"}
            else:
                return {
                    "valid": False,
                    "error": result.stdout.replace("VALIDATION_ERROR: ", "")
                }

        except subprocess.TimeoutExpired:
            return {"valid": False, "error": "Validation timeout"}
        except Exception as e:
            return {"valid": False, "error": f"Execution error: {str(e)}"}
        finally:
            # Clean up
            os.unlink(temp_file)
```

### 5. Action Buttons Implementation

#### ✓ Validate Button:
```javascript
async function validateStrategy() {
    const form = new StrategyBuilderForm();
    const code = strategyEditor.getCode();

    // Show loading state
    const validateBtn = document.querySelector('[onclick="validateStrategy()"]');
    validateBtn.innerHTML = '🔄 Validating...';
    validateBtn.disabled = true;

    try {
        // 1. Form validation
        const formErrors = form.validateForm();
        if (formErrors.length > 0) {
            showValidationErrors(formErrors);
            return;
        }

        // 2. Code validation
        const response = await fetch('/api/strategies/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: form.formData.strategyName,
                base_strategy: form.formData.baseStrategy,
                risk_params: form.formData.riskParams,
                code: code
            })
        });

        const result = await response.json();

        if (result.valid) {
            showSuccess('Strategy validation passed! ✅');
            enableBacktestButton();
        } else {
            showValidationErrors(result.errors);
        }

    } catch (error) {
        showError(`Validation failed: ${error.message}`);
    } finally {
        validateBtn.innerHTML = '✓ Validate';
        validateBtn.disabled = false;
    }
}
```

#### 📊 Backtest Button:
```python
@app.post("/api/strategies/backtest")
def run_strategy_backtest() -> tuple[Any, int]:
    """Execute strategy backtest with historical data."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    payload = cast(dict[str, Any], request.get_json(silent=True) or {})

    # Validate strategy first
    validator = StrategyCodeValidator()
    validation_result = validator.validate_code(payload.get('code', ''))

    if not validation_result.is_valid:
        return jsonify({
            "error": "validation_failed",
            "details": validation_result.errors
        }), 400

    # Create backtest configuration
    backtest_config = {
        'start_date': payload.get('start_date', '2024-01-01'),
        'end_date': payload.get('end_date', '2024-12-31'),
        'initial_balance': payload.get('initial_balance', 100000),
        'instruments': payload.get('instruments', ['EURUSD.SIM']),
        'strategy_params': payload.get('risk_params', {})
    }

    # Queue backtest job
    backtest_service = BacktestService()
    job_id = backtest_service.submit_backtest(
        strategy_code=payload.get('code'),
        config=backtest_config
    )

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "message": "Backtest submitted successfully"
    }), 202
```

#### 🚀 Deploy Live Button:
```python
@app.post("/api/strategies/deploy")
def deploy_strategy_live() -> tuple[Any, int]:
    """Deploy strategy to live trading."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    payload = cast(dict[str, Any], request.get_json(silent=True) or {})

    # Enhanced security check for live deployment
    deployment_service = DeploymentService()

    # 1. Risk validation
    risk_check = deployment_service.validate_risk_parameters(payload)
    if not risk_check.approved:
        return jsonify({
            "error": "risk_validation_failed",
            "details": risk_check.reasons
        }), 400

    # 2. Code security audit
    security_audit = deployment_service.security_audit(payload.get('code'))
    if not security_audit.passed:
        return jsonify({
            "error": "security_audit_failed",
            "details": security_audit.issues
        }), 400

    # 3. Backtest requirement check
    if not deployment_service.has_recent_backtest(payload.get('strategy_name')):
        return jsonify({
            "error": "backtest_required",
            "message": "Recent backtest required before live deployment"
        }), 400

    # 4. Deploy to staging environment first
    staging_deployment = deployment_service.deploy_to_staging(payload)

    return jsonify({
        "deployment_id": staging_deployment.id,
        "status": "deployed_to_staging",
        "message": "Strategy deployed to staging environment",
        "staging_url": staging_deployment.monitoring_url
    }), 201
```

### 6. Backtesting Infrastructure

**Backtest Service Architecture:**

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import uuid

@dataclass
class BacktestRequest:
    """Backtest execution request."""
    job_id: str
    strategy_code: str
    strategy_name: str
    config: Dict[str, Any]
    user_id: str
    created_at: float

@dataclass
class BacktestResult:
    """Backtest execution result."""
    job_id: str
    status: str  # 'running', 'completed', 'failed'
    performance_metrics: Dict[str, float]
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    error_message: Optional[str]
    execution_time_seconds: float

class BacktestService:
    """Production-ready backtesting service."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.active_jobs: Dict[str, BacktestRequest] = {}
        self.results_cache: Dict[str, BacktestResult] = {}
        self.data_loader = BacktestDataLoader()

    def submit_backtest(self, strategy_code: str, config: Dict[str, Any]) -> str:
        """Submit backtest job and return job ID."""
        job_id = str(uuid.uuid4())

        request = BacktestRequest(
            job_id=job_id,
            strategy_code=strategy_code,
            strategy_name=config.get('strategy_name', f'strategy_{job_id[:8]}'),
            config=config,
            user_id=config.get('user_id', 'dashboard'),
            created_at=time.time()
        )

        self.active_jobs[job_id] = request

        # Submit to thread pool
        future = self.executor.submit(self._execute_backtest, request)

        return job_id

    def _execute_backtest(self, request: BacktestRequest) -> BacktestResult:
        """Execute backtest in background thread."""
        start_time = time.time()

        try:
            # 1. Load historical data
            data = self.data_loader.load_data(
                instruments=request.config.get('instruments', ['EURUSD.SIM']),
                start_date=request.config.get('start_date'),
                end_date=request.config.get('end_date')
            )

            # 2. Setup backtest engine
            engine = self._create_backtest_engine(request, data)

            # 3. Execute backtest
            engine.run()

            # 4. Extract results
            performance = self._calculate_performance_metrics(engine)
            equity_curve = self._extract_equity_curve(engine)
            trades = self._extract_trades(engine)

            result = BacktestResult(
                job_id=request.job_id,
                status='completed',
                performance_metrics=performance,
                equity_curve=equity_curve,
                trades=trades,
                error_message=None,
                execution_time_seconds=time.time() - start_time
            )

        except Exception as e:
            result = BacktestResult(
                job_id=request.job_id,
                status='failed',
                performance_metrics={},
                equity_curve=[],
                trades=[],
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )

        # Cache result
        self.results_cache[request.job_id] = result

        # Clean up active job
        if request.job_id in self.active_jobs:
            del self.active_jobs[request.job_id]

        return result

    def _create_backtest_engine(self, request: BacktestRequest, data) -> BacktestEngine:
        """Create configured backtest engine."""
        from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
        from nautilus_trader.config import LoggingConfig

        # Dynamically create strategy class from code
        strategy_class = self._compile_strategy_class(request.strategy_code)

        # Create strategy config
        strategy_config = MLStrategyConfig(
            instrument_id=InstrumentId.from_str(request.config.get('instruments', ['EURUSD.SIM'])[0]),
            ml_signal_source='backtest_signal_actor',
            **request.config.get('strategy_params', {})
        )

        # Configure backtest engine
        config = BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            log_level="INFO",
            logging=LoggingConfig(),
        )

        engine = BacktestEngine(config=config)

        # Add instruments and data
        for instrument_id_str in request.config.get('instruments', []):
            instrument = self._create_instrument(instrument_id_str)
            engine.add_instrument(instrument)

        # Add data
        engine.add_data(data)

        # Add strategy
        engine.add_strategy(strategy_class(strategy_config))

        return engine

    def _compile_strategy_class(self, code: str) -> type:
        """Safely compile strategy code into class."""

        # Security validation already done, but double-check
        validator = StrategyCodeValidator()
        validation = validator.validate_code(code)
        if not validation.is_valid:
            raise ValueError(f"Strategy validation failed: {validation.errors}")

        # Create isolated namespace
        namespace = self._create_safe_namespace()

        # Execute code in namespace
        exec(code, namespace)

        # Find strategy class
        strategy_classes = [
            obj for name, obj in namespace.items()
            if isinstance(obj, type) and 'Strategy' in name
        ]

        if not strategy_classes:
            raise ValueError("No strategy class found in code")

        return strategy_classes[0]

    def _create_safe_namespace(self) -> dict:
        """Create safe namespace for strategy execution."""
        return {
            '__builtins__': {
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'print': print  # For debugging
            },
            # Import allowed modules
            'MLStrategyConfig': MLStrategyConfig,
            'BaseMLStrategy': BaseMLStrategy,
            'MLSignal': MLSignal,
            'OrderSide': OrderSide,
            'Position': Position,
            'np': np,
            'pd': pd,
        }

    def _calculate_performance_metrics(self, engine) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        account = engine.trader.generate_account_report()

        # Basic metrics
        metrics = {
            'total_return': float(account.total_pnl.as_double()),
            'return_percentage': (float(account.total_pnl.as_double()) / 100000) * 100,  # Assuming 100k initial
            'total_trades': account.total_orders,
            'winning_trades': 0,  # Will calculate from trade history
            'losing_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0
        }

        # Calculate advanced metrics from equity curve
        equity_curve = self._extract_equity_curve(engine)
        if len(equity_curve) > 1:
            returns = self._calculate_returns(equity_curve)
            metrics.update(self._calculate_risk_metrics(returns))

        return metrics

    def get_backtest_status(self, job_id: str) -> Dict[str, Any]:
        """Get backtest job status."""
        if job_id in self.results_cache:
            result = self.results_cache[job_id]
            return {
                'job_id': job_id,
                'status': result.status,
                'progress': 100 if result.status == 'completed' else 0,
                'message': result.error_message or 'Backtest completed successfully'
            }
        elif job_id in self.active_jobs:
            return {
                'job_id': job_id,
                'status': 'running',
                'progress': 50,  # Estimated progress
                'message': 'Backtest in progress...'
            }
        else:
            return {
                'job_id': job_id,
                'status': 'not_found',
                'progress': 0,
                'message': 'Job not found'
            }
```

### 7. Strategy Performance Chart Integration

**Chart.js Implementation:**

```javascript
class StrategyPerformanceChart {
    constructor(canvasId) {
        this.chart = null;
        this.canvasId = canvasId;
    }

    initialize() {
        const ctx = document.getElementById(this.canvasId).getContext('2d');

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cumulative P&L',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Drawdown',
                    data: [],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: false,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'P&L ($)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Drawdown (%)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;

                                if (label === 'Cumulative P&L') {
                                    return `${label}: $${value.toLocaleString()}`;
                                } else {
                                    return `${label}: ${value.toFixed(2)}%`;
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    updateData(backtestResult) {
        if (!backtestResult.equity_curve) return;

        const labels = backtestResult.equity_curve.map(point => point.timestamp);
        const pnlData = backtestResult.equity_curve.map(point => point.balance);
        const drawdownData = backtestResult.equity_curve.map(point => point.drawdown);

        this.chart.data.labels = labels;
        this.chart.data.datasets[0].data = pnlData;
        this.chart.data.datasets[1].data = drawdownData;

        this.chart.update();
    }

    showMetrics(metrics) {
        // Update metrics display
        document.getElementById('total-return').textContent = `$${metrics.total_return.toLocaleString()}`;
        document.getElementById('win-rate').textContent = `${metrics.win_rate.toFixed(1)}%`;
        document.getElementById('sharpe-ratio').textContent = metrics.sharpe_ratio.toFixed(2);
        document.getElementById('max-drawdown').textContent = `${metrics.max_drawdown.toFixed(2)}%`;
    }
}

// Backtest execution with real-time updates
async function backtestStrategy() {
    const backtestBtn = document.querySelector('[onclick="backtestStrategy()"]');
    backtestBtn.innerHTML = '🔄 Running Backtest...';
    backtestBtn.disabled = true;

    try {
        // Submit backtest
        const response = await fetch('/api/strategies/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: strategyEditor.getCode(),
                strategy_name: document.getElementById('strategy-name').value,
                start_date: '2024-01-01',
                end_date: '2024-12-31',
                initial_balance: 100000,
                instruments: ['EURUSD.SIM'],
                strategy_params: {
                    max_position_size: document.getElementById('max-position').value,
                    stop_loss_pct: document.getElementById('stop-loss').value / 100,
                    take_profit_pct: document.getElementById('take-profit').value / 100
                }
            })
        });

        const result = await response.json();

        if (response.ok) {
            // Poll for results
            pollBacktestStatus(result.job_id);
        } else {
            showError(result.error || 'Backtest failed');
        }

    } catch (error) {
        showError(`Backtest error: ${error.message}`);
    } finally {
        backtestBtn.innerHTML = '📊 Backtest';
        backtestBtn.disabled = false;
    }
}

async function pollBacktestStatus(jobId) {
    const maxAttempts = 60; // 5 minutes max
    let attempts = 0;

    const poll = async () => {
        try {
            const response = await fetch(`/api/strategies/backtest/${jobId}/status`);
            const status = await response.json();

            if (status.status === 'completed') {
                // Get full results
                const resultsResponse = await fetch(`/api/strategies/backtest/${jobId}/results`);
                const results = await resultsResponse.json();

                // Update chart and metrics
                const chart = new StrategyPerformanceChart('strategy-chart');
                chart.updateData(results);
                chart.showMetrics(results.performance_metrics);

                showSuccess('Backtest completed successfully! ✅');

            } else if (status.status === 'failed') {
                showError(`Backtest failed: ${status.message}`);

            } else if (status.status === 'running' && attempts < maxAttempts) {
                // Continue polling
                attempts++;
                setTimeout(poll, 5000); // Poll every 5 seconds

            } else {
                showError('Backtest timeout or unknown error');
            }

        } catch (error) {
            showError(`Status polling error: ${error.message}`);
        }
    };

    poll();
}
```

### 8. Deployment Pipeline to Live Trading

**Multi-Stage Deployment Architecture:**

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json
import time
import subprocess
import docker

class DeploymentStage(Enum):
    VALIDATION = "validation"
    STAGING = "staging"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"

@dataclass
class DeploymentConfig:
    strategy_id: str
    environment: DeploymentStage
    risk_limits: Dict[str, float]
    monitoring_config: Dict[str, Any]
    rollback_config: Dict[str, Any]

class DeploymentService:
    """Production deployment service with safety controls."""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.active_deployments: Dict[str, DeploymentConfig] = {}

    def deploy_strategy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-stage strategy deployment with safety checks."""

        # Stage 1: Validation
        validation_result = self._validate_for_deployment(payload)
        if not validation_result['passed']:
            return {
                "success": False,
                "stage": "validation",
                "error": validation_result['errors']
            }

        # Stage 2: Deploy to staging
        staging_result = self._deploy_to_staging(payload)
        if not staging_result['success']:
            return staging_result

        # Stage 3: Paper trading (optional)
        if payload.get('paper_trading_required', True):
            paper_result = self._deploy_to_paper_trading(payload)
            if not paper_result['success']:
                return paper_result

        # Stage 4: Live deployment (requires manual approval)
        live_result = self._prepare_live_deployment(payload)

        return live_result

    def _validate_for_deployment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive pre-deployment validation."""

        errors = []
        warnings = []

        # 1. Risk parameter validation
        risk_params = payload.get('risk_params', {})

        if risk_params.get('max_position_size', 0) > 1_000_000:
            errors.append("Position size exceeds $1M limit")

        if risk_params.get('stop_loss_pct', 0) < 0.01:
            warnings.append("Stop loss below 1% - high risk")

        # 2. Backtest requirement
        strategy_name = payload.get('strategy_name')
        recent_backtest = self._check_recent_backtest(strategy_name)

        if not recent_backtest:
            errors.append("Recent backtest required (within 7 days)")
        elif recent_backtest['sharpe_ratio'] < 0.5:
            warnings.append(f"Low Sharpe ratio: {recent_backtest['sharpe_ratio']:.2f}")

        # 3. Code security audit
        code_audit = self._security_audit(payload.get('code', ''))
        errors.extend(code_audit['critical_issues'])
        warnings.extend(code_audit['warnings'])

        # 4. Resource requirements check
        resource_check = self._check_resource_availability()
        if not resource_check['available']:
            errors.append("Insufficient system resources")

        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def _deploy_to_staging(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to staging environment with simulated data."""

        try:
            # Create staging container
            container_config = {
                'image': 'nautilus-ml:latest',
                'environment': {
                    'ENVIRONMENT': 'staging',
                    'STRATEGY_CODE': payload.get('code'),
                    'RISK_PARAMS': json.dumps(payload.get('risk_params')),
                    'LOG_LEVEL': 'DEBUG'
                },
                'ports': {'8080/tcp': None},  # Random port
                'mem_limit': '512m',
                'cpu_count': 1
            }

            container = self.docker_client.containers.run(
                detach=True,
                name=f"staging-{payload.get('strategy_name')}-{int(time.time())}",
                **container_config
            )

            # Wait for health check
            health_check = self._wait_for_health(container, timeout=60)

            if health_check['healthy']:
                return {
                    "success": True,
                    "stage": "staging",
                    "container_id": container.id,
                    "monitoring_url": f"http://localhost:{health_check['port']}/metrics"
                }
            else:
                container.remove(force=True)
                return {
                    "success": False,
                    "stage": "staging",
                    "error": "Health check failed"
                }

        except Exception as e:
            return {
                "success": False,
                "stage": "staging",
                "error": f"Container deployment failed: {str(e)}"
            }

    def _deploy_to_paper_trading(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to paper trading with real market data."""

        # Similar to staging but with live data feeds
        container_config = {
            'image': 'nautilus-ml:latest',
            'environment': {
                'ENVIRONMENT': 'paper',
                'STRATEGY_CODE': payload.get('code'),
                'RISK_PARAMS': json.dumps(payload.get('risk_params')),
                'DATA_SOURCE': 'live',
                'PAPER_TRADING': 'true'
            },
            'mem_limit': '1g',
            'cpu_count': 2
        }

        # Implementation similar to staging but with extended monitoring
        return {"success": True, "stage": "paper_trading", "duration_hours": 24}

    def _prepare_live_deployment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare live deployment (requires manual approval)."""

        deployment_id = f"live-{payload.get('strategy_name')}-{int(time.time())}"

        # Create deployment record for manual approval
        deployment_record = {
            'deployment_id': deployment_id,
            'strategy_name': payload.get('strategy_name'),
            'risk_params': payload.get('risk_params'),
            'staging_results': 'passed',
            'paper_trading_results': 'pending',
            'requires_approval': True,
            'created_at': time.time(),
            'status': 'pending_approval'
        }

        # Store in deployment queue
        self._store_deployment_record(deployment_record)

        return {
            "success": True,
            "stage": "live_preparation",
            "deployment_id": deployment_id,
            "status": "pending_approval",
            "message": "Strategy prepared for live deployment. Manual approval required.",
            "approval_url": f"/dashboard/deployments/{deployment_id}/approve"
        }

    def approve_live_deployment(self, deployment_id: str, approver_id: str) -> Dict[str, Any]:
        """Approve and execute live deployment."""

        record = self._get_deployment_record(deployment_id)
        if not record:
            return {"success": False, "error": "Deployment not found"}

        if record['status'] != 'pending_approval':
            return {"success": False, "error": "Invalid deployment status"}

        try:
            # Final safety checks
            safety_check = self._final_safety_check(record)
            if not safety_check['passed']:
                return {"success": False, "error": safety_check['errors']}

            # Deploy to production
            production_container = self._deploy_to_production(record)

            # Update record
            record.update({
                'status': 'live',
                'approved_by': approver_id,
                'approved_at': time.time(),
                'container_id': production_container['id'],
                'monitoring_url': production_container['monitoring_url']
            })

            self._update_deployment_record(record)

            return {
                "success": True,
                "stage": "live_deployment",
                "deployment_id": deployment_id,
                "status": "live",
                "monitoring_url": production_container['monitoring_url']
            }

        except Exception as e:
            return {"success": False, "error": f"Live deployment failed: {str(e)}"}

    def _final_safety_check(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Final safety check before live deployment."""

        checks = []

        # 1. Market hours check
        if not self._is_market_open():
            checks.append("Market is closed - deployment not recommended")

        # 2. System load check
        if self._get_system_load() > 0.8:
            checks.append("High system load - deployment not recommended")

        # 3. Recent news/volatility check
        volatility = self._get_market_volatility()
        if volatility > 2.0:  # 2x normal volatility
            checks.append("High market volatility detected")

        return {
            'passed': len(checks) == 0,
            'errors': checks
        }
```

## Security Architecture

### 1. Code Execution Sandbox

```python
import docker
import tempfile
import os
from pathlib import Path

class SecureExecutionSandbox:
    """Docker-based secure execution environment."""

    def __init__(self):
        self.client = docker.from_env()

    def create_sandbox(self, strategy_code: str) -> str:
        """Create isolated container for strategy execution."""

        # Create temporary directory with strategy code
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy_path = Path(temp_dir) / "strategy.py"
            strategy_path.write_text(strategy_code)

            # Create Dockerfile for sandbox
            dockerfile = """
FROM python:3.11-slim

# Install only required packages
RUN pip install nautilus-trader[ml] numpy pandas

# Create non-root user
RUN useradd -m -s /bin/bash trader
USER trader

# Copy strategy
COPY strategy.py /home/trader/strategy.py

# Resource limits
ENV PYTHONPATH=/home/trader
WORKDIR /home/trader

# Security: disable network, limit resources
ENTRYPOINT ["python", "strategy.py"]
"""

            dockerfile_path = Path(temp_dir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile)

            # Build image
            image = self.client.images.build(
                path=temp_dir,
                tag=f"strategy-sandbox:{int(time.time())}",
                rm=True
            )

            return image[0].id
```

### 2. API Security

```python
from functools import wraps
import jwt
from datetime import datetime, timedelta

def require_strategy_permissions(f):
    """Decorator to require strategy deployment permissions."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')

        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])

            # Check permissions
            if 'strategy:deploy' not in payload.get('permissions', []):
                return jsonify({"error": "insufficient_permissions"}), 403

            # Check rate limits
            if not check_rate_limit(payload['user_id'], 'strategy_operations'):
                return jsonify({"error": "rate_limit_exceeded"}), 429

            return f(*args, **kwargs)

        except jwt.InvalidTokenError:
            return jsonify({"error": "invalid_token"}), 401

    return decorated_function

@app.post("/api/strategies/deploy")
@require_strategy_permissions
def deploy_strategy_secure():
    """Secure strategy deployment endpoint."""
    # Implementation here...
    pass
```

## Monitoring & Observability

### 1. Strategy Performance Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge

# Strategy-specific metrics
strategy_deployments = Counter(
    'ml_strategy_deployments_total',
    'Total strategy deployments',
    ['strategy_name', 'environment', 'status']
)

strategy_performance = Gauge(
    'ml_strategy_performance',
    'Strategy performance metrics',
    ['strategy_id', 'metric_name']
)

deployment_latency = Histogram(
    'ml_deployment_latency_seconds',
    'Strategy deployment latency',
    ['stage']
)

class StrategyMonitoringService:
    """Real-time strategy monitoring."""

    def __init__(self):
        self.active_strategies: Dict[str, Dict[str, Any]] = {}

    def monitor_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Monitor strategy performance in real-time."""

        metrics = {}

        # Get container stats
        container = self._get_strategy_container(strategy_id)
        if container:
            stats = container.stats(stream=False)

            # CPU and Memory
            metrics['cpu_usage'] = self._calculate_cpu_usage(stats)
            metrics['memory_usage'] = stats['memory_stats']['usage']

        # Get strategy performance from StrategyStore
        performance = self._get_strategy_performance(strategy_id)
        metrics.update(performance)

        # Update Prometheus metrics
        for metric_name, value in metrics.items():
            strategy_performance.labels(
                strategy_id=strategy_id,
                metric_name=metric_name
            ).set(value)

        return metrics
```

### 2. Real-time Alerts

```python
class StrategyAlertService:
    """Alert service for strategy monitoring."""

    def __init__(self):
        self.alert_rules = self._load_alert_rules()

    def check_alerts(self, strategy_id: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any alert conditions are met."""

        alerts = []

        # Performance alerts
        if metrics.get('drawdown', 0) > 0.1:  # 10% drawdown
            alerts.append({
                'level': 'warning',
                'type': 'high_drawdown',
                'message': f"Strategy {strategy_id} drawdown: {metrics['drawdown']:.2%}",
                'action': 'reduce_position_size'
            })

        if metrics.get('daily_pnl', 0) < -10000:  # $10K daily loss
            alerts.append({
                'level': 'critical',
                'type': 'large_loss',
                'message': f"Strategy {strategy_id} large loss: ${metrics['daily_pnl']:,.0f}",
                'action': 'emergency_stop'
            })

        # System alerts
        if metrics.get('cpu_usage', 0) > 80:
            alerts.append({
                'level': 'warning',
                'type': 'high_cpu',
                'message': f"Strategy {strategy_id} high CPU: {metrics['cpu_usage']:.1f}%"
            })

        return alerts

    def handle_alert(self, alert: Dict[str, Any]) -> bool:
        """Handle alert with appropriate action."""

        if alert['action'] == 'emergency_stop':
            return self._emergency_stop_strategy(alert['strategy_id'])
        elif alert['action'] == 'reduce_position_size':
            return self._reduce_position_size(alert['strategy_id'])

        return False
```

## Risk Management Enforcement

### 1. Pre-Trade Risk Checks

```python
class RiskGuardService:
    """Real-time risk management service."""

    def __init__(self):
        self.position_limits = self._load_position_limits()
        self.correlation_matrix = self._load_correlation_matrix()

    def validate_trade(self, strategy_id: str, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade against risk limits."""

        checks = []

        # 1. Position size check
        position_size = abs(trade_request.get('quantity', 0))
        max_position = self.position_limits.get(strategy_id, {}).get('max_position_size', 100000)

        if position_size > max_position:
            checks.append({
                'type': 'position_limit',
                'severity': 'block',
                'message': f'Position size {position_size} exceeds limit {max_position}'
            })

        # 2. Portfolio concentration check
        portfolio_exposure = self._calculate_portfolio_exposure(trade_request)
        if portfolio_exposure > 0.3:  # 30% max exposure to single instrument
            checks.append({
                'type': 'concentration_risk',
                'severity': 'warn',
                'message': f'High portfolio concentration: {portfolio_exposure:.1%}'
            })

        # 3. Correlation check
        correlation_risk = self._check_correlation_risk(strategy_id, trade_request)
        if correlation_risk > 0.8:
            checks.append({
                'type': 'correlation_risk',
                'severity': 'warn',
                'message': f'High correlation with existing positions: {correlation_risk:.2f}'
            })

        # 4. VaR check
        var_impact = self._calculate_var_impact(trade_request)
        if var_impact > 0.05:  # 5% portfolio VaR increase
            checks.append({
                'type': 'var_limit',
                'severity': 'block',
                'message': f'VaR impact too high: {var_impact:.1%}'
            })

        blocking_checks = [c for c in checks if c['severity'] == 'block']

        return {
            'approved': len(blocking_checks) == 0,
            'checks': checks,
            'risk_score': self._calculate_risk_score(checks)
        }
```

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- ✅ Strategy Builder Form UI components
- ✅ Monaco Editor integration with Python syntax highlighting
- ✅ Basic form validation and error handling
- ✅ API endpoints for strategy creation and validation

### Phase 2: Security & Validation (Week 2-3)
- 🔄 AST-based code validation system
- 🔄 Docker-based execution sandbox
- 🔄 Security audit pipeline
- 🔄 Risk parameter validation

### Phase 3: Backtesting Infrastructure (Week 3-4)
- 📋 BacktestService with job queue
- 📋 Integration with Nautilus BacktestEngine
- 📋 Performance metrics calculation
- 📋 Real-time progress tracking

### Phase 4: Visualization & UI Polish (Week 4)
- 📋 Strategy performance charts (Chart.js)
- 📋 Real-time backtest progress indicators
- 📋 Enhanced error messages and user feedback
- 📋 Mobile-responsive design improvements

### Phase 5: Deployment Pipeline (Week 5-6)
- 📋 Multi-stage deployment system
- 📋 Staging environment setup
- 📋 Paper trading integration
- 📋 Manual approval workflow for live deployment

### Phase 6: Monitoring & Risk Management (Week 6-7)
- 📋 Real-time strategy monitoring
- 📋 Performance alerts and notifications
- 📋 Risk management enforcement
- 📋 Emergency stop mechanisms

### Phase 7: Production Hardening (Week 7-8)
- 📋 Load testing and optimization
- 📋 Security penetration testing
- 📋 Documentation and user guides
- 📋 Integration testing with existing ML pipeline

## Success Metrics

### Technical Metrics
- **Code Validation Speed**: < 2 seconds for syntax validation
- **Backtest Execution Time**: < 5 minutes for 1-year daily data
- **Deployment Success Rate**: > 95% successful deployments
- **API Response Time**: < 200ms for form submissions

### User Experience Metrics
- **Form Completion Rate**: > 80% of started forms completed
- **Error Resolution Time**: < 30 seconds average to resolve validation errors
- **Feature Adoption**: > 50% of strategies use custom code editor
- **User Satisfaction**: > 4.5/5 rating in dashboard feedback

### Security & Risk Metrics
- **Security Violation Rate**: 0 critical security bypasses
- **Risk Limit Breaches**: < 1% of trades exceed risk parameters
- **Alert Response Time**: < 30 seconds for critical alerts
- **System Uptime**: > 99.5% availability for strategy services

This comprehensive plan provides a production-ready strategy builder that balances flexibility with security, enabling users to create, test, and deploy custom trading strategies through an intuitive dashboard interface.