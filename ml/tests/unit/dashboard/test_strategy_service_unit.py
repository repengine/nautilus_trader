"""
Unit tests for Strategy Service - comprehensive coverage.

Tests code validation, security sandboxing, backtest submission, deployment,
and performance tracking without requiring external dependencies.
"""

from __future__ import annotations

import pytest

from ml.dashboard.services.strategy_service import (
    BacktestRequest,
    CodeValidationRequest,
    DeploymentRequest,
    StrategyCodeValidator,
    StrategyService,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def validator() -> StrategyCodeValidator:
    """Create a fresh validator instance."""
    return StrategyCodeValidator()


@pytest.fixture
def service() -> StrategyService:
    """Create a strategy service instance."""
    return StrategyService(integration_manager=None)


@pytest.fixture
def valid_strategy_code() -> str:
    """Return valid strategy code for testing."""
    return """
from ml.strategies.base import BaseMLStrategy
from ml.actors.base import MLSignal
from nautilus_trader.model.enums import OrderSide


class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal: MLSignal) -> None:
        self.log.info(f"Processing signal: {signal.prediction}")
        if signal.confidence > 0.7:
            self._enter_position(OrderSide.BUY, signal)
"""


@pytest.fixture
def dangerous_code_eval() -> str:
    """Code with dangerous eval call."""
    return """
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        result = eval("1 + 1")  # DANGEROUS
        return result
"""


@pytest.fixture
def dangerous_code_import() -> str:
    """Code with dangerous import."""
    return """
import os
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        os.system("ls")  # DANGEROUS
"""


@pytest.fixture
def code_with_forbidden_attrs() -> str:
    """Code accessing forbidden attributes."""
    return """
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        x = self.__globals__  # DANGEROUS
"""


@pytest.fixture
def syntax_error_code() -> str:
    """Code with syntax errors."""
    return """
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal) ->  # Missing body
"""


@pytest.fixture
def unauthorized_import_code() -> str:
    """Code with unauthorized imports."""
    return """
import requests  # Not allowed
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        pass
"""


@pytest.fixture
def no_strategy_class_code() -> str:
    """Code without a strategy class."""
    return """
def some_function():
    pass
"""


# ============================================================================
# CODE VALIDATOR TESTS
# ============================================================================


def test_validator_accepts_valid_code(validator: StrategyCodeValidator, valid_strategy_code: str) -> None:
    """Test that validator accepts valid strategy code."""
    result = validator.validate(valid_strategy_code)

    assert result.valid is True
    assert len(result.errors) == 0
    assert result.security_risk is False
    assert result.syntax_error is False


def test_validator_detects_eval_call(validator: StrategyCodeValidator, dangerous_code_eval: str) -> None:
    """Test that validator detects dangerous eval calls."""
    result = validator.validate(dangerous_code_eval)

    assert result.valid is False
    assert result.security_risk is True
    assert any("eval" in str(err).lower() for err in result.errors)


def test_validator_detects_dangerous_import(validator: StrategyCodeValidator, dangerous_code_import: str) -> None:
    """Test that validator detects dangerous imports (os, sys, etc)."""
    result = validator.validate(dangerous_code_import)

    assert result.valid is False
    assert result.security_risk is True
    assert any("os" in str(err).lower() for err in result.errors)


def test_validator_detects_forbidden_attrs(
    validator: StrategyCodeValidator, code_with_forbidden_attrs: str
) -> None:
    """Test that validator detects forbidden attribute access."""
    result = validator.validate(code_with_forbidden_attrs)

    assert result.valid is False
    assert result.security_risk is True
    assert any("__globals__" in str(err) for err in result.errors)


def test_validator_detects_syntax_error(validator: StrategyCodeValidator, syntax_error_code: str) -> None:
    """Test that validator detects syntax errors."""
    result = validator.validate(syntax_error_code)

    assert result.valid is False
    assert result.syntax_error is True
    assert len(result.errors) > 0


def test_validator_detects_unauthorized_import(
    validator: StrategyCodeValidator, unauthorized_import_code: str
) -> None:
    """Test that validator detects unauthorized imports."""
    result = validator.validate(unauthorized_import_code)

    assert result.valid is False
    # Should have security issue for dangerous import
    assert result.security_risk is True


def test_validator_detects_missing_strategy_class(
    validator: StrategyCodeValidator, no_strategy_class_code: str
) -> None:
    """Test that validator detects missing strategy class."""
    result = validator.validate(no_strategy_class_code)

    assert result.valid is False
    assert any("strategy class" in str(err).lower() for err in result.errors)


def test_validator_allows_numpy_and_pandas(validator: StrategyCodeValidator) -> None:
    """Test that numpy and pandas are allowed imports."""
    code = """
import numpy as np
import pandas as pd
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        arr = np.array([1, 2, 3])
        df = pd.DataFrame({"a": [1, 2, 3]})
"""
    result = validator.validate(code)

    assert result.valid is True
    assert not result.security_risk


def test_validator_detects_open_call(validator: StrategyCodeValidator) -> None:
    """Test that file I/O operations are blocked."""
    code = """
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        with open("test.txt", "r") as f:
            data = f.read()
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True
    assert any("file i/o" in str(err).lower() for err in result.errors)


def test_validator_detects_exec(validator: StrategyCodeValidator) -> None:
    """Test that exec is blocked."""
    code = """
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        exec("print('hello')")
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True
    assert any("exec" in str(err).lower() for err in result.errors)


def test_validator_detects_compile(validator: StrategyCodeValidator) -> None:
    """Test that compile is blocked."""
    code = """
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        compile("1 + 1", "<string>", "eval")
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True
    assert any("compile" in str(err).lower() for err in result.errors)


def test_validator_detects_import_call(validator: StrategyCodeValidator) -> None:
    """Test that __import__ is blocked."""
    code = """
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        mod = __import__("os")
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True
    assert any("__import__" in str(err).lower() for err in result.errors)


def test_validator_detects_getattr(validator: StrategyCodeValidator) -> None:
    """Test that getattr is blocked."""
    code = """
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        func = getattr(self, "_some_method")
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True
    assert any("getattr" in str(err).lower() for err in result.errors)


# ============================================================================
# STRATEGY SERVICE TESTS
# ============================================================================


def test_service_get_service_name(service: StrategyService) -> None:
    """Test service name for metrics."""
    assert service.get_service_name() == "strategy_service"


@pytest.mark.asyncio
async def test_service_health_check(service: StrategyService) -> None:
    """Test service health check."""
    health = await service.health_check()

    assert health["service"] == "strategy_service"
    assert health["status"] == "ok"
    assert "active_backtests" in health
    assert "integration_available" in health


def test_service_validate_strategy_code_success(service: StrategyService, valid_strategy_code: str) -> None:
    """Test successful code validation."""
    request = CodeValidationRequest(
        code=valid_strategy_code,
        strategy_name="TestStrategy",
    )

    result = service.validate_strategy_code(request)

    assert result.valid is True
    assert len(result.errors) == 0


def test_service_validate_strategy_code_security_failure(service: StrategyService, dangerous_code_eval: str) -> None:
    """Test code validation with security issues."""
    request = CodeValidationRequest(
        code=dangerous_code_eval,
        strategy_name="DangerousStrategy",
    )

    result = service.validate_strategy_code(request)

    assert result.valid is False
    assert result.security_risk is True


def test_service_validate_strategy_code_syntax_error(service: StrategyService, syntax_error_code: str) -> None:
    """Test code validation with syntax errors."""
    request = CodeValidationRequest(
        code=syntax_error_code,
        strategy_name="BrokenStrategy",
    )

    result = service.validate_strategy_code(request)

    assert result.valid is False
    assert result.syntax_error is True


def test_service_submit_backtest_success(service: StrategyService, valid_strategy_code: str) -> None:
    """Test successful backtest submission."""
    request = BacktestRequest(
        strategy_code=valid_strategy_code,
        strategy_name="TestStrategy",
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_balance=100000.0,
        instruments=["EURUSD.SIM"],
        risk_params={},
    )

    result = service.submit_backtest(request)

    assert result.status == "queued"
    assert result.job_id != ""
    assert result.error is None


def test_service_submit_backtest_validation_failure(service: StrategyService, dangerous_code_eval: str) -> None:
    """Test backtest submission with invalid code."""
    request = BacktestRequest(
        strategy_code=dangerous_code_eval,
        strategy_name="DangerousStrategy",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    result = service.submit_backtest(request)

    assert result.status == "failed"
    assert result.error is not None
    assert "validation" in result.error.lower()


def test_service_get_backtest_status_success(service: StrategyService, valid_strategy_code: str) -> None:
    """Test getting backtest status."""
    # Submit backtest
    request = BacktestRequest(
        strategy_code=valid_strategy_code,
        strategy_name="TestStrategy",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    submit_result = service.submit_backtest(request)
    job_id = submit_result.job_id

    # Get status
    result = service.get_backtest_status(job_id)

    assert result is not None
    assert result.job_id == job_id
    assert result.status == "queued"


def test_service_get_backtest_status_not_found(service: StrategyService) -> None:
    """Test getting backtest status for non-existent job."""
    result = service.get_backtest_status("non-existent-job-id")

    assert result is None


def test_service_deploy_strategy_staging(service: StrategyService, valid_strategy_code: str) -> None:
    """Test strategy deployment to staging."""
    request = DeploymentRequest(
        strategy_name="TestStrategy",
        strategy_code=valid_strategy_code,
        environment="staging",
        risk_params={
            "max_position_size": 10000,
            "stop_loss_pct": 0.02,
        },
        instruments=["EURUSD.SIM"],
    )

    result = service.deploy_strategy(request)

    assert result.status == "deployed"
    assert result.environment == "staging"
    assert result.deployment_id != ""
    assert result.monitoring_url is not None
    assert result.error is None


def test_service_deploy_strategy_paper(service: StrategyService, valid_strategy_code: str) -> None:
    """Test strategy deployment to paper trading."""
    request = DeploymentRequest(
        strategy_name="TestStrategy",
        strategy_code=valid_strategy_code,
        environment="paper",
    )

    result = service.deploy_strategy(request)

    assert result.status == "deployed"
    assert result.environment == "paper"


def test_service_deploy_strategy_live_requires_approval(service: StrategyService, valid_strategy_code: str) -> None:
    """Test strategy deployment to live requires approval."""
    request = DeploymentRequest(
        strategy_name="TestStrategy",
        strategy_code=valid_strategy_code,
        environment="live",
    )

    result = service.deploy_strategy(request)

    assert result.status == "pending_approval"
    assert result.environment == "live"
    assert "approval" in result.message.lower()


def test_service_deploy_strategy_validation_failure(service: StrategyService, dangerous_code_eval: str) -> None:
    """Test deployment with invalid code."""
    request = DeploymentRequest(
        strategy_name="DangerousStrategy",
        strategy_code=dangerous_code_eval,
        environment="staging",
    )

    result = service.deploy_strategy(request)

    assert result.status == "failed"
    assert result.error is not None
    assert "validation" in result.error.lower()


def test_service_deploy_strategy_risk_validation_failure(service: StrategyService, valid_strategy_code: str) -> None:
    """Test deployment with excessive risk parameters."""
    request = DeploymentRequest(
        strategy_name="RiskyStrategy",
        strategy_code=valid_strategy_code,
        environment="staging",
        risk_params={
            "max_position_size": 10_000_000,  # Exceeds $1M limit
        },
    )

    result = service.deploy_strategy(request)

    assert result.status == "failed"
    assert result.error is not None
    assert "risk" in result.error.lower()


def test_service_deploy_strategy_invalid_environment(service: StrategyService, valid_strategy_code: str) -> None:
    """Test deployment with invalid environment."""
    request = DeploymentRequest(
        strategy_name="TestStrategy",
        strategy_code=valid_strategy_code,
        environment="invalid",
    )

    result = service.deploy_strategy(request)

    assert result.status == "failed"
    assert "invalid environment" in result.message.lower()


def test_service_get_strategy_performance(service: StrategyService) -> None:
    """Test getting strategy performance."""
    result = service.get_strategy_performance("test-strategy-123")

    assert "strategy_id" in result
    assert result["strategy_id"] == "test-strategy-123"


def test_service_list_strategies(service: StrategyService) -> None:
    """Test listing strategies."""
    result = service.list_strategies()

    assert "strategies" in result
    assert "count" in result
    assert isinstance(result["strategies"], list)


def test_service_validate_risk_parameters_success(service: StrategyService) -> None:
    """Test risk parameter validation with valid parameters."""
    risk_params = {
        "max_position_size": 50000,
        "max_leverage": 2.0,
        "max_drawdown": 0.10,
        "stop_loss_pct": 0.02,
    }

    result = service._validate_risk_parameters(risk_params)

    assert result["valid"] is True
    assert len(result["errors"]) == 0


def test_service_validate_risk_parameters_excessive_position(service: StrategyService) -> None:
    """Test risk validation with excessive position size."""
    risk_params = {
        "max_position_size": 5_000_000,  # Exceeds $1M limit
    }

    result = service._validate_risk_parameters(risk_params)

    assert result["valid"] is False
    assert any("position size" in err.lower() for err in result["errors"])


def test_service_validate_risk_parameters_excessive_leverage(service: StrategyService) -> None:
    """Test risk validation with excessive leverage."""
    risk_params = {
        "max_leverage": 10.0,  # Exceeds 3x limit
    }

    result = service._validate_risk_parameters(risk_params)

    assert result["valid"] is False
    assert any("leverage" in err.lower() for err in result["errors"])


def test_service_validate_risk_parameters_excessive_drawdown(service: StrategyService) -> None:
    """Test risk validation with excessive drawdown."""
    risk_params = {
        "max_drawdown": 0.50,  # Exceeds 20% limit
    }

    result = service._validate_risk_parameters(risk_params)

    assert result["valid"] is False
    assert any("drawdown" in err.lower() for err in result["errors"])


def test_service_validate_risk_parameters_insufficient_stop_loss(service: StrategyService) -> None:
    """Test risk validation with insufficient stop loss."""
    risk_params = {
        "stop_loss_pct": 0.005,  # Below 1% minimum
    }

    result = service._validate_risk_parameters(risk_params)

    assert result["valid"] is False
    assert any("stop loss" in err.lower() for err in result["errors"])


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


def test_validator_empty_code(validator: StrategyCodeValidator) -> None:
    """Test validator with empty code."""
    result = validator.validate("")

    assert result.valid is False
    # Empty code results in missing strategy class, not syntax error
    assert len(result.errors) > 0


def test_validator_whitespace_only(validator: StrategyCodeValidator) -> None:
    """Test validator with whitespace only."""
    result = validator.validate("   \n\n   ")

    assert result.valid is False


def test_service_backtest_empty_code(service: StrategyService) -> None:
    """Test backtest with empty code."""
    request = BacktestRequest(
        strategy_code="",
        strategy_name="EmptyStrategy",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    result = service.submit_backtest(request)

    assert result.status == "failed"
    assert result.error is not None


def test_service_deploy_empty_code(service: StrategyService) -> None:
    """Test deployment with empty code."""
    request = DeploymentRequest(
        strategy_name="EmptyStrategy",
        strategy_code="",
        environment="staging",
    )

    result = service.deploy_strategy(request)

    assert result.status == "failed"
    assert result.error is not None


# ============================================================================
# SECURITY PENETRATION TESTS
# ============================================================================


def test_security_no_subprocess(validator: StrategyCodeValidator) -> None:
    """Test that subprocess module is blocked."""
    code = """
import subprocess
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        subprocess.run(["ls"])
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True
    assert any("subprocess" in str(err).lower() for err in result.errors)


def test_security_no_socket(validator: StrategyCodeValidator) -> None:
    """Test that socket module is blocked."""
    code = """
import socket
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        s = socket.socket()
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True
    assert any("socket" in str(err).lower() for err in result.errors)


def test_security_no_pickle(validator: StrategyCodeValidator) -> None:
    """Test that pickle module is blocked."""
    code = """
import pickle
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        data = pickle.loads(b"data")
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True
    assert any("pickle" in str(err).lower() for err in result.errors)


def test_security_no_urllib(validator: StrategyCodeValidator) -> None:
    """Test that urllib is blocked."""
    code = """
import urllib
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        pass
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True


def test_security_no_requests(validator: StrategyCodeValidator) -> None:
    """Test that requests is blocked."""
    code = """
import requests
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        requests.get("http://example.com")
"""
    result = validator.validate(code)

    assert result.valid is False
    assert result.security_risk is True


def test_security_allowed_nautilus_imports(validator: StrategyCodeValidator) -> None:
    """Test that Nautilus Trader imports are allowed."""
    code = """
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.position import Position
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import Bar
from nautilus_trader.model.events import OrderFilled
from ml.strategies.base import BaseMLStrategy

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        pass
"""
    result = validator.validate(code)

    assert result.valid is True
    assert not result.security_risk


def test_security_allowed_ml_imports(validator: StrategyCodeValidator) -> None:
    """Test that ML module imports are allowed."""
    code = """
from ml.strategies.base import BaseMLStrategy
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.actors.base import MLSignal
from ml.config.base import MLStrategyConfig
from dataclasses import dataclass
from typing import Any
from enum import Enum

class TestStrategy(BaseMLStrategy):
    def _process_ml_signal(self, signal):
        pass
"""
    result = validator.validate(code)

    assert result.valid is True
    assert not result.security_risk
