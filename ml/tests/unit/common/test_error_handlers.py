"""Tests for error handling utilities."""

import logging

import pytest

from ml.common.error_handlers import (
    db_operation_handler,
    registry_operation_handler,
    with_db_error_handling,
    with_fallback,
)


def test_db_operation_handler_success(caplog: pytest.LogCaptureFixture) -> None:
    """Context manager passes through on success."""
    logger = logging.getLogger(__name__)
    result = None

    with db_operation_handler("test operation", logger):
        result = 42

    assert result == 42
    assert len(caplog.records) == 0


def test_db_operation_handler_error_re_raises(caplog: pytest.LogCaptureFixture) -> None:
    """Context manager logs and re-raises by default."""
    logger = logging.getLogger(__name__)

    with pytest.raises(ValueError, match="test error"):
        with db_operation_handler("test operation", logger):
            raise ValueError("test error")

    assert any("Failed to test operation" in r.message for r in caplog.records)


def test_db_operation_handler_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Context manager returns fallback when re_raise=False."""
    logger = logging.getLogger(__name__)

    with db_operation_handler("test", logger, fallback=[], re_raise=False):
        raise ValueError("error")

    # Should have logged error
    assert any("Failed to test" in r.message for r in caplog.records)


def test_registry_operation_handler_success(caplog: pytest.LogCaptureFixture) -> None:
    """Registry handler passes through on success."""
    logger = logging.getLogger(__name__)
    result = None

    with registry_operation_handler("load", "TestRegistry", logger):
        result = {"data": "value"}

    assert result == {"data": "value"}
    assert len(caplog.records) == 0


def test_registry_operation_handler_no_reraise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Registry handler does not re-raise by default."""
    logger = logging.getLogger(__name__)

    # Should not raise
    with registry_operation_handler("load", "TestRegistry", logger):
        raise ValueError("registry error")

    # Should have logged warning
    assert any("TestRegistry failed to load" in r.message for r in caplog.records)


def test_with_db_error_handling_decorator_success() -> None:
    """Decorator passes through on success."""

    class TestClass:
        logger = logging.getLogger(__name__)

        @with_db_error_handling("test op")
        def method(self) -> int:
            return 42

    obj = TestClass()
    assert obj.method() == 42


def test_with_db_error_handling_decorator_error() -> None:
    """Decorator logs and re-raises by default."""

    class TestClass:
        logger = logging.getLogger(__name__)

        @with_db_error_handling("test op")
        def method(self) -> int:
            raise ValueError("error")

    obj = TestClass()
    with pytest.raises(ValueError):
        obj.method()


def test_with_db_error_handling_fallback() -> None:
    """Decorator returns fallback when re_raise=False."""

    class TestClass:
        logger = logging.getLogger(__name__)

        @with_db_error_handling(fallback_value=[], re_raise=False)
        def method(self) -> list[int]:
            raise ValueError("error")

    obj = TestClass()
    result = obj.method()
    assert result == []


def test_with_db_error_handling_uses_function_name() -> None:
    """Decorator uses function name if operation_name not provided."""

    class TestClass:
        logger = logging.getLogger(__name__)

        @with_db_error_handling(fallback_value=None, re_raise=False)
        def my_custom_method(self) -> None:
            raise ValueError("error")

    obj = TestClass()
    # Should not raise, uses function name
    result = obj.my_custom_method()
    assert result is None


def test_with_fallback_decorator() -> None:
    """Fallback decorator returns fallback on error."""

    class TestClass:
        logger = logging.getLogger(__name__)

        @with_fallback(fallback_value={}, log_level="debug")
        def method(self) -> dict[str, str]:
            raise ValueError("error")

    obj = TestClass()
    result = obj.method()
    assert result == {}


def test_with_fallback_success() -> None:
    """Fallback decorator passes through on success."""

    class TestClass:
        logger = logging.getLogger(__name__)

        @with_fallback(fallback_value={}, log_level="debug")
        def method(self) -> dict[str, str]:
            return {"key": "value"}

    obj = TestClass()
    result = obj.method()
    assert result == {"key": "value"}


def test_with_fallback_log_levels(caplog: pytest.LogCaptureFixture) -> None:
    """Fallback decorator respects log level."""

    class TestClass:
        logger = logging.getLogger(__name__)

        @with_fallback(fallback_value=None, log_level="error")
        def method_error(self) -> None:
            raise ValueError("error")

        @with_fallback(fallback_value=None, log_level="warning")
        def method_warning(self) -> None:
            raise ValueError("warning")

        @with_fallback(fallback_value=None, log_level="info")
        def method_info(self) -> None:
            raise ValueError("info")

    obj = TestClass()

    with caplog.at_level(logging.ERROR):
        obj.method_error()
        assert any(r.levelname == "ERROR" for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        obj.method_warning()
        assert any(r.levelname == "WARNING" for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        obj.method_info()
        assert any(r.levelname == "INFO" for r in caplog.records)


def test_with_fallback_operation_name(caplog: pytest.LogCaptureFixture) -> None:
    """Fallback decorator uses custom operation name."""

    class TestClass:
        logger = logging.getLogger(__name__)

        @with_fallback(fallback_value=None, operation_name="custom operation")
        def method(self) -> None:
            raise ValueError("error")

    obj = TestClass()
    with caplog.at_level(logging.WARNING):
        obj.method()
        assert any("Failed to custom operation" in r.message for r in caplog.records)


def test_error_handlers_preserve_exc_info(caplog: pytest.LogCaptureFixture) -> None:
    """Error handlers include exception info in logs."""
    logger = logging.getLogger(__name__)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            with db_operation_handler("test op", logger):
                raise ValueError("test error")

        # Should have exc_info in log record
        assert any(r.exc_info is not None for r in caplog.records)
