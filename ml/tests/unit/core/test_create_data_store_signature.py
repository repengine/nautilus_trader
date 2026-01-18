"""
Test Suite: Protocol Remediation Task 2.3 - create_data_store Explicit Signature

Verifies that create_data_store uses explicit parameters with proper type annotations
instead of **kwargs: object type erasure.

Coverage Target: ≥90%
Pattern: Protocol-First (Pattern 2)

Test Design Principles (from Task 1.1 success):
1. Verify BEHAVIOR (types not object, function callable) not implementation details
2. Handle runtime variations (Python versions, type aliasing, feature flags)
3. Use flexible assertions (not exact type name matching)
4. All tests initially marked @pytest.mark.skip awaiting implementation

Phase 1: All tests marked @pytest.mark.skip awaiting implementation.
Phase 2: Implementation agent will remove skips and make tests pass.
Phase 4: Validation agent will verify "11 passed" (not "11 collected").
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from typing import get_type_hints

import pytest
from ml.tests.utils.db import build_postgres_url

if TYPE_CHECKING:
    from unittest.mock import Mock


TEST_DB_CONNECTION = build_postgres_url(user="test", password="test", database="test")


class TestCreateDataStoreSignature:
    """Test suite verifying create_data_store has explicit typed parameters."""

    def test_no_kwargs_in_signature(self) -> None:
        """
        Verify function does not use **kwargs: object type erasure pattern.

        The original implementation used **kwargs: object which erases all
        parameter type information. This test verifies the anti-pattern is
        removed by checking the signature has no VAR_KEYWORD parameters.

        Behavior tested: Signature structure (no **kwargs), not exact parameters.

        Expected: Function signature has explicit named parameters only.
        """
        from ml.core.integration import create_data_store

        # Get function signature
        sig = inspect.signature(create_data_store)

        # Verify no VAR_KEYWORD parameters (**kwargs)
        for param_name, param_obj in sig.parameters.items():
            assert param_obj.kind != inspect.Parameter.VAR_KEYWORD, (
                f"Function should not use **kwargs type erasure pattern, "
                f"found parameter '{param_name}' with VAR_KEYWORD kind"
            )

    def test_has_required_parameters(self) -> None:
        """
        Verify function has all required explicit parameters.

        After removing **kwargs, the function should have explicit parameters:
        - registry: DataRegistry
        - connection_string: str
        - feature_store/model_store/strategy_store/earnings_store: optional protocols
        - raw_reader: RawReaderProtocol | None
        - raw_writer: RawIngestionWriterProtocol | None

        Behavior tested: Presence of parameters, not exact types.

        Expected: All 4 parameters present in signature.
        """
        from ml.core.integration import create_data_store

        # Get function signature
        sig = inspect.signature(create_data_store)
        param_names = list(sig.parameters.keys())

        # Verify all required parameters present
        required_params = [
            "registry",
            "connection_string",
            "feature_store",
            "model_store",
            "strategy_store",
            "earnings_store",
            "raw_reader",
            "raw_writer",
        ]

        for param in required_params:
            assert param in param_names, (
                f"Function should have '{param}' parameter, "
                f"found parameters: {param_names}"
            )

    def test_uses_keyword_only_arguments(self) -> None:
        """
        Verify function uses keyword-only arguments (has * separator).

        Keyword-only arguments force callers to use parameter names, preventing
        accidental positional argument errors. This improves API clarity.

        Behavior tested: Signature structure (keyword-only), not exact order.

        Expected: At least one parameter is KEYWORD_ONLY.
        """
        from ml.core.integration import create_data_store

        # Get function signature
        sig = inspect.signature(create_data_store)

        # Check for keyword-only parameters
        keyword_only_params = [
            p
            for p in sig.parameters.values()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]

        assert len(keyword_only_params) > 0, (
            "Function should use keyword-only arguments (signature should have * separator), "
            f"found {len(keyword_only_params)} keyword-only parameters"
        )

    def test_registry_param_type_not_object(self) -> None:
        """
        Verify registry parameter is not generic object type.

        The type should be DataRegistry or similar concrete type, NOT object.
        Uses flexible assertion to handle runtime type aliasing (e.g.,
        DataRegistry → DataRegistryLegacy when feature flags active).

        Behavior tested: Type is concrete (not object), not exact type name.

        Expected: Type contains "Registry" and is not "object".
        """
        from ml.core.integration import create_data_store

        # Get type hints (what mypy sees)
        hints = get_type_hints(create_data_store)

        # Verify registry parameter has type annotation
        assert "registry" in hints, "registry parameter should have type annotation"

        # Get type name (flexible - handles aliasing)
        registry_type = hints["registry"]
        type_name = getattr(registry_type, "__name__", str(registry_type))

        # Flexible assertion 1: NOT object
        assert type_name != "object", (
            f"registry parameter should not be generic object type, got {type_name}"
        )

        # Flexible assertion 2: Registry-related (handles aliasing)
        assert "Registry" in type_name or "DataRegistry" in str(registry_type), (
            f"Expected registry-related type (e.g., DataRegistry), got {type_name}"
        )

    def test_connection_string_param_is_str(self) -> None:
        """
        Verify connection_string parameter has str type.

        This parameter should be exactly str type. Unlike registry parameter,
        str doesn't get aliased, so we can use exact type checking.

        Behavior tested: Exact type is str.

        Expected: Type annotation is str.
        """
        from ml.core.integration import create_data_store

        # Get type hints
        hints = get_type_hints(create_data_store)

        # Verify connection_string parameter has type annotation
        assert "connection_string" in hints, (
            "connection_string parameter should have type annotation"
        )

        # Get type (str is stable - safe to check exactly)
        conn_str_type = hints["connection_string"]

        # str type doesn't vary across Python versions - exact check OK
        assert conn_str_type is str or str(conn_str_type) == "<class 'str'>", (
            f"connection_string should be str type, got {conn_str_type}"
        )

    def test_raw_reader_param_is_optional_protocol(self) -> None:
        """
        Verify raw_reader parameter is optional protocol type.

        Type should be RawReaderProtocol | None (or Optional[RawReaderProtocol]).
        Uses flexible assertion to handle Python 3.11 vs 3.12+ Union syntax.

        Behavior tested: Type is optional protocol, not object.

        Expected: Type references protocol and is optional (has None).
        """
        from ml.core.integration import create_data_store

        # Get type hints
        hints = get_type_hints(create_data_store)

        # Verify raw_reader parameter has type annotation
        assert "raw_reader" in hints, "raw_reader parameter should have type annotation"

        # Get type (handle Union syntax variations)
        reader_type = hints["raw_reader"]
        type_str = str(reader_type)
        type_name = getattr(reader_type, "__name__", type_str)

        # Should NOT be object
        assert type_name != "object", (
            f"raw_reader should not be object type, got {type_name}"
        )

        # Should be optional (Union with None or using | syntax)
        assert "None" in type_str or "Optional" in type_str, (
            f"raw_reader should be optional (have None in type), got {type_str}"
        )

        # Should reference protocol or RawReader
        assert "Protocol" in type_str or "RawReader" in type_str, (
            f"raw_reader should be protocol type (RawReaderProtocol), got {type_str}"
        )

    def test_raw_writer_param_is_optional_protocol(self) -> None:
        """
        Verify raw_writer parameter is optional protocol type.

        Type should be RawIngestionWriterProtocol | None (or Optional).
        Uses flexible assertion to handle Union syntax variations.

        Behavior tested: Type is optional protocol, not object.

        Expected: Type references protocol and is optional (has None).
        """
        from ml.core.integration import create_data_store

        # Get type hints
        hints = get_type_hints(create_data_store)

        # Verify raw_writer parameter has type annotation
        assert "raw_writer" in hints, "raw_writer parameter should have type annotation"

        # Get type (handle Union syntax variations)
        writer_type = hints["raw_writer"]
        type_str = str(writer_type)
        type_name = getattr(writer_type, "__name__", type_str)

        # Should NOT be object
        assert type_name != "object", (
            f"raw_writer should not be object type, got {type_name}"
        )

        # Should be optional (Union with None or using | syntax)
        assert "None" in type_str or "Optional" in type_str, (
            f"raw_writer should be optional (have None in type), got {type_str}"
        )

        # Should reference protocol or writer
        assert "Protocol" in type_str or "Writer" in type_str or "RawIngestion" in type_str, (
            f"raw_writer should be protocol type (RawIngestionWriterProtocol), got {type_str}"
        )

    def test_store_params_are_optional_protocols(self) -> None:
        """
        Verify store parameters are optional protocol types.

        Feature/Model/Strategy/Earnings store parameters should be optional
        protocol types, not object, and allow None.
        """
        from ml.core.integration import create_data_store

        hints = get_type_hints(create_data_store)
        store_params = {
            "feature_store": "FeatureStore",
            "model_store": "ModelStore",
            "strategy_store": "StrategyStore",
            "earnings_store": "EarningsStore",
        }

        for param, label in store_params.items():
            assert param in hints, f"{param} parameter should have type annotation"

            param_type = hints[param]
            type_str = str(param_type)
            type_name = getattr(param_type, "__name__", type_str)

            assert type_name != "object", (
                f"{param} should not be object type, got {type_name}"
            )
            assert "None" in type_str or "Optional" in type_str, (
                f"{param} should be optional (have None in type), got {type_str}"
            )
            assert "Protocol" in type_str or label in type_str, (
                f"{param} should be protocol type ({label}), got {type_str}"
            )

    def test_return_type_not_object(self) -> None:
        """
        Verify return type is concrete DataStore (not object or string literal).

        The original used unsafe string literal cast. After refactoring, return
        type should be DataStore (or related type, handling aliasing).

        Behavior tested: Return type is concrete Store, not object.

        Expected: Type contains "Store" and is not "object".
        """
        from ml.core.integration import create_data_store

        # Get type hints
        hints = get_type_hints(create_data_store)

        # Verify return annotation exists
        assert "return" in hints, "Function should have return type annotation"

        # Get return type (flexible - handles aliasing)
        return_type = hints["return"]
        type_name = getattr(return_type, "__name__", str(return_type))

        # Should NOT be object
        assert type_name != "object", (
            f"return type should not be object, got {type_name}"
        )

        # Should be Store-related (flexible for DataStore/DataStoreLegacy aliasing)
        assert "Store" in type_name or "DataStore" in str(return_type), (
            f"Expected Store-related return type (e.g., DataStore), got {type_name}"
        )


class TestCreateDataStoreRuntime:
    """Test suite verifying create_data_store runtime behavior with typed parameters."""

    def test_call_with_all_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Verify function accepts all typed parameters and executes.

        Tests that after refactoring, the function can be called with all
        parameters (registry, connection_string, store deps, raw_reader/raw_writer)
        and returns a valid result.

        Uses mocks to avoid requiring PostgreSQL database for unit test.

        Behavior tested: Function callable with all parameters.

        Expected: Function returns non-None result without TypeError.
        """
        from unittest.mock import Mock

        from ml.core.integration import create_data_store

        # Create mock dependencies
        mock_registry = Mock(spec=["register_dataset", "get_dataset"])
        mock_reader = Mock(spec=["read_range"])
        mock_writer = Mock(spec=["write"])
        mock_feature_store = Mock()
        mock_model_store = Mock()
        mock_strategy_store = Mock()
        mock_earnings_store = Mock()

        # Mock DataStore construction to avoid database requirement
        from ml.stores.data_store import DataStore
        from ml.stores.data_store import DataStoreConfig

        original_init = DataStore.__init__

        def mock_init(self: DataStore, **kwargs: object) -> None:
            """Mock __init__ to avoid database connection."""
            self._config = DataStoreConfig(
                connection_string=str(kwargs.get("connection_string", "")),
                registry=kwargs.get("registry"),
                feature_store=kwargs.get("feature_store"),
                model_store=kwargs.get("model_store"),
                strategy_store=kwargs.get("strategy_store"),
                earnings_store=kwargs.get("earnings_store"),
                raw_reader=kwargs.get("raw_reader"),
                raw_writer=kwargs.get("raw_writer"),
            )

        monkeypatch.setattr(DataStore, "__init__", mock_init)

        try:
            # Call function with all parameters
            result = create_data_store(
                registry=mock_registry,
                connection_string=TEST_DB_CONNECTION,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
                raw_reader=mock_reader,
                raw_writer=mock_writer,
            )

            # Verify function executed and returned result
            assert result is not None, "Function should return DataStore instance"

            # Verify config captured by mock __init__
            assert result._config.registry is mock_registry
            assert result._config.connection_string == TEST_DB_CONNECTION
        finally:
            # Restore original __init__
            monkeypatch.setattr(DataStore, "__init__", original_init)

    def test_call_with_minimal_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Verify function works with only required parameters (defaults for optional).

        Tests that raw_reader=None and raw_writer=None defaults work correctly.
        This matches the call pattern at line 2058 which only passes registry
        and connection_string.

        Behavior tested: Function callable with minimal parameters.

        Expected: Function returns non-None result with defaults.
        """
        from unittest.mock import Mock

        from ml.core.integration import create_data_store

        # Create mock registry only
        mock_registry = Mock(spec=["register_dataset"])

        # Mock DataStore construction
        from ml.stores.data_store import DataStore
        from ml.stores.data_store import DataStoreConfig

        original_init = DataStore.__init__

        def mock_init(self: DataStore, **kwargs: object) -> None:
            """Mock __init__ to avoid database connection."""
            self._config = DataStoreConfig(
                connection_string=str(kwargs.get("connection_string", "")),
                registry=kwargs.get("registry"),
                feature_store=kwargs.get("feature_store"),
                model_store=kwargs.get("model_store"),
                strategy_store=kwargs.get("strategy_store"),
                earnings_store=kwargs.get("earnings_store"),
                raw_reader=kwargs.get("raw_reader"),
                raw_writer=kwargs.get("raw_writer"),
            )

        monkeypatch.setattr(DataStore, "__init__", mock_init)

        try:
            # Call with only required parameters (defaults for optional)
            result = create_data_store(
                registry=mock_registry,
                connection_string=TEST_DB_CONNECTION,
            )

            # Verify function executed
            assert result is not None, "Function should work with minimal parameters"

            # Verify optional parameters defaulted to None
            assert result._config.feature_store is None, "feature_store should default to None"
            assert result._config.model_store is None, "model_store should default to None"
            assert result._config.strategy_store is None, "strategy_store should default to None"
            assert result._config.earnings_store is None, "earnings_store should default to None"
            assert result._config.raw_reader is None, "raw_reader should default to None"
            assert result._config.raw_writer is None, "raw_writer should default to None"
        finally:
            # Restore original __init__
            monkeypatch.setattr(DataStore, "__init__", original_init)


class TestCreateDataStoreImportStyle:
    """Test suite verifying create_data_store uses direct import (not dynamic)."""

    def test_no_dynamic_import(self) -> None:
        """
        Verify function uses direct import (not importlib + getattr).

        The original implementation used dynamic import to bypass mypy:
        - import importlib
        - DataStore = getattr(importlib.import_module("ml.stores.data_store"), "DataStore")

        This test verifies the workaround is removed and replaced with direct import:
        - from ml.stores.data_store import DataStore

        Behavior tested: Import style (direct vs dynamic), not exact formatting.

        Expected: Function uses direct import, no importlib/getattr pattern.
        """
        from ml.core.integration import create_data_store

        # Get function source code
        source = inspect.getsource(create_data_store)

        # Should NOT have dynamic import pattern
        assert "importlib" not in source, (
            "Function should not use dynamic import (importlib module). "
            "Use direct import: from ml.stores.data_store import DataStore"
        )

        assert "getattr(importlib" not in source, (
            "Function should not use getattr on module imports. "
            "This pattern bypasses mypy type checking."
        )

        # SHOULD have direct import (flexible - allow variations)
        has_direct_import = (
            "from ml.stores.data_store import DataStore" in source
            or "from ml.stores.data_store import" in source
            or "import DataStore" in source
        )

        assert has_direct_import, (
            "Function should use direct import of DataStore. "
            "Expected pattern: from ml.stores.data_store import DataStore"
        )


class TestCreateDataStoreCompatibility:
    """Test suite verifying backward compatibility with existing call sites."""

    def test_existing_call_sites_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Verify existing call patterns (lines 628-633, 2058) still work.

        Existing call sites:
        - Line 628-633: registry, connection_string, raw_reader, raw_writer
        - Line 2058: registry, connection_string

        Both patterns already use keyword arguments, so signature change should
        be transparent (no breaking changes).

        Behavior tested: Existing call patterns work unchanged.

        Expected: Both call patterns execute without TypeError.
        """
        from unittest.mock import Mock

        from ml.core.integration import create_data_store

        # Create mocks for dependencies
        mock_registry = Mock(spec=["register_dataset", "get_dataset"])
        mock_reader = Mock(spec=["read_range"])
        mock_writer = Mock(spec=["write"])

        # Mock DataStore construction
        from ml.stores.data_store import DataStore
        from ml.stores.data_store import DataStoreConfig

        original_init = DataStore.__init__

        def mock_init(self: DataStore, **kwargs: object) -> None:
            """Mock __init__ to avoid database connection."""
            self._config = DataStoreConfig(
                connection_string=str(kwargs.get("connection_string", "")),
                registry=kwargs.get("registry"),
                feature_store=kwargs.get("feature_store"),
                model_store=kwargs.get("model_store"),
                strategy_store=kwargs.get("strategy_store"),
                earnings_store=kwargs.get("earnings_store"),
                raw_reader=kwargs.get("raw_reader"),
                raw_writer=kwargs.get("raw_writer"),
            )

        monkeypatch.setattr(DataStore, "__init__", mock_init)

        try:
            # Pattern 1: All 4 parameters (line 628-633 style)
            result1 = create_data_store(
                registry=mock_registry,
                connection_string="postgresql://test",
                raw_reader=mock_reader,
                raw_writer=mock_writer,
            )
            assert result1 is not None, "Pattern 1 (all params) should work"

            # Pattern 2: Minimal parameters (line 2058 style)
            result2 = create_data_store(
                registry=mock_registry,
                connection_string="postgresql://test",
            )
            assert result2 is not None, "Pattern 2 (minimal params) should work"

            # Both patterns should return same type
            assert isinstance(result1, type(result2)), (
                "Both call patterns should return same type"
            )
        finally:
            # Restore original __init__
            monkeypatch.setattr(DataStore, "__init__", original_init)
