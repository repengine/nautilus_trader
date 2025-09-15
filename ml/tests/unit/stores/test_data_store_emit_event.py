from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import EventStatus, Source, Stage
from ml.stores.data_store import DataStore


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


class RegistryMockAdapter:
    """
    Tiny adapter to make a MagicMock behave like a typed RegistryProtocol.

    Converts enum arguments to their `.value` strings before forwarding to the
    underlying mock. Keeps tests decoupled from production-only branches.

    """

    def __init__(self, mock: MagicMock) -> None:
        self._mock = mock

    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._mock.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage.value,
            source=source.value,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status=status.value,
            error=error,
            metadata=metadata,
        )

    def update_watermark(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        # Allow tests to assert watermark behavior if needed
        self._mock.update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=source.value,
            last_success_ns=last_success_ns,
            count=count,
            completeness_pct=completeness_pct,
        )


class TestDataStoreEmitEvent:
    def test_emit_event_attaches_correlation_id_and_normalizes(self) -> None:
        # Arrange
        mock_registry = MagicMock()
        # Avoid DB connections by injecting store mocks
        feature_store = cast(Any, MagicMock())
        model_store = cast(Any, MagicMock())
        strategy_store = cast(Any, MagicMock())
        capture = CapturePublisher()
        store = DataStore(
            connection_string="sqlite:///:memory:",
            registry=RegistryMockAdapter(mock_registry),
            feature_store=feature_store,
            model_store=model_store,
            strategy_store=strategy_store,
            publisher=capture,
            enable_publishing=True,
        )

        dataset_id = "features"
        instrument_id = "EURUSD.SIM"
        run_id = "run_123"
        ts_min = 1000
        ts_max = 2000
        count = 42

        # Act
        store.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status="success",
        )

        # Assert
        assert mock_registry.emit_event.call_count == 1
        kwargs = mock_registry.emit_event.call_args.kwargs
        assert kwargs["dataset_id"] == dataset_id
        assert kwargs["instrument_id"] == instrument_id
        assert kwargs["stage"] == Stage.FEATURE_COMPUTED.value
        assert kwargs["source"] == Source.LIVE.value
        assert kwargs["run_id"] == run_id
        assert kwargs["ts_min"] == ts_min
        assert kwargs["ts_max"] == ts_max
        assert kwargs["count"] == count
        assert kwargs["status"] == "success"
        metadata = cast(dict[str, Any], kwargs["metadata"])
        assert "correlation_id" in metadata and isinstance(metadata["correlation_id"], str)
        # Publisher was invoked with canonical topic
        assert len(capture.calls) == 1
        topic, payload = capture.calls[0]
        assert topic.startswith("ml.features.updated.")
        assert payload["dataset_id"] == dataset_id

    def test_emit_event_respects_external_metadata_without_overwriting_correlation(self) -> None:
        mock_registry = MagicMock()
        feature_store = cast(Any, MagicMock())
        model_store = cast(Any, MagicMock())
        strategy_store = cast(Any, MagicMock())
        capture = CapturePublisher()
        store = DataStore(
            connection_string="sqlite:///:memory:",
            registry=RegistryMockAdapter(mock_registry),
            feature_store=feature_store,
            model_store=model_store,
            strategy_store=strategy_store,
            publisher=capture,
            enable_publishing=True,
        )

        store.emit_event(
            dataset_id="predictions",
            instrument_id="BTCUSDT.BINANCE",
            stage=Stage.PREDICTION_EMITTED,
            source="unknown",  # should normalize to 'live'
            run_id="run_abc",
            ts_min=0,
            ts_max=10,
            count=1,
            status="partial",
            metadata={"custom": 1, "correlation_id": "SHOULD_NOT_WIN"},
        )

        kwargs = mock_registry.emit_event.call_args.kwargs
        assert kwargs["source"] == "live"  # normalized
        meta = kwargs["metadata"]
        assert meta["custom"] == 1
        # correlation_id must be generated by the store
        assert meta["correlation_id"] != "SHOULD_NOT_WIN"
        assert len(capture.calls) == 1
        topic, payload = capture.calls[0]
        assert topic.startswith("ml.models.created.")
