"""Shared typed stubs for unit tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, TYPE_CHECKING, cast

from ml.actors.base import MLSignal
from ml.actors.signal import MLSignalActor
from ml.config.events import EventStatus, Source, Stage
from ml.core.integration import MLIntegrationManager
from ml.core.common.actor_factory import ActorFactoryComponent
from ml.core.common.observability import ObservabilityComponent
from ml.observability.service import ObservabilityService
from ml.registry.dataclasses import (
    DataContract,
    DatasetManifest,
    DatasetType,
    QualityFlag,
    StorageKind,
    ValidationRule,
    ValidationRuleType,
)
from ml.registry.protocols import RegistryProtocol
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.strategies.protocols import OrderExecutorProtocol

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:  # pragma: no cover - typing only imports
    from collections.abc import Mapping

    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.identifiers import (
        ClientOrderId,
        InstrumentId,
        StrategyId,
        TraderId,
    )
    from nautilus_trader.core.uuid import UUID4
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.model.orders import Order


class RegistryTestStub(RegistryProtocol):
    """Simple in-memory implementation of :class:`RegistryProtocol` for tests."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.watermarks: list[dict[str, object]] = []

    def emit_event(
        self,
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
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage,
                "source": source,
                "run_id": run_id,
                "ts_min": ts_min,
                "ts_max": ts_max,
                "count": count,
                "status": status,
                "error": error,
                "metadata": metadata or {},
            },
        )

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.watermarks.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "source": source,
                "last_success_ns": last_success_ns,
                "count": count,
                "completeness_pct": completeness_pct,
            },
        )

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        return DatasetManifest(
            dataset_id=dataset_id,
            dataset_type=DatasetType.FEATURES,
            storage_kind=StorageKind.POSTGRES,
            location="/tmp",
            partitioning={},
            retention_days=1,
            schema={"ts_event": "int64"},
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["ts_event"],
            schema_hash="",
            constraints={},
            lineage=[],
            pipeline_signature="test",
            version="1.0.0",
        )

    def get_contract(self, dataset_id: str) -> DataContract:
        return DataContract(
            contract_id=f"contract-{dataset_id}",
            dataset_id=dataset_id,
            version="1.0.0",
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.MONOTONICITY,
                    field_name="ts_event",
                    parameters={"direction": "increasing"},
                    severity=QualityFlag.FAIL,
                    description="ts_event must increase",
                ),
            ],
        )

    def register_dataset(self, manifest: DatasetManifest) -> str:
        return manifest.dataset_id

    def update_manifest(self, dataset_id: str, changes: dict[str, object]) -> None:
        del dataset_id, changes


@dataclass(slots=True)
class FeatureStoreNoOp:
    """Minimal in-memory store stub satisfying FeatureStore protocol methods."""

    feature_batches: list[list[object]] = field(default_factory=list)

    def write_features(self, *args: object, **kwargs: object) -> None:  # noqa: D401
        del args, kwargs

    def write_batch(self, data: Sequence[object], emit_events: bool = True) -> None:
        """Record feature batches without performing any persistence."""
        del emit_events
        self.feature_batches.append(list(data))


@dataclass(slots=True)
class ModelStoreNoOp:
    def write_prediction(self, *args: object, **kwargs: object) -> None:  # noqa: D401
        del args, kwargs

    def write_batch(
        self,
        data: Sequence[object],
        emit_events: bool = True,
        publish_bus: bool = True,
    ) -> None:
        del data, emit_events, publish_bus


@dataclass(slots=True)
class StrategyStoreNoOp:
    def write_signal(self, *args: object, **kwargs: object) -> None:  # noqa: D401
        del args, kwargs

    def write_batch(
        self,
        data: Sequence[object],
        emit_events: bool = True,
        publish_bus: bool = True,
    ) -> None:
        del data, emit_events, publish_bus


@dataclass(slots=True)
class LoggerStub:
    """Collects log invocations without performing any I/O."""

    records: list[tuple[str, tuple[object, ...], dict[str, object]]] = field(default_factory=list)

    def _capture(self, level: str, *args: object, **kwargs: object) -> None:
        self.records.append((level, args, dict(kwargs)))

    def info(self, *args: object, **kwargs: object) -> None:
        self._capture("info", *args, **kwargs)

    def warning(self, *args: object, **kwargs: object) -> None:
        self._capture("warning", *args, **kwargs)

    def debug(self, *args: object, **kwargs: object) -> None:
        self._capture("debug", *args, **kwargs)


@dataclass(slots=True)
class StrategyDecisionRecord:
    signal: MLSignal
    decision_type: str
    position_size: object
    risk_metrics: Mapping[str, float]
    execution_params: Mapping[str, object]


@dataclass(slots=True)
class StrategyDecisionRecorder:
    """Callable helper to collect strategy decision payloads."""

    records: list[StrategyDecisionRecord] = field(default_factory=list)

    def __call__(
        self,
        *,
        signal: MLSignal,
        decision_type: str,
        position_size: object,
        risk_metrics: Mapping[str, float],
        execution_params: Mapping[str, object],
    ) -> None:
        self.records.append(
            StrategyDecisionRecord(
                signal=signal,
                decision_type=decision_type,
                position_size=position_size,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
            ),
        )


@dataclass(slots=True)
class SignalActorHarness:
    """Typed stub exposing the MLSignalActor surface required by unit tests."""

    _signal_strategy: object
    _signal_config: object
    _config: object
    id: object
    _model_id: str = "ml_model"
    _last_signal_bar: int = 0
    _bars_processed: int = 0
    _prediction_history: list[Any] = field(default_factory=list)
    _confidence_history: list[Any] = field(default_factory=list)
    _adaptive_threshold: float = 0.0
    _market_regime: Any | None = None
    _feature_set_id: str | None = None
    clock: object = field(default_factory=lambda: SimpleNamespace(timestamp_ns=lambda: 0))
    _performance_monitor: Any | None = None
    _signals_generated_metric: Any | None = None
    log: object = field(default_factory=lambda: SimpleNamespace(debug=lambda *args, **kwargs: None))
    _prediction_window: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(1, dtype=np.float32),
    )
    _window_index: int = 0
    _window_count: int = 0
    _model_store: Any | None = field(default_factory=ModelStoreNoOp)
    _strategy_store: Any | None = field(default_factory=StrategyStoreNoOp)
    _data_store: Any | None = None
    _feature_store: Any | None = None
    _indicator_manager: Any | None = None
    _feature_engineer: Any | None = None
    _registry_feature_calculator: Any | None = None
    _feature_buffer: npt.NDArray[np.float32] | None = None
    _last_feature_time_ns: int = 0
    _publish_signal: Callable[[Any], None] = field(default_factory=lambda: lambda _sig: None)

    def as_actor(self) -> MLSignalActor:
        """Return the harness as an MLSignalActor-compatible instance."""
        return cast(MLSignalActor, self)


@dataclass(slots=True)
class StrategyCacheStub:
    """Minimal strategy cache with deterministic fixtures."""

    instrument_id: InstrumentId
    client_order_seed: int = 0

    def instrument(self, inst: InstrumentId) -> Instrument | None:
        if inst != self.instrument_id:
            return None

        class _Instrument:
            def __init__(self, iid: InstrumentId) -> None:
                self.id = iid
                self.venue = iid.venue
                self.size_precision = 6
                self.price_precision = 5

            class _MinQuantity:
                def as_double(self) -> float:
                    return 0.0001

            min_quantity = _MinQuantity()

        return _Instrument(inst)

    def account_for_venue(self, _venue: object) -> object:
        class _Account:
            class _Balance:
                def as_double(self) -> float:
                    return 10_000.0

            def balance_total(self) -> object:
                return self._Balance()

        return _Account()

    def positions_open(self, _venue: object, _instrument_id: InstrumentId) -> list[object]:
        return []

    def quote_tick(self, _instrument_id: InstrumentId) -> object:
        class _Px:
            def __init__(self, value: float) -> None:
                self._value = value

            def as_double(self) -> float:
                return self._value

        class _Tick:
            bid_price = _Px(1.0)
            ask_price = _Px(1.0002)

        return _Tick()

    def trade_tick(self, _instrument_id: InstrumentId) -> object:
        return None

    def client_order_id(self) -> ClientOrderId:
        from nautilus_trader.test_kit.stubs.identifiers import TestIdStubs

        self.client_order_seed += 1
        return TestIdStubs.client_order_id(counter=self.client_order_seed)


@dataclass(slots=True)
class StrategyPortfolioStub:
    """Minimal portfolio facade exposing account lookup."""

    def account(self, _venue: object) -> object:
        class _Account:
            class _Balance:
                def as_double(self) -> float:
                    return 10_000.0

            def balance_total(self) -> object:
                return self._Balance()

        return _Account()


@dataclass(slots=True)
class OrderExecutorStub(OrderExecutorProtocol):
    """Configurable order executor doubles used in strategy tests."""

    on_create: Callable[[OrderSide, Quantity, MLSignal, dict[str, float], Instrument], Order | None] | None = None

    def create_order(
        self,
        side: OrderSide,
        quantity: Quantity,
        signal: MLSignal,
        market_state: dict[str, float],
        instrument: Instrument,
        *,
        trader_id: TraderId | None = None,
        strategy_id: StrategyId | None = None,
        client_order_id: ClientOrderId | None = None,
        init_id: UUID4 | None = None,
        ts_init: int | None = None,
    ) -> Order | None:
        del trader_id, strategy_id, client_order_id, init_id, ts_init
        if self.on_create is not None:
            return self.on_create(side, quantity, signal, market_state, instrument)
        return None


def build_ml_trading_strategy_stub(
    *,
    execute_trades: bool = False,
    decision_recorder: StrategyDecisionRecorder | None = None,
) -> MLTradingStrategy:
    """Construct a lightweight :class:`MLTradingStrategy` test double."""

    class _StrategyShim:
        def __init__(self) -> None:
            self.log = LoggerStub()
            self._config = SimpleNamespace(
                execute_trades=execute_trades,
                serialize_order_intents=False,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                exit_policy_config=None,
            )
            self._active_positions = 0
            self._dry_run_trades = 0
            self.track_performance = False
            self._decision_recorder = decision_recorder or StrategyDecisionRecorder()
            self._pending_exit_metadata = None
            self._resolve_exit_policy_config = MLTradingStrategy._resolve_exit_policy_config.__get__(
                self,
                MLTradingStrategy,
            )
            self._timestamp_ns = MLTradingStrategy._timestamp_ns.__get__(self, MLTradingStrategy)
            self._position_entry_price = MLTradingStrategy._position_entry_price.__get__(
                self,
                MLTradingStrategy,
            )
            self._time_in_trade_ns = MLTradingStrategy._time_in_trade_ns.__get__(
                self,
                MLTradingStrategy,
            )
            self._build_exit_metadata = MLTradingStrategy._build_exit_metadata.__get__(
                self,
                MLTradingStrategy,
            )
            self._exit_side_for_position = MLTradingStrategy._exit_side_for_position.__get__(
                self,
                MLTradingStrategy,
            )
            self._set_exit_intent_metadata = MLTradingStrategy._set_exit_intent_metadata.__get__(
                self,
                MLTradingStrategy,
            )
            self._evaluate_exit_policy = MLTradingStrategy._evaluate_exit_policy.__get__(
                self,
                MLTradingStrategy,
            )

        def _get_current_position(self) -> object:
            return None

        def _calculate_position_size(self) -> object:
            return None

        def _should_reverse_position(self, _current: object, _target: object) -> bool:
            return False

        def _enter_position(self, _side: object, _signal: MLSignal) -> None:
            return None

        def _reverse_position(self, _current: object, _side: object, _signal: MLSignal) -> None:
            return None

        def _persist_strategy_decision(
            self,
            *,
            signal: MLSignal,
            decision_type: str,
            position_size: object,
            risk_metrics: Mapping[str, float],
            execution_params: Mapping[str, object],
        ) -> None:
            self._decision_recorder(
                signal=signal,
                decision_type=decision_type,
                position_size=position_size,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
            )

    return cast(MLTradingStrategy, _StrategyShim())


class _StubEventIngestion:
    """Minimal event ingestion stub for MLIntegrationManager tests."""

    def ingest_events(self, config: object) -> Path:
        out_dir = getattr(config, "out_dir", Path("."))
        target = Path(out_dir) / "events.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
        return target

    def maybe_run_backfill_on_start(self) -> None:
        return None


def build_integration_manager_stub(
    *,
    service: ObservabilityService | None = None,
) -> MLIntegrationManager:
    """Construct an :class:`MLIntegrationManager` bypassing heavy initialization."""

    mgr = cast(MLIntegrationManager, object.__new__(MLIntegrationManager))
    mgr.db_connection = "sqlite://"
    mgr.feature_store = None
    mgr.model_store = None
    mgr.strategy_store = None
    mgr.data_store = None
    mgr._data_store = None
    mgr.feature_registry = RegistryTestStub()
    mgr.model_registry = RegistryTestStub()
    mgr.strategy_registry = RegistryTestStub()
    mgr.data_registry = RegistryTestStub()
    mgr.partition_manager = None
    mgr._observability = ObservabilityComponent(stores=[])
    mgr.observability_service = service or ObservabilityService()
    mgr._event_ingestion = _StubEventIngestion()
    mgr._actor_factory = ActorFactoryComponent(
        db_connection=mgr.db_connection,
        feature_store=mgr.feature_store,
        model_store=mgr.model_store,
        strategy_store=mgr.strategy_store,
        data_store=mgr.data_store,
    )
    mgr._obs_flusher = None
    mgr._obs_async_worker = None
    mgr._obs_stop_event = None
    mgr._obs_thread = None
    return mgr


@dataclass(slots=True)
class DatabentoServiceStub:
    """Provide typed surface compatible with :class:`DatabentoIngestionService`."""

    start_ns: int
    end_ns: int
    frame_factory: Callable[[object], object] | None = None
    requests: list[object] = field(default_factory=list, init=False)
    frames: list[object] = field(default_factory=list, init=False)
    metadata_client: object = field(default_factory=object, init=False)

    def get_available_range_ns(
        self,
        *,
        dataset: str,
        schema: str | None = None,
    ) -> tuple[int | None, int | None]:
        del dataset, schema
        return (self.start_ns, self.end_ns)

    def ingest(
        self,
        request: object,
        *,
        on_chunk: Callable[[object], None] | None = None,
    ) -> list[object]:
        self.requests.append(request)
        frame_obj: object
        if self.frame_factory is not None:
            frame_obj = self.frame_factory(request)
        else:
            try:
                import pandas as _pd

                start = getattr(request, "start", None)
                ts_ns = 0
                if start is not None:
                    ts_ns = int(getattr(start, "timestamp", lambda: 0.0)() * 1_000_000_000)
                frame_obj = _pd.DataFrame({"ts_event": [ts_ns]})
            except Exception:  # pragma: no cover - pandas optional
                frame_obj = []
        self.frames.append(frame_obj)
        if on_chunk is not None:
            try:
                from ml.data.ingest.orchestrator import IngestionChunk, IngestionWindow
            except Exception:
                IngestionChunk = SimpleNamespace  # type: ignore[assignment]
                IngestionWindow = SimpleNamespace  # type: ignore[assignment]

            start = getattr(request, "start", None)
            end = getattr(request, "end", None)
            window = IngestionWindow(start=start, end=end)
            chunk = IngestionChunk(
                symbol=(getattr(request, "symbols", ["stub"])[0]),
                window=window,
                frame=frame_obj,
            )
            on_chunk(chunk)  # type: ignore[arg-type]
        return []
