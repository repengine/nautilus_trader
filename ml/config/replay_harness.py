"""
Configuration for the parquet live replay harness.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import msgspec

from ml.config.base import AccountMode
from ml.config.base import ExecutionValidationMode
from ml.config.base import MLFeatureConfig
from ml.config.base import ModelExitConfig
from ml.config.base import ReturnsConfig
from ml.config.base import ShortEntryPolicy
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ml.strategies.risk import RiskConfig
    from ml.strategies.risk import RiskLiquidationConfig


class ActorReplayConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Actor template configuration for the replay harness.

    Parameters
    ----------
    component_id_prefix : str, default "MLSignalActor"
        Prefix used to build per-instrument actor component IDs.
    prediction_threshold : NonNegativeFloat, default 0.5
        Minimum confidence threshold for ML predictions.
    warm_up_period : NonNegativeInt, default 50
        Bars required before emitting predictions.
    publish_signals : bool, default True
        Whether to publish ML signals on the message bus.
    log_predictions : bool, default False
        Whether to log individual predictions.
    signal_strategy : str | None, optional
        Optional signal strategy override (e.g., "threshold", "momentum").
    min_signal_separation_bars : int | None, optional
        Optional minimum bars between signals override.
    feature_config : MLFeatureConfig | None, optional
        Feature configuration overrides.
    feature_set_id : str | None, optional
        Feature registry identifier when using registry-aligned features.
    registry_path : str | None, optional
        Registry root path (used when feature registry overrides are enabled).
    use_registry_features : bool, default False
        Whether to align inference features to a registry manifest.
    db_connection : str | None, optional
        Explicit database connection for store initialization.
    use_dummy_stores : bool, default False
        Whether to use dummy stores (no persistence).
    """

    component_id_prefix: str = "MLSignalActor"
    prediction_threshold: NonNegativeFloat = 0.5
    warm_up_period: NonNegativeInt = 50
    publish_signals: bool = True
    log_predictions: bool = False
    signal_strategy: (
        Literal["threshold", "extremes", "momentum", "ensemble", "adaptive"] | None
    ) = None
    min_signal_separation_bars: PositiveInt | None = None
    feature_config: MLFeatureConfig | None = None
    feature_set_id: str | None = None
    registry_path: str | None = None
    use_registry_features: bool = False
    db_connection: str | None = None
    use_dummy_stores: bool = False

    def __post_init__(self) -> None:
        if not self.component_id_prefix.strip():
            raise ValueError("component_id_prefix must be non-empty")
        if self.use_registry_features and not self.feature_set_id:
            raise ValueError("feature_set_id is required when use_registry_features=True")


class StrategyReplayConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Strategy template configuration for the replay harness.

    Parameters
    ----------
    id_prefix : str, default "MLStrategy"
        Prefix used to build per-instrument strategy IDs.
    position_size_pct : PositiveFloat, default 0.1
        Fractional position size for each trade.
    min_confidence : NonNegativeFloat, default 0.7
        Minimum model confidence for trade decisions.
    max_positions : PositiveInt, default 1
        Maximum number of concurrent positions.
    account_mode : AccountMode, default AccountMode.CASH
        Account mode for short-entry defaults in replay runs.
    short_entry_policy : ShortEntryPolicy | None, optional
        Optional short-entry policy override for replay runs.
    stop_loss_pct : NonNegativeFloat, default 0.02
        Stop loss threshold.
    take_profit_pct : NonNegativeFloat, default 0.04
        Take profit threshold.
    max_holding_ms : NonNegativeInt | None, optional
        Maximum holding time in milliseconds before forcing an exit.
    model_exit_config : ModelExitConfig | None, optional
        Optional model-driven exit configuration.
    persist_all_signals : bool, default False
        Whether to persist HOLD signals as well.
    execute_trades : bool, default False
        Whether to execute trades (required to emit order intents).
    serialize_order_intents : bool, default False
        Whether to serialize order intents to JSONL instead of executing. Intended
        for live safety runs without broker access; in replay/backtest this disables
        simulated fills and exits. Enable quote tick subscriptions for pricing context.
    order_intent_path : str | None, optional
        Explicit path for order intent JSONL outputs.
    subscribe_quote_ticks : bool, default False
        Whether to subscribe to quote ticks for execution market state. Recommended
        for execution validation and for richer intent pricing metadata.
    quote_schema : str | None, optional
        Optional quote schema parameter passed to data clients (e.g., "mbp-1").
    max_quote_age_ms : NonNegativeInt | None, optional
        Maximum quote age in milliseconds allowed for execution market state.
    positions_log_degraded_in_backtest : bool, default False
        Whether to log degraded positions readiness during replay/backtest.
    execution_validation_mode : ExecutionValidationMode | None, optional
        Replay-only execution mode to force marketable orders for fill validation.
    returns_config : ReturnsConfig | None, optional
        Optional returns update configuration for sizing/portfolio volatility.
    use_strategy_store : bool, default True
        Whether to persist strategy decisions to StrategyStore.
    risk_config : RiskConfig | None, optional
        Optional risk configuration override.
    liquidation_config : RiskLiquidationConfig | None, optional
        Optional liquidation config to merge into default risk settings.
    allow_reduce_only_when_halted : bool, default True
        Allow reduce-only orders to bypass risk halts when configured.
    """

    id_prefix: str = "MLStrategy"
    position_size_pct: PositiveFloat = 0.1
    min_confidence: NonNegativeFloat = 0.7
    max_positions: PositiveInt = 1
    account_mode: AccountMode = AccountMode.CASH
    short_entry_policy: ShortEntryPolicy | None = None
    stop_loss_pct: NonNegativeFloat = 0.02
    take_profit_pct: NonNegativeFloat = 0.04
    max_holding_ms: NonNegativeInt | None = None
    model_exit_config: ModelExitConfig | None = None
    persist_all_signals: bool = False
    execute_trades: bool = False
    serialize_order_intents: bool = False
    order_intent_path: str | None = None
    subscribe_quote_ticks: bool = False
    quote_schema: str | None = None
    max_quote_age_ms: NonNegativeInt | None = None
    positions_log_degraded_in_backtest: bool = False
    execution_validation_mode: ExecutionValidationMode | None = None
    returns_config: ReturnsConfig | None = None
    use_strategy_store: bool = True
    risk_config: RiskConfig | None = None
    liquidation_config: RiskLiquidationConfig | None = None
    allow_reduce_only_when_halted: bool = True

    def __post_init__(self) -> None:
        if not self.id_prefix.strip():
            raise ValueError("id_prefix must be non-empty")
        if self.quote_schema is not None and not self.quote_schema.strip():
            raise ValueError("quote_schema must be non-empty when provided")
        if self.risk_config is not None and self.liquidation_config is not None:
            raise ValueError("risk_config and liquidation_config cannot both be set")


class ParquetLiveReplayHarnessConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for the parquet live replay harness.

    Parameters
    ----------
    catalog_path : str
        Path to the parquet data catalog.
    instrument_ids : list[str]
        Instrument identifiers to replay (e.g., ["SPY.XNAS"]).
    model_id : str
        Model identifier for MLSignalActor.
    model_path : str
        ONNX model artifact path.
    bar_spec : str, default "1-MINUTE-LAST"
        Bar specification string used to build BarType.
    start_time : str | int | None, optional
        Optional start time for the replay window.
    end_time : str | int | None, optional
        Optional end time for the replay window.
    run_id : str | None, optional
        Optional run identifier used for logging and output paths.
    output_dir : str | None, optional
        Base output directory for JSONL file stores. When set, the harness writes
        to ``output_dir`` or ``output_dir/run_id`` when ``run_id`` is provided.
    allow_parquet_fallback : bool, default True
        Whether to set ML_TFT_ALLOW_PARQUET_FALLBACK=1 for catalog reads.
    fallback_venue : str, default "SIM"
        Venue used when instrument IDs omit venue segments.
    trader_id : str, default "REPLAY-001"
        Trader ID for the backtest engine.
    engine_log_level : str, default "INFO"
        Log level for the backtest engine.
    starting_balance : PositiveFloat, default 100000.0
        Starting cash balance for each venue.
    actor : ActorReplayConfig
        Actor template configuration.
    strategy : StrategyReplayConfig
        Strategy template configuration.
    """

    catalog_path: str
    instrument_ids: list[str]
    model_id: str
    model_path: str
    bar_spec: str = "1-MINUTE-LAST"
    start_time: str | int | None = None
    end_time: str | int | None = None
    run_id: str | None = None
    output_dir: str | None = None
    allow_parquet_fallback: bool = True
    fallback_venue: str = "SIM"
    trader_id: str = "REPLAY-001"
    engine_log_level: str = "INFO"
    starting_balance: PositiveFloat = 100000.0
    actor: ActorReplayConfig = msgspec.field(default_factory=ActorReplayConfig)
    strategy: StrategyReplayConfig = msgspec.field(default_factory=StrategyReplayConfig)

    def __post_init__(self) -> None:
        if not self.catalog_path.strip():
            raise ValueError("catalog_path must be non-empty")
        if not self.instrument_ids:
            raise ValueError("instrument_ids must be non-empty")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.model_path.strip():
            raise ValueError("model_path must be non-empty")
        if not self.bar_spec.strip():
            raise ValueError("bar_spec must be non-empty")
        if not self.fallback_venue.strip():
            raise ValueError("fallback_venue must be non-empty")
        try:
            from nautilus_trader.model.data import BarSpecification

            _ = BarSpecification.from_str(self.bar_spec)
        except Exception as exc:
            logger.debug(
                "bar_spec_validation_failed",
                exc_info=True,
                extra={"error": str(exc), "bar_spec": self.bar_spec},
            )
            raise ValueError(f"Invalid bar_spec: {self.bar_spec}") from exc


__all__ = [
    "ActorReplayConfig",
    "ParquetLiveReplayHarnessConfig",
    "StrategyReplayConfig",
]
