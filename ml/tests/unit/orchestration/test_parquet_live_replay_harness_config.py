from __future__ import annotations

import pytest

from ml.config.base import AccountMode
from ml.config.base import ExecutionValidationMode
from ml.config.base import ShortEntryPolicy
from ml.config.replay_harness import ActorReplayConfig
from ml.config.replay_harness import StrategyReplayConfig
from ml.orchestration.parquet_live_replay_harness import _build_actor_config
from ml.orchestration.parquet_live_replay_harness import _build_strategy_config
from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskLiquidationConfig
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
)


def test_build_strategy_config_sets_exit_policy_max_holding_ms() -> None:
    strategy = StrategyReplayConfig(max_holding_ms=120_000)
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    cfg = _build_strategy_config(
        strategy_config=strategy,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
        strategy_id="STRAT-1",
    )

    assert cfg.exit_policy_config is not None
    assert cfg.exit_policy_config.max_holding_ms == 120_000
    assert cfg.exit_policy_config.stop_loss_pct == strategy.stop_loss_pct
    assert cfg.exit_policy_config.take_profit_pct == strategy.take_profit_pct


def test_build_strategy_config_includes_risk_config() -> None:
    liquidation = RiskLiquidationConfig(
        enabled=True,
        daily_loss_limit_pct=0.1,
    )
    risk_config = RiskConfig(
        allow_reduce_only_when_halted=False,
        liquidation_config=liquidation,
    )
    strategy = StrategyReplayConfig(risk_config=risk_config)
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    cfg = _build_strategy_config(
        strategy_config=strategy,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
        strategy_id="STRAT-1",
    )

    assert cfg.risk_config is not None
    assert cfg.risk_config.allow_reduce_only_when_halted is False
    assert cfg.risk_config.liquidation_config is not None
    assert cfg.risk_config.liquidation_config.daily_loss_limit_pct == 0.1


def test_build_strategy_config_includes_account_and_validation_mode() -> None:
    strategy = StrategyReplayConfig(
        account_mode=AccountMode.MARGIN,
        short_entry_policy=ShortEntryPolicy.ALLOW,
        execution_validation_mode=ExecutionValidationMode.MARKET,
    )
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    cfg = _build_strategy_config(
        strategy_config=strategy,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
        strategy_id="STRAT-1",
    )

    assert cfg.account_mode is AccountMode.MARGIN
    assert cfg.short_entry_policy is ShortEntryPolicy.ALLOW
    assert cfg.execution_config is not None
    assert cfg.execution_config.validation_mode is ExecutionValidationMode.MARKET


def test_build_actor_config_includes_actor_overrides() -> None:
    actor = ActorReplayConfig(
        publish_signals=False,
        log_predictions=True,
        signal_strategy="momentum",
        min_signal_separation_bars=1,
        use_dummy_stores=True,
    )
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")
    bar_type = BarType(instrument_id, BarSpecification.from_str("1-MINUTE-LAST"))

    cfg = _build_actor_config(
        actor_config=actor,
        model_id="model-1",
        model_path="model.onnx",
        bar_type=bar_type,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
    )

    assert cfg.publish_signals is False
    assert cfg.log_predictions is True
    assert cfg.signal_strategy == "momentum"
    assert cfg.min_signal_separation_bars == 1
    assert cfg.use_dummy_stores is True


def test_build_actor_config_uses_defaults_when_overrides_missing() -> None:
    actor = ActorReplayConfig()
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")
    bar_type = BarType(instrument_id, BarSpecification.from_str("1-MINUTE-LAST"))

    cfg = _build_actor_config(
        actor_config=actor,
        model_id="model-1",
        model_path="model.onnx",
        bar_type=bar_type,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
    )

    assert cfg.signal_strategy == "threshold"
    assert cfg.min_signal_separation_bars == 3


def test_strategy_replay_config_rejects_risk_and_liquidation_config() -> None:
    with pytest.raises(ValueError, match="risk_config and liquidation_config"):
        StrategyReplayConfig(
            risk_config=RiskConfig(),
            liquidation_config=RiskLiquidationConfig(enabled=True),
        )
