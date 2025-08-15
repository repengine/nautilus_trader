#!/usr/bin/env python3

from __future__ import annotations

import tempfile
from pathlib import Path

from ml.registry.strategy_registry import MarketRegime
from ml.registry.strategy_registry import StrategyInfo
from ml.registry.strategy_registry import StrategyManifest
from ml.registry.strategy_registry import StrategyRegistry
from ml.registry.strategy_registry import StrategyType


def _write_dummy_strategy(path: Path) -> None:
    path.write_text(
        """
def run():
    return "ok"
""",
    )


def test_strategy_registry_register_and_query() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        reg = StrategyRegistry(base)

        strategy_file = base / "my_strategy.py"
        _write_dummy_strategy(strategy_file)

        manifest = StrategyManifest(
            strategy_id="momentum_v1",
            strategy_type=StrategyType.MOMENTUM,
            version="1.0.0",
            required_models=["student_v1"],
            required_features=["student_features_v1"],
            suitable_regimes=[MarketRegime.TRENDING_UP, MarketRegime.VOLATILE],
            instrument_types=["FX", "CRYPTO"],
            timeframe_range=("1m", "1h"),
            max_position_size=1000.0,
            max_leverage=1.0,
            max_drawdown=0.2,
            stop_loss_type="atr",
            min_sharpe_ratio=0.5,
            min_win_rate=0.45,
            max_correlation_with_portfolio=0.8,
            parent_strategy_id=None,
            incompatible_strategies=[],
            config_schema={"threshold": "float"},
            default_config={"threshold": 0.7},
            backtest_metrics={"sharpe": 1.2},
            live_metrics=None,
            created_at=0.0,
            last_modified=0.0,
            author="test",
            description="test strategy",
        )

        sid = reg.register_strategy(strategy_file, manifest)
        info = reg.get_strategy(sid)
        assert isinstance(info, StrategyInfo)
        assert info.manifest.strategy_type == StrategyType.MOMENTUM

        # Regime filtering
        candidates = reg.get_strategies_for_regime(MarketRegime.TRENDING_UP)
        assert any(s.manifest.strategy_id == sid for s in candidates)

        # Requirements validation
        assert reg.validate_requirements(sid, ["student_v1"], ["student_features_v1"]) is True
        assert reg.validate_requirements(sid, [], []) is False
