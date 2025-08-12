"""
Strategy Registry for managing trading strategies.

This module provides a registry system for trading strategies with:
- Self-describing manifests
- Performance tracking
- Compatibility checking
- Lineage tracking
- Filtering by various criteria

"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


# =================================================================================================
# Enums
# =================================================================================================


class StrategyType(Enum):
    """
    Strategy type enumeration.
    """

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    ENSEMBLE = "ensemble"
    META = "meta"


class MarketRegime(Enum):
    """
    Market regime enumeration.
    """

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


# =================================================================================================
# Data Classes
# =================================================================================================


@dataclass
class StrategyManifest:
    """
    Self-describing strategy manifest.
    """

    # Identity
    strategy_id: str
    strategy_type: StrategyType
    version: str

    # Requirements
    required_models: list[str] | None
    required_features: list[str]

    # Market conditions
    suitable_regimes: list[MarketRegime]
    instrument_types: list[str]
    timeframe_range: tuple[str, str]

    # Risk parameters
    max_position_size: float
    max_leverage: float
    max_drawdown: float
    stop_loss_type: str

    # Performance constraints
    min_sharpe_ratio: float
    min_win_rate: float
    max_correlation_with_portfolio: float

    # Dependencies
    parent_strategy_id: str | None
    incompatible_strategies: list[str]

    # Configuration
    config_schema: dict[str, str]
    default_config: dict[str, Any]

    # Performance metrics
    backtest_metrics: dict[str, float]
    live_metrics: dict[str, float] | None

    # Metadata
    created_at: float
    last_modified: float
    author: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """
        Convert manifest to dictionary.
        """
        data = asdict(self)
        # Convert enums to strings
        data["strategy_type"] = self.strategy_type.value
        data["suitable_regimes"] = [r.value for r in self.suitable_regimes]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyManifest:
        """
        Create manifest from dictionary.
        """
        # Convert strings back to enums
        data["strategy_type"] = StrategyType(data["strategy_type"])
        data["suitable_regimes"] = [MarketRegime(r) for r in data["suitable_regimes"]]
        # Convert tuple if needed
        if isinstance(data["timeframe_range"], list):
            data["timeframe_range"] = tuple(data["timeframe_range"])
        return cls(**data)


@dataclass
class StrategyInfo:
    """
    Strategy information including manifest and file path.
    """

    manifest: StrategyManifest
    file_path: Path

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.
        """
        return {
            "manifest": self.manifest.to_dict(),
            "file_path": str(self.file_path),
        }


# =================================================================================================
# Strategy Registry
# =================================================================================================


class LocalStrategyRegistry:
    """
    Local file-based strategy registry.
    """

    def __init__(self, base_path: Path) -> None:
        """
        Initialize strategy registry.

        Parameters
        ----------
        base_path : Path
            Base directory for storing strategies.

        """
        self.base_path = Path(base_path)
        self.strategies_dir = self.base_path / "strategies"
        self.registry_file = self.strategies_dir / "registry.json"

        # Create directories
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

        # Initialize registry file if it doesn't exist
        if not self.registry_file.exists():
            self._save_registry({})

    def register_strategy(
        self,
        strategy_path: Path,
        manifest: StrategyManifest,
    ) -> str:
        """
        Register a new strategy.

        Parameters
        ----------
        strategy_path : Path
            Path to the strategy implementation file.
        manifest : StrategyManifest
            Strategy manifest with metadata.

        Returns
        -------
        str
            The strategy ID.

        """
        # Create strategy directory
        strategy_dir = self.strategies_dir / manifest.strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Copy strategy file
        dest_path = strategy_dir / f"{manifest.strategy_id}.py"
        shutil.copy2(strategy_path, dest_path)

        # Save manifest
        manifest_path = strategy_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # Update registry
        registry = self._load_registry()
        registry[manifest.strategy_id] = {
            "manifest_path": str(manifest_path),
            "file_path": str(dest_path),
            "registered_at": manifest.created_at,
        }
        self._save_registry(registry)

        return manifest.strategy_id

    def get_strategy(self, strategy_id: str) -> StrategyInfo | None:
        """
        Get strategy by ID.

        Parameters
        ----------
        strategy_id : str
            The strategy ID.

        Returns
        -------
        StrategyInfo | None
            Strategy information or None if not found.

        """
        registry = self._load_registry()

        if strategy_id not in registry:
            return None

        entry = registry[strategy_id]
        manifest_path = Path(entry["manifest_path"])

        with open(manifest_path) as f:
            manifest_data = json.load(f)

        manifest = StrategyManifest.from_dict(manifest_data)
        file_path = Path(entry["file_path"])

        return StrategyInfo(manifest=manifest, file_path=file_path)

    def is_registered(self, strategy_id: str) -> bool:
        """
        Check if strategy is registered.

        Parameters
        ----------
        strategy_id : str
            The strategy ID.

        Returns
        -------
        bool
            True if registered.

        """
        registry = self._load_registry()
        return strategy_id in registry

    def get_strategies_for_regime(self, regime: MarketRegime) -> list[StrategyInfo]:
        """
        Get strategies suitable for a market regime.

        Parameters
        ----------
        regime : MarketRegime
            The market regime.

        Returns
        -------
        list[StrategyInfo]
            List of suitable strategies.

        """
        result = []
        registry = self._load_registry()

        for strategy_id in registry:
            strategy_info = self.get_strategy(strategy_id)
            if strategy_info and regime in strategy_info.manifest.suitable_regimes:
                result.append(strategy_info)

        return result

    def get_strategies_for_instrument_type(self, instrument_type: str) -> list[StrategyInfo]:
        """
        Get strategies suitable for an instrument type.

        Parameters
        ----------
        instrument_type : str
            The instrument type (e.g., "FX", "EQUITY").

        Returns
        -------
        list[StrategyInfo]
            List of suitable strategies.

        """
        result = []
        registry = self._load_registry()

        for strategy_id in registry:
            strategy_info = self.get_strategy(strategy_id)
            if strategy_info and instrument_type in strategy_info.manifest.instrument_types:
                result.append(strategy_info)

        return result

    def update_live_metrics(
        self,
        strategy_id: str,
        metrics: dict[str, float],
    ) -> None:
        """
        Update live performance metrics for a strategy.

        Parameters
        ----------
        strategy_id : str
            The strategy ID.
        metrics : dict[str, float]
            The performance metrics.

        """
        strategy_info = self.get_strategy(strategy_id)
        if not strategy_info:
            raise ValueError(f"Strategy {strategy_id} not found")

        # Update manifest
        strategy_info.manifest.live_metrics = metrics

        # Save updated manifest
        strategy_dir = self.strategies_dir / strategy_id
        manifest_path = strategy_dir / "manifest.json"

        with open(manifest_path, "w") as f:
            json.dump(strategy_info.manifest.to_dict(), f, indent=2)

    def get_strategies_ranked_by_performance(
        self,
        metric: str,
        use_live_metrics: bool = True,
    ) -> list[StrategyInfo]:
        """
        Get strategies ranked by a performance metric.

        Parameters
        ----------
        metric : str
            The metric to rank by.
        use_live_metrics : bool
            Whether to use live metrics (True) or backtest metrics (False).

        Returns
        -------
        list[StrategyInfo]
            Strategies ranked by the metric (descending).

        """
        strategies = []
        registry = self._load_registry()

        for strategy_id in registry:
            strategy_info = self.get_strategy(strategy_id)
            if strategy_info:
                strategies.append(strategy_info)

        # Sort by metric
        def get_metric_value(info: StrategyInfo) -> float:
            if use_live_metrics and info.manifest.live_metrics:
                return info.manifest.live_metrics.get(metric, 0.0)
            return info.manifest.backtest_metrics.get(metric, 0.0)

        strategies.sort(key=get_metric_value, reverse=True)

        return strategies

    def validate_requirements(
        self,
        strategy_id: str,
        available_models: list[str],
        available_features: list[str],
    ) -> bool:
        """
        Validate if strategy requirements are met.

        Parameters
        ----------
        strategy_id : str
            The strategy ID.
        available_models : list[str]
            Available model IDs.
        available_features : list[str]
            Available feature names.

        Returns
        -------
        bool
            True if all requirements are met.

        """
        strategy_info = self.get_strategy(strategy_id)
        if not strategy_info:
            return False

        manifest = strategy_info.manifest

        # Check model requirements
        if manifest.required_models:
            for required_model in manifest.required_models:
                if required_model not in available_models:
                    return False

        # Check feature requirements
        for required_feature in manifest.required_features:
            if required_feature not in available_features:
                return False

        return True

    def check_compatibility(
        self,
        strategy_id: str,
        active_strategies: list[str],
    ) -> bool:
        """
        Check if strategy is compatible with active strategies.

        Parameters
        ----------
        strategy_id : str
            The strategy to check.
        active_strategies : list[str]
            Currently active strategy IDs.

        Returns
        -------
        bool
            True if compatible with all active strategies.

        """
        strategy_info = self.get_strategy(strategy_id)
        if not strategy_info:
            return False

        # Check if any active strategy is incompatible
        for active_id in active_strategies:
            if active_id in strategy_info.manifest.incompatible_strategies:
                return False

        return True

    def get_strategy_lineage(self, strategy_id: str) -> list[StrategyInfo]:
        """
        Get strategy lineage (parent chain).

        Parameters
        ----------
        strategy_id : str
            The strategy ID.

        Returns
        -------
        list[StrategyInfo]
            List of strategies from root parent to the given strategy.

        """
        lineage: list[StrategyInfo] = []
        current_id: str | None = strategy_id
        visited: set[str] = set()  # Prevent infinite loops

        # Build lineage from child to parent
        while current_id and current_id not in visited:
            visited.add(current_id)
            strategy_info = self.get_strategy(current_id)

            if not strategy_info:
                break

            lineage.insert(0, strategy_info)  # Insert at beginning
            current_id = strategy_info.manifest.parent_strategy_id

        return lineage

    def _load_registry(self) -> dict[str, Any]:
        """
        Load registry from file.
        """
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data: dict[str, Any] = json.load(f)
                return data
        return {}

    def _save_registry(self, registry: dict[str, Any]) -> None:
        """
        Save registry to file.
        """
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)
