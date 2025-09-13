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
import time
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

from ml.registry.abstract_registry import AbstractRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager
from ml.registry.persistence import StrategyTable


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


class StrategyRegistry(AbstractRegistry):
    """
    Strategy registry with configurable persistence backend.

    Supports both JSON files and PostgreSQL for persistence, making it suitable for both
    development and production environments.

    """

    def __init__(
        self,
        base_path: Path,
        persistence_config: PersistenceConfig | None = None,
    ) -> None:
        """
        Initialize strategy registry with configurable persistence backend.

        Parameters
        ----------
        base_path : Path
            Base directory for storing strategies.
        persistence_config : PersistenceConfig | None
            Persistence configuration. If None, defaults to JSON backend.

        """
        self.base_path = Path(base_path)
        self.strategies_dir = self.base_path / "strategies"
        self.registry_file = self.strategies_dir / "registry.json"

        # Create directories
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

        # Setup persistence
        if persistence_config is None:
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=self.strategies_dir,
            )
        persistence = PersistenceManager(persistence_config)
        super().__init__(persistence)

        # Initialize registry
        if self.backend == BackendType.JSON:
            if not self.registry_file.exists():
                self._save_registry({})
        elif self.backend == BackendType.POSTGRES:
            # PostgreSQL tables are created automatically by SQLAlchemy
            pass

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
        if self.backend == BackendType.JSON:
            registry = self._load_registry()
            registry[manifest.strategy_id] = {
                "manifest_path": str(manifest_path),
                "file_path": str(dest_path),
                "registered_at": manifest.created_at,
            }
            self._save_registry(registry)
        elif self.backend == BackendType.POSTGRES:
            self._save_strategy_to_db(manifest, dest_path)

        # Log audit
        self.log_audit(
            entity_type="strategy",
            entity_id=manifest.strategy_id,
            action="registered",
            changes={"type": manifest.strategy_type.value, "version": manifest.version},
        )

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
        if self.backend == BackendType.JSON:
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
        elif self.backend == BackendType.POSTGRES:
            session = self.persistence.get_session()
            if session is None:
                return None

            try:
                strategy = session.query(StrategyTable).filter_by(strategy_id=strategy_id).first()

                if strategy is None:
                    return None

                return self._db_to_strategy_info(strategy)
            finally:
                session.close()

        return None

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
        if self.backend == BackendType.JSON:
            registry = self._load_registry()
            return strategy_id in registry
        elif self.backend == BackendType.POSTGRES:
            session = self.persistence.get_session()
            if session is None:
                return False

            try:
                exists = (
                    session.query(StrategyTable).filter_by(strategy_id=strategy_id).first()
                    is not None
                )
                return exists
            finally:
                session.close()

        return False

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
            data = self._json_load("registry.json")
            return data or {}
        return {}

    def _save_registry(self, registry: dict[str, Any]) -> None:
        """
        Save registry to file.
        """
        if self.backend == BackendType.JSON:
            self._json_save("registry.json", registry)
        # PostgreSQL doesn't need this as it's saved per operation

    # ------------------------------ health hook ------------------------------
    def _health_snapshot(self) -> tuple[int, float | None]:
        if self.backend == BackendType.JSON:
            registry = self._load_registry()
            if not registry:
                return 0, None
            # Attempt to read manifests for last_modified
            last: float | None = None
            for entry in registry.values():
                try:
                    manifest_path = Path(entry.get("manifest_path", ""))
                    if manifest_path.exists():
                        with open(manifest_path) as f:
                            data = json.load(f)
                            lm = float(data.get("last_modified", 0.0))
                            if last is None or lm > last:
                                last = lm
                except Exception:
                    # Best-effort only
                    continue
            return len(registry), last

        # POSTGRES: count and max(last_modified)
        session = self.persistence.get_session()
        if session is None:
            return 0, None
        try:
            count = session.query(StrategyTable).count()
            latest = (
                session.query(StrategyTable.last_modified)
                .order_by(StrategyTable.last_modified.desc())
                .first()
            )
            if latest and latest[0] is not None:
                return count, latest[0].timestamp()
            return count, None
        finally:
            session.close()

    def _db_to_strategy_info(self, db_strategy: StrategyTable) -> StrategyInfo:
        """
        Convert database strategy to StrategyInfo.

        Parameters
        ----------
        db_strategy : StrategyTable
            Database strategy record

        Returns
        -------
        StrategyInfo
            Strategy information object

        """
        # Parse timeframe_range
        range_str = db_strategy.timeframe_range if db_strategy.timeframe_range else ""
        if range_str:
            parts = range_str.split(",")
            timeframe_range = (parts[0], parts[1] if len(parts) > 1 else "")
        else:
            timeframe_range = ("", "")

        manifest = StrategyManifest(
            strategy_id=cast(str, db_strategy.strategy_id),
            strategy_type=StrategyType(cast(str, db_strategy.strategy_type)),
            version=cast(str, db_strategy.version),
            required_models=cast(list[str] | None, db_strategy.required_models),
            required_features=cast(list[str], db_strategy.required_features) or [],
            suitable_regimes=[
                MarketRegime(r) for r in (cast(list[str], db_strategy.suitable_regimes) or [])
            ],
            instrument_types=cast(list[str], db_strategy.instrument_types) or [],
            timeframe_range=timeframe_range,
            max_position_size=cast(float, db_strategy.max_position_size) or 0.0,
            max_leverage=cast(float, db_strategy.max_leverage) or 0.0,
            max_drawdown=cast(float, db_strategy.max_drawdown) or 0.0,
            stop_loss_type=cast(str, db_strategy.stop_loss_type) or "",
            min_sharpe_ratio=cast(float, db_strategy.min_sharpe_ratio) or 0.0,
            min_win_rate=cast(float, db_strategy.min_win_rate) or 0.0,
            max_correlation_with_portfolio=cast(float, db_strategy.max_correlation_with_portfolio)
            or 0.0,
            parent_strategy_id=db_strategy.parent_strategy_id,
            incompatible_strategies=cast(list[str], db_strategy.incompatible_strategies) or [],
            config_schema=cast(dict[str, str], db_strategy.config_schema) or {},
            default_config=cast(dict[str, Any], db_strategy.default_config) or {},
            backtest_metrics=cast(dict[str, float], db_strategy.backtest_metrics) or {},
            live_metrics=cast(dict[str, float] | None, db_strategy.live_metrics),
            created_at=(
                db_strategy.created_at.timestamp() if db_strategy.created_at else time.time()
            ),
            last_modified=(
                db_strategy.last_modified.timestamp() if db_strategy.last_modified else time.time()
            ),
            author=cast(str, db_strategy.author) or "",
            description=cast(str, db_strategy.description) or "",
        )

        # Extract file path from metadata
        metadata = db_strategy.extra_metadata or {}  # type: ignore[attr-defined]
        file_path = Path(metadata.get("file_path", ""))

        return StrategyInfo(manifest=manifest, file_path=file_path)

    def _save_strategy_to_db(self, manifest: StrategyManifest, file_path: Path) -> None:
        """
        Save strategy to PostgreSQL database.

        Parameters
        ----------
        manifest : StrategyManifest
            Strategy manifest to save
        file_path : Path
            Path to strategy file

        """
        session = self.persistence.get_session()
        if session is None:
            return

        try:
            # Check if strategy exists
            existing = (
                session.query(StrategyTable).filter_by(strategy_id=manifest.strategy_id).first()
            )

            # Convert timeframe_range tuple to string
            timeframe_range_str = (
                ",".join(manifest.timeframe_range) if manifest.timeframe_range else ""
            )

            # Store file path in metadata
            metadata = {"file_path": str(file_path)}

            if existing:
                # Update existing strategy
                existing.strategy_type = manifest.strategy_type.value
                existing.version = manifest.version
                existing.required_models = manifest.required_models
                existing.required_features = manifest.required_features
                existing.suitable_regimes = [r.value for r in manifest.suitable_regimes]
                existing.instrument_types = manifest.instrument_types
                existing.timeframe_range = timeframe_range_str
                existing.max_position_size = manifest.max_position_size
                existing.max_leverage = manifest.max_leverage
                existing.max_drawdown = manifest.max_drawdown
                existing.stop_loss_type = manifest.stop_loss_type
                existing.min_sharpe_ratio = manifest.min_sharpe_ratio
                existing.min_win_rate = manifest.min_win_rate
                existing.max_correlation_with_portfolio = manifest.max_correlation_with_portfolio
                existing.parent_strategy_id = manifest.parent_strategy_id
                existing.incompatible_strategies = manifest.incompatible_strategies
                existing.config_schema = manifest.config_schema
                existing.default_config = manifest.default_config
                existing.backtest_metrics = manifest.backtest_metrics
                existing.live_metrics = manifest.live_metrics
                existing.author = manifest.author
                existing.description = manifest.description
                existing.extra_metadata = metadata  # type: ignore[attr-defined]
            else:
                # Create new strategy
                new_strategy = StrategyTable(
                    strategy_id=manifest.strategy_id,
                    strategy_type=manifest.strategy_type.value,
                    version=manifest.version,
                    required_models=manifest.required_models,
                    required_features=manifest.required_features,
                    suitable_regimes=[r.value for r in manifest.suitable_regimes],
                    instrument_types=manifest.instrument_types,
                    timeframe_range=timeframe_range_str,
                    max_position_size=manifest.max_position_size,
                    max_leverage=manifest.max_leverage,
                    max_drawdown=manifest.max_drawdown,
                    stop_loss_type=manifest.stop_loss_type,
                    min_sharpe_ratio=manifest.min_sharpe_ratio,
                    min_win_rate=manifest.min_win_rate,
                    max_correlation_with_portfolio=manifest.max_correlation_with_portfolio,
                    parent_strategy_id=manifest.parent_strategy_id,
                    incompatible_strategies=manifest.incompatible_strategies,
                    config_schema=manifest.config_schema,
                    default_config=manifest.default_config,
                    backtest_metrics=manifest.backtest_metrics,
                    live_metrics=manifest.live_metrics,
                    author=manifest.author,
                    description=manifest.description,
                    extra_metadata=metadata,  # type: ignore[call-arg]
                )
                session.add(new_strategy)

            session.commit()
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Failed to save strategy to database: {e}") from e
        finally:
            session.close()
