#!/usr/bin/env python3

"""
Persistence layer for registry with configurable backends (JSON or PostgreSQL).

This module provides a unified interface for persisting registry data to either local
JSON files or PostgreSQL database based on configuration.

"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON
from sqlalchemy import TIMESTAMP
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Session
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

from ml.common.db_utils import get_or_create_engine


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm.session import sessionmaker as SessionMaker

Base = declarative_base()


def utcnow() -> datetime:
    """
    Timezone-aware UTC now to avoid deprecation warnings.
    """
    return datetime.now(UTC)


class BackendType(Enum):
    """
    Registry backend type.
    """

    JSON = "json"
    POSTGRES = "postgres"


class ModelTable(Base):
    """
    SQLAlchemy model for model registry.
    """

    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(50), nullable=False, index=True)
    data_requirements = Column(String(50), nullable=False)
    architecture = Column(String(100), nullable=False)
    feature_schema = Column(JSON, nullable=False)
    feature_schema_hash = Column(String(64), nullable=False)
    parent_id = Column(String(255), index=True)
    children_ids: Column[list[str]] = Column(ARRAY(Text))
    training_config: Column[dict[str, Any]] = Column(JSON)
    performance_metrics: Column[dict[str, Any]] = Column(JSON)
    deployment_constraints: Column[dict[str, Any]] = Column(JSON)
    output_schema: Column[dict[str, Any]] = Column(JSON)
    calibration: Column[dict[str, Any]] = Column(JSON)
    deployment_status: Column[str] = Column(String(50), nullable=False)
    deployed_to: Column[list[str]] = Column(ARRAY(Text))
    version = Column(String(50), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=utcnow)
    last_modified = Column(
        TIMESTAMP(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )
    extra_metadata = Column("metadata", JSON)
    model_path = Column(Text, nullable=False)
    performance_history = Column(JSON)
    # Cold-path and feature linkage fields
    serveable = Column(String(5))  # store as 'true'/'false' for portability
    artifact_format = Column(Text)
    feature_set_id = Column(Text)
    pipeline_signature = Column(Text)
    pipeline_version = Column(Text)
    # Artifact integrity field for security
    artifact_sha256_digest = Column(String(64))  # SHA-256 hash is 64 hex characters


class FeatureTable(Base):
    """
    SQLAlchemy model for feature registry.
    """

    __tablename__ = "features"

    id = Column(Integer, primary_key=True)
    feature_set_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    data_requirements = Column(String(50), nullable=False)
    feature_names: Column[list[str]] = Column(ARRAY(Text))
    feature_dtypes: Column[list[str]] = Column(ARRAY(Text))
    schema_hash = Column(String(64), nullable=False)
    pipeline_signature = Column(String(255))
    pipeline_version = Column(String(50))
    capability_flags = Column(JSON)
    constraints = Column(JSON)
    parity_tolerance = Column(Float)
    parity_digest = Column(JSON)
    perf_digest = Column(JSON)
    parent_feature_set_id = Column(String(255))
    stage = Column(String(50), nullable=False, index=True)
    created_at = Column(TIMESTAMP(timezone=True), default=utcnow)
    last_modified = Column(
        TIMESTAMP(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )
    extra_metadata = Column("metadata", JSON)


class StrategyTable(Base):
    """
    SQLAlchemy model for strategy registry.
    """

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(255), unique=True, nullable=False, index=True)
    strategy_type = Column(String(50), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    required_models: Column[list[str]] = Column(ARRAY(Text))
    required_features: Column[list[str]] = Column(ARRAY(Text))
    suitable_regimes: Column[list[str]] = Column(ARRAY(Text))
    instrument_types: Column[list[str]] = Column(ARRAY(Text))
    timeframe_range = Column(String(100))
    max_position_size = Column(Float)
    max_leverage = Column(Float)
    max_drawdown = Column(Float)
    stop_loss_type = Column(String(50))
    min_sharpe_ratio = Column(Float)
    min_win_rate = Column(Float)
    max_correlation_with_portfolio = Column(Float)
    parent_strategy_id = Column(String(255))
    incompatible_strategies: Column[list[str]] = Column(ARRAY(Text))
    config_schema = Column(JSON)
    default_config = Column(JSON)
    backtest_metrics = Column(JSON)
    live_metrics = Column(JSON)
    created_at = Column(TIMESTAMP(timezone=True), default=utcnow)
    last_modified = Column(
        TIMESTAMP(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )
    author = Column(String(255))
    description = Column(Text)


class AuditLogTable(Base):
    """
    SQLAlchemy model for audit logging.
    """

    __tablename__ = "registry_audit_log"

    id = Column(Integer, primary_key=True)
    entity_type = Column(String(50), nullable=False, index=True)
    entity_id = Column(String(255), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    changes = Column(JSON)
    user_id = Column(String(255))
    timestamp = Column(TIMESTAMP(timezone=True), default=utcnow, index=True)


@dataclass(frozen=True)
class PersistenceConfig:
    """
    Configuration for registry persistence.
    """

    backend: BackendType = BackendType.JSON
    connection_string: str | None = None
    json_path: Path | None = None
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False

    def __post_init__(self) -> None:
        """
        Validate and finalize configuration values.
        """
        # apply env default for connection string
        conn = self.connection_string or os.getenv("NAUTILUS_REGISTRY_DB_URL")
        if self.backend == BackendType.POSTGRES and not conn:
            conn = "postgresql://postgres:postgres@localhost:5432/nautilus"
        object.__setattr__(self, "connection_string", conn)
        if self.backend == BackendType.JSON and self.json_path is None:
            raise ValueError("json_path is required for JSON backend")


class PersistenceManager:
    """
    Manages persistence operations for registries.
    """

    def __init__(self, config: PersistenceConfig) -> None:
        """
        Initialize persistence manager.

        Parameters
        ----------
        config : PersistenceConfig
            Persistence configuration

        """
        self.config = config
        self._engine: Engine | None = None
        self._session_factory: SessionMaker[Session] | None = None

        if config.backend == BackendType.POSTGRES:
            self._init_postgres()

    def _init_postgres(self) -> None:
        """
        Initialize PostgreSQL connection.
        """
        if self.config.connection_string is None:
            raise ValueError("Connection string is required for PostgreSQL backend")
        self._engine = get_or_create_engine(
            self.config.connection_string,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            echo=self.config.echo,
        )
        self._session_factory = sessionmaker(bind=self._engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self._engine)

    def get_session(self) -> Session | None:
        """
        Get database session (PostgreSQL only).

        Returns
        -------
        Session | None
            Database session or None for JSON backend

        """
        if self.config.backend == BackendType.POSTGRES and self._session_factory is not None:
            return self._session_factory()
        return None

    def close(self) -> None:
        """
        Close database connections.
        """
        if self._engine:
            self._engine.dispose()

    # JSON operations
    def save_json(self, data: dict[str, Any], filename: str) -> None:
        """
        Save data to JSON file.

        Parameters
        ----------
        data : dict[str, Any]
            Data to save
        filename : str
            JSON filename

        """
        if self.config.backend != BackendType.JSON:
            return

        if self.config.json_path is None:
            return

        filepath = self.config.json_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, filename: str) -> dict[str, Any] | None:
        """
        Load data from JSON file.

        Parameters
        ----------
        filename : str
            JSON filename

        Returns
        -------
        dict[str, Any] | None
            Loaded data or None if file doesn't exist

        """
        if self.config.backend != BackendType.JSON:
            return None

        if self.config.json_path is None:
            return None

        filepath = self.config.json_path / filename
        if not filepath.exists():
            return None

        with open(filepath) as f:
            data: dict[str, Any] = json.load(f)
            return data

    def log_audit(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        changes: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        """
        Log an audit entry.

        Parameters
        ----------
        entity_type : str
            Type of entity (model, feature, strategy)
        entity_id : str
            Entity identifier
        action : str
            Action performed
        changes : dict[str, Any] | None
            Changes made
        user_id : str | None
            User performing the action

        """
        if self.config.backend == BackendType.POSTGRES:
            session = self.get_session()
            if session is not None:
                try:
                    audit_entry = AuditLogTable(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        action=action,
                        changes=changes,
                        user_id=user_id,
                    )
                    session.add(audit_entry)
                    session.commit()
                finally:
                    session.close()
        elif self.config.backend == BackendType.JSON and self.config.json_path is not None:
            # Append to audit log file
            audit_file = self.config.json_path / "audit_log.jsonl"
            audit_entry_dict = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
                "changes": changes,
                "user_id": user_id,
                "timestamp": utcnow().isoformat(),
            }
            with open(audit_file, "a") as f:
                f.write(json.dumps(audit_entry_dict, default=str) + "\n")
