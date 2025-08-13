#!/usr/bin/env python3

"""
Persistence layer for registry with configurable backends (JSON or PostgreSQL).

This module provides a unified interface for persisting registry data to either
local JSON files or PostgreSQL database based on configuration.

"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import JSON
from sqlalchemy import TIMESTAMP
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker


Base = declarative_base()


class BackendType(Enum):
    """Registry backend type."""

    JSON = "json"
    POSTGRES = "postgres"


class ModelTable(Base):
    """SQLAlchemy model for model registry."""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(50), nullable=False, index=True)
    data_requirements = Column(String(50), nullable=False)
    architecture = Column(String(100), nullable=False)
    feature_schema = Column(JSON, nullable=False)
    feature_schema_hash = Column(String(64), nullable=False)
    parent_id = Column(String(255), index=True)
    children_ids = Column(ARRAY(Text))
    training_config = Column(JSON)
    performance_metrics = Column(JSON)
    deployment_constraints = Column(JSON)
    deployment_status = Column(String(50), nullable=False)
    deployed_to = Column(ARRAY(Text))
    version = Column(String(50), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    last_modified = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)
    model_path = Column(Text, nullable=False)
    performance_history = Column(JSON)


class FeatureTable(Base):
    """SQLAlchemy model for feature registry."""

    __tablename__ = "features"

    id = Column(Integer, primary_key=True)
    feature_set_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    data_requirements = Column(String(50), nullable=False)
    feature_names = Column(ARRAY(Text))
    feature_dtypes = Column(ARRAY(Text))
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
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    last_modified = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)


class StrategyTable(Base):
    """SQLAlchemy model for strategy registry."""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(255), unique=True, nullable=False, index=True)
    strategy_type = Column(String(50), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    required_models = Column(ARRAY(Text))
    required_features = Column(ARRAY(Text))
    suitable_regimes = Column(ARRAY(Text))
    instrument_types = Column(ARRAY(Text))
    timeframe_range = Column(String(100))
    max_position_size = Column(Float)
    max_leverage = Column(Float)
    max_drawdown = Column(Float)
    stop_loss_type = Column(String(50))
    min_sharpe_ratio = Column(Float)
    min_win_rate = Column(Float)
    max_correlation_with_portfolio = Column(Float)
    parent_strategy_id = Column(String(255))
    incompatible_strategies = Column(ARRAY(Text))
    config_schema = Column(JSON)
    default_config = Column(JSON)
    backtest_metrics = Column(JSON)
    live_metrics = Column(JSON)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    last_modified = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    author = Column(String(255))
    description = Column(Text)


class AuditLogTable(Base):
    """SQLAlchemy model for audit logging."""

    __tablename__ = "registry_audit_log"

    id = Column(Integer, primary_key=True)
    entity_type = Column(String(50), nullable=False, index=True)
    entity_id = Column(String(255), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    changes = Column(JSON)
    user_id = Column(String(255))
    timestamp = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, index=True)


class PersistenceConfig:
    """Configuration for registry persistence."""

    def __init__(
        self,
        backend: BackendType = BackendType.JSON,
        connection_string: str | None = None,
        json_path: Path | None = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
    ) -> None:
        """
        Initialize persistence configuration.

        Parameters
        ----------
        backend : BackendType
            Backend type (JSON or POSTGRES)
        connection_string : str | None
            PostgreSQL connection string (required for POSTGRES backend)
        json_path : Path | None
            Path for JSON storage (required for JSON backend)
        pool_size : int
            Database connection pool size (PostgreSQL only)
        max_overflow : int
            Maximum overflow connections (PostgreSQL only)
        echo : bool
            Echo SQL statements for debugging

        """
        self.backend = backend
        self.connection_string = connection_string or os.getenv("NAUTILUS_REGISTRY_DB_URL")
        self.json_path = json_path
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo

        # Validate configuration
        if backend == BackendType.POSTGRES and not self.connection_string:
            # Default to local Nautilus PostgreSQL container
            self.connection_string = "postgresql://postgres:postgres@localhost:5432/nautilus"

        if backend == BackendType.JSON and not json_path:
            msg = "json_path is required for JSON backend"
            raise ValueError(msg)


class PersistenceManager:
    """Manages persistence operations for registries."""

    def __init__(self, config: PersistenceConfig) -> None:
        """
        Initialize persistence manager.

        Parameters
        ----------
        config : PersistenceConfig
            Persistence configuration

        """
        self.config = config
        self._engine = None
        self._session_factory = None

        if config.backend == BackendType.POSTGRES:
            self._init_postgres()

    def _init_postgres(self) -> None:
        """Initialize PostgreSQL connection."""
        self._engine = create_engine(
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
        if self.config.backend == BackendType.POSTGRES:
            return self._session_factory()
        return None

    def close(self) -> None:
        """Close database connections."""
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

        filepath = self.config.json_path / filename
        if not filepath.exists():
            return None

        with open(filepath) as f:
            return json.load(f)

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
        elif self.config.backend == BackendType.JSON:
            # Append to audit log file
            audit_file = self.config.json_path / "audit_log.jsonl"
            audit_entry = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
                "changes": changes,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
            with open(audit_file, "a") as f:
                f.write(json.dumps(audit_entry, default=str) + "\n")