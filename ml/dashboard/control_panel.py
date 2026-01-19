"""
High-level dashboard control helpers with typed integration points.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import Any

from ml.common.logging_config import configure_logging
from ml.core.integration import MLIntegrationManager
from ml.dashboard.control_simple import SimpleControlPanel
from ml.dashboard.services import PipelineIntegrationService
from ml.dashboard.services import PipelineTriggerRequest
from ml.dashboard.services import PipelineTriggerResult
from ml.registry import DataRegistry
from ml.registry import FeatureRegistry
from ml.registry import ModelRegistry
from ml.registry import StrategyRegistry
from ml.stores.protocols import DataStoreFacadeProtocol
from ml.stores.protocols import ModelStoreProtocol
from ml.stores.protocols import StrategyStoreProtocol


try:
    configure_logging()
except Exception:  # pragma: no cover - defensive
    pass

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ControlPanelConfig:
    """
    Configuration flags for dashboard control features.
    """

    enable_actor_control: bool = True
    enable_pipeline_control: bool = True
    enable_ingestion_control: bool = True
    enable_model_deployment: bool = True
    enable_strategy_control: bool = True
    enable_emergency_stop: bool = True
    max_concurrent_actors: int = 10
    max_pipeline_runs: int = 5
    max_backfill_days: int = 90
    db_connection: str | None = None


class DashboardControlPanel:
    """
    Typed control façade used by dashboard service endpoints.
    """

    def __init__(self, config: ControlPanelConfig) -> None:
        self.config = config
        self._state = SimpleControlPanel()
        self._pipeline_tasks: MutableMapping[str, asyncio.Task[None]] = {}
        self._integration = self._init_integration()
        self._pipeline_service: PipelineIntegrationService | None = None
        if self._integration is not None:
            try:
                self._pipeline_service = PipelineIntegrationService(self._integration)
            except Exception:
                logger.debug("Failed to initialise PipelineIntegrationService", exc_info=True)
                self._pipeline_service = None
        self.data_store: DataStoreFacadeProtocol | None = self._get_attr("data_store")
        self.model_store: ModelStoreProtocol | None = self._get_attr("model_store")
        self.strategy_store: StrategyStoreProtocol | None = self._get_attr("strategy_store")
        self.model_registry: ModelRegistry | None = self._get_attr("model_registry")
        self.feature_registry: FeatureRegistry | None = self._get_attr("feature_registry")
        self.strategy_registry: StrategyRegistry | None = self._get_attr("strategy_registry")
        self.data_registry: DataRegistry | None = self._get_attr("data_registry")

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _init_integration(self) -> MLIntegrationManager | None:
        try:
            return MLIntegrationManager(
                db_connection=self.config.db_connection,
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )
        except Exception:
            logger.debug("Failed to initialise MLIntegrationManager", exc_info=True)
            return None

    def _get_attr(self, name: str) -> Any:
        if self._integration is None:
            return None
        return getattr(self._integration, name, None)

    # ------------------------------------------------------------------
    # Actor control
    # ------------------------------------------------------------------
    async def start_actor(
        self,
        actor_id: str,
        actor_type: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Register an actor as started.
        """
        if not self.config.enable_actor_control:
            return {"success": False, "error": "Actor control disabled"}
        if self._state.actor_count() >= self.config.max_concurrent_actors:
            return {"success": False, "error": "Max actors limit reached"}
        return self._state.start_actor(actor_id, actor_type, config)

    async def stop_actor(self, actor_id: str) -> dict[str, Any]:
        if not self.config.enable_actor_control:
            return {"success": False, "error": "Actor control disabled"}
        return self._state.stop_actor(actor_id)

    async def hot_reload_model(self, actor_id: str, model_id: str) -> dict[str, Any]:
        if not self.config.enable_actor_control:
            return {"success": False, "error": "Actor control disabled"}
        return self._state.record_hot_reload(actor_id, model_id)

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------
    async def trigger_pipeline(
        self,
        mode: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.config.enable_pipeline_control:
            return {"success": False, "error": "Pipeline control disabled"}
        if len(self._pipeline_tasks) >= self.config.max_pipeline_runs:
            return {"success": False, "error": "Max pipeline runs limit reached"}
        job_result = await self._trigger_pipeline_via_integration(mode, config)
        job_id = job_result.job_id if job_result else None
        status_text = job_result.status.lower() if job_result else "queued"

        run = self._state.trigger_pipeline(
            mode,
            config=config,
            job_id=job_id,
            status=status_text,
        )
        run_id = str(run.get("run_id", ""))
        if job_result is not None:
            run.update(
                {
                    "success": job_result.success,
                    "pipeline_type": job_result.pipeline_type,
                    "status": job_result.status,
                    "message": job_result.message,
                    "error": job_result.error,
                },
            )
        if run_id:
            self._pipeline_tasks[run_id] = asyncio.create_task(
                self._pipeline_worker(run_id, job_id=job_id),
            )
        return run

    async def _pipeline_worker(self, run_id: str, *, job_id: str | None) -> None:
        try:
            if job_id is None or self._pipeline_service is None:
                await asyncio.sleep(0)
                self._state.set_pipeline_status(run_id, "completed")
                return
            while True:
                await asyncio.sleep(1.0)
                progress = await self._pipeline_service.get_pipeline_progress(job_id)
                status_lower = progress.status.lower()
                status_repr = status_lower
                if progress.status.upper() == "RUNNING" and progress.current_stage:
                    status_repr = f"{status_lower}:{progress.current_stage.lower()}"
                self._state.set_pipeline_status(run_id, status_repr)
                if progress.status.upper() in {"COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"}:
                    break
        except Exception:
            logger.debug("Pipeline worker polling failed", exc_info=True)
        finally:
            self._pipeline_tasks.pop(run_id, None)

    async def _trigger_pipeline_via_integration(
        self,
        pipeline_type: str,
        config: Mapping[str, Any],
    ) -> PipelineTriggerResult | None:
        if self._pipeline_service is None:
            return None
        try:
            request = PipelineTriggerRequest(
                pipeline_type=pipeline_type,
                config=dict(config),
            )
            return await self._pipeline_service.trigger_pipeline(request)
        except Exception:
            logger.debug("Pipeline trigger through integration failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Ingestion control
    # ------------------------------------------------------------------
    async def start_ingestion(
        self,
        symbols: list[str],
        source: str = "databento",
    ) -> dict[str, Any]:
        if not self.config.enable_ingestion_control:
            return {"success": False, "error": "Ingestion control disabled"}
        return self._state.start_ingestion(symbols, source)

    async def trigger_backfill(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        if not self.config.enable_ingestion_control:
            return {"success": False, "error": "Ingestion control disabled"}
        days_diff = (end_date - start_date).days
        if days_diff > self.config.max_backfill_days:
            return {
                "success": False,
                "error": f"Backfill range exceeds {self.config.max_backfill_days} days",
            }
        return self._state.trigger_backfill(symbols, start_date, end_date)

    # ------------------------------------------------------------------
    # Model deployment
    # ------------------------------------------------------------------
    async def deploy_model(
        self,
        model_id: str,
        target: str,
        validation_gates: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.config.enable_model_deployment:
            return {"success": False, "error": "Model deployment disabled"}
        if self.model_registry is None:
            return {"success": False, "error": "Model registry unavailable"}
        if validation_gates:
            for gate in validation_gates:
                if not await self._run_validation_gate(model_id, gate):
                    return {"success": False, "error": f"Validation gate failed: {gate}"}
        ok = False
        try:
            ok = self.model_registry.deploy_model(model_id, target)
        except Exception as exc:
            logger.debug("Model deployment failed", exc_info=True)
            return {"success": False, "error": str(exc)}
        if ok and self.model_store is not None:
            try:
                self.model_store.write_batch([])
            except Exception:
                logger.debug("Model store flush after deployment failed", exc_info=True)
        return {
            "success": ok,
            "model_id": model_id,
            "target": target,
            "deployed_at": datetime.now(UTC).isoformat(),
        }

    async def rollback_model(
        self,
        target: str,
        to_version: str | None = None,
    ) -> dict[str, Any]:
        if self.model_registry is None:
            return {"success": False, "error": "Model registry unavailable"}
        history_fn = getattr(self.model_registry, "get_deployment_history", None)
        deploy_fn = getattr(self.model_registry, "deploy_model", None)
        if not callable(history_fn) or not callable(deploy_fn):
            return {"success": False, "error": "Deployment history unavailable"}
        history = history_fn(target)
        if not history or len(history) < 2:
            return {"success": False, "error": "No previous version to rollback to"}
        previous = to_version or getattr(history[-2], "model_id", None)
        if not isinstance(previous, str):
            return {"success": False, "error": "Unable to determine previous deployment"}
        ok = bool(deploy_fn(previous, target))
        return {
            "success": ok,
            "target": target,
            "rolled_back_to": previous,
            "rollback_time": datetime.now(UTC).isoformat(),
        }

    async def configure_strategy(
        self,
        strategy_id: str,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.config.enable_strategy_control:
            return {"success": False, "error": "Strategy control disabled"}
        if self.strategy_registry is None or self.strategy_store is None:
            return {"success": False, "error": "Strategy registry unavailable"}
        get_strategy = getattr(self.strategy_registry, "get_strategy", None)
        update_strategy = getattr(self.strategy_registry, "update_strategy", None)
        if not callable(get_strategy) or not callable(update_strategy):
            return {"success": False, "error": "Strategy registry unavailable"}
        strategy = get_strategy(strategy_id)
        if strategy is None:
            return {"success": False, "error": "Strategy not found"}
        config_obj = getattr(strategy, "config", None)
        if not isinstance(config_obj, dict):
            return {"success": False, "error": "Strategy config not mutable"}
        config_obj.update(dict(parameters))
        updated = bool(update_strategy(strategy_id, strategy))
        try:
            self.strategy_store.write_batch([])
        except Exception:
            logger.debug("Strategy store flush failed", exc_info=True)
        return {
            "success": updated,
            "strategy_id": strategy_id,
            "updated_params": dict(parameters),
            "update_time": datetime.now(UTC).isoformat(),
        }

    async def emergency_stop_all(self) -> dict[str, Any]:
        if not self.config.enable_emergency_stop:
            return {"success": False, "error": "Emergency stop disabled"}
        for task in list(self._pipeline_tasks.values()):
            task.cancel()
        self._pipeline_tasks.clear()
        return self._state.emergency_stop_all()

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------
    def get_system_status(self) -> dict[str, Any]:
        status = self._state.get_system_status()
        status["pipelines"]["active"] = len(self._pipeline_tasks)
        return status

    async def _run_validation_gate(self, model_id: str, gate: str) -> bool:
        if gate == "performance" and self.model_store is not None:
            try:
                perf = self.model_store.get_model_performance(model_id)
            except Exception:
                return False
            sharpe = perf.get("sharpe_ratio", 0.0)
            try:
                sharpe_value = float(sharpe)
            except (TypeError, ValueError):
                return False
            return sharpe_value > 1.0
        if gate == "drift" and self.model_store is not None:
            drift_score = getattr(self.model_store, "get_drift_score", None)
            if callable(drift_score):
                try:
                    score = drift_score(model_id)
                    score_value = float(score)
                except (TypeError, ValueError):
                    return False
                except Exception:
                    return False
                return score_value < 0.1
        if gate == "coverage":
            return True
        logger.warning("Unknown validation gate: %s", gate)
        return True

    @classmethod
    def from_env(cls) -> DashboardControlPanel:
        config = ControlPanelConfig(
            db_connection=os.getenv("ML_DB_CONNECTION"),
        )
        return cls(config)


__all__ = ["ControlPanelConfig", "DashboardControlPanel"]
