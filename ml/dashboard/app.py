"""
Flask app factory for the Dashboard API.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any, cast

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.common.metrics_export import CONTENT_TYPE_LATEST
from ml.common.metrics_export import generate_latest
from ml.dashboard.config import DashboardConfig
from ml.dashboard.service import DashboardService


def _status_to_http(status: str, success_code: int = 200) -> int:
    normalized = status.lower()
    if normalized in {"success", "purged"}:
        return success_code
    if normalized == "not_found":
        return 404
    if normalized == "unavailable":
        return 503
    if normalized in {"failed", "error"}:
        return 500
    return 500


def create_app(config: DashboardConfig | None = None) -> Flask:
    """
    Create a Flask application exposing the dashboard API.
    """
    # Configure Flask with static file serving
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    app = Flask(__name__, static_folder=static_folder, static_url_path="/static")
    # Light, idempotent logging configuration for API usage
    logger = logging.getLogger(__name__)
    try:
        configure_logging()
        bind_log_context(component="ml.dashboard.api")
    except Exception:  # pragma: no cover - defensive
        logger.debug("dashboard_logging_config_failed", exc_info=True)
    cfg = config or DashboardConfig.from_env()
    svc = DashboardService.from_config(cfg)
    if cfg.events_poll_interval_seconds > 0.0:
        svc.start_event_polling(cfg.events_poll_interval_seconds)

    @app.teardown_appcontext
    def _shutdown(_: object | None) -> None:  # pragma: no cover - teardown path
        svc.stop_event_polling()

    def _require_token() -> bool:
        """
        Return True when dashboard authentication (if enabled) is satisfied.
        """
        provided = request.headers.get("X-ML-DASHBOARD-TOKEN")
        if not provided:
            auth_header = request.headers.get("Authorization") or ""
            if auth_header.lower().startswith("bearer "):
                provided = auth_header[7:].strip() or None
        return svc.validate_token(provided)

    @app.get("/api/health/system")
    def health_system() -> tuple[Any, int]:
        data = svc.get_system_health()
        return jsonify(data), 200

    @app.get("/api/services")
    def services_list() -> tuple[Any, int]:
        data = svc.list_services()
        return jsonify(data), 200

    @app.post("/api/services/<name>:action")
    def services_action(name: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"start", "stop", "restart"}:
            return jsonify({"error": "invalid_action"}), 400
        res = svc.control_service(name, action)
        # 202 Accepted for async/semi-async operations
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/pipeline/run")
    def pipeline_run() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        pipeline_type_raw = payload.get("pipeline_type") or payload.get("mode") or "full"
        pipeline_type = str(pipeline_type_raw).strip() or "full"

        config_payload = payload.get("config")
        if isinstance(config_payload, Mapping):
            config_dict: Mapping[str, Any] = dict(config_payload)
        else:
            config_dict = {
                key: value
                for key, value in payload.items()
                if key not in {"pipeline_type", "mode", "config"}
            }

        res = svc.trigger_pipeline(pipeline_type, config_dict)
        status_token = str(res.get("status", "ERROR")).upper()
        if status_token == "QUEUED" and res.get("success"):
            status_code = 202
        elif status_token == "UNAVAILABLE":
            status_code = 503
        elif status_token == "INVALID":
            status_code = 400
        elif res.get("success"):
            status_code = 202
        else:
            status_code = 500
        return jsonify(res), status_code

    @app.get("/api/pipeline/jobs")
    def pipeline_jobs_list() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        res = svc.list_pipeline_jobs()
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @app.get("/api/pipeline/jobs/<job_id>")
    def pipeline_job_detail(job_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        res = svc.get_pipeline_job(job_id)
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @app.delete("/api/pipeline/jobs/<job_id>")
    def pipeline_job_purge(job_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        res = svc.purge_pipeline_job(job_id)
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @app.post("/api/orchestrator/<task>")
    def orchestrator_task(task: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        res = svc.trigger_orchestrator_task(task, payload)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/api/training/streaming/state")
    def streaming_training_state() -> tuple[Any, int]:
        data = svc.get_streaming_training_state()
        return jsonify(data), 200

    @app.get("/api/registry/models")
    def registry_models() -> tuple[Any, int]:
        return jsonify(svc.list_models()), 200

    @app.get("/api/registry/models/performance")
    def registry_models_performance() -> tuple[Any, int]:
        """Get all models with performance metrics for dashboard display.

        Returns:
            {
                "models": [
                    {
                        "model_id": "tft-signal-001",
                        "type": "TFT",
                        "daily_pnl": 1234.56,
                        "sharpe": 1.5,
                        "win_rate": 0.55,
                        "status": "deployed"
                    },
                    ...
                ]
            }
        """
        models = svc.list_models_with_performance()
        return jsonify({"models": models}), 200

    @app.get("/api/registry/models/<model_id>/history")
    def registry_model_history(model_id: str) -> tuple[Any, int]:
        limit_raw = request.args.get("limit")
        limit_value: int | None = None
        if limit_raw:
            try:
                limit_value = int(limit_raw)
                if limit_value < 0:
                    raise ValueError
            except Exception:
                return jsonify({"error": "invalid_limit"}), 400
        data = svc.get_model_performance_history(model_id, limit=limit_value)
        return jsonify(data), 200

    @app.get("/api/registry/deployments")
    def registry_deployments() -> tuple[Any, int]:
        return jsonify(svc.list_deployments()), 200

    @app.get("/api/registry/features")
    def registry_features() -> tuple[Any, int]:
        role = request.args.get("role")
        stage = request.args.get("stage")
        return jsonify(svc.list_features(role=role or None, stage=stage or None)), 200

    @app.get("/api/registry/strategies")
    def registry_strategies() -> tuple[Any, int]:
        return jsonify(svc.list_strategies()), 200

    @app.get("/api/registry/datasets")
    def registry_datasets() -> tuple[Any, int]:
        return jsonify(svc.list_datasets()), 200

    @app.get("/api/registry/strategies/<strategy_id>")
    def strategy_details(strategy_id: str) -> tuple[Any, int]:
        data = svc.get_strategy_details(strategy_id)
        if data is None:
            return jsonify({}), 200
        return jsonify(data), 200

    @app.get("/api/registry/strategies/<strategy_id>/compatibility")
    def strategy_compatibility(strategy_id: str) -> tuple[Any, int]:
        active_raw = request.args.get("active", "")
        active = [s.strip() for s in active_raw.split(",") if s.strip()]
        return jsonify(svc.check_strategy_compatibility(strategy_id, active)), 200

    @app.get("/api/registry/features/<feature_set_id>/lineage")
    def feature_lineage(feature_set_id: str) -> tuple[Any, int]:
        return jsonify(svc.get_feature_lineage(feature_set_id)), 200

    @app.get("/api/registry/datasets/watermarks")
    def dataset_watermarks() -> tuple[Any, int]:
        ds = request.args.get("dataset_id")
        if not ds:
            return jsonify([]), 200
        instr = request.args.get("instrument")
        source = request.args.get("source")
        try:
            limit = int(str(request.args.get("limit", "100")))
        except Exception:
            limit = 100
        return (
            jsonify(
                svc.list_watermarks(
                    dataset_id=ds, instrument=instr or None, source=source or None, limit=limit
                )
            ),
            200,
        )

    @app.get("/api/registry/datasets/lineage")
    def dataset_lineage() -> tuple[Any, int]:
        child = request.args.get("child") or None
        parent = request.args.get("parent") or None
        try:
            limit = int(str(request.args.get("limit", "100")))
        except Exception:
            limit = 100
        return jsonify(svc.list_dataset_lineage(child=child, parent=parent, limit=limit)), 200

    @app.post("/api/registry/features/<feature_set_id>:promote")
    def registry_feature_promote(feature_set_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        stage = str(payload.get("stage") or "") or None
        gates = payload.get("gates")
        if gates is not None and not isinstance(gates, list):
            return jsonify({"error": "invalid_gates"}), 400
        res = svc.promote_feature(
            feature_set_id, stage=stage, gates=cast(list[dict[str, Any]] | None, gates)
        )
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/registry/features/<feature_set_id>:deprecate")
    def registry_feature_deprecate(feature_set_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        reason = str(payload.get("reason") or "") or None
        res = svc.deprecate_feature(feature_set_id, reason=reason)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/registry/models/<model_id>:deploy")
    def registry_deploy(model_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        target = str(payload.get("target", "ml_signal_actor")).strip()
        if not target:
            return jsonify({"error": "invalid_target"}), 400
        res = svc.deploy_model(model_id, target)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/registry/models/<model_id>:hot_reload")
    def registry_hot_reload(model_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        target = str(payload.get("target", "ml_signal_actor")).strip()
        if not target:
            return jsonify({"error": "invalid_target"}), 400
        res = svc.hot_reload_model(target=target, new_model_id=model_id)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/registry/deployments:rollback")
    def registry_rollback() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        target = str(payload.get("target", "")).strip()
        to_model_id = str(payload.get("to_model_id", "")).strip()
        if not target or not to_model_id:
            return jsonify({"error": "invalid_params"}), 400
        res = svc.rollback_deployment(target=target, to_model_id=to_model_id)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/api/events")
    def events_list() -> tuple[Any, int]:
        args = request.args
        try:
            limit = int(str(args.get("limit", "100")))
        except Exception:
            limit = 100
        stage = str(args.get("stage", "")) or None
        source = str(args.get("source", "")) or None
        instrument = str(args.get("instrument", "")) or None
        data = svc.list_events(
            limit=limit, stage=stage, source=source, instrument_substr=instrument
        )
        return jsonify(data), 200

    # ===== Control Panel Endpoints =====

    @app.post("/api/control/actors/start")
    def control_start_actor() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_id = str(payload.get("actor_id", "")).strip()
        actor_type = str(payload.get("actor_type", "signal")).strip()
        config = payload.get("config", {})
        if not actor_id:
            return jsonify({"error": "invalid_actor_id"}), 400

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.start_actor(actor_id, actor_type, config)
        return jsonify(result), 202 if result.get("success") else 400

    @app.post("/api/control/actors/stop")
    def control_stop_actor() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_id = str(payload.get("actor_id", "")).strip()
        if not actor_id:
            return jsonify({"error": "invalid_actor_id"}), 400

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.stop_actor(actor_id)
        return jsonify(result), 200

    @app.post("/api/control/pipeline/trigger")
    def control_trigger_pipeline() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        pipeline_type_raw = payload.get("pipeline_type") or payload.get("mode") or "full"
        pipeline_type = str(pipeline_type_raw).strip() or "full"

        config_payload = payload.get("config")
        if isinstance(config_payload, Mapping):
            config_dict: Mapping[str, Any] = dict(config_payload)
        else:
            config_dict = {
                key: value
                for key, value in payload.items()
                if key not in {"pipeline_type", "mode", "config"}
            }

        pipeline_result = svc.trigger_pipeline(pipeline_type, config_dict)

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        control_state = control_panel.trigger_pipeline(
            pipeline_type,
            config_dict,
            job_id=str(pipeline_result.get("job_id")) if pipeline_result.get("job_id") else None,
            status=str(pipeline_result.get("status", "QUEUED")).lower(),
        )

        response_payload = {
            **pipeline_result,
            "control_run_id": control_state["run_id"],
            "control_status": control_state["status"],
        }

        status_token = str(pipeline_result.get("status", "ERROR")).upper()
        if status_token == "QUEUED" and pipeline_result.get("success"):
            status_code = 202
        elif status_token == "UNAVAILABLE":
            status_code = 503
        elif status_token == "INVALID":
            status_code = 400
        elif pipeline_result.get("success"):
            status_code = 202
        else:
            status_code = 500

        return jsonify(response_payload), status_code

    @app.post("/api/control/ingestion/start")
    def control_start_ingestion() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        symbols = payload.get("symbols", [])
        source = str(payload.get("source", "databento")).strip()
        if not symbols:
            return jsonify({"error": "no_symbols"}), 400

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.start_ingestion(symbols, source)
        return jsonify(result), 202 if result.get("success") else 400

    @app.post("/api/control/ingestion/backfill")
    def control_backfill() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        symbols = payload.get("symbols", [])
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        if not symbols or not start_date or not end_date:
            return jsonify({"error": "missing_params"}), 400

        from datetime import datetime

        from ml.dashboard.control_panel import DashboardControlPanel

        control_panel = DashboardControlPanel.from_env()
        import asyncio

        result = asyncio.run(
            control_panel.trigger_backfill(
                symbols,
                datetime.fromisoformat(start_date),
                datetime.fromisoformat(end_date),
            ),
        )
        return jsonify(result), 202 if result.get("success") else 400

    @app.post("/api/control/emergency/stop")
    def control_emergency_stop() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.emergency_stop_all()
        return jsonify(result), 200

    @app.get("/api/control/status")
    def control_system_status() -> tuple[Any, int]:
        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        status = control_panel.get_system_status()
        return jsonify(status), 200

    # ========== METRICS & MONITORING ROUTES (Agent 1) ==========

    @app.get("/api/metrics/snapshot")
    def metrics_snapshot() -> tuple[Any, int]:
        """Get real-time metrics snapshot with KPIs."""
        import asyncio

        from ml.dashboard.services.metrics_service import StoreIntegrationService

        service = StoreIntegrationService(svc._pipeline_integration_manager)
        snapshot = asyncio.run(service.get_metrics_snapshot())

        return jsonify(snapshot.to_dict()), 200

    @app.get("/api/metrics/portfolio")
    def metrics_portfolio() -> tuple[Any, int]:
        """Get portfolio summary with positions."""
        import asyncio

        from ml.dashboard.services.metrics_service import StoreIntegrationService

        service = StoreIntegrationService(svc._pipeline_integration_manager)
        portfolio = asyncio.run(service.get_portfolio_snapshot())

        return jsonify(portfolio.to_dict()), 200

    @app.get("/api/metrics/ingestion")
    def metrics_ingestion() -> tuple[Any, int]:
        """Get data ingestion rates."""
        import asyncio

        from ml.dashboard.services.metrics_service import StoreIntegrationService

        service = StoreIntegrationService(svc._pipeline_integration_manager)
        ingestion = asyncio.run(service.get_ingestion_snapshot())

        return jsonify(ingestion.to_dict()), 200

    @app.get("/api/metrics/experiments")
    def metrics_experiments() -> tuple[Any, int]:
        """Get active experiments status."""
        import asyncio

        from ml.dashboard.services.metrics_service import StoreIntegrationService

        service = StoreIntegrationService(svc._pipeline_integration_manager)
        experiments = asyncio.run(service.get_experiments_snapshot())

        return jsonify({"experiments": experiments}), 200

    # ========== END METRICS & MONITORING ROUTES ==========

    # ========== TRADING ROUTES (Agent 2) ==========

    @app.post("/api/trading/toggle")
    def trading_toggle() -> tuple[Any, int]:
        """Toggle live trading mode."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        import asyncio
        from dataclasses import asdict

        from ml.dashboard.services.trading_service import TradingIntegrationService
        from ml.dashboard.services.trading_service import TradingToggleRequest

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        enable = bool(payload.get("enable", False))
        safety_checks = payload.get("safety_checks")

        toggle_request = TradingToggleRequest(
            enable=enable,
            safety_checks=safety_checks,
        )

        trading_service = TradingIntegrationService(svc._pipeline_integration_manager)
        result = asyncio.run(trading_service.toggle_live_trading(toggle_request))

        return jsonify(asdict(result)), 200 if result.success else 400

    @app.post("/api/trading/emergency")
    def trading_emergency() -> tuple[Any, int]:
        """Emergency stop all trading."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        import asyncio
        from dataclasses import asdict

        from ml.dashboard.services.trading_service import TradingIntegrationService

        trading_service = TradingIntegrationService(svc._pipeline_integration_manager)
        result = asyncio.run(trading_service.emergency_stop())

        return jsonify(asdict(result)), 200 if result.success else 500

    @app.get("/api/trading/health")
    def trading_health() -> tuple[Any, int]:
        """Get trading system health."""
        import asyncio

        from ml.dashboard.services.trading_service import TradingIntegrationService

        trading_service = TradingIntegrationService(svc._pipeline_integration_manager)
        health_data = asyncio.run(trading_service.health_check())

        return jsonify(health_data), 200

    @app.get("/api/trading/market-data")
    def trading_market_data() -> tuple[Any, int]:
        """Get live market data stream."""
        import asyncio
        from dataclasses import asdict

        from ml.dashboard.services.trading_service import TradingIntegrationService

        trading_service = TradingIntegrationService(svc._pipeline_integration_manager)
        metrics = asyncio.run(trading_service.get_trading_metrics())

        return jsonify(asdict(metrics)), 200

    # ========== END TRADING ROUTES ==========

    # ========== MARKET DATA ROUTES ==========

    @app.get("/api/market/tickers")
    def market_tickers() -> tuple[Any, int]:
        """Get latest prices for market symbols.

        Query Parameters:
            symbols: Comma-separated list of symbols (default: SPY,QQQ)

        Returns:
            Dict mapping symbol to price data:
            {
                "SPY": {"price": 452.50, "change_pct": 0.55, "timestamp": "..."},
                "QQQ": null  // if data not available
            }
        """
        import os
        from pathlib import Path

        from ml._imports import HAS_POLARS, pl

        symbols_param = request.args.get("symbols", "SPY,QQQ")
        symbols = [s.strip().upper() for s in symbols_param.split(",") if s.strip()]

        data_dir = Path(os.environ.get("ML_MARKET_DATA_DIR", "/data/catalog"))
        result: dict[str, Any] = {}

        if not HAS_POLARS:
            # Return empty result if polars not available
            for symbol in symbols:
                result[symbol] = None
            return jsonify(result), 200

        for symbol in symbols:
            parquet_path = data_dir / symbol / "l0" / f"{symbol}_ohlcv.parquet"

            if not parquet_path.exists():
                result[symbol] = None
                continue

            try:
                df = pl.read_parquet(str(parquet_path))
                if df.is_empty():
                    result[symbol] = None
                    continue

                # Get the last row
                last_row = df.tail(1)

                # Extract values
                close_col = "close" if "close" in df.columns else None
                open_col = "open" if "open" in df.columns else None
                ts_col = next(
                    (c for c in ("timestamp", "ts_event", "ts") if c in df.columns),
                    None,
                )

                if close_col is None:
                    result[symbol] = None
                    continue

                close_price = float(last_row[close_col][0])
                open_price = float(last_row[open_col][0]) if open_col else close_price

                # Calculate change percentage
                change_pct = (
                    ((close_price - open_price) / open_price * 100) if open_price else 0.0
                )

                # Get timestamp
                timestamp_val = None
                if ts_col:
                    ts_raw = last_row[ts_col][0]
                    if hasattr(ts_raw, "isoformat"):
                        timestamp_val = ts_raw.isoformat()
                    elif isinstance(ts_raw, (int, float)):
                        from datetime import datetime, timezone

                        # Assume nanoseconds
                        timestamp_val = datetime.fromtimestamp(
                            ts_raw / 1_000_000_000, tz=timezone.utc
                        ).isoformat()

                result[symbol] = {
                    "price": round(close_price, 2),
                    "change_pct": round(change_pct, 2),
                    "timestamp": timestamp_val,
                }

            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to read market data for %s: %s", symbol, exc)
                result[symbol] = None

        return jsonify(result), 200

    # ========== END MARKET DATA ROUTES ==========

    # ========== API_EXPLORER ROUTES (Agent 3) ==========

    @app.get("/api/openapi.json")
    def api_openapi_json() -> tuple[Any, int]:
        """Get OpenAPI specification."""
        from ml.dashboard.services.api_explorer_service import APIExplorerService

        api_explorer = APIExplorerService(svc._pipeline_integration_manager, app)
        spec = api_explorer.get_openapi_spec()
        return jsonify(spec), 200

    @app.get("/api/docs")
    def api_docs() -> tuple[str, int]:
        """Get interactive API documentation."""
        from ml.dashboard.services.api_explorer_service import APIExplorerService

        api_explorer = APIExplorerService(svc._pipeline_integration_manager, app)
        html = api_explorer.get_swagger_ui_html()
        return html, 200

    @app.post("/api/explorer/test")
    def api_test_endpoint() -> tuple[Any, int]:
        """Test any dashboard endpoint."""
        from ml.dashboard.services.api_explorer_service import APIExplorerService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        method = str(payload.get("method", "GET")).strip().upper()
        endpoint = str(payload.get("endpoint", "")).strip()
        headers = payload.get("headers")
        body = payload.get("body")

        if not endpoint:
            return jsonify({"error": "missing_endpoint"}), 400

        api_explorer = APIExplorerService(svc._pipeline_integration_manager, app)
        result = api_explorer.test_endpoint(
            method=method, endpoint=endpoint, headers=headers, body=body
        )

        return jsonify(result), 200 if result.get("success") else 400

    # ========== END API_EXPLORER ROUTES ==========

    # ========== TERMINAL ROUTES (Agent 4) ==========

    @app.post("/api/terminal/execute")
    def terminal_execute() -> tuple[Any, int]:
        """Execute terminal command."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        from pathlib import Path

        from ml.dashboard.services.terminal_service import TerminalService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        command = str(payload.get("command", "")).strip()
        user_id = str(payload.get("user_id", "")) or None

        if not command:
            return jsonify({"error": "empty_command"}), 400

        # Initialize service
        history_file = Path("ml_data/terminal_history.json")
        service = TerminalService(svc._pipeline_integration_manager, history_file=history_file)

        # Execute command
        result = service.execute_command(command, user_id=user_id)

        return (
            jsonify(
                {
                    "command": result.command,
                    "output": result.output,
                    "exit_code": result.exit_code,
                    "duration_seconds": result.duration_seconds,
                    "timestamp": result.timestamp,
                    "command_type": result.command_type,
                    "success": result.success,
                    "error": result.error,
                }
            ),
            200 if result.success else 400,
        )

    @app.get("/api/terminal/history")
    def terminal_history() -> tuple[Any, int]:
        """Get command history."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        from pathlib import Path

        from ml.dashboard.services.terminal_service import TerminalService

        # Get limit parameter
        limit_raw = request.args.get("limit")
        limit_value: int | None = None
        if limit_raw:
            try:
                limit_value = int(limit_raw)
                if limit_value < 0:
                    raise ValueError
            except Exception:
                return jsonify({"error": "invalid_limit"}), 400

        # Initialize service
        history_file = Path("ml_data/terminal_history.json")
        service = TerminalService(svc._pipeline_integration_manager, history_file=history_file)

        # Get history
        history = service.get_command_history(limit=limit_value)

        return jsonify({"history": history, "total": len(history)}), 200

    @app.get("/api/settings")
    def terminal_settings_get() -> tuple[Any, int]:
        """Get dashboard settings."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        from pathlib import Path

        from ml.dashboard.services.terminal_service import TerminalService

        # Get section parameter
        section = request.args.get("section") or None

        # Initialize service
        config_file = Path("ml_data/dashboard_settings.json")
        service = TerminalService(svc._pipeline_integration_manager, config_file=config_file)

        # Get settings
        settings = service.get_settings(section=section)

        return jsonify({"settings": settings, "section": section}), 200

    @app.post("/api/settings")
    def terminal_settings_update() -> tuple[Any, int]:
        """Update dashboard settings."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        from pathlib import Path

        from ml.dashboard.services.terminal_service import TerminalService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        section = str(payload.get("section", "")).strip()
        updates = payload.get("updates", {})
        validate = bool(payload.get("validate", True))

        if not section:
            return jsonify({"error": "missing_section"}), 400

        if not isinstance(updates, dict):
            return jsonify({"error": "invalid_updates"}), 400

        # Initialize service
        config_file = Path("ml_data/dashboard_settings.json")
        service = TerminalService(svc._pipeline_integration_manager, config_file=config_file)

        # Update settings
        result = service.update_settings(section, updates, validate=validate)

        if result.get("success"):
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    # ========== END TERMINAL ROUTES ==========

    # ========== ACTORS ROUTES (Agent 5) ==========
    @app.post("/api/actors/deploy")
    def actors_deploy() -> tuple[Any, int]:
        """Deploy new ML actor."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        from ml.dashboard.services.actors_service import ActorDeploymentRequest
        from ml.dashboard.services.actors_service import ActorIntegrationService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        # Build deployment request from payload
        actor_type = payload.get("actor_type", "MLSignalActor")
        config = payload.get("config", {})
        run_id = payload.get("run_id")

        deploy_request = ActorDeploymentRequest(
            actor_type=actor_type,
            config=config,
            run_id=run_id,
        )

        # Execute deployment asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(actor_service.deploy_actor(deploy_request))
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "status": result.status,
            "message": result.message,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @app.post("/api/actors/hot-reload")
    def actors_hot_reload() -> tuple[Any, int]:
        """Hot reload actor model."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        from ml.dashboard.services.actors_service import ActorIntegrationService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        actor_id = payload.get("actor_id")
        new_model_id = payload.get("model_id")

        if not actor_id or not new_model_id:
            return jsonify({"error": "actor_id and model_id are required"}), 400

        # Execute hot reload asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(actor_service.hot_reload_model(actor_id, new_model_id))
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "new_model_id": result.new_model_id,
            "status": result.status,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @app.post("/api/actors/pause")
    def actors_pause() -> tuple[Any, int]:
        """Pause actor."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        from ml.dashboard.services.actors_service import ActorIntegrationService
        from ml.dashboard.services.actors_service import ActorPauseRequest

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        actor_id = payload.get("actor_id")
        if not actor_id:
            return jsonify({"error": "actor_id is required"}), 400

        pause_request = ActorPauseRequest(
            actor_id=actor_id,
            reason=payload.get("reason"),
            metadata=payload.get("metadata"),
        )

        # Execute pause asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(actor_service.pause_actor(pause_request))
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "status": result.status,
            "message": result.message,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @app.post("/api/actors/resume")
    def actors_resume() -> tuple[Any, int]:
        """Resume actor."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        from ml.dashboard.services.actors_service import ActorIntegrationService
        from ml.dashboard.services.actors_service import ActorResumeRequest

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        actor_id = payload.get("actor_id")
        if not actor_id:
            return jsonify({"error": "actor_id is required"}), 400

        resume_request = ActorResumeRequest(
            actor_id=actor_id,
            metadata=payload.get("metadata"),
        )

        # Execute resume asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(actor_service.resume_actor(resume_request))
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "status": result.status,
            "message": result.message,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @app.post("/api/actors/stop")
    def actors_stop() -> tuple[Any, int]:
        """Stop and dispose actor."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        from ml.dashboard.services.actors_service import ActorIntegrationService
        from ml.dashboard.services.actors_service import ActorStopRequest

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        actor_id = payload.get("actor_id")
        if not actor_id:
            return jsonify({"error": "actor_id is required"}), 400

        stop_request = ActorStopRequest(
            actor_id=actor_id,
            force=payload.get("force", False),
            reason=payload.get("reason"),
            metadata=payload.get("metadata"),
        )

        # Execute stop asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(actor_service.stop_actor(stop_request))
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "status": result.status,
            "message": result.message,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @app.get("/api/actors/health")
    def actors_health() -> tuple[Any, int]:
        """Get all actors health."""
        from ml.dashboard.services.actors_service import ActorIntegrationService

        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        # Execute health check asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            snapshot = loop.run_until_complete(actor_service.get_actor_health())
        finally:
            loop.close()

        response_data = {
            "total_actors": snapshot.total_actors,
            "healthy_actors": snapshot.healthy_actors,
            "unhealthy_actors": snapshot.unhealthy_actors,
            "paused_actors": snapshot.paused_actors,
            "actors": snapshot.actors,
        }
        return jsonify(response_data), 200

    # ========== END ACTORS ROUTES ==========

    # ========== PIPELINES ROUTES (Agent 6) ==========
    @app.post("/api/pipeline/build-dataset")
    def pipelines_build_dataset() -> tuple[Any, int]:
        """Build training dataset via pipeline orchestration."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        res = svc.build_dataset_pipeline(config=payload)
        status = res.get("status", "ERROR").upper()
        if status == "QUEUED" and res.get("success"):
            code = 202
        elif status == "UNAVAILABLE":
            code = 503
        elif status == "INVALID":
            code = 400
        elif res.get("success"):
            code = 202
        else:
            code = 500
        return jsonify(res), code

    @app.post("/api/pipeline/train-model")
    def pipelines_train_model() -> tuple[Any, int]:
        """Train ML model via pipeline orchestration."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        res = svc.train_model_pipeline(config=payload)
        status = res.get("status", "ERROR").upper()
        if status == "QUEUED" and res.get("success"):
            code = 202
        elif status == "UNAVAILABLE":
            code = 503
        elif status == "INVALID":
            code = 400
        elif res.get("success"):
            code = 202
        else:
            code = 500
        return jsonify(res), code

    @app.post("/api/pipeline/run-hpo")
    def pipelines_run_hpo() -> tuple[Any, int]:
        """Run hyperparameter optimization via pipeline orchestration."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        res = svc.run_hpo_pipeline(config=payload)
        status = res.get("status", "ERROR").upper()
        if status == "QUEUED" and res.get("success"):
            code = 202
        elif status == "UNAVAILABLE":
            code = 503
        elif status == "INVALID":
            code = 400
        elif res.get("success"):
            code = 202
        else:
            code = 500
        return jsonify(res), code

    @app.get("/api/pipeline/jobs/<job_id>/progress")
    def pipelines_progress(job_id: str) -> tuple[Any, int]:
        """Get pipeline job progress."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        res = svc.get_pipeline_progress(job_id)
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @app.post("/api/pipeline/jobs/<job_id>/cancel")
    def pipelines_cancel(job_id: str) -> tuple[Any, int]:
        """Cancel pipeline job."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        res = svc.cancel_pipeline_job(job_id)
        # Normalize status for case-insensitive comparison
        status = str(res.get("status", "")).upper()
        if res.get("success"):
            code = 200
        elif status == "NOT_FOUND":
            code = 404
        elif status == "UNAVAILABLE":
            code = 503
        else:
            code = 500
        return jsonify(res), code

    # ========== END PIPELINES ROUTES ==========

    # ========== FEATURES ROUTES (Agent 7) ==========
    @app.post("/api/features/designer/generate")
    def features_designer_generate() -> tuple[Any, int]:
        """Generate feature set from designer UI configuration."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})

        # Import service lazily
        from ml.dashboard.services.features_service import FeatureEngineeringService
        from ml.dashboard.services.features_service import FeatureGenerationRequest

        # Build request
        req = FeatureGenerationRequest(
            feature_set_name=str(payload.get("feature_set_name", "")).strip(),
            price_features=bool(payload.get("price_features", False)),
            volume_features=bool(payload.get("volume_features", False)),
            microstructure=bool(payload.get("microstructure", False)),
            order_flow=bool(payload.get("order_flow", False)),
            technical_indicators=payload.get("technical_indicators", []),
            lookback_periods=str(payload.get("lookback_periods", "10,20,50,100,200")),
            custom_code=payload.get("custom_code"),
        )

        # Validate feature set name
        if not req.feature_set_name:
            return jsonify({"success": False, "error": "feature_set_name required"}), 400

        # Execute generation
        integration_manager = svc.get_integration_manager()
        service = FeatureEngineeringService(integration_manager)
        result = asyncio.run(service.generate_features(req))

        # Convert result to dict
        result_dict = {
            "success": result.success,
            "feature_set_id": result.feature_set_id,
            "feature_count": result.feature_count,
            "feature_names": list(result.feature_names),
            "manifest": result.manifest,
            "error": result.error,
            "validation_errors": list(result.validation_errors),
        }

        return jsonify(result_dict), 200 if result.success else 400

    @app.post("/api/features/validate-code")
    def features_validate_code() -> tuple[Any, int]:
        """Validate custom feature code with security analysis."""
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})

        # Import service lazily
        from ml.dashboard.services.features_service import CodeValidationRequest
        from ml.dashboard.services.features_service import FeatureEngineeringService

        # Build request
        req = CodeValidationRequest(
            code=str(payload.get("code", "")).strip(),
            test_execution=bool(payload.get("test_execution", False)),
        )

        if not req.code:
            return jsonify({"valid": False, "errors": ["No code provided"]}), 400

        # Execute validation
        integration_manager = svc.get_integration_manager()
        service = FeatureEngineeringService(integration_manager)
        result = asyncio.run(service.validate_code(req))

        # Convert result to dict
        result_dict = {
            "valid": result.valid,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "security_risk": result.security_risk,
            "syntax_error": result.syntax_error,
            "signature_error": result.signature_error,
        }

        return jsonify(result_dict), 200

    @app.post("/api/features/analyze")
    def features_analyze() -> tuple[Any, int]:
        """Analyze feature importance and correlations."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})

        # Import service lazily
        from ml.dashboard.services.features_service import FeatureAnalysisRequest
        from ml.dashboard.services.features_service import FeatureEngineeringService

        # Build request
        req = FeatureAnalysisRequest(
            feature_set_id=str(payload.get("feature_set_id", "")).strip(),
            method=str(payload.get("method", "shap")),
            limit=int(payload.get("limit", 1000)),
        )

        if not req.feature_set_id:
            return (
                jsonify({"success": False, "error": "feature_set_id required"}),
                400,
            )

        # Execute analysis
        integration_manager = svc.get_integration_manager()
        service = FeatureEngineeringService(integration_manager)
        result = asyncio.run(service.analyze_features(req))

        # Convert result to dict
        result_dict = {
            "success": result.success,
            "total_features": result.total_features,
            "feature_names": list(result.feature_names),
            "avg_correlation": result.avg_correlation,
            "max_correlation": result.max_correlation,
            "feature_importance_method": result.feature_importance_method,
            "top_features": list(result.top_features),
            "data_quality": result.data_quality,
            "error": result.error,
        }

        return jsonify(result_dict), 200 if result.success else 400

    @app.get("/api/features/manifests")
    def features_manifests() -> tuple[Any, int]:
        """List all feature manifests in the registry."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        # Import service lazily
        from ml.dashboard.services.features_service import FeatureEngineeringService

        # Execute list
        integration_manager = svc.get_integration_manager()
        service = FeatureEngineeringService(integration_manager)
        result = asyncio.run(service.list_manifests())

        return jsonify(result), 200 if result.get("success") else 500

    # ========== END FEATURES ROUTES ==========

    # ========== STRATEGIES ROUTES (Agent 8) ==========
    @app.post("/api/strategies")
    def strategies_create() -> tuple[Any, int]:
        """Placeholder endpoint for strategy registry integration."""
        return (
            jsonify({
                "error": "not_implemented",
                "message": "Strategy creation is managed by the strategy registry CLI.",
            }),
            501,
        )

    @app.post("/api/strategies/optimize")
    def strategies_optimize() -> tuple[Any, int]:
        """Placeholder endpoint for future strategy optimization workflows."""
        return (
            jsonify({
                "error": "not_implemented",
                "message": "Strategy optimization is handled by dedicated pipeline jobs.",
            }),
            501,
        )

    @app.post("/api/strategies/validate")
    def strategies_validate() -> tuple[Any, int]:
        """Validate strategy code with security checks."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        code = str(payload.get("code", ""))
        strategy_name = payload.get("strategy_name")
        base_strategy = str(payload.get("base_strategy", "MLTradingStrategy"))

        if not code.strip():
            return jsonify({"error": "empty_code", "valid": False}), 400

        # Import strategy service
        from ml.dashboard.services.strategy_service import CodeValidationRequest
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)

        # Validate code
        validation_request = CodeValidationRequest(
            code=code,
            strategy_name=strategy_name,
            base_strategy=base_strategy,
        )
        result = strategy_svc.validate_strategy_code(validation_request)

        response = {
            "valid": result.valid,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "security_risk": result.security_risk,
            "syntax_error": result.syntax_error,
            "signature_error": result.signature_error,
            "allowed_imports": list(result.allowed_imports),
        }

        return jsonify(response), 200

    @app.post("/api/strategies/backtest")
    def strategies_backtest() -> tuple[Any, int]:
        """Run strategy backtest."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        code = str(payload.get("code", ""))
        strategy_name = str(payload.get("strategy_name", ""))

        if not code.strip():
            return jsonify({"error": "empty_code"}), 400

        if not strategy_name.strip():
            return jsonify({"error": "empty_strategy_name"}), 400

        # Import strategy service
        from ml.dashboard.services.strategy_service import BacktestRequest
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)

        # Create backtest request
        backtest_request = BacktestRequest(
            strategy_code=code,
            strategy_name=strategy_name,
            start_date=str(payload.get("start_date", "2024-01-01")),
            end_date=str(payload.get("end_date", "2024-12-31")),
            initial_balance=float(payload.get("initial_balance", 100000.0)),
            instruments=list(payload.get("instruments", ["EURUSD.SIM"])),
            risk_params=dict(payload.get("risk_params", {})),
        )

        result = strategy_svc.submit_backtest(backtest_request)

        response = {
            "job_id": result.job_id,
            "status": result.status,
            "error": result.error,
        }

        # 202 Accepted for queued jobs, 400 for validation failures
        status_code = 202 if result.status == "queued" else 400

        return jsonify(response), status_code

    @app.get("/api/strategies/backtest/<job_id>/status")
    def strategies_backtest_status(job_id: str) -> tuple[Any, int]:
        """Get backtest status by job ID."""
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)
        result = strategy_svc.get_backtest_status(job_id)

        if result is None:
            return jsonify({"error": "not_found", "job_id": job_id}), 404

        response = {
            "job_id": result.job_id,
            "status": result.status,
            "error": result.error,
        }

        return jsonify(response), 200

    @app.get("/api/strategies/backtest/<job_id>/results")
    def strategies_backtest_results(job_id: str) -> tuple[Any, int]:
        """Get backtest results by job ID."""
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)
        result = strategy_svc.get_backtest_status(job_id)

        if result is None:
            return jsonify({"error": "not_found", "job_id": job_id}), 404

        if result.status != "completed":
            return jsonify({"error": "not_completed", "status": result.status}), 400

        response = {
            "job_id": result.job_id,
            "status": result.status,
            "performance_metrics": dict(result.performance_metrics),
            "trades": list(result.trades),
            "equity_curve": list(result.equity_curve),
            "execution_time_seconds": result.execution_time_seconds,
        }

        return jsonify(response), 200

    @app.post("/api/strategies/deploy")
    def strategies_deploy() -> tuple[Any, int]:
        """Deploy strategy live."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        code = str(payload.get("code", ""))
        strategy_name = str(payload.get("strategy_name", ""))
        environment = str(payload.get("environment", "staging"))

        if not code.strip():
            return jsonify({"error": "empty_code"}), 400

        if not strategy_name.strip():
            return jsonify({"error": "empty_strategy_name"}), 400

        # Import strategy service
        from ml.dashboard.services.strategy_service import DeploymentRequest
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)

        # Create deployment request
        deployment_request = DeploymentRequest(
            strategy_name=strategy_name,
            strategy_code=code,
            environment=environment,
            risk_params=dict(payload.get("risk_params", {})),
            instruments=list(payload.get("instruments", ["EURUSD.SIM"])),
        )

        result = strategy_svc.deploy_strategy(deployment_request)

        response = {
            "deployment_id": result.deployment_id,
            "status": result.status,
            "environment": result.environment,
            "message": result.message,
            "monitoring_url": result.monitoring_url,
            "error": result.error,
        }

        # 201 Created for successful deployments, 400 for failures
        status_code = 201 if result.status in ("deployed", "pending_approval") else 400

        return jsonify(response), status_code

    @app.get("/api/strategies/<strategy_id>/performance")
    def strategies_performance(strategy_id: str) -> tuple[Any, int]:
        """Get strategy performance."""
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)
        result = strategy_svc.get_strategy_performance(strategy_id)

        return jsonify(result), 200

    @app.get("/api/strategies")
    def strategies_list() -> tuple[Any, int]:
        """List all strategies."""
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)
        result = strategy_svc.list_strategies()

        return jsonify(result), 200

    # ========== END STRATEGIES ROUTES ==========

    @app.post("/api/observability/grafana/provision")
    def grafana_provision() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        title = str(payload.get("title") or "") or None
        res = svc.provision_grafana_dashboard(title=title)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/api/observability/status")
    def observability_status() -> tuple[Any, int]:
        data = svc.get_grafana_status()
        return jsonify(data), 200

    @app.get("/api/observability/summary")
    def observability_summary() -> tuple[Any, int]:
        data = svc.get_prometheus_summary()
        return jsonify(data), 200

    @app.get("/api/observability/stores")
    def observability_stores() -> tuple[Any, int]:
        data = svc.get_store_summary()
        return jsonify(data), 200

    @app.get("/health")
    def health() -> tuple[Any, int]:  # pragma: no cover - simple readiness
        return jsonify({"healthy": True}), 200

    @app.get("/metrics")
    def metrics() -> tuple[bytes, int, dict[str, str]]:  # pragma: no cover - passthrough
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    @app.get("/")
    def index() -> tuple[str, int]:
        # Check for UI preference via query param or cookie
        ui_type = request.args.get("ui") or request.cookies.get("ui_preference") or "standard"

        # Map UI types to template files
        # DEPRECATED: control, enhanced, advanced UIs are deprecated and redirect to unified
        # See reports/ui_consolidation_analysis.md for details
        template_map = {
            "unified": "index_unified.html",  # Primary UI - full control center
            "control": "index_unified.html",  # DEPRECATED: redirects to unified
            "enhanced": "index_unified.html",  # DEPRECATED: redirects to unified
            "advanced": "index_unified.html",  # DEPRECATED: redirects to unified
            "standard": "index.html",  # Minimal fallback UI
        }

        # Check if requested template exists, fallback to standard if not
        template = template_map.get(ui_type, "index.html")
        template_path = os.path.join(app.root_path, "templates", template)
        if not os.path.exists(template_path):
            template = "index.html"

        # Minimal visual UI for local/dev usage
        return (
            render_template(
                template,
                grafana_embed_enabled=cfg.grafana_embed_enabled,
                grafana_embed_urls=cfg.grafana_embed_urls(),
                grafana_dashboard_url=cfg.grafana_dashboard_url(),
                grafana_theme=cfg.grafana_embed_theme,
            ),
            200,
        )

    return app


__all__ = ["create_app"]
