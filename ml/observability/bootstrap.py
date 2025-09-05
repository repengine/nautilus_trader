from __future__ import annotations

import os

from ml.core.integration import MLIntegrationManager


def auto_start_if_configured(mgr: MLIntegrationManager) -> None:
    """
    Auto-start background observability flushing based on env config.

    Reads `ML_OBS_SINK`, `ML_OBS_BASE_PATH`, `ML_OBS_FILE_FORMAT`, `ML_OBS_DB_URL`,
    and `ML_OBS_INTERVAL_SECONDS`. If any are present, starts background flushing
    accordingly. Safe to call at startup; no-op if misconfigured.
    """
    # Quick check to avoid touching env if not necessary
    if not any(k in os.environ for k in ("ML_OBS_SINK", "ML_OBS_DB_URL", "ML_OBS_BASE_PATH")):
        return None
    try:
        from ml.config.observability import ObservabilityConfig

        cfg = ObservabilityConfig.from_env()
        MLIntegrationManager.start_observability_from_config(mgr, cfg)
    except Exception:
        return None
