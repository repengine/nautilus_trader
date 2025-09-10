from __future__ import annotations

import os
from contextlib import contextmanager

from ml.core.integration import MLIntegrationManager
from ml.observability.bootstrap import auto_start_if_configured


@contextmanager
def env_vars(vars: dict[str, str]):
    old = {k: os.environ.get(k) for k in vars}
    try:
        os.environ.update(vars)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_auto_start_if_configured_starts_flusher(tmp_path) -> None:
    mgr = object.__new__(MLIntegrationManager)  # type: ignore[misc]
    with env_vars(
        {
            "ML_OBS_SINK": "db",
            "ML_OBS_DB_URL": f"sqlite:///{tmp_path}/obs.db",
            "ML_OBS_INTERVAL_SECONDS": "0.01",
        },
    ):
        auto_start_if_configured(mgr)
        assert getattr(mgr, "_obs_flusher", None) is not None
        MLIntegrationManager.stop_observability_flush(mgr)
