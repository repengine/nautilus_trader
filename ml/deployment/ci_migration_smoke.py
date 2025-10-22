from __future__ import annotations

import logging
import time
from pathlib import Path

from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command


logger = logging.getLogger(__name__)

COMPOSE_FILE = Path("ml/deployment/docker-compose.yml")


def _compose(*args: str, timeout: float | None = None) -> None:
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), *args]
    run_command(cmd, timeout=timeout, log=logger)


def wait_for_postgres(timeout: int = 30) -> None:
    start = time.time()
    while time.time() - start < timeout:
        result = run_command(
            [
                "docker",
                "compose",
                "-f",
                str(COMPOSE_FILE),
                "exec",
                "-T",
                "postgres",
                "pg_isready",
                "-U",
                "postgres",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            log=logger,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("PostgreSQL did not become ready in time")


def check_views() -> None:
    sql = """
    SELECT 1 FROM pg_views WHERE viewname IN (
        'pipeline_health',
        'data_collection_stats',
        'model_performance_summary',
        'strategy_signal_summary'
    ) AND schemaname = 'ml';
    """
    cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "postgres",
        "nautilus",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]
    try:
        run_command(cmd, timeout=30, log=logger)
    except SubprocessExecutionError as exc:
        raise RuntimeError(f"View validation failed: {exc}") from exc


def main() -> None:
    # Start postgres only
    _compose("up", "-d", "postgres", timeout=30)
    wait_for_postgres()

    # Apply migrations
    from ml.deployment.migrations import apply_migrations_via_compose

    apply_migrations_via_compose(compose_file=COMPOSE_FILE)

    # Validate core views exist
    check_views()

    print("Migration smoke OK")


if __name__ == "__main__":  # pragma: no cover - CI helper
    main()
