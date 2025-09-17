"""
Tiny HTTP server factory for /health and /metrics endpoints (cold path).

Used by deployment entrypoints to expose health and Prometheus metrics without
direct prometheus_client imports (uses ml.common.metrics_export facade).
"""

from __future__ import annotations

from collections.abc import Callable

from flask import Flask
from flask import Response
from flask import jsonify

from ml.common.metrics_export import CONTENT_TYPE_LATEST
from ml.common.metrics_export import generate_latest


def build_app(is_healthy: Callable[[], bool]) -> Flask:
    """
    Create a Flask app with /health and /metrics endpoints.

    Parameters
    ----------
    is_healthy : Callable[[], bool]
        A callable returning True when the service is healthy.
    """
    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[Response, int]:
        status = {"healthy": bool(is_healthy())}
        return jsonify(status), 200 if status["healthy"] else 503

    @app.get("/metrics")
    def metrics() -> Response:  # pragma: no cover - trivial passthrough
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    return app


__all__ = ["build_app"]
