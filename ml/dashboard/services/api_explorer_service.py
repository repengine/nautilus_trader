"""
API Explorer Service for Dashboard.

Provides OpenAPI/Swagger spec generation, interactive documentation,
and endpoint testing capabilities.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.dashboard.services.base_service import BaseIntegrationService


if TYPE_CHECKING:
    from flask import Flask

    from ml.core.integration import MLIntegrationManager


# ===== Metrics =====
api_explorer_requests = get_counter(
    "ml_dashboard_api_explorer_requests_total",
    "API Explorer requests by operation",
    labelnames=["operation", "status"],
)

api_explorer_latency = get_histogram(
    "ml_dashboard_api_explorer_latency_seconds",
    "API Explorer operation latency",
    labelnames=["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
)


class APIExplorerService(BaseIntegrationService):
    """Service for API exploration, documentation, and testing."""

    def __init__(
        self,
        integration_manager: MLIntegrationManager | None = None,
        flask_app: Flask | None = None,
    ) -> None:
        super().__init__(integration_manager)
        self._flask_app = flask_app
        self._openapi_spec_cache: dict[str, Any] | None = None
        self._cache_timestamp: float = 0.0
        self._cache_ttl_seconds: float = 300.0  # 5 minutes

    def get_service_name(self) -> str:
        """Return service name for metrics."""
        return "api_explorer"

    async def health_check(self) -> dict[str, Any]:
        """Return health status of API Explorer service."""
        return {
            "service": "api_explorer",
            "status": "healthy",
            "flask_app_available": self._flask_app is not None,
            "cache_valid": self._is_cache_valid(),
        }

    def set_flask_app(self, app: Flask) -> None:
        """Set the Flask app reference for introspection."""
        self._flask_app = app
        # Invalidate cache when app changes
        self._openapi_spec_cache = None

    def _is_cache_valid(self) -> bool:
        """Check if cached OpenAPI spec is still valid."""
        if self._openapi_spec_cache is None:
            return False
        elapsed = time.time() - self._cache_timestamp
        return elapsed < self._cache_ttl_seconds

    def get_openapi_spec(self) -> dict[str, Any]:
        """
        Generate OpenAPI 3.0 specification from Flask routes.

        Returns cached spec if still valid.
        """
        start_time = time.perf_counter()
        operation = "get_openapi_spec"

        try:
            # Return cached spec if valid
            if self._is_cache_valid() and self._openapi_spec_cache is not None:
                api_explorer_requests.labels(operation=operation, status="cache_hit").inc()
                return self._openapi_spec_cache

            # Generate new spec
            spec = self._generate_openapi_spec()
            self._openapi_spec_cache = spec
            self._cache_timestamp = time.time()

            api_explorer_requests.labels(operation=operation, status="success").inc()
            self._track_operation(operation=operation, status="success")

            return spec

        except Exception as e:
            api_explorer_requests.labels(operation=operation, status="error").inc()
            self._track_operation(operation=operation, status="error")
            # Return minimal fallback spec on error
            return self._get_fallback_spec(error=str(e))

        finally:
            elapsed = time.perf_counter() - start_time
            api_explorer_latency.labels(operation=operation).observe(elapsed)

    def _generate_openapi_spec(self) -> dict[str, Any]:
        """Generate complete OpenAPI 3.0.3 specification."""
        spec: dict[str, Any] = {
            "openapi": "3.0.3",
            "info": {
                "title": "Nautilus ML Dashboard API",
                "version": "1.0.0",
                "description": "REST API for Nautilus ML Platform Dashboard",
                "contact": {"name": "Nautilus ML Team"},
            },
            "servers": [{"url": "/api", "description": "Dashboard API"}],
            "components": {
                "schemas": self._get_common_schemas(),
                "securitySchemes": {
                    "BearerAuth": {"type": "http", "scheme": "bearer"},
                    "TokenHeader": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-ML-DASHBOARD-TOKEN",
                    },
                },
            },
            "tags": [
                {"name": "Health", "description": "System health and status"},
                {"name": "Registry", "description": "Model, feature, and dataset registry"},
                {"name": "Pipeline", "description": "Pipeline orchestration and jobs"},
                {"name": "Control", "description": "Actor and system control"},
                {"name": "Observability", "description": "Metrics and monitoring"},
                {"name": "API Explorer", "description": "API documentation and testing"},
            ],
            "paths": {},
        }

        # Add paths from Flask app if available
        if self._flask_app is not None:
            spec["paths"] = self._extract_paths_from_flask()

        return spec

    def _extract_paths_from_flask(self) -> dict[str, Any]:
        """Extract API paths from Flask app routes."""
        paths: dict[str, Any] = {}

        if self._flask_app is None:
            return paths

        # Iterate through all URL rules
        for rule in self._flask_app.url_map.iter_rules():
            # Only document /api/* endpoints (excluding metrics)
            if not rule.rule.startswith("/api/") or rule.rule == "/api/metrics":
                continue

            # Convert Flask route to OpenAPI path
            path = rule.rule.replace("<", "{").replace(">", "}")

            # Initialize path if not exists
            if path not in paths:
                paths[path] = {}

            # Add operations for each HTTP method
            for method in rule.methods or []:
                method_lower = method.lower()
                # Skip HEAD and OPTIONS (auto-generated)
                if method_lower in {"head", "options"}:
                    continue

                operation = self._create_operation_spec(rule, method_lower)
                paths[path][method_lower] = operation

        return paths

    def _create_operation_spec(self, rule: Any, method: str) -> dict[str, Any]:
        """Create OpenAPI operation specification for a route."""
        # Extract endpoint name and convert to summary
        endpoint = str(rule.endpoint or "")
        summary = endpoint.replace("_", " ").title()

        # Determine tags based on path
        path = rule.rule
        tags: list[str] = []
        if "/health" in path:
            tags = ["Health"]
        elif "/registry" in path:
            tags = ["Registry"]
        elif "/pipeline" in path:
            tags = ["Pipeline"]
        elif "/control" in path:
            tags = ["Control"]
        elif "/observability" in path:
            tags = ["Observability"]
        elif "/openapi" in path or "/docs" in path or "/explorer" in path:
            tags = ["API Explorer"]
        else:
            tags = ["Other"]

        # Determine if auth is required (most POST/DELETE endpoints require auth)
        requires_auth = method in {"post", "delete", "put", "patch"}

        operation: dict[str, Any] = {
            "summary": summary,
            "tags": tags,
            "responses": {
                "200": {"description": "Success", "content": {"application/json": {}}},
                "400": {"description": "Bad Request"},
                "401": {"description": "Unauthorized"},
                "404": {"description": "Not Found"},
                "500": {"description": "Internal Server Error"},
            },
        }

        if requires_auth:
            operation["security"] = [{"BearerAuth": []}, {"TokenHeader": []}]

        # Add parameters for path variables
        params = []
        # Extract path parameters from <param> or {param}
        import re

        param_pattern = r"<(?:(?:string|int|float|path):)?(\w+)>"
        for match in re.finditer(param_pattern, rule.rule):
            param_name = match.group(1)
            params.append(
                {
                    "name": param_name,
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            )

        if params:
            operation["parameters"] = params

        # Add request body for POST/PUT/PATCH
        if method in {"post", "put", "patch"}:
            operation["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"type": "object", "additionalProperties": True}
                    }
                },
            }

        return operation

    def _get_common_schemas(self) -> dict[str, Any]:
        """Return common JSON schemas used across the API."""
        return {
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                    "details": {"type": "object"},
                },
                "required": ["error"],
            },
            "SuccessResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {"type": "object"},
                },
                "required": ["success"],
            },
            "HealthStatus": {
                "type": "object",
                "properties": {
                    "healthy": {"type": "boolean"},
                    "status": {"type": "string"},
                    "components": {"type": "object"},
                },
            },
        }

    def _get_fallback_spec(self, error: str) -> dict[str, Any]:
        """Return minimal fallback OpenAPI spec when generation fails."""
        return {
            "openapi": "3.0.3",
            "info": {
                "title": "Nautilus ML Dashboard API (Fallback)",
                "version": "1.0.0",
                "description": f"OpenAPI spec generation failed: {error}",
            },
            "servers": [{"url": "/api"}],
            "paths": {},
            "components": {"schemas": {}, "securitySchemes": {}},
        }

    def get_swagger_ui_html(self) -> str:
        """
        Generate HTML for Swagger UI documentation page.

        Returns HTML that loads Swagger UI and displays the OpenAPI spec.
        """
        operation = "get_swagger_ui"
        start_time = time.perf_counter()

        try:
            html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Documentation - Nautilus ML Dashboard</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui.css">
    <style>
        body { margin: 0; padding: 0; }
        .topbar { display: none; }
        .swagger-ui .info .title { font-size: 2em; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "/api/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                defaultModelsExpandDepth: 1,
                defaultModelExpandDepth: 1,
                docExpansion: "list",
                filter: true,
                tryItOutEnabled: true
            });
        };
    </script>
</body>
</html>"""

            api_explorer_requests.labels(operation=operation, status="success").inc()
            self._track_operation(operation=operation, status="success")

            return html

        except Exception:
            api_explorer_requests.labels(operation=operation, status="error").inc()
            self._track_operation(operation=operation, status="error")
            raise

        finally:
            elapsed = time.perf_counter() - start_time
            api_explorer_latency.labels(operation=operation).observe(elapsed)

    def test_endpoint(
        self,
        method: str,
        endpoint: str,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Test an API endpoint with provided parameters.

        This is a simplified implementation that validates the request
        but does not actually execute it. Real testing should use the
        Flask test client.
        """
        operation = "test_endpoint"
        start_time = time.perf_counter()

        try:
            # Validate method
            valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH"}
            method_upper = method.upper()
            if method_upper not in valid_methods:
                api_explorer_requests.labels(operation=operation, status="invalid_method").inc()
                return {
                    "success": False,
                    "error": "invalid_method",
                    "valid_methods": list(valid_methods),
                }

            # Validate endpoint format
            if not endpoint.startswith("/"):
                endpoint = f"/{endpoint}"

            # Build test request info
            request_info = {
                "method": method_upper,
                "endpoint": endpoint,
                "headers": headers or {},
                "body": body,
            }

            # Return mock response for now
            # In a real implementation, this would use Flask test client
            result = {
                "success": True,
                "request": request_info,
                "message": "Endpoint test validated successfully",
                "note": "This is a dry-run validation. Use /api/docs for live testing.",
            }

            api_explorer_requests.labels(operation=operation, status="success").inc()
            self._track_operation(operation=operation, status="success")

            return result

        except Exception as e:
            api_explorer_requests.labels(operation=operation, status="error").inc()
            self._track_operation(operation=operation, status="error")
            return {"success": False, "error": "test_failed", "details": str(e)}

        finally:
            elapsed = time.perf_counter() - start_time
            api_explorer_latency.labels(operation=operation).observe(elapsed)

    def invalidate_cache(self) -> None:
        """Invalidate the cached OpenAPI spec."""
        self._openapi_spec_cache = None
        self._cache_timestamp = 0.0


__all__ = [
    "APIExplorerService",
]
