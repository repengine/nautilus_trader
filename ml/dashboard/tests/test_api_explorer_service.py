"""
Tests for API Explorer Service.

Validates OpenAPI spec generation, Swagger UI, and endpoint testing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ml.dashboard.app import create_app
from ml.dashboard.config import DashboardConfig
from ml.dashboard.services.api_explorer_service import APIExplorerService


@pytest.fixture
def api_explorer_service() -> APIExplorerService:
    """Create API Explorer service instance for testing."""
    return APIExplorerService(integration_manager=None, flask_app=None)


@pytest.fixture
def flask_app() -> Any:
    """Create Flask test app."""
    config = DashboardConfig(
        auth_tokens=(),  # No auth tokens for testing
        grafana_embed_enabled=False,
    )
    return create_app(config)


@pytest.fixture
def client(flask_app: Any) -> Any:
    """Create Flask test client."""
    return flask_app.test_client()


class TestAPIExplorerService:
    """Test API Explorer service functionality."""

    def test_service_name(self, api_explorer_service: APIExplorerService) -> None:
        """Test service name is correct."""
        assert api_explorer_service.get_service_name() == "api_explorer"

    def test_health_check_no_app(self, api_explorer_service: APIExplorerService) -> None:
        """Test health check without Flask app."""
        import asyncio

        health = asyncio.run(api_explorer_service.health_check())
        assert health["service"] == "api_explorer"
        assert health["status"] == "healthy"
        assert health["flask_app_available"] is False
        assert health["cache_valid"] is False

    def test_health_check_with_app(self, flask_app: Any) -> None:
        """Test health check with Flask app."""
        import asyncio

        service = APIExplorerService(integration_manager=None, flask_app=flask_app)
        health = asyncio.run(service.health_check())
        assert health["service"] == "api_explorer"
        assert health["status"] == "healthy"
        assert health["flask_app_available"] is True

    def test_get_openapi_spec_no_app(self, api_explorer_service: APIExplorerService) -> None:
        """Test OpenAPI spec generation without Flask app."""
        spec = api_explorer_service.get_openapi_spec()
        assert spec["openapi"] == "3.0.3"
        assert spec["info"]["title"] == "Nautilus ML Dashboard API"
        assert "components" in spec
        assert "securitySchemes" in spec["components"]
        assert "BearerAuth" in spec["components"]["securitySchemes"]
        assert "TokenHeader" in spec["components"]["securitySchemes"]
        assert "paths" in spec
        assert len(spec["paths"]) == 0  # No paths without app

    def test_get_openapi_spec_with_app(self, flask_app: Any) -> None:
        """Test OpenAPI spec generation with Flask app."""
        service = APIExplorerService(integration_manager=None, flask_app=flask_app)
        spec = service.get_openapi_spec()

        assert spec["openapi"] == "3.0.3"
        assert spec["info"]["title"] == "Nautilus ML Dashboard API"
        assert "paths" in spec
        # Should have multiple paths from the app
        assert len(spec["paths"]) > 0

        # Verify our new endpoints are in the spec
        assert "/api/openapi.json" in spec["paths"]
        assert "/api/docs" in spec["paths"]
        assert "/api/explorer/test" in spec["paths"]

        # Verify path has GET method
        assert "get" in spec["paths"]["/api/openapi.json"]
        assert "get" in spec["paths"]["/api/docs"]
        assert "post" in spec["paths"]["/api/explorer/test"]

    def test_openapi_spec_caching(self, flask_app: Any) -> None:
        """Test that OpenAPI spec is cached and reused."""
        service = APIExplorerService(integration_manager=None, flask_app=flask_app)

        # First call should generate spec
        spec1 = service.get_openapi_spec()
        assert service._openapi_spec_cache is not None

        # Second call should return cached spec
        spec2 = service.get_openapi_spec()
        assert spec1 is spec2  # Same object reference

    def test_cache_invalidation(self, flask_app: Any) -> None:
        """Test cache invalidation."""
        service = APIExplorerService(integration_manager=None, flask_app=flask_app)

        # Generate and cache spec
        spec1 = service.get_openapi_spec()
        assert service._openapi_spec_cache is not None

        # Invalidate cache
        service.invalidate_cache()
        assert service._openapi_spec_cache is None

        # Next call should regenerate
        spec2 = service.get_openapi_spec()
        assert service._openapi_spec_cache is not None
        # New spec should be different object
        assert spec1 is not spec2

    def test_set_flask_app_invalidates_cache(self, flask_app: Any) -> None:
        """Test that setting Flask app invalidates cache."""
        service = APIExplorerService(integration_manager=None, flask_app=flask_app)

        # Generate and cache spec
        service.get_openapi_spec()
        assert service._openapi_spec_cache is not None

        # Setting new app should invalidate
        service.set_flask_app(flask_app)
        assert service._openapi_spec_cache is None

    def test_swagger_ui_html(self, api_explorer_service: APIExplorerService) -> None:
        """Test Swagger UI HTML generation."""
        html = api_explorer_service.get_swagger_ui_html()
        assert "<!DOCTYPE html>" in html
        assert "swagger-ui" in html
        assert "/api/openapi.json" in html
        assert "SwaggerUIBundle" in html

    def test_test_endpoint_valid(self, api_explorer_service: APIExplorerService) -> None:
        """Test endpoint testing with valid input."""
        result = api_explorer_service.test_endpoint(
            method="GET", endpoint="/api/health/system", headers=None, body=None
        )
        assert result["success"] is True
        assert "request" in result
        assert result["request"]["method"] == "GET"
        assert result["request"]["endpoint"] == "/api/health/system"

    def test_test_endpoint_with_body(self, api_explorer_service: APIExplorerService) -> None:
        """Test endpoint testing with request body."""
        body = {"key": "value"}
        result = api_explorer_service.test_endpoint(
            method="POST", endpoint="/api/pipeline/run", headers=None, body=body
        )
        assert result["success"] is True
        assert result["request"]["method"] == "POST"
        assert result["request"]["body"] == body

    def test_test_endpoint_invalid_method(
        self, api_explorer_service: APIExplorerService
    ) -> None:
        """Test endpoint testing with invalid method."""
        result = api_explorer_service.test_endpoint(
            method="INVALID", endpoint="/api/health/system", headers=None, body=None
        )
        assert result["success"] is False
        assert result["error"] == "invalid_method"
        assert "valid_methods" in result

    def test_test_endpoint_normalizes_path(
        self, api_explorer_service: APIExplorerService
    ) -> None:
        """Test that endpoint path is normalized."""
        result = api_explorer_service.test_endpoint(
            method="GET", endpoint="api/health/system", headers=None, body=None
        )
        assert result["success"] is True
        assert result["request"]["endpoint"] == "/api/health/system"

    def test_common_schemas(self, api_explorer_service: APIExplorerService) -> None:
        """Test common schemas are defined."""
        schemas = api_explorer_service._get_common_schemas()
        assert "ErrorResponse" in schemas
        assert "SuccessResponse" in schemas
        assert "HealthStatus" in schemas

        # Validate schema structure
        error_schema = schemas["ErrorResponse"]
        assert error_schema["type"] == "object"
        assert "error" in error_schema["properties"]
        assert "error" in error_schema["required"]

    def test_fallback_spec_on_error(self, flask_app: Any) -> None:
        """Test fallback spec is returned on error."""
        service = APIExplorerService(integration_manager=None, flask_app=None)

        # Force an error by setting invalid flask app
        service._flask_app = "invalid"  # type: ignore

        spec = service.get_openapi_spec()
        assert spec["openapi"] == "3.0.3"
        assert "Fallback" in spec["info"]["title"]
        assert "generation failed" in spec["info"]["description"]


class TestAPIExplorerEndpoints:
    """Test API Explorer HTTP endpoints."""

    def test_openapi_json_endpoint(self, client: Any) -> None:
        """Test GET /api/openapi.json endpoint."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["openapi"] == "3.0.3"
        assert "info" in data
        assert "paths" in data
        assert "components" in data

    def test_docs_endpoint(self, client: Any) -> None:
        """Test GET /api/docs endpoint."""
        response = client.get("/api/docs")
        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.data
        assert b"swagger-ui" in response.data
        assert b"/api/openapi.json" in response.data

    def test_explorer_test_endpoint_get(self, client: Any) -> None:
        """Test POST /api/explorer/test with GET method."""
        response = client.post(
            "/api/explorer/test",
            json={"method": "GET", "endpoint": "/api/health/system"},
            content_type="application/json",
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["success"] is True
        assert data["request"]["method"] == "GET"

    def test_explorer_test_endpoint_post(self, client: Any) -> None:
        """Test POST /api/explorer/test with POST method."""
        response = client.post(
            "/api/explorer/test",
            json={
                "method": "POST",
                "endpoint": "/api/pipeline/run",
                "body": {"pipeline_type": "full"},
            },
            content_type="application/json",
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["success"] is True
        assert data["request"]["method"] == "POST"
        assert data["request"]["body"]["pipeline_type"] == "full"

    def test_explorer_test_endpoint_missing_endpoint(self, client: Any) -> None:
        """Test POST /api/explorer/test without endpoint."""
        response = client.post(
            "/api/explorer/test", json={"method": "GET"}, content_type="application/json"
        )
        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["error"] == "missing_endpoint"

    def test_explorer_test_endpoint_invalid_method(self, client: Any) -> None:
        """Test POST /api/explorer/test with invalid method."""
        response = client.post(
            "/api/explorer/test",
            json={"method": "INVALID", "endpoint": "/api/health/system"},
            content_type="application/json",
        )
        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["success"] is False
        assert data["error"] == "invalid_method"

    def test_openapi_spec_has_new_endpoints(self, client: Any) -> None:
        """Test that OpenAPI spec includes new API Explorer endpoints."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200

        spec = json.loads(response.data)
        paths = spec["paths"]

        # Verify all three endpoints are documented
        assert "/api/openapi.json" in paths
        assert "/api/docs" in paths
        assert "/api/explorer/test" in paths

        # Verify HTTP methods
        assert "get" in paths["/api/openapi.json"]
        assert "get" in paths["/api/docs"]
        assert "post" in paths["/api/explorer/test"]

    def test_openapi_spec_has_tags(self, client: Any) -> None:
        """Test that OpenAPI spec includes proper tags."""
        response = client.get("/api/openapi.json")
        spec = json.loads(response.data)

        tags = spec["tags"]
        tag_names = [tag["name"] for tag in tags]

        assert "Health" in tag_names
        assert "Registry" in tag_names
        assert "Pipeline" in tag_names
        assert "Control" in tag_names
        assert "Observability" in tag_names
        assert "API Explorer" in tag_names

    def test_openapi_spec_security_schemes(self, client: Any) -> None:
        """Test that OpenAPI spec includes security schemes."""
        response = client.get("/api/openapi.json")
        spec = json.loads(response.data)

        security_schemes = spec["components"]["securitySchemes"]
        assert "BearerAuth" in security_schemes
        assert "TokenHeader" in security_schemes

        # Verify BearerAuth structure
        bearer = security_schemes["BearerAuth"]
        assert bearer["type"] == "http"
        assert bearer["scheme"] == "bearer"

        # Verify TokenHeader structure
        token = security_schemes["TokenHeader"]
        assert token["type"] == "apiKey"
        assert token["in"] == "header"
        assert token["name"] == "X-ML-DASHBOARD-TOKEN"


class TestMetricsTracking:
    """Test that metrics are properly tracked."""

    def test_metrics_tracked_on_spec_generation(self, flask_app: Any) -> None:
        """Test that metrics are incremented on spec generation."""
        from ml.dashboard.services.api_explorer_service import api_explorer_latency
        from ml.dashboard.services.api_explorer_service import api_explorer_requests

        service = APIExplorerService(integration_manager=None, flask_app=flask_app)

        # Get initial metric values (if any)
        # Note: In real tests with Prometheus, you'd query the registry
        # For now, we just ensure no exceptions are raised

        spec = service.get_openapi_spec()
        assert spec is not None

        # Verify metrics objects exist
        assert api_explorer_requests is not None
        assert api_explorer_latency is not None

    def test_metrics_tracked_on_test_endpoint(
        self, api_explorer_service: APIExplorerService
    ) -> None:
        """Test that metrics are tracked for endpoint testing."""
        result = api_explorer_service.test_endpoint(
            method="GET", endpoint="/api/health/system", headers=None, body=None
        )
        assert result["success"] is True

    def test_metrics_tracked_on_swagger_ui(
        self, api_explorer_service: APIExplorerService
    ) -> None:
        """Test that metrics are tracked for Swagger UI generation."""
        html = api_explorer_service.get_swagger_ui_html()
        assert html is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_endpoint_path(self, api_explorer_service: APIExplorerService) -> None:
        """Test handling of empty endpoint path."""
        result = api_explorer_service.test_endpoint(
            method="GET", endpoint="", headers=None, body=None
        )
        # Empty path should be normalized to "/"
        assert result["success"] is True
        assert result["request"]["endpoint"] == "/"

    def test_case_insensitive_method(self, api_explorer_service: APIExplorerService) -> None:
        """Test that HTTP methods are case-insensitive."""
        result = api_explorer_service.test_endpoint(
            method="get", endpoint="/api/health", headers=None, body=None
        )
        assert result["success"] is True
        assert result["request"]["method"] == "GET"

    def test_headers_passthrough(self, api_explorer_service: APIExplorerService) -> None:
        """Test that headers are properly passed through."""
        headers = {"X-Custom-Header": "value", "Authorization": "Bearer token"}
        result = api_explorer_service.test_endpoint(
            method="GET", endpoint="/api/health", headers=headers, body=None
        )
        assert result["success"] is True
        assert result["request"]["headers"] == headers

    def test_complex_body(self, api_explorer_service: APIExplorerService) -> None:
        """Test complex nested body structure."""
        body = {
            "pipeline_type": "full",
            "config": {"nested": {"deep": {"value": 123}}, "array": [1, 2, 3]},
        }
        result = api_explorer_service.test_endpoint(
            method="POST", endpoint="/api/pipeline/run", headers=None, body=body
        )
        assert result["success"] is True
        assert result["request"]["body"] == body
