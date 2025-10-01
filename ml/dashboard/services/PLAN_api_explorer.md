# API Explorer Implementation Plan

## Executive Summary

The API Explorer tab provides a comprehensive interface for exploring, testing, and documenting the Nautilus ML Dashboard API. Based on analysis of the existing codebase in `/home/nate/projects/nautilus_trader/ml/dashboard/`, this plan outlines the implementation strategy for enhancing the current API Explorer functionality with OpenAPI documentation, advanced testing capabilities, and robust authentication handling.

## Current State Analysis

### Existing API Explorer Functionality
- **Location**: `/home/nate/projects/nautilus_trader/ml/dashboard/templates/index_unified.html` (lines 1040-1178)
- **Current Features**:
  - Accordion-based endpoint organization by category
  - Basic API tester with method selection (GET, POST, PUT, DELETE)
  - JSON request body editor
  - Response display with syntax highlighting
  - Pre-loaded example payloads for common endpoints

### Existing API Endpoints Structure
Based on `/home/nate/projects/nautilus_trader/ml/dashboard/app.py`, the dashboard exposes:

#### Actor Management APIs
- `POST /api/control/actors/start` - Deploy new ML actor
- `POST /api/control/actors/stop` - Stop running actor
- `GET /api/control/status` - System status

#### Pipeline Control APIs
- `POST /api/control/pipeline/trigger` - Trigger pipeline execution
- `POST /api/pipeline/run` - Run pipeline with mode
- `POST /api/orchestrator/<task>` - Trigger orchestrator task

#### Model Registry APIs
- `GET /api/registry/models` - List models
- `GET /api/registry/models/<model_id>/history` - Model performance history
- `POST /api/registry/models/<model_id>:deploy` - Deploy model
- `POST /api/registry/models/<model_id>:hot_reload` - Hot reload model
- `POST /api/registry/deployments:rollback` - Rollback deployment

#### Data Ingestion APIs
- `POST /api/control/ingestion/start` - Start data ingestion
- `POST /api/control/ingestion/backfill` - Backfill historical data
- `GET /api/registry/datasets/watermarks` - Dataset watermarks
- `GET /api/registry/datasets/lineage` - Dataset lineage

### Authentication System
- Token-based authentication via `X-ML-DASHBOARD-TOKEN` header or `Authorization: Bearer`
- Token validation in `_require_token()` function
- Support for multiple tokens with expiration timestamps
- Configuration via `DashboardToken` dataclass

## Implementation Plan

### Phase 1: OpenAPI/Swagger Integration

#### 1.1 OpenAPI Specification Generation
**Files to Create/Modify**:
- `/home/nate/projects/nautilus_trader/ml/dashboard/openapi/spec_generator.py`
- `/home/nate/projects/nautilus_trader/ml/dashboard/openapi/__init__.py`
- `/home/nate/projects/nautilus_trader/ml/dashboard/openapi/schemas.py`

**Implementation Strategy**:
```python
# spec_generator.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import inspect
from flask import Flask

@dataclass
class EndpointSpec:
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    tags: List[str]
    security: Optional[List[Dict[str, List[str]]]]

class OpenAPISpecGenerator:
    def __init__(self, app: Flask):
        self.app = app
        self.spec = self._init_base_spec()

    def _init_base_spec(self) -> Dict[str, Any]:
        return {
            "openapi": "3.0.3",
            "info": {
                "title": "Nautilus ML Dashboard API",
                "version": "1.0.0",
                "description": "REST API for Nautilus ML Platform Dashboard",
                "contact": {
                    "name": "Nautilus ML Team"
                }
            },
            "servers": [
                {"url": "/api", "description": "Dashboard API"}
            ],
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer"
                    },
                    "TokenHeader": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-ML-DASHBOARD-TOKEN"
                    }
                }
            },
            "paths": {},
            "tags": [
                {"name": "Actor Management", "description": "ML Actor deployment and control"},
                {"name": "Pipeline Control", "description": "Pipeline orchestration and execution"},
                {"name": "Model Registry", "description": "Model management and deployment"},
                {"name": "Data Ingestion", "description": "Data ingestion and backfill operations"},
                {"name": "Observability", "description": "System health and metrics"},
            ]
        }

    def generate_spec(self) -> Dict[str, Any]:
        """Generate complete OpenAPI specification from Flask routes."""
        for rule in self.app.url_map.iter_rules():
            if rule.rule.startswith('/api/'):
                self._process_route(rule)
        return self.spec
```

#### 1.2 Schema Definition
**File**: `/home/nate/projects/nautilus_trader/ml/dashboard/openapi/schemas.py`
```python
# Common request/response schemas
ACTOR_START_REQUEST = {
    "type": "object",
    "required": ["actor_id", "actor_type"],
    "properties": {
        "actor_id": {"type": "string", "example": "momentum_signal_v1"},
        "actor_type": {"type": "string", "enum": ["signal", "trading"], "example": "signal"},
        "config": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "example": "TFT_v3.2.1"},
                "signal_strategy": {"type": "string", "example": "adaptive"},
                "prediction_threshold": {"type": "number", "minimum": 0, "maximum": 1, "example": 0.75}
            }
        }
    }
}

ERROR_RESPONSE = {
    "type": "object",
    "properties": {
        "error": {"type": "string"},
        "details": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"}
    }
}
```

#### 1.3 Integration with Flask App
**Modify**: `/home/nate/projects/nautilus_trader/ml/dashboard/app.py`
```python
# Add to imports
from ml.dashboard.openapi.spec_generator import OpenAPISpecGenerator

# Add new endpoint
@app.get("/api/openapi.json")
def openapi_spec() -> tuple[Any, int]:
    """Generate OpenAPI specification."""
    generator = OpenAPISpecGenerator(app)
    spec = generator.generate_spec()
    return jsonify(spec), 200

@app.get("/api/docs")
def api_docs() -> tuple[str, int]:
    """Serve Swagger UI for API documentation."""
    return render_template("api_docs.html"), 200
```

### Phase 2: Enhanced API Tester UI

#### 2.1 Advanced Request Builder
**Enhance**: Existing API tester in unified template
```javascript
class APITester {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.history = JSON.parse(localStorage.getItem('api_test_history') || '[]');
        this.setupUI();
    }

    setupUI() {
        this.container.innerHTML = `
            <div class="api-tester-layout">
                <div class="request-builder">
                    <div class="method-url-row">
                        <select id="api-method" class="method-select">
                            <option value="GET">GET</option>
                            <option value="POST">POST</option>
                            <option value="PUT">PUT</option>
                            <option value="DELETE">DELETE</option>
                        </select>
                        <input type="text" id="api-endpoint" class="endpoint-input"
                               placeholder="/api/control/status">
                        <button id="send-request" class="btn btn-primary">Send</button>
                    </div>

                    <div class="request-tabs">
                        <button class="tab-btn active" data-tab="headers">Headers</button>
                        <button class="tab-btn" data-tab="body">Body</button>
                        <button class="tab-btn" data-tab="params">Query Params</button>
                        <button class="tab-btn" data-tab="auth">Auth</button>
                    </div>

                    <div class="tab-content active" id="headers-tab">
                        <div class="key-value-editor" id="headers-editor"></div>
                    </div>

                    <div class="tab-content" id="body-tab">
                        <select id="body-type">
                            <option value="json">JSON</option>
                            <option value="form">Form Data</option>
                            <option value="raw">Raw Text</option>
                        </select>
                        <div id="body-editor" class="code-editor"></div>
                    </div>

                    <div class="tab-content" id="params-tab">
                        <div class="key-value-editor" id="params-editor"></div>
                    </div>

                    <div class="tab-content" id="auth-tab">
                        <select id="auth-type">
                            <option value="none">No Auth</option>
                            <option value="bearer">Bearer Token</option>
                            <option value="header">Header Token</option>
                        </select>
                        <input type="text" id="auth-token" placeholder="Token value">
                    </div>
                </div>

                <div class="response-panel">
                    <div class="response-header">
                        <span class="status-code" id="response-status"></span>
                        <span class="response-time" id="response-time"></span>
                    </div>
                    <div class="response-tabs">
                        <button class="tab-btn active" data-tab="body">Response Body</button>
                        <button class="tab-btn" data-tab="headers">Headers</button>
                    </div>
                    <div class="response-content">
                        <pre id="response-body" class="response-body"></pre>
                    </div>
                </div>
            </div>
        `;
    }

    async sendRequest() {
        const method = document.getElementById('api-method').value;
        const endpoint = document.getElementById('api-endpoint').value;
        const startTime = performance.now();

        try {
            const options = this.buildRequestOptions(method);
            const response = await fetch(endpoint, options);
            const endTime = performance.now();

            this.displayResponse(response, endTime - startTime);
            this.saveToHistory({method, endpoint, options, timestamp: Date.now()});
        } catch (error) {
            this.displayError(error);
        }
    }
}
```

#### 2.2 Code Generation Features
```javascript
class CodeGenerator {
    generateCurl(method, endpoint, options) {
        let curl = `curl -X ${method}`;

        if (options.headers) {
            Object.entries(options.headers).forEach(([key, value]) => {
                curl += ` -H "${key}: ${value}"`;
            });
        }

        if (options.body) {
            curl += ` -d '${options.body}'`;
        }

        curl += ` "${window.location.origin}${endpoint}"`;
        return curl;
    }

    generatePython(method, endpoint, options) {
        return `import requests
import json

url = "${window.location.origin}${endpoint}"
headers = ${JSON.stringify(options.headers || {}, null, 2)}
${options.body ? `data = ${JSON.stringify(JSON.parse(options.body), null, 2)}` : ''}

response = requests.${method.toLowerCase()}(url, headers=headers${options.body ? ', json=data' : ''})
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")`;
    }

    generateJavascript(method, endpoint, options) {
        return `const response = await fetch("${endpoint}", {
    method: "${method}",
    headers: ${JSON.stringify(options.headers || {}, null, 2)},
    ${options.body ? `body: JSON.stringify(${JSON.parse(options.body)})` : ''}
});

const data = await response.json();
console.log("Status:", response.status);
console.log("Data:", data);`;
    }
}
```

### Phase 3: Request/Response Logging and Analytics

#### 3.1 Request Logger Service
**Create**: `/home/nate/projects/nautilus_trader/ml/dashboard/services/api_logger.py`
```python
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import sqlite3
from pathlib import Path

@dataclass
class APIRequest:
    timestamp: datetime
    method: str
    endpoint: str
    headers: Dict[str, str]
    body: Optional[str]
    query_params: Dict[str, str]
    user_agent: Optional[str]
    ip_address: str

@dataclass
class APIResponse:
    status_code: int
    headers: Dict[str, str]
    body: str
    response_time_ms: float

@dataclass
class APILogEntry:
    id: Optional[int]
    request: APIRequest
    response: Optional[APIResponse]
    session_id: Optional[str]

class APILogger:
    def __init__(self, db_path: Path = Path("api_requests.db")):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for API logging."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    method TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    headers TEXT,
                    body TEXT,
                    query_params TEXT,
                    user_agent TEXT,
                    ip_address TEXT,
                    status_code INTEGER,
                    response_headers TEXT,
                    response_body TEXT,
                    response_time_ms REAL,
                    session_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON api_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_endpoint ON api_logs(endpoint)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON api_logs(status_code)")

    def log_request(self, entry: APILogEntry) -> int:
        """Log API request/response entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO api_logs (
                    timestamp, method, endpoint, headers, body, query_params,
                    user_agent, ip_address, status_code, response_headers,
                    response_body, response_time_ms, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.request.timestamp.isoformat(),
                entry.request.method,
                entry.request.endpoint,
                json.dumps(entry.request.headers),
                entry.request.body,
                json.dumps(entry.request.query_params),
                entry.request.user_agent,
                entry.request.ip_address,
                entry.response.status_code if entry.response else None,
                json.dumps(entry.response.headers) if entry.response else None,
                entry.response.body if entry.response else None,
                entry.response.response_time_ms if entry.response else None,
                entry.session_id
            ))
            return cursor.lastrowid
```

#### 3.2 Flask Middleware Integration
**Modify**: `/home/nate/projects/nautilus_trader/ml/dashboard/app.py`
```python
from ml.dashboard.services.api_logger import APILogger, APIRequest, APIResponse, APILogEntry

# Initialize logger
api_logger = APILogger()

@app.before_request
def log_request_start():
    request.start_time = time.time()

@app.after_request
def log_request_end(response):
    if request.path.startswith('/api/'):
        end_time = time.time()
        response_time = (end_time - getattr(request, 'start_time', end_time)) * 1000

        api_request = APIRequest(
            timestamp=datetime.now(),
            method=request.method,
            endpoint=request.path,
            headers=dict(request.headers),
            body=request.get_data(as_text=True) if request.data else None,
            query_params=dict(request.args),
            user_agent=request.headers.get('User-Agent'),
            ip_address=request.remote_addr or 'unknown'
        )

        api_response = APIResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            body=response.get_data(as_text=True)[:1000],  # Truncate large responses
            response_time_ms=response_time
        )

        entry = APILogEntry(
            id=None,
            request=api_request,
            response=api_response,
            session_id=session.get('session_id')
        )

        try:
            api_logger.log_request(entry)
        except Exception as e:
            logger.warning(f"Failed to log API request: {e}")

    return response
```

### Phase 4: Rate Limiting and Throttling

#### 4.1 Rate Limiter Implementation
**Create**: `/home/nate/projects/nautilus_trader/ml/dashboard/services/rate_limiter.py`
```python
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Tuple
import threading
from functools import wraps
from flask import request, jsonify

class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

        # Rate limit configurations
        self.limits = {
            "default": (100, timedelta(minutes=1)),  # 100 requests per minute
            "auth": (10, timedelta(minutes=1)),      # 10 auth requests per minute
            "heavy": (5, timedelta(minutes=1)),      # 5 heavy operations per minute
        }

    def is_rate_limited(self, identifier: str, limit_type: str = "default") -> Tuple[bool, Dict[str, any]]:
        """Check if identifier is rate limited."""
        max_requests, window = self.limits.get(limit_type, self.limits["default"])
        now = datetime.now()
        cutoff = now - window

        with self.lock:
            # Remove old requests outside the window
            requests_queue = self.requests[f"{identifier}:{limit_type}"]
            while requests_queue and requests_queue[0] < cutoff:
                requests_queue.popleft()

            current_count = len(requests_queue)

            if current_count >= max_requests:
                # Calculate when the next request will be allowed
                oldest_request = requests_queue[0] if requests_queue else now
                retry_after = int((oldest_request + window - now).total_seconds())

                return True, {
                    "error": "rate_limit_exceeded",
                    "limit": max_requests,
                    "window_seconds": int(window.total_seconds()),
                    "retry_after": max(retry_after, 1),
                    "current_usage": current_count
                }

            # Record this request
            requests_queue.append(now)

            return False, {
                "limit": max_requests,
                "remaining": max_requests - current_count - 1,
                "reset_at": int((now + window).timestamp())
            }

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(limit_type: str = "default"):
    """Decorator to apply rate limiting to Flask routes."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Use IP address as identifier, could also use authenticated user ID
            identifier = request.remote_addr or "unknown"

            is_limited, info = rate_limiter.is_rate_limited(identifier, limit_type)

            if is_limited:
                response = jsonify(info)
                response.status_code = 429  # Too Many Requests
                response.headers['Retry-After'] = str(info['retry_after'])
                return response

            # Add rate limit headers to successful responses
            response = f(*args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit'] = str(info['limit'])
                response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                response.headers['X-RateLimit-Reset'] = str(info['reset_at'])

            return response
        return decorated_function
    return decorator
```

#### 4.2 Apply Rate Limiting to Routes
**Modify**: `/home/nate/projects/nautilus_trader/ml/dashboard/app.py`
```python
from ml.dashboard.services.rate_limiter import rate_limit

# Apply to authentication-sensitive endpoints
@app.post("/api/control/actors/start")
@rate_limit("heavy")
def control_start_actor():
    # ... existing implementation

@app.post("/api/control/emergency/stop")
@rate_limit("heavy")
def control_emergency_stop():
    # ... existing implementation

# Apply to high-frequency endpoints
@app.get("/api/health/system")
@rate_limit("default")
def health_system():
    # ... existing implementation
```

### Phase 5: API Versioning Strategy

#### 5.1 URL-Based Versioning
**Implementation**: Modify URL patterns to include version prefix
```python
# In app.py - Add versioned API routes
@app.route('/api/v1/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_v1_proxy(path):
    """Proxy v1 API requests to current implementation."""
    return redirect(f'/api/{path}', code=308)

@app.route('/api/v2/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_v2_handler(path):
    """Handle v2 API requests with enhanced features."""
    # Enhanced API implementation with additional validation,
    # improved error responses, etc.
    pass
```

#### 5.2 Content Negotiation
```python
from flask import request

def get_api_version():
    """Determine API version from Accept header or default."""
    accept_header = request.headers.get('Accept', '')
    if 'application/vnd.nautilus.v2+json' in accept_header:
        return 'v2'
    elif 'application/vnd.nautilus.v1+json' in accept_header:
        return 'v1'
    else:
        return 'v1'  # Default version

@app.before_request
def set_api_version():
    """Set API version context for request processing."""
    if request.path.startswith('/api/'):
        g.api_version = get_api_version()
```

### Phase 6: API Testing Framework

#### 6.1 Integration Test Suite
**Create**: `/home/nate/projects/nautilus_trader/ml/dashboard/tests/test_api_integration.py`
```python
import pytest
import json
from datetime import datetime
from ml.dashboard.app import create_app
from ml.dashboard.config import DashboardConfig, DashboardToken

@pytest.fixture
def app():
    """Create test Flask app."""
    config = DashboardConfig(
        auth_tokens=(DashboardToken(value="test_token"),),
        # ... other test config
    )
    return create_app(config)

@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {"X-ML-DASHBOARD-TOKEN": "test_token"}

class TestActorManagementAPI:
    def test_start_actor_success(self, client, auth_headers):
        """Test successful actor deployment."""
        payload = {
            "actor_id": "test_actor",
            "actor_type": "signal",
            "config": {
                "model_id": "test_model",
                "prediction_threshold": 0.75
            }
        }

        response = client.post(
            "/api/control/actors/start",
            headers=auth_headers,
            data=json.dumps(payload),
            content_type="application/json"
        )

        assert response.status_code == 202
        data = json.loads(response.data)
        assert data["success"] is True

    def test_start_actor_invalid_payload(self, client, auth_headers):
        """Test actor deployment with invalid payload."""
        payload = {"invalid": "payload"}

        response = client.post(
            "/api/control/actors/start",
            headers=auth_headers,
            data=json.dumps(payload),
            content_type="application/json"
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_start_actor_unauthorized(self, client):
        """Test actor deployment without authentication."""
        payload = {
            "actor_id": "test_actor",
            "actor_type": "signal"
        }

        response = client.post(
            "/api/control/actors/start",
            data=json.dumps(payload),
            content_type="application/json"
        )

        assert response.status_code == 401

class TestRateLimiting:
    def test_rate_limit_enforcement(self, client, auth_headers):
        """Test that rate limiting is enforced."""
        endpoint = "/api/control/status"

        # Make requests up to the limit
        for i in range(100):  # Assuming 100/min limit
            response = client.get(endpoint)
            if response.status_code == 429:
                break
        else:
            pytest.fail("Rate limit was not enforced")

        # Verify rate limit response
        assert response.status_code == 429
        data = json.loads(response.data)
        assert data["error"] == "rate_limit_exceeded"
        assert "retry_after" in data
```

#### 6.2 Performance Testing
**Create**: `/home/nate/projects/nautilus_trader/ml/dashboard/tests/test_api_performance.py`
```python
import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

class APIPerformanceTester:
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token

    async def test_endpoint_latency(self, endpoint: str, num_requests: int = 100):
        """Test endpoint latency with concurrent requests."""
        latencies = []

        async with aiohttp.ClientSession() as session:
            headers = {"X-ML-DASHBOARD-TOKEN": self.auth_token}

            tasks = []
            for _ in range(num_requests):
                task = self._make_request(session, endpoint, headers)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    status, latency = result
                    if status == 200:
                        latencies.append(latency)

        if latencies:
            return {
                "endpoint": endpoint,
                "total_requests": len(latencies),
                "successful_requests": len(latencies),
                "mean_latency_ms": statistics.mean(latencies) * 1000,
                "median_latency_ms": statistics.median(latencies) * 1000,
                "p95_latency_ms": self._percentile(latencies, 95) * 1000,
                "p99_latency_ms": self._percentile(latencies, 99) * 1000,
                "min_latency_ms": min(latencies) * 1000,
                "max_latency_ms": max(latencies) * 1000,
            }
        return None

    async def _make_request(self, session, endpoint, headers):
        """Make single request and measure latency."""
        start = time.time()
        try:
            async with session.get(f"{self.base_url}{endpoint}", headers=headers) as response:
                await response.text()  # Consume response body
                end = time.time()
                return response.status, end - start
        except Exception:
            return None, None

    def _percentile(self, data, percentile):
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

# Usage example
async def run_performance_tests():
    tester = APIPerformanceTester("http://localhost:8010", "test_token")

    endpoints = [
        "/api/health/system",
        "/api/registry/models",
        "/api/control/status",
        "/api/events"
    ]

    results = []
    for endpoint in endpoints:
        result = await tester.test_endpoint_latency(endpoint)
        if result:
            results.append(result)

    # Print results
    for result in results:
        print(f"\nEndpoint: {result['endpoint']}")
        print(f"Mean latency: {result['mean_latency_ms']:.2f}ms")
        print(f"P95 latency: {result['p95_latency_ms']:.2f}ms")
        print(f"P99 latency: {result['p99_latency_ms']:.2f}ms")
```

### Phase 7: Enhanced UI Features

#### 7.1 Request History and Collections
```javascript
class RequestHistory {
    constructor() {
        this.history = this.loadHistory();
        this.collections = this.loadCollections();
    }

    loadHistory() {
        const history = localStorage.getItem('api_request_history');
        return history ? JSON.parse(history) : [];
    }

    saveHistory() {
        localStorage.setItem('api_request_history', JSON.stringify(this.history));
    }

    addRequest(request) {
        const entry = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body,
            response: request.response,
            status: request.status,
            duration: request.duration
        };

        this.history.unshift(entry);

        // Keep only last 100 requests
        if (this.history.length > 100) {
            this.history = this.history.slice(0, 100);
        }

        this.saveHistory();
        this.updateHistoryUI();
    }

    createCollection(name, requests) {
        const collection = {
            id: Date.now(),
            name: name,
            created: new Date().toISOString(),
            requests: requests
        };

        this.collections.push(collection);
        this.saveCollections();
        return collection.id;
    }
}
```

#### 7.2 Response Data Visualization
```javascript
class ResponseVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }

    visualizeResponse(response, contentType) {
        if (contentType.includes('application/json')) {
            this.renderJsonTree(response);
        } else if (contentType.includes('text/csv')) {
            this.renderDataTable(response);
        } else if (contentType.includes('image/')) {
            this.renderImage(response);
        } else {
            this.renderRawText(response);
        }
    }

    renderJsonTree(jsonData) {
        // Create interactive JSON tree viewer
        const tree = this.createJsonTree(JSON.parse(jsonData));
        this.container.innerHTML = '';
        this.container.appendChild(tree);
    }

    createJsonTree(obj, level = 0) {
        const container = document.createElement('div');
        container.className = 'json-tree';

        if (typeof obj === 'object' && obj !== null) {
            Object.entries(obj).forEach(([key, value]) => {
                const item = document.createElement('div');
                item.className = 'json-item';
                item.style.paddingLeft = `${level * 20}px`;

                if (typeof value === 'object' && value !== null) {
                    const toggle = document.createElement('span');
                    toggle.textContent = '▼ ';
                    toggle.className = 'json-toggle';
                    toggle.onclick = () => this.toggleJsonNode(toggle);

                    item.appendChild(toggle);
                    item.appendChild(document.createTextNode(`"${key}": {`));

                    const nested = this.createJsonTree(value, level + 1);
                    nested.className += ' json-nested';

                    container.appendChild(item);
                    container.appendChild(nested);

                    const closing = document.createElement('div');
                    closing.style.paddingLeft = `${level * 20}px`;
                    closing.textContent = '}';
                    container.appendChild(closing);
                } else {
                    item.innerHTML = `<span class="json-key">"${key}"</span>: <span class="json-value">${JSON.stringify(value)}</span>`;
                    container.appendChild(item);
                }
            });
        }

        return container;
    }
}
```

## Security Considerations

### Authentication and Authorization
- **Token Management**: Secure token storage and rotation
- **Session Validation**: Verify token expiration and scope
- **Request Signing**: Optional HMAC request signing for critical operations
- **CORS Policy**: Restrict cross-origin requests to authorized domains

### Input Validation and Sanitization
- **JSON Schema Validation**: Validate request payloads against predefined schemas
- **SQL Injection Prevention**: Use parameterized queries in logging layer
- **XSS Prevention**: Sanitize all user inputs in UI components
- **File Upload Security**: Validate and scan uploaded files (if applicable)

### Rate Limiting and DDoS Protection
- **Adaptive Rate Limiting**: Adjust limits based on system load
- **IP-based Throttling**: Track requests per IP address
- **Token-based Limits**: Different limits for authenticated vs anonymous users
- **Circuit Breaker Pattern**: Prevent cascade failures during high load

## Monitoring and Observability

### Metrics Collection
```python
# Additional metrics for API Explorer
API_EXPLORER_METRICS = {
    "requests_total": get_counter(
        "ml_api_explorer_requests_total",
        "API Explorer requests by endpoint",
        labels=["endpoint", "method", "status"]
    ),
    "test_requests_total": get_counter(
        "ml_api_explorer_test_requests_total",
        "API test requests from UI",
        labels=["endpoint", "success"]
    ),
    "documentation_views": get_counter(
        "ml_api_explorer_doc_views_total",
        "API documentation page views",
        labels=["endpoint"]
    )
}
```

### Health Checks
```python
@app.get("/api/explorer/health")
def api_explorer_health():
    """Health check for API Explorer components."""
    checks = {
        "openapi_spec": _check_openapi_generation(),
        "rate_limiter": _check_rate_limiter(),
        "request_logger": _check_request_logger(),
        "documentation": _check_documentation_accessibility()
    }

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return jsonify({
        "healthy": all_healthy,
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }), status_code
```

## Implementation Timeline

### Phase 1 (Week 1): Foundation
- [ ] OpenAPI specification generation
- [ ] Basic schema definitions
- [ ] Integration with existing Flask app

### Phase 2 (Week 2): Enhanced UI
- [ ] Advanced request builder
- [ ] Response visualization
- [ ] Code generation features

### Phase 3 (Week 3): Logging and Analytics
- [ ] Request/response logging infrastructure
- [ ] API analytics dashboard
- [ ] Performance monitoring

### Phase 4 (Week 4): Security and Testing
- [ ] Rate limiting implementation
- [ ] Authentication enhancements
- [ ] Comprehensive test suite

### Phase 5 (Week 5): Advanced Features
- [ ] API versioning
- [ ] Request history and collections
- [ ] Documentation improvements

## File Structure Summary

```
ml/dashboard/
├── services/
│   ├── api_logger.py              # Request/response logging
│   ├── rate_limiter.py            # Rate limiting implementation
│   └── PLAN_api_explorer.md       # This file
├── openapi/
│   ├── __init__.py
│   ├── spec_generator.py          # OpenAPI spec generation
│   └── schemas.py                 # Common schemas
├── templates/
│   ├── api_docs.html              # Swagger UI template
│   └── index_unified.html         # Enhanced with API Explorer
├── tests/
│   ├── test_api_integration.py    # Integration tests
│   └── test_api_performance.py    # Performance tests
├── app.py                         # Enhanced with new endpoints
└── config.py                      # Configuration enhancements
```

This implementation plan provides a comprehensive roadmap for implementing a production-ready API Explorer with robust authentication, monitoring, and testing capabilities while integrating seamlessly with the existing Nautilus ML Dashboard architecture.